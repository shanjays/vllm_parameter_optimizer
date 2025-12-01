import csv
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Callable, Any, Dict


class TrainingCallback(BaseCallback):
    """
    Custom callback for per-step logging during PPO training.
    
    This callback integrates with TrainingLogger to capture per-step metrics
    during the exploration phase, enabling detailed training analysis.
    """
    
    def __init__(
        self,
        training_logger: Optional[Any] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: int = 0
    ):
        """
        Initialize the training callback.
        
        Args:
            training_logger: Optional TrainingLogger instance for structured logging
            step_callback: Optional callable invoked after each step with step data
            verbose: Verbosity level (0 = silent, 1 = info)
        """
        super().__init__(verbose)
        self.training_logger = training_logger
        self.step_callback = step_callback
        self._step_count = 0
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop
        """
        self._step_count += 1
        
        # Collect step data
        step_data = {
            'step': self._step_count,
            'num_timesteps': self.num_timesteps,
        }
        
        # Get reward from last step (if available)
        if len(self.locals.get('rewards', [])) > 0:
            step_data['reward'] = float(self.locals['rewards'][-1])
        
        # Get info from environment (may contain config and metrics)
        infos = self.locals.get('infos', [])
        if infos and isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if isinstance(info, dict):
                step_data.update(info)
        
        # Log to TrainingLogger if available
        if self.training_logger is not None:
            self.training_logger.log_step(
                step=self._step_count,
                reward=step_data.get('reward', 0.0),
                config=step_data.get('config'),
                metrics=step_data.get('metrics'),
                token_count=step_data.get('token_count'),
            )
        
        # Invoke custom callback if provided
        if self.step_callback is not None:
            self.step_callback(step_data)
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f"[TrainingCallback] Training ended after {self._step_count} steps")


class EarlyStoppingCallback(BaseCallback):
    """Stop training when reward stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.01, verbose: int = 1):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of steps without improvement before stopping
            min_delta: Minimum improvement required to reset patience counter
            verbose: Verbosity level (0 = silent, 1 = info)
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            True to continue training, False to stop early
        """
        infos = self.locals.get('infos', [])
        if infos and isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if isinstance(info, dict):
                reward = info.get('reward', 0)
                if reward > self.best_reward + self.min_delta:
                    self.best_reward = reward
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print(f"[EarlyStopping] Stopping after {self.patience} steps without improvement")
                    return False
        return True


class ExplorationAgent:
    """
    Exploration Agent for kernel configuration optimization.

    This agent uses Proximal Policy Optimization (PPO) to explore the
    kernel configuration search space. It learns to select configurations
    that maximize the objective function defined by the meta-controller.

    The agent's neural network (MlpPolicy) runs on CPU to avoid GPU conflicts
    with the profiling worker that executes benchmarks on dedicated GPUs.

    References:
        Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    
    Note:
        We use small n_steps because each environment step runs NCU profiler
        which takes ~30 seconds. Default n_steps=2048 would cause 17+ hour waits!
    """
    def __init__(self, env, log_dir="./logs/exploration_agent/", device=None,
                 training_logger=None, load_existing: bool = True):
        """
        Initialize the exploration agent.
        
        Args:
            env: The kernel tuning environment
            log_dir: Directory for logs and model checkpoints
            device: Device for training ('cpu', 'cuda', or None for auto-detect)
            training_logger: Optional TrainingLogger for structured metrics logging
            load_existing: If True, load existing model if available
        """
        self.log_dir = log_dir
        self.model_path = os.path.join(self.log_dir, "exploration_ppo_model.zip")
        self.best_model_path = os.path.join(self.log_dir, "exploration_ppo_best.zip")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Store training logger for callback integration
        self.training_logger = training_logger
        
        # Wrap the kernel tuning environment in a format SB3 understands
        self.env = DummyVecEnv([lambda: env])
        
        # Ensure the agent trains on the correct device
        # If device is explicitly passed, use that; otherwise auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.best_reward = float('-inf')
        
        # TRY TO LOAD EXISTING MODEL
        if load_existing and os.path.exists(self.model_path):
            print(f"[ExplorationAgent] Loading existing model from {self.model_path}")
            try:
                # First check if action spaces match by loading without environment
                loaded_model = PPO.load(self.model_path, env=None, device=device)
                
                # Get action space dimensions
                current_action_space = self.env.action_space
                
                # Check if they're compatible using structural comparison
                if hasattr(loaded_model, 'action_space'):
                    saved_action_space = loaded_model.action_space
                    spaces_compatible = self._check_action_space_compatibility(
                        saved_action_space, current_action_space
                    )
                    if not spaces_compatible:
                        print(f"[ExplorationAgent] Action space changed: {saved_action_space} -> {current_action_space}")
                        print(f"[ExplorationAgent] Creating new model (can't reuse old weights)")
                        self._create_new_model(device)
                    else:
                        # Action spaces match, load with environment
                        self.model = PPO.load(
                            self.model_path,
                            env=self.env,
                            device=device,
                            tensorboard_log=self.log_dir,
                        )
                        print(f"[ExplorationAgent] Model loaded successfully!")
                        self._load_best_reward()
                else:
                    # Can't check, try loading directly
                    self.model = PPO.load(
                        self.model_path,
                        env=self.env,
                        device=device,
                        tensorboard_log=self.log_dir,
                    )
                    print(f"[ExplorationAgent] Model loaded successfully!")
                    self._load_best_reward()
                    
            except Exception as e:
                print(f"[ExplorationAgent] Failed to load: {e}, creating new")
                self._create_new_model(device)
        else:
            print(f"[ExplorationAgent] Creating new PPO agent")
            self._create_new_model(device)
        
        # Training history for analysis
        self.training_history = []
    
    def _check_action_space_compatibility(self, saved_space, current_space) -> bool:
        """
        Check if two action spaces are compatible for model loading.
        
        Compares the type and structure of the spaces to determine if a saved
        model can be loaded with the current environment's action space.
        
        Args:
            saved_space: Action space from the saved model
            current_space: Action space from the current environment
            
        Returns:
            True if the spaces are compatible, False otherwise
        """
        # Check if types match
        if type(saved_space) != type(current_space):
            return False
        
        # For Box spaces (continuous), check shape and bounds
        if hasattr(saved_space, 'shape') and hasattr(current_space, 'shape'):
            if saved_space.shape != current_space.shape:
                return False
        
        # For Discrete spaces, check n (number of actions)
        if hasattr(saved_space, 'n') and hasattr(current_space, 'n'):
            if saved_space.n != current_space.n:
                return False
        
        # For MultiDiscrete spaces, check nvec
        if hasattr(saved_space, 'nvec') and hasattr(current_space, 'nvec'):
            import numpy as np
            if not np.array_equal(saved_space.nvec, current_space.nvec):
                return False
        
        return True

    def _create_new_model(self, device: str) -> None:
        """
        Create a new PPO model.
        
        Args:
            device: Device for training ('cpu' or 'cuda')
        """
        print(f"[ExplorationAgent] Initializing PPO agent on device: {device}")
        
        # CRITICAL: Use small n_steps for NCU-based environments
        # Each step takes ~30 seconds (NCU profiler), so we need tiny buffers
        # Default n_steps=2048 would take 17+ hours to fill!
        # Optimized for 5-hour aggressive training: n_steps=6, batch_size=6
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            device=device,
            n_steps=6,        # Collect 6 samples before update (optimized for 5-hour run)
            batch_size=6,     # Match batch size to n_steps
            n_epochs=2,       # Only 2 epochs per update (was 10)
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,    # Small entropy bonus for exploration
        )
    
    def _load_best_reward(self) -> None:
        """Load best reward from file."""
        path = os.path.join(self.log_dir, "best_reward.txt")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.best_reward = float(f.read().strip())
            print(f"[ExplorationAgent] Best reward: {self.best_reward:.2f}")
    
    def save_best_if_improved(self, current_reward: float) -> bool:
        """
        Save model if current reward is best.
        
        Args:
            current_reward: The current reward value to compare
            
        Returns:
            True if this was a new best reward and model was saved, False otherwise
        """
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.model.save(self.best_model_path)
            with open(os.path.join(self.log_dir, "best_reward.txt"), 'w') as f:
                f.write(str(self.best_reward))
            print(f"[ExplorationAgent] New best! Reward: {self.best_reward:.2f}")
            return True
        return False
    
    def load_best(self) -> bool:
        """
        Load best saved model.
        
        Returns:
            True if best model was loaded successfully, False otherwise
        """
        if os.path.exists(self.best_model_path):
            self.model = PPO.load(self.best_model_path, env=self.env, device=self.device)
            return True
        return False
    
    @classmethod
    def load_for_inference(cls, model_path: str, env, device: Optional[str] = None):
        """
        Load trained model for inference.
        
        Args:
            model_path: Path to the saved model file
            env: Environment to use with the model
            device: Device to load model on (auto-detect if None)
            
        Returns:
            Loaded PPO model ready for inference
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return PPO.load(model_path, env=env, device=device)
    
    def predict(self, observation, deterministic: bool = True):
        """
        Use trained model to predict action.
        
        Args:
            observation: Current environment observation
            deterministic: If True, use deterministic actions (no exploration)
            
        Returns:
            Tuple of (action, info_dict) where info_dict contains metadata
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, {"deterministic": deterministic}

    def train_epoch(self, steps=100, callback=None):
        """
        Run exploration training for one epoch.
        
        With n_steps=4, this means:
        - steps=10 → 2-3 policy updates, ~10 NCU runs (~5 minutes)
        - steps=20 → 5 policy updates, ~20 NCU runs (~10 minutes)
        
        Args:
            steps: Total number of environment steps for this epoch
            callback: Optional custom callback or list of callbacks
        """
        print(f"[ExplorationAgent] Training on {self.model.device.type} for {steps} steps...")
        
        # Build callback list
        callbacks = []
        
        # Add TrainingLogger callback if logger is configured
        if self.training_logger is not None:
            callbacks.append(TrainingCallback(
                training_logger=self.training_logger,
                verbose=1
            ))
        
        # Add user-provided callback(s)
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)
        
        # `reset_num_timesteps=False` allows TensorBoard to have a continuous x-axis
        self.model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=callbacks if callbacks else None
        )
        self.model.save(self.model_path)
        print(f"[ExplorationAgent] Epoch complete. Model checkpoint saved.")

    def save_training_history(self, path: Optional[str] = None) -> str:
        """
        Save training history to CSV file.
        
        Args:
            path: Optional path to save to. If None, uses default location.
            
        Returns:
            Path to the saved CSV file
        """
        if path is None:
            path = os.path.join(self.log_dir, "training_history.csv")
        
        # If training logger has entries, save those
        if self.training_logger is not None and self.training_logger.entries:
            return self.training_logger.save_csv(path)
        
        # Otherwise save basic training history
        if not self.training_history:
            print("[ExplorationAgent] No training history to save")
            return path
            
        with open(path, 'w', newline='') as f:
            if self.training_history:
                writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
                writer.writeheader()
                writer.writerows(self.training_history)
        
        print(f"[ExplorationAgent] Saved training history to: {path}")
        return path

    def close(self):
        """Clean up resources."""
        # Save training history before closing
        if self.training_logger is not None:
            try:
                self.training_logger.save()
            except Exception as e:
                print(f"[ExplorationAgent] Warning: Could not save training log: {e}")
        
        try:
            self.env.close()
        except Exception:
            pass

    def set_env(self, env):
        """Updates the environment for the agent."""
        self.env = DummyVecEnv([lambda: env])
        self.model.set_env(self.env)
        print("[ExplorationAgent] Environment updated.")


# Backward compatibility alias
FighterPilot = ExplorationAgent