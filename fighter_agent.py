import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class FighterPilot:
    """
    This class is a simple wrapper around the Stable-Baselines3 PPO agent.
    Its neural network (MlpPolicy) lives and trains on CPU to avoid GPU conflicts.
    
    IMPORTANT: We use small n_steps because each environment step runs NCU profiler
    which takes ~30 seconds. Default n_steps=2048 would cause 17+ hour waits!
    """
    def __init__(self, env, log_dir="./hakt_logs/fighter_pilot/", device=None):
        self.log_dir = log_dir
        self.model_path = os.path.join(self.log_dir, "hakt_ppo_model.zip")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Wrap the 'Fast Gym' in a format SB3 understands
        self.env = DummyVecEnv([lambda: env])
        
        # Ensure the agent trains on the correct device
        # If device is explicitly passed, use that; otherwise auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FighterPilot] Initializing PPO agent on device: {device}")
        
        # CRITICAL FIX: Use small n_steps for NCU-based environments
        # Each step takes ~30 seconds (NCU profiler), so we need tiny buffers
        # Default n_steps=2048 would take 17+ hours to fill!
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            device=device,
            n_steps=4,        # Collect only 4 samples before update (was 2048!)
            batch_size=4,     # Match batch size to n_steps
            n_epochs=2,       # Only 2 epochs per update (was 10)
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,    # Small entropy bonus for exploration
        )

    def train_epoch(self, steps=100):
        """
        Runs the "Fast Loop" training for one epoch.
        
        With n_steps=4, this means:
        - steps=10 → 2-3 policy updates, ~10 NCU runs (~5 minutes)
        - steps=20 → 5 policy updates, ~20 NCU runs (~10 minutes)
        """
        print(f"[FighterPilot] Training on {self.model.device.type} for {steps} steps...")
        # `reset_num_timesteps=False` allows TensorBoard to have a continuous x-axis
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False)
        self.model.save(self.model_path)
        print(f"[FighterPilot] Epoch complete. Model checkpoint saved.")

    def close(self):
        """Clean up resources."""
        try:
            self.env.close()
        except Exception:
            pass

    def set_env(self, env):
        """Updates the environment for the agent."""
        self.env = DummyVecEnv([lambda: env])
        self.model.set_env(self.env)
        print("[FighterPilot] Environment updated.")