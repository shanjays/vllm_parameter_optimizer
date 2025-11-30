import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class FighterPilot:
    """
    This class is a simple wrapper around the Stable-Baselines3 PPO agent.
    Its neural network (MlpPolicy) lives and trains on GPU 0.
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
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            device=device 
        )

    def train_epoch(self, steps=100):
        """
        Runs the "Fast Loop" training for one epoch.
        The agent will run, learn, and update its own policy on GPU 0.
        """
        print(f"[FighterPilot] Training on {self.model.device.type} for {steps} steps...")
        # `reset_num_timesteps=False` allows TensorBoard to have a continuous x-axis
        self.model.learn(total_timesteps=steps, reset_num_timesteps=False)
        self.model.save(self.model_path)
        print(f"[FighterPilot] Epoch complete. Model checkpoint saved.")

    def set_env(self, env):
        """
        Updates the environment for the agent. This is not strictly
        needed for PPO if the observation/action spaces don't change,
        but it's good practice.
        """
        self.env = DummyVecEnv([lambda: env])
        self.model.set_env(self.env)
        print("[FighterPilot] Environment updated.")