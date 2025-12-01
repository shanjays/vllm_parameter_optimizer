"""
Tests for ExplorationAgent PPO persistence functionality.

Tests the following features:
1. load_existing parameter in __init__
2. _create_new_model() method
3. _load_best_reward() method
4. save_best_if_improved() method
5. load_best() method
6. load_for_inference() class method
7. predict() method
8. EarlyStoppingCallback class
"""

import os
import tempfile
import shutil
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from exploration_agent import ExplorationAgent, EarlyStoppingCallback


class DummyEnv(gym.Env):
    """A simple dummy environment for testing."""
    
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)
        self._step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32), {}
    
    def step(self, action):
        self._step_count += 1
        obs = np.random.uniform(0, 100, size=(4,)).astype(np.float32)
        reward = np.random.uniform(0, 1)
        done = self._step_count >= 10
        truncated = False
        info = {"reward": reward}
        return obs, reward, done, truncated, info


def test_exploration_agent_init_creates_new():
    """Test that ExplorationAgent creates new model when no existing model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=True  # Should still create new since no model exists
        )
        
        # Verify model was created
        assert agent.model is not None
        assert agent.best_reward == float('-inf')
        assert agent.device == "cpu"
        assert agent.best_model_path == os.path.join(tmpdir, "exploration_ppo_best.zip")
        
        agent.close()
        print("✅ test_exploration_agent_init_creates_new PASSED")


def test_exploration_agent_init_load_existing_false():
    """Test that load_existing=False always creates new model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        
        # First create and save a model
        agent1 = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        agent1.model.save(agent1.model_path)
        agent1.close()
        
        # Now create with load_existing=False - should create new, not load
        agent2 = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        
        # Model should exist but be newly created
        assert agent2.model is not None
        agent2.close()
        print("✅ test_exploration_agent_init_load_existing_false PASSED")


def test_exploration_agent_loads_existing_model():
    """Test that ExplorationAgent loads existing model when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        
        # First create and save a model
        agent1 = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        agent1.model.save(agent1.model_path)
        agent1.close()
        
        # Verify model file exists
        assert os.path.exists(os.path.join(tmpdir, "exploration_ppo_model.zip"))
        
        # Now create with load_existing=True - should load the saved model
        agent2 = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=True
        )
        
        assert agent2.model is not None
        agent2.close()
        print("✅ test_exploration_agent_loads_existing_model PASSED")


def test_save_best_if_improved():
    """Test save_best_if_improved() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        
        # Initial best_reward should be -inf
        assert agent.best_reward == float('-inf')
        
        # First improvement should save
        result = agent.save_best_if_improved(0.5)
        assert result is True
        assert agent.best_reward == 0.5
        assert os.path.exists(agent.best_model_path)
        assert os.path.exists(os.path.join(tmpdir, "best_reward.txt"))
        
        # Second improvement should save
        result = agent.save_best_if_improved(0.8)
        assert result is True
        assert agent.best_reward == 0.8
        
        # No improvement should not save
        result = agent.save_best_if_improved(0.6)
        assert result is False
        assert agent.best_reward == 0.8  # Still 0.8
        
        # Equal value should not save
        result = agent.save_best_if_improved(0.8)
        assert result is False
        
        agent.close()
        print("✅ test_save_best_if_improved PASSED")


def test_load_best_reward_from_file():
    """Test _load_best_reward() loads from file correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create best_reward.txt manually
        reward_path = os.path.join(tmpdir, "best_reward.txt")
        with open(reward_path, 'w') as f:
            f.write("0.75")
        
        # Also need a model file to trigger load
        env = DummyEnv()
        agent_temp = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        agent_temp.model.save(os.path.join(tmpdir, "exploration_ppo_model.zip"))
        agent_temp.close()
        
        # Now create agent that loads existing
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=True
        )
        
        assert agent.best_reward == 0.75
        agent.close()
        print("✅ test_load_best_reward_from_file PASSED")


def test_load_best():
    """Test load_best() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        
        # No best model yet, should return False
        assert agent.load_best() is False
        
        # Save as best
        agent.save_best_if_improved(0.9)
        
        # Now should be able to load best
        assert agent.load_best() is True
        
        agent.close()
        print("✅ test_load_best PASSED")


def test_load_for_inference():
    """Test load_for_inference() class method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        
        # Create and save a model
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        model_path = os.path.join(tmpdir, "test_model.zip")
        agent.model.save(model_path)
        agent.close()
        
        # Load for inference
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv([lambda: DummyEnv()])
        model = ExplorationAgent.load_for_inference(model_path, vec_env, device="cpu")
        
        assert model is not None
        
        # Clean up
        vec_env.close()
        print("✅ test_load_for_inference PASSED")


def test_predict():
    """Test predict() method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu",
            load_existing=False
        )
        
        # Get an observation
        obs = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
        
        # Predict with deterministic=True
        action, info = agent.predict(obs, deterministic=True)
        assert action is not None
        assert info["deterministic"] is True
        
        # Predict with deterministic=False
        action, info = agent.predict(obs, deterministic=False)
        assert action is not None
        assert info["deterministic"] is False
        
        agent.close()
        print("✅ test_predict PASSED")


def test_early_stopping_callback():
    """Test EarlyStoppingCallback class."""
    callback = EarlyStoppingCallback(patience=3, min_delta=0.1, verbose=0)
    
    # Simulate steps with rewards in infos
    callback.locals = {'infos': [{'reward': 0.5}]}
    assert callback._on_step() is True
    assert callback.best_reward == 0.5
    assert callback.no_improvement_count == 0
    
    # Another step with same reward (no improvement)
    callback.locals = {'infos': [{'reward': 0.5}]}
    assert callback._on_step() is True
    assert callback.no_improvement_count == 1
    
    # Another step with small improvement (less than min_delta)
    callback.locals = {'infos': [{'reward': 0.55}]}
    assert callback._on_step() is True
    assert callback.no_improvement_count == 2
    
    # Another step with no improvement - should stop now (patience=3)
    callback.locals = {'infos': [{'reward': 0.55}]}
    assert callback._on_step() is False  # Should return False to stop training
    
    print("✅ test_early_stopping_callback PASSED")


def test_early_stopping_callback_improvement_resets():
    """Test that EarlyStoppingCallback resets count on improvement."""
    callback = EarlyStoppingCallback(patience=3, min_delta=0.1, verbose=0)
    
    # Initial reward
    callback.locals = {'infos': [{'reward': 0.5}]}
    callback._on_step()
    
    # Two steps without improvement
    callback.locals = {'infos': [{'reward': 0.5}]}
    callback._on_step()
    callback._on_step()
    assert callback.no_improvement_count == 2
    
    # Big improvement - should reset counter
    callback.locals = {'infos': [{'reward': 0.7}]}
    callback._on_step()
    assert callback.no_improvement_count == 0
    assert callback.best_reward == 0.7
    
    print("✅ test_early_stopping_callback_improvement_resets PASSED")


def test_backward_compatibility():
    """Test that existing code without load_existing still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = DummyEnv()
        # Old-style initialization without load_existing parameter
        agent = ExplorationAgent(
            env=env,
            log_dir=tmpdir,
            device="cpu"
        )
        
        assert agent.model is not None
        agent.close()
        print("✅ test_backward_compatibility PASSED")


if __name__ == "__main__":
    print("--- EXPLORATION AGENT PERSISTENCE TESTS ---")
    
    all_passed = True
    
    try:
        test_exploration_agent_init_creates_new()
    except Exception as e:
        print(f"❌ test_exploration_agent_init_creates_new FAILED: {e}")
        all_passed = False
    
    try:
        test_exploration_agent_init_load_existing_false()
    except Exception as e:
        print(f"❌ test_exploration_agent_init_load_existing_false FAILED: {e}")
        all_passed = False
    
    try:
        test_exploration_agent_loads_existing_model()
    except Exception as e:
        print(f"❌ test_exploration_agent_loads_existing_model FAILED: {e}")
        all_passed = False
    
    try:
        test_save_best_if_improved()
    except Exception as e:
        print(f"❌ test_save_best_if_improved FAILED: {e}")
        all_passed = False
    
    try:
        test_load_best_reward_from_file()
    except Exception as e:
        print(f"❌ test_load_best_reward_from_file FAILED: {e}")
        all_passed = False
    
    try:
        test_load_best()
    except Exception as e:
        print(f"❌ test_load_best FAILED: {e}")
        all_passed = False
    
    try:
        test_load_for_inference()
    except Exception as e:
        print(f"❌ test_load_for_inference FAILED: {e}")
        all_passed = False
    
    try:
        test_predict()
    except Exception as e:
        print(f"❌ test_predict FAILED: {e}")
        all_passed = False
    
    try:
        test_early_stopping_callback()
    except Exception as e:
        print(f"❌ test_early_stopping_callback FAILED: {e}")
        all_passed = False
    
    try:
        test_early_stopping_callback_improvement_resets()
    except Exception as e:
        print(f"❌ test_early_stopping_callback_improvement_resets FAILED: {e}")
        all_passed = False
    
    try:
        test_backward_compatibility()
    except Exception as e:
        print(f"❌ test_backward_compatibility FAILED: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL EXPLORATION AGENT PERSISTENCE TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("="*60)
