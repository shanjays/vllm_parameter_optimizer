import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import ray

class FastGymEnv(gym.Env):
    """
    This is the "Fast Gym" environment (client) that runs on GPU 0.
    It receives actions from the PPO agent and dispatches benchmark
    jobs to the BenchmarkWorker on GPU 1 via Ray.
    """
    metadata = {'render_modes': []}

    def __init__(self, mission_plan_path, benchmark_worker, static_args, initial_state, 
                 config_saver=None, current_token_count=None):
        super(FastGymEnv, self).__init__()

        self.benchmark_worker = benchmark_worker # Ray handle for GPU 1
        self.static_args = static_args
        self.epoch_results = []
        self.initial_state = initial_state
        self.config_saver = config_saver
        self.current_token_count = current_token_count or static_args.get('num_tokens', 16088)
        
        # Define the State Space (our 4 ncu metrics)
        # [sm_throughput, dram_throughput, l1_hit_rate, l2_hit_rate]
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(4,), dtype=np.float32
        )
        
        # Load the first mission plan
        self.set_mission_plan(mission_plan_path)

    def set_mission_plan(self, mission_plan_path):
        """Called by the supervisor to update the plan between epochs."""
        # Default fallback plan with multiple options for exploration
        DEFAULT_FALLBACK_PLAN = {
            'reward_function': {
                'R_sm_throughput': 0.4,
                'R_dram_throughput': 0.3,
                'R_l1_hit_rate': 0.15,
                'R_l2_hit_rate': 0.15
            }, 
            'pruned_action_space': {
                'BLOCK_SIZE_M': [32, 64, 128], 
                'BLOCK_SIZE_N': [32, 64, 128], 
                'BLOCK_SIZE_K': [32, 64],
                'num_warps': [4, 8, 16], 
                'num_stages': [2, 3, 4]
            }
        }
        
        try:
            with open(mission_plan_path, 'r') as f:
                self.mission_plan = json.load(f)
        except Exception as e:
            print(f"[FastGymEnv] ERROR: Failed to load mission plan '{mission_plan_path}'. {e}")
            # Fallback to a safe plan with multiple options
            self.mission_plan = DEFAULT_FALLBACK_PLAN

        self.reward_weights = self.mission_plan['reward_function']
        self.pruned_space = self.mission_plan['pruned_action_space']
        self.param_keys = list(self.pruned_space.keys())

        # Define the Action Space (based on the *new* plan)
        self.action_space = spaces.MultiDiscrete([
            len(self.pruned_space['BLOCK_SIZE_M']),
            len(self.pruned_space['BLOCK_SIZE_N']),
            len(self.pruned_space['BLOCK_SIZE_K']),
            len(self.pruned_space['num_warps']),
            len(self.pruned_space['num_stages']),
        ])
        
        total_combinations = self.action_space.nvec.prod()
        print(f"[FastGymEnv] Mission Plan set. Action space has {total_combinations} combinations.")
        
        # Validate action space has at least 2 combinations for PPO exploration
        if total_combinations < 2:
            print(f"[FastGymEnv] WARNING: Action space too small ({total_combinations}), using default with multiple options")
            self.mission_plan = DEFAULT_FALLBACK_PLAN
            self.reward_weights = self.mission_plan['reward_function']
            self.pruned_space = self.mission_plan['pruned_action_space']
            self.param_keys = list(self.pruned_space.keys())
            self.action_space = spaces.MultiDiscrete([
                len(self.pruned_space['BLOCK_SIZE_M']),
                len(self.pruned_space['BLOCK_SIZE_N']),
                len(self.pruned_space['BLOCK_SIZE_K']),
                len(self.pruned_space['num_warps']),
                len(self.pruned_space['num_stages']),
            ])
            total_combinations = self.action_space.nvec.prod()
            print(f"[FastGymEnv] Updated action space to {total_combinations} combinations.")

    def _action_to_params(self, action):
        """Converts an action (indices) into a kernel config dict."""
        try:
            num_warps_val = self.pruned_space['num_warps'][action[3]]
            
            # Validate num_warps is a power of 2
            if num_warps_val <= 0 or (num_warps_val & (num_warps_val - 1)) != 0:
                print(f"[FastGymEnv] WARNING: Invalid num_warps={num_warps_val}, using 4")
                num_warps_val = 4  # Default to valid power of 2
            
            return {
                "BLOCK_SIZE_M": self.pruned_space['BLOCK_SIZE_M'][action[0]],
                "BLOCK_SIZE_N": self.pruned_space['BLOCK_SIZE_N'][action[1]],
                "BLOCK_SIZE_K": self.pruned_space['BLOCK_SIZE_K'][action[2]],
                "num_warps": num_warps_val,
                "num_stages": self.pruned_space['num_stages'][action[4]],
            }
        except IndexError as e:
            print(f"[FastGymEnv] ERROR: Action space mismatch. {e}")
            # Fallback to a default action
            return {
                "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "num_warps": 4, "num_stages": 4,
            }

    def step(self, action):
        params = self._action_to_params(action)
        
        # Send the benchmark request to the worker on GPU 1
        # Pass the current token count for multi-token testing
        result_id = self.benchmark_worker.run_fast_gym_benchmark.remote(
            params, self.static_args, self.reward_weights, self.current_token_count
        )
        # Wait for the result
        state, reward, csv_data = ray.get(result_id)
        
        done = False
        truncated = False
        
        if state is None:
            # The benchmark worker reported a failure (e.g., OOM, timeout)
            state = self.initial_state # Reset to avoid errors
            # reward is already -100.0 (as set by worker)
        else:
            # Update config saver with the result if available
            if self.config_saver is not None:
                metrics = {
                    'sm_throughput': state[0],
                    'dram_throughput': state[1],
                    'l1_hit_rate': state[2],
                    'l2_hit_rate': state[3]
                }
                self.config_saver.update_best_config(
                    self.current_token_count,
                    params,
                    reward,
                    metrics
                )
        
        # Log this result for the "Professor" to review
        self.epoch_results.append((params, state.tolist(), reward))
        
        return state, reward, done, truncated, {} # 'info' dict is empty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.epoch_results = [] # Clear results for the new epoch
        return self.initial_state, {} # 'info' dict is empty
    
    def get_top_results(self, n=5):
        if not self.epoch_results:
            return []
        sorted_results = sorted(
            self.epoch_results, key=lambda x: x[2], reverse=True
        )
        return sorted_results[:n]
    
    def close(self):
        """Called when the environment is no longer needed."""
        print("[FastGymEnv] Closing environment.")
        pass
