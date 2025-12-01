import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import ray

class KernelTuningEnvironment(gym.Env):
    """
    Kernel Tuning Environment for configuration optimization.
    
    This Gymnasium environment enables reinforcement learning agents to explore
    the kernel configuration search space. It interfaces with a profiling worker
    on a dedicated GPU to collect performance metrics for each configuration.
    
    The environment state consists of normalized performance metrics:
    - SM throughput (% of peak)
    - DRAM throughput (% of peak)
    - L1 cache hit rate (%)
    - L2 cache hit rate (%)
    
    Actions correspond to selecting values for kernel parameters from the
    search space defined in the optimization policy.
    """
    metadata = {'render_modes': []}

    def __init__(self, policy_config_path, profiling_worker, static_args, initial_state, 
                 config_exporter=None, current_token_count=None):
        """
        Initialize the kernel tuning environment.
        
        Args:
            policy_config_path: Path to the optimization policy JSON file
            profiling_worker: Ray handle to the ProfilingWorker
            static_args: Static benchmark arguments
            initial_state: Initial observation state
            config_exporter: Optional VLLMConfigExporter for saving best configs
            current_token_count: Token count for the current benchmark
        """
        super(KernelTuningEnvironment, self).__init__()

        self.profiling_worker = profiling_worker
        self.static_args = static_args
        self.epoch_results = []
        self.initial_state = initial_state
        self.config_exporter = config_exporter
        self.current_token_count = current_token_count or static_args.get('num_tokens', 16088)
        
        # Best config and reward tracking
        self.best_config = None
        self.best_reward = float('-inf')
        
        # Define the State Space (4 NCU metrics)
        # [sm_throughput, dram_throughput, l1_hit_rate, l2_hit_rate]
        self.observation_space = spaces.Box(
            low=0.0, high=100.0, shape=(4,), dtype=np.float32
        )
        
        # Load the optimization policy
        self.set_optimization_policy(policy_config_path)

    def set_optimization_policy(self, policy_config_path):
        """
        Load and apply a new optimization policy.
        
        Args:
            policy_config_path: Path to the optimization policy JSON file
        """
        # Default fallback policy with multiple options for exploration
        DEFAULT_FALLBACK_POLICY = {
            'objective_weights': {
                'R_sm_throughput': 0.4,
                'R_dram_throughput': 0.3,
                'R_l1_hit_rate': 0.15,
                'R_l2_hit_rate': 0.15
            }, 
            'search_space': {
                'BLOCK_SIZE_M': [32, 64, 128], 
                'BLOCK_SIZE_N': [32, 64, 128], 
                'BLOCK_SIZE_K': [32, 64],
                'num_warps': [4, 8, 16], 
                'num_stages': [2, 3, 4]
            }
        }
        
        try:
            with open(policy_config_path, 'r') as f:
                self.optimization_policy = json.load(f)
        except Exception as e:
            print(f"[KernelTuningEnv] ERROR: Failed to load optimization policy '{policy_config_path}'. {e}")
            # Fallback to a safe policy with multiple options
            self.optimization_policy = DEFAULT_FALLBACK_POLICY

        # Support both old and new key names for backward compatibility
        if 'objective_weights' in self.optimization_policy:
            self.objective_weights = self.optimization_policy['objective_weights']
        else:
            self.objective_weights = self.optimization_policy.get('reward_function', DEFAULT_FALLBACK_POLICY['objective_weights'])
        
        if 'search_space' in self.optimization_policy:
            self.search_space = self.optimization_policy['search_space']
        else:
            self.search_space = self.optimization_policy.get('pruned_action_space', DEFAULT_FALLBACK_POLICY['search_space'])
        
        self.param_keys = list(self.search_space.keys())

        # Define the Action Space (based on the search space)
        self.action_space = spaces.MultiDiscrete([
            len(self.search_space['BLOCK_SIZE_M']),
            len(self.search_space['BLOCK_SIZE_N']),
            len(self.search_space['BLOCK_SIZE_K']),
            len(self.search_space['num_warps']),
            len(self.search_space['num_stages']),
        ])
        
        total_combinations = self.action_space.nvec.prod()
        print(f"[KernelTuningEnv] Optimization policy set. Search space has {total_combinations} combinations.")
        
        # Validate action space has at least 2 combinations for PPO exploration
        if total_combinations < 2:
            print(f"[KernelTuningEnv] WARNING: Search space too small ({total_combinations}), using default with multiple options")
            self.optimization_policy = DEFAULT_FALLBACK_POLICY
            self.objective_weights = self.optimization_policy['objective_weights']
            self.search_space = self.optimization_policy['search_space']
            self.param_keys = list(self.search_space.keys())
            self.action_space = spaces.MultiDiscrete([
                len(self.search_space['BLOCK_SIZE_M']),
                len(self.search_space['BLOCK_SIZE_N']),
                len(self.search_space['BLOCK_SIZE_K']),
                len(self.search_space['num_warps']),
                len(self.search_space['num_stages']),
            ])
            total_combinations = self.action_space.nvec.prod()
            print(f"[KernelTuningEnv] Updated search space to {total_combinations} combinations.")

    def set_mission_plan(self, mission_plan_path):
        """Legacy method for backward compatibility."""
        self.set_optimization_policy(mission_plan_path)

    def _action_to_params(self, action):
        """Converts an action (indices) into a kernel config dict."""
        try:
            num_warps_val = self.search_space['num_warps'][action[3]]
            
            # Validate num_warps is a power of 2
            if num_warps_val <= 0 or (num_warps_val & (num_warps_val - 1)) != 0:
                print(f"[KernelTuningEnv] WARNING: Invalid num_warps={num_warps_val}, using 4")
                num_warps_val = 4  # Default to valid power of 2
            
            return {
                "BLOCK_SIZE_M": self.search_space['BLOCK_SIZE_M'][action[0]],
                "BLOCK_SIZE_N": self.search_space['BLOCK_SIZE_N'][action[1]],
                "BLOCK_SIZE_K": self.search_space['BLOCK_SIZE_K'][action[2]],
                "num_warps": num_warps_val,
                "num_stages": self.search_space['num_stages'][action[4]],
            }
        except IndexError as e:
            print(f"[KernelTuningEnv] ERROR: Search space mismatch. {e}")
            # Fallback to a default action
            return {
                "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "num_warps": 4, "num_stages": 4,
            }

    def step(self, action):
        params = self._action_to_params(action)
        
        # Send the profiling request to the worker
        # Pass the current token count for multi-token testing
        try:
            result_id = self.profiling_worker.run_kernel_profiling.remote(
                params, self.static_args, self.objective_weights, self.current_token_count
            )
            # Wait for the result
            state, reward, csv_data = ray.get(result_id)
        except Exception as e:
            # Handle VRAM crashes and other exceptions gracefully
            print(f"[KernelTuningEnv] Profiling failed with exception: {e}")
            state = None
            reward = -20.0  # Use aggressive penalty for OOM-like errors
            csv_data = None
        
        done = False
        truncated = False
        
        if state is None:
            # The profiling worker reported a failure (e.g., OOM, timeout)
            # This is valuable boundary information for aggressive optimization!
            state = self.initial_state  # Reset to avoid errors
            # reward is already negative (as set by worker or exception handler)
            print(f"[KernelTuningEnv] Config failed (boundary found): {params}, reward={reward}")
        else:
            # Track best config and reward
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_config = params.copy()
            
            # Update config exporter with the result if available
            if self.config_exporter is not None:
                metrics = {
                    'sm_throughput': state[0],
                    'dram_throughput': state[1],
                    'l1_hit_rate': state[2],
                    'l2_hit_rate': state[3]
                }
                self.config_exporter.update_best_config(
                    self.current_token_count,
                    params,
                    reward,
                    metrics
                )
        
        # Log this result for the meta-controller to review
        self.epoch_results.append((params, state.tolist(), reward))
        
        return state, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.epoch_results = []  # Clear results for the new epoch
        return self.initial_state, {}
    
    def get_top_results(self, n=5):
        if not self.epoch_results:
            return []
        sorted_results = sorted(
            self.epoch_results, key=lambda x: x[2], reverse=True
        )
        return sorted_results[:n]
    
    def close(self):
        """Called when the environment is no longer needed."""
        print("[KernelTuningEnv] Closing environment.")
        pass


# Backward compatibility alias
FastGymEnv = KernelTuningEnvironment
