"""
VLLMConfigExporter - Exports optimized kernel configurations in vLLM format.

This module handles the export of best-performing kernel configurations
discovered during the hierarchical optimization process. The output format
is compatible with vLLM's fused_moe kernel configuration system.

Output format matches vLLM's expected config:
{
    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, ...},
    "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, ...},
    ...
}
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# All token counts that vLLM expects
TOKEN_COUNTS_ALL = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024,
    1536, 2048, 3072, 4096, 5120, 9216, 13312, 17408,
    25600, 33792, 41984, 50176, 58368
]


class VLLMConfigExporter:
    """
    Exports optimized kernel configurations in vLLM format for fused_moe kernel.
    
    This class tracks the best-performing configurations discovered during
    kernel optimization and exports them in the format expected by vLLM's
    autotuning system.
    
    Output format matches vLLM's expected config:
    {
        "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, ...},
        "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, ...},
        ...
    }
    """
    
    # Default output directory for saving configs
    DEFAULT_OUTPUT_DIR = "./optimized_configs"
    
    # Default kernel configuration that works on H100
    DEFAULT_KERNEL_CONFIG = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 4
    }
    
    def __init__(self, num_experts, inter_size, device_name="NVIDIA_H100_80GB_HBM3",
                 output_dir: str = None):
        """
        Initialize the config saver.
        
        Args:
            num_experts: E value (number of experts in MoE model)
            inter_size: N value (intermediate size)
            device_name: GPU device name for config filename
            output_dir: Output directory for local config backups
        """
        self.num_experts = num_experts  # E value
        self.inter_size = inter_size    # N value
        self.device_name = device_name
        self._last_output_dir = output_dir if output_dir else self.DEFAULT_OUTPUT_DIR
        
        # All token counts that vLLM expects
        self.all_token_counts = TOKEN_COUNTS_ALL
        
        # Best config for each token count
        self.best_configs: Dict[str, Dict[str, Any]] = {}
        self.best_rewards: Dict[str, float] = {}
        
        # All tested configs for analysis
        self.all_results = []
        
        # Create output directory
        os.makedirs(self._last_output_dir, exist_ok=True)
        
        # Get vLLM config directory
        self.vllm_config_dir = self._get_vllm_config_dir()
        
        # Config filename that vLLM expects
        self.config_filename = f"E={num_experts},N={inter_size},device_name={device_name}.json"
        
        # Initialize config file with defaults if it doesn't exist
        self._initialize_config_with_defaults()
    
    def _get_vllm_config_dir(self) -> str:
        """Get the vLLM fused_moe config directory."""
        try:
            import vllm
            vllm_lib_path = os.path.dirname(vllm.__file__)
            config_dir = os.path.join(
                vllm_lib_path, 
                "model_executor/layers/fused_moe/configs/"
            )
            os.makedirs(config_dir, exist_ok=True)
            return config_dir
        except ImportError:
            print("[ConfigExporter] vLLM not installed, using local directory")
            fallback_dir = os.path.join(self._last_output_dir, "vllm_configs")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir
        except OSError as e:
            print(f"[ConfigExporter] Could not create vLLM config dir: {e}")
            fallback_dir = os.path.join(self._last_output_dir, "vllm_configs")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir
    
    def _initialize_config_with_defaults(self):
        """Create config file with default values for all token counts."""
        config_path = os.path.join(self.vllm_config_dir, self.config_filename)
        
        # Check if config already exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                # Load existing best configs and rewards
                # Use float('-inf') as sentinel for untested configs
                for tc_str, config in existing_config.items():
                    self.best_configs[tc_str] = config
                    self.best_rewards[tc_str] = float('-inf')
                print(f"[ConfigExporter] Loaded existing config with {len(existing_config)} token counts from {config_path}")
                return
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ConfigExporter] Could not parse existing config: {e}, creating new")
        
        # Create default config for all token counts
        # Use float('-inf') as sentinel for untested configs
        default_config = {}
        for tc in TOKEN_COUNTS_ALL:
            default_config[str(tc)] = self.DEFAULT_KERNEL_CONFIG.copy()
            self.best_configs[str(tc)] = self.DEFAULT_KERNEL_CONFIG.copy()
            self.best_rewards[str(tc)] = float('-inf')
        
        # Write to vLLM config directory
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"[ConfigExporter] Initialized config with defaults at {config_path}")
        except OSError as e:
            print(f"[ConfigExporter] Could not write to vLLM dir: {e}")
            # Fallback to local directory
            local_path = os.path.join(self._last_output_dir, self.config_filename)
            with open(local_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"[ConfigExporter] Initialized config at {local_path}")
        
        # Also write to local output directory as backup
        local_path = os.path.join(self._last_output_dir, self.config_filename)
        try:
            with open(local_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        except OSError as e:
            print(f"[ConfigExporter] Warning: Could not write local backup: {e}")
    
    def _update_config_file(self, token_count: int, config: Dict[str, int]):
        """Update the config file with new best config for a token count."""
        config_path = os.path.join(self.vllm_config_dir, self.config_filename)
        
        # Load existing config
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    full_config = json.load(f)
            else:
                full_config = {}
        except (json.JSONDecodeError, ValueError):
            full_config = {}
        
        # Update with new config for this token count
        full_config[str(token_count)] = config
        
        # Write back
        try:
            with open(config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
        except OSError as e:
            print(f"[ConfigExporter] Warning: Could not update vLLM config: {e}")
        
        # Also update local backup
        local_path = os.path.join(self._last_output_dir, self.config_filename)
        try:
            if os.path.exists(local_path):
                with open(local_path, 'r') as f:
                    local_config = json.load(f)
            else:
                local_config = {}
            local_config[str(token_count)] = config
            with open(local_path, 'w') as f:
                json.dump(local_config, f, indent=2)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"[ConfigExporter] Warning: Could not update local config: {e}")
        
    def get_config_filename(self):
        """Generate vLLM config filename."""
        return f"E={self.num_experts},N={self.inter_size},device_name={self.device_name}.json"
    
    def update_best_config(self, token_count, config, reward, metrics=None):
        """
        Update best config for a token count if this one is better.
        
        Args:
            token_count: Number of tokens (1, 2, 4, 8, 16, ...)
            config: Dict with BLOCK_SIZE_M, BLOCK_SIZE_N, etc.
            reward: Reward value from benchmark
            metrics: Optional dict with sm_throughput, dram_throughput, etc.
            
        Returns:
            bool: True if this config is the new best for this token count
        """
        token_key = str(token_count)
        
        # Record all results
        self.all_results.append({
            'token_count': token_count,
            'config': config.copy(),
            'reward': reward,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update best if this is better
        current_best = self.best_rewards.get(token_key, float('-inf'))
        
        if reward > current_best:
            new_config = {
                "BLOCK_SIZE_M": config.get("BLOCK_SIZE_M", 64),
                "BLOCK_SIZE_N": config.get("BLOCK_SIZE_N", 64),
                "BLOCK_SIZE_K": config.get("BLOCK_SIZE_K", 32),
                "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 8),
                "num_warps": config.get("num_warps", 4),
                "num_stages": config.get("num_stages", 4)
            }
            self.best_configs[token_key] = new_config
            self.best_rewards[token_key] = reward
            
            # Immediately update the config file
            self._update_config_file(token_count, new_config)
            
            print(f"[ConfigExporter] New best config for {token_count} tokens: reward={reward:.2f}")
            return True
        return False
    
    def save_vllm_config(self, output_dir="./optimized_configs"):
        """
        Save configs in vLLM format.
        
        Creates:
        - E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json (vLLM format)
        - best_configs_detailed.json (with rewards and metrics)
        - all_results.json (complete experiment log)
        
        Also saves to vLLM config directory if available.
        
        Args:
            output_dir: Directory to save config files
            
        Returns:
            str: Path to the saved vLLM config file
        """
        os.makedirs(output_dir, exist_ok=True)
        self._last_output_dir = output_dir  # Track for copy_to_vllm
        
        # Build config from best configs, using defaults for any missing token counts
        full_config = {}
        for tc in TOKEN_COUNTS_ALL:
            tc_str = str(tc)
            if tc_str in self.best_configs:
                full_config[tc_str] = self.best_configs[tc_str]
            else:
                # Use default if no best found
                full_config[tc_str] = self.DEFAULT_KERNEL_CONFIG.copy()
        
        # 1. Save to vLLM config directory
        vllm_config_path = os.path.join(self.vllm_config_dir, self.config_filename)
        try:
            with open(vllm_config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            print(f"[ConfigExporter] Saved vLLM config to: {vllm_config_path}")
        except OSError as e:
            print(f"[ConfigExporter] Could not write to vLLM dir: {e}")
        
        # 2. Also save to local output directory
        local_vllm_path = os.path.join(output_dir, self.get_config_filename())
        with open(local_vllm_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        print(f"[ConfigExporter] Saved local config to: {local_vllm_path}")
        
        # 3. Save detailed config with rewards
        detailed = {
            "metadata": {
                "num_experts": self.num_experts,
                "inter_size": self.inter_size,
                "device_name": self.device_name,
                "generated_at": datetime.now().isoformat(),
                "total_experiments": len(self.all_results)
            },
            "best_configs": {}
        }
        for token_key, config in self.best_configs.items():
            detailed["best_configs"][token_key] = {
                "config": config,
                "reward": self.best_rewards.get(token_key, 0)
            }
        
        detailed_path = os.path.join(output_dir, "best_configs_detailed.json")
        with open(detailed_path, 'w') as f:
            json.dump(detailed, f, indent=2)
        print(f"[ConfigExporter] Saved detailed config to: {detailed_path}")
        
        # 4. Save all results for analysis
        all_results_path = os.path.join(output_dir, "all_results.json")
        with open(all_results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"[ConfigExporter] Saved all results to: {all_results_path}")
        
        return vllm_config_path
    
    def get_summary(self):
        """Get summary of best configs found."""
        # Only count token counts that have been actually tested (reward > -inf)
        tested_token_counts = len([tc for tc, r in self.best_rewards.items() if r > float('-inf')])
        return {
            "total_token_counts": len(self.best_configs),
            "total_experiments": len(self.all_results),
            "tested_token_counts": tested_token_counts,
            "best_rewards": {tc: r for tc, r in self.best_rewards.items() if r > float('-inf')},
            "config_filename": self.get_config_filename(),
            "config_path": os.path.join(self.vllm_config_dir, self.config_filename)
        }
    
    def export_visualization_data(self):
        """
        Export training metrics in visualization-friendly format.
        
        Returns:
            Dict containing visualization-friendly data structures including:
            - steps: List of experiment indices
            - rewards: List of reward values
            - token_counts: List of token counts
            - timestamps: List of ISO timestamps
            - configs: List of configuration dicts
            - metrics: List of NCU metrics dicts
            - cumulative_stats: Dict with running statistics
        """
        if not self.all_results:
            return {
                'steps': [],
                'rewards': [],
                'token_counts': [],
                'timestamps': [],
                'configs': [],
                'metrics': [],
                'cumulative_stats': {
                    'total_experiments': 0,
                    'best_reward_by_token': {},
                    'mean_reward': None,
                }
            }
        
        steps = list(range(len(self.all_results)))
        rewards = [r.get('reward', 0) for r in self.all_results]
        token_counts = [r.get('token_count') for r in self.all_results]
        timestamps = [r.get('timestamp') for r in self.all_results]
        configs = [r.get('config', {}) for r in self.all_results]
        metrics = [r.get('metrics', {}) for r in self.all_results]
        
        # Compute cumulative statistics
        valid_rewards = [r for r in rewards if r is not None]
        cumulative_stats = {
            'total_experiments': len(self.all_results),
            'best_reward_by_token': self.best_rewards.copy(),
            'mean_reward': sum(valid_rewards) / len(valid_rewards) if valid_rewards else None,
            'best_reward': max(valid_rewards) if valid_rewards else None,
            'min_reward': min(valid_rewards) if valid_rewards else None,
        }
        
        return {
            'steps': steps,
            'rewards': rewards,
            'token_counts': token_counts,
            'timestamps': timestamps,
            'configs': configs,
            'metrics': metrics,
            'cumulative_stats': cumulative_stats,
        }
    
    def export_complete_config(self, output_path=None):
        """
        Export config with ALL token counts, interpolating missing ones.
        
        Args:
            output_path: Optional output file path. If None, uses default filename.
            
        Returns:
            str: Path to the saved config file, or None if no configs available
        """
        if output_path is None:
            filename = f"E={self.num_experts},N={self.inter_size},device_name={self.device_name}.json"
            output_path = os.path.join(self._last_output_dir, filename)
        
        # Get tested token counts
        tested_counts = sorted([int(k) for k in self.best_configs.keys()])
        
        if not tested_counts:
            print("[ConfigExporter] ERROR: No configs to export!")
            return None
        
        # Build complete config with interpolation
        complete_config = {}
        
        for token_count in self.all_token_counts:
            if str(token_count) in self.best_configs:
                # Use actual tested config
                complete_config[str(token_count)] = self.best_configs[str(token_count)].copy()
            else:
                # Interpolate from nearest tested config
                nearest = self._find_nearest_config(token_count, tested_counts)
                complete_config[str(token_count)] = self.best_configs[str(nearest)].copy()
                print(f"[ConfigExporter] Token {token_count} → using config from token {nearest}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save complete config
        with open(output_path, "w") as f:
            json.dump(complete_config, f, indent=2)
        
        print(f"[ConfigExporter] Exported complete config with {len(complete_config)} token counts")
        print(f"[ConfigExporter] Saved to: {output_path}")
        
        return output_path
    
    def _find_nearest_config(self, target, tested_counts):
        """
        Find nearest tested token count for interpolation.
        
        Strategy:
        - For targets beyond tested range: use nearest boundary
        - For targets within range: use nearest tested count, with tie-breaker
          preferring the lower count for safety (proven configs work)
        
        Args:
            target: Target token count to find config for
            tested_counts: List of token counts that have been tested
            
        Returns:
            int: Nearest tested token count
        """
        if not tested_counts:
            return 1  # Fallback to smallest
        
        # Handle boundary cases first
        if target > max(tested_counts):
            return max(tested_counts)
        elif target < min(tested_counts):
            return min(tested_counts)
        
        # Find bracketing values for targets within range
        lower = max([c for c in tested_counts if c <= target], default=min(tested_counts))
        upper = min([c for c in tested_counts if c >= target], default=max(tested_counts))
        
        # Prefer lower for safety when equidistant (proven configs work)
        return lower if (target - lower) <= (upper - target) else upper
    
    def copy_to_vllm(self, vllm_config_dir=None):
        """
        Copy the generated config to vLLM's config directory.
        
        Args:
            vllm_config_dir: Path to vLLM's fused_moe/configs/ directory
                            If None, tries to auto-detect
                            
        Returns:
            str: Path to destination file if successful, None otherwise
        """
        if vllm_config_dir is None:
            try:
                import vllm
                vllm_config_dir = os.path.join(
                    os.path.dirname(vllm.__file__),
                    "model_executor/layers/fused_moe/configs/"
                )
            except ImportError:
                print("[ConfigExporter] WARNING: Could not find vLLM installation")
                return None
        
        if not os.path.exists(vllm_config_dir):
            os.makedirs(vllm_config_dir, exist_ok=True)
        
        src_path = os.path.join(self._last_output_dir, self.get_config_filename())
        dst_path = os.path.join(vllm_config_dir, self.get_config_filename())
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"[ConfigExporter] Copied config to vLLM: {dst_path}")
            return dst_path
        else:
            print(f"[ConfigExporter] WARNING: Source config not found: {src_path}")
            return None


# Backward compatibility alias
VLLMConfigSaver = VLLMConfigExporter
