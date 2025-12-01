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
    
    def __init__(self, num_experts, inter_size, device_name="NVIDIA_H100_80GB_HBM3"):
        """
        Initialize the config saver.
        
        Args:
            num_experts: E value (number of experts in MoE model)
            inter_size: N value (intermediate size)
            device_name: GPU device name for config filename
        """
        self.num_experts = num_experts  # E value
        self.inter_size = inter_size    # N value
        self.device_name = device_name
        self._last_output_dir = self.DEFAULT_OUTPUT_DIR
        
        # All token counts that vLLM expects
        self.all_token_counts = TOKEN_COUNTS_ALL
        
        # Best config for each token count
        self.best_configs = {}
        self.best_rewards = {}
        
        # All tested configs for analysis
        self.all_results = []
        
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
        if token_key not in self.best_rewards or reward > self.best_rewards[token_key]:
            self.best_configs[token_key] = {
                "BLOCK_SIZE_M": config.get("BLOCK_SIZE_M", 64),
                "BLOCK_SIZE_N": config.get("BLOCK_SIZE_N", 64),
                "BLOCK_SIZE_K": config.get("BLOCK_SIZE_K", 32),
                "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 8),
                "num_warps": config.get("num_warps", 4),
                "num_stages": config.get("num_stages", 4)
            }
            self.best_rewards[token_key] = reward
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
        
        Args:
            output_dir: Directory to save config files
            
        Returns:
            str: Path to the saved vLLM config file
        """
        os.makedirs(output_dir, exist_ok=True)
        self._last_output_dir = output_dir  # Track for copy_to_vllm
        
        # 1. Save in vLLM format (just configs, no rewards)
        vllm_config = {}
        for token_key in sorted(self.best_configs.keys(), key=lambda x: int(x)):
            vllm_config[token_key] = self.best_configs[token_key]
        
        vllm_path = os.path.join(output_dir, self.get_config_filename())
        with open(vllm_path, 'w') as f:
            json.dump(vllm_config, f, indent=2)
        print(f"[ConfigExporter] Saved vLLM config to: {vllm_path}")
        
        # 2. Save detailed config with rewards
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
        
        # 3. Save all results for analysis
        all_results_path = os.path.join(output_dir, "all_results.json")
        with open(all_results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"[ConfigExporter] Saved all results to: {all_results_path}")
        
        return vllm_path
    
    def get_summary(self):
        """Get summary of best configs found."""
        return {
            "total_token_counts": len(self.best_configs),
            "total_experiments": len(self.all_results),
            "best_rewards": self.best_rewards.copy(),
            "config_filename": self.get_config_filename()
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
