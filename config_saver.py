"""
VLLMConfigSaver - Saves best kernel configs in vLLM format for fused_moe kernel.

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


class VLLMConfigSaver:
    """
    Saves best kernel configs in vLLM format for fused_moe kernel.
    
    Output format matches vLLM's expected config:
    {
        "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, ...},
        "2": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, ...},
        ...
    }
    """
    
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
            print(f"[ConfigSaver] New best config for {token_count} tokens: reward={reward:.2f}")
            return True
        return False
    
    def save_vllm_config(self, output_dir="./hakt_configs"):
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
        
        # 1. Save in vLLM format (just configs, no rewards)
        vllm_config = {}
        for token_key in sorted(self.best_configs.keys(), key=lambda x: int(x)):
            vllm_config[token_key] = self.best_configs[token_key]
        
        vllm_path = os.path.join(output_dir, self.get_config_filename())
        with open(vllm_path, 'w') as f:
            json.dump(vllm_config, f, indent=2)
        print(f"[ConfigSaver] Saved vLLM config to: {vllm_path}")
        
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
        print(f"[ConfigSaver] Saved detailed config to: {detailed_path}")
        
        # 3. Save all results for analysis
        all_results_path = os.path.join(output_dir, "all_results.json")
        with open(all_results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"[ConfigSaver] Saved all results to: {all_results_path}")
        
        return vllm_path
    
    def get_summary(self):
        """Get summary of best configs found."""
        return {
            "total_token_counts": len(self.best_configs),
            "total_experiments": len(self.all_results),
            "best_rewards": self.best_rewards.copy(),
            "config_filename": self.get_config_filename()
        }
    
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
                print("[ConfigSaver] WARNING: Could not find vLLM installation")
                return None
        
        if not os.path.exists(vllm_config_dir):
            os.makedirs(vllm_config_dir, exist_ok=True)
        
        src_path = os.path.join("./hakt_configs", self.get_config_filename())
        dst_path = os.path.join(vllm_config_dir, self.get_config_filename())
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            print(f"[ConfigSaver] Copied config to vLLM: {dst_path}")
            return dst_path
        else:
            print(f"[ConfigSaver] WARNING: Source config not found: {src_path}")
            return None
