"""
Test Script for DeepSeek-MoE-16B-Base Optimization

This script finds optimal MoE kernel parameters for the DeepSeek-MoE-16B-Base model
using NCU profiling and the trained hierarchical kernel optimizer.

Usage:
    # Quick test (no NCU, simulated metrics)
    python test_deepseek_moe_optimization.py --quick --no-ncu
    
    # Full optimization with NCU profiling on GPU 7
    python test_deepseek_moe_optimization.py --gpu 7
    
    # Quick optimization with NCU + benchmark
    python test_deepseek_moe_optimization.py --gpu 7 --quick --run-benchmark
    
    # Full optimization with limited configs per token
    python test_deepseek_moe_optimization.py --gpu 7 --max-configs 10
"""

import argparse
import json
import os
import subprocess
import csv
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DeepSeekMoEConfig:
    """Configuration for DeepSeek-MoE-16B-Base model."""
    model_name: str = "deepseek-ai/deepseek-moe-16b-base"
    hidden_size: int = 2048
    moe_intermediate_size: int = 1408
    dense_intermediate_size: int = 10944
    num_routed_experts: int = 64
    num_shared_experts: int = 2
    top_k: int = 6
    total_layers: int = 27  # 1 dense + 26 MoE
    attention_heads: int = 16
    context_length: int = 4096
    dtype: str = "fp16"
    
    @property
    def total_experts(self) -> int:
        """Total number of experts (routed + shared)."""
        return self.num_routed_experts + self.num_shared_experts
    
    @property
    def inter_size(self) -> int:
        """Intermediate size for vLLM config filename."""
        return self.moe_intermediate_size


# Token counts to test
TOKEN_COUNTS_FULL = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024,
    1536, 2048, 3072, 4096
]

TOKEN_COUNTS_QUICK = [1, 16, 64, 256, 1024, 4096]


# Default search space for kernel parameters
DEFAULT_SEARCH_SPACE = {
    "BLOCK_SIZE_M": [32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [32, 64],
    "num_warps": [4, 8, 16],
    "num_stages": [2, 3, 4]
}


# H100 hardware limits
H100_SHARED_MEM_LIMIT = 232448  # bytes (~227KB)

# Data type sizes
FP16_BYTES = 2  # bytes per FP16 element


class DeepSeekMoEOptimizer:
    """
    Optimizer for finding optimal MoE kernel parameters for DeepSeek-MoE-16B-Base.
    
    This class provides methods to:
    - Generate valid kernel configurations from a search space
    - Run NCU profiling to collect performance metrics
    - Calculate reward scores based on metrics
    - Find optimal configurations for different token counts
    - Export results in vLLM format
    """
    
    def __init__(
        self,
        config: Optional[DeepSeekMoEConfig] = None,
        search_space: Optional[Dict[str, List[int]]] = None,
        gpu_id: int = 7,
        use_ncu: bool = True,
        output_dir: str = "./optimized_configs",
        log_dir: str = "./logs"
    ):
        """
        Initialize the DeepSeek MoE optimizer.
        
        Args:
            config: DeepSeekMoEConfig with model specifications
            search_space: Dictionary of kernel parameter search space
            gpu_id: GPU ID to use for profiling
            use_ncu: Whether to use NCU for profiling (False for simulation)
            output_dir: Directory to save optimized configs
            log_dir: Directory to save experiment logs
        """
        self.config = config or DeepSeekMoEConfig()
        self.search_space = search_space or DEFAULT_SEARCH_SPACE.copy()
        self.gpu_id = gpu_id
        self.use_ncu = use_ncu
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        # Results storage
        self.best_configs: Dict[int, Dict[str, Any]] = {}
        self.best_rewards: Dict[int, float] = {}
        self.all_experiments: List[Dict[str, Any]] = []
        
        # NCU metrics (used for reward calculation)
        self.objective_weights = {
            "R_sm_throughput": 0.4,
            "R_dram_throughput": 0.3,
            "R_l1_hit_rate": 0.15,
            "R_l2_hit_rate": 0.15
        }
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Temporary files
        pid = os.getpid()
        self.temp_config_path = f"/tmp/deepseek_temp_config_{pid}.json"
        self.ncu_log_path = f"/tmp/deepseek_ncu_log_{pid}.csv"
        
        self._print_init_info()
    
    def _print_init_info(self):
        """Print initialization information."""
        print(f"[DeepSeekOptimizer] Initialized for {self.config.model_name}")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  MoE Intermediate Size: {self.config.moe_intermediate_size}")
        print(f"  Num Experts: {self.config.num_routed_experts} routed + {self.config.num_shared_experts} shared")
        print(f"  Top-K: {self.config.top_k}")
        print(f"  Profiling GPU: {self.gpu_id}")
        if not self.use_ncu:
            print(f"  Mode: Simulation (no NCU)")
    
    def validate_config(self, config: Dict[str, int]) -> bool:
        """
        Check if a kernel configuration fits H100 shared memory limit.
        
        Args:
            config: Kernel configuration dictionary
            
        Returns:
            True if config is valid, False otherwise
        """
        M = config.get("BLOCK_SIZE_M", 64)
        N = config.get("BLOCK_SIZE_N", 64)
        K = config.get("BLOCK_SIZE_K", 32)
        stages = config.get("num_stages", 4)
        
        # Calculate shared memory usage
        shared_mem = (M * K + K * N) * FP16_BYTES * stages
        
        if shared_mem > H100_SHARED_MEM_LIMIT:
            return False
        
        # Validate num_warps is power of 2
        num_warps = config.get("num_warps", 4)
        if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            return False
        
        return True
    
    def generate_all_configs(self) -> List[Dict[str, int]]:
        """
        Generate all valid kernel configurations from the search space.
        
        Returns:
            List of valid kernel configuration dictionaries
        """
        configs = []
        
        for M in self.search_space["BLOCK_SIZE_M"]:
            for N in self.search_space["BLOCK_SIZE_N"]:
                for K in self.search_space["BLOCK_SIZE_K"]:
                    for warps in self.search_space["num_warps"]:
                        for stages in self.search_space["num_stages"]:
                            config = {
                                "BLOCK_SIZE_M": M,
                                "BLOCK_SIZE_N": N,
                                "BLOCK_SIZE_K": K,
                                "GROUP_SIZE_M": 8,  # Default value
                                "num_warps": warps,
                                "num_stages": stages
                            }
                            if self.validate_config(config):
                                configs.append(config)
        
        return configs
    
    def run_ncu_profile(
        self,
        config: Dict[str, int],
        num_tokens: int
    ) -> Optional[Dict[str, float]]:
        """
        Run NCU profiling for a specific configuration.
        
        Args:
            config: Kernel configuration dictionary
            num_tokens: Number of tokens for the benchmark
            
        Returns:
            Dictionary with metrics (sm, dram, l1, l2) or None on failure
        """
        if not self.use_ncu:
            return self._simulate_metrics(config, num_tokens)
        
        # Write config to temp file
        with open(self.temp_config_path, "w") as f:
            json.dump(config, f)
        
        # Build NCU command
        ncu_command = [
            "ncu", "--csv",
            "--kernel-name", "fused_moe_kernel",
            "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", self.ncu_log_path,
        ]
        
        # Build Python command
        python_command = [
            "python", "run_kernel_benchmark.py",
            "--config-path", self.temp_config_path,
            "--num-experts", str(self.config.total_experts),
            "--top-k", str(self.config.top_k),
            "--hidden-size", str(self.config.hidden_size),
            "--inter-size", str(self.config.moe_intermediate_size),
            "--num-tokens", str(num_tokens),
            "--dtype", self.config.dtype,
            "--num-iters", "1",
            "--num-warmup-iters", "1",
        ]
        
        full_command = ncu_command + python_command
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        try:
            subprocess.run(
                full_command,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env=env
            )
            
            return self._parse_ncu_log()
            
        except subprocess.CalledProcessError as e:
            print(f"    NCU failed: {e.stderr[:200] if e.stderr else 'Unknown error'}")
            return None
        except subprocess.TimeoutExpired:
            print(f"    NCU timed out")
            return None
    
    def _parse_ncu_log(self) -> Optional[Dict[str, float]]:
        """Parse NCU log file and extract metrics."""
        if not os.path.exists(self.ncu_log_path):
            return None
        
        try:
            kernel_invocations = {}
            clean_csv_lines = []
            header_found = False
            
            with open(self.ncu_log_path, 'r') as f:
                for line in f:
                    if not header_found and line.strip().startswith('"ID"'):
                        header_found = True
                    if header_found:
                        clean_csv_lines.append(line)
            
            if not header_found:
                return None
            
            csv_reader = csv.DictReader(clean_csv_lines)
            
            for row in csv_reader:
                try:
                    invocation_key = f"{row['Kernel Name']}_{row['ID']}"
                    if invocation_key not in kernel_invocations:
                        kernel_invocations[invocation_key] = {}
                    
                    metric_name = row['Metric Name']
                    metric_value_str = row.get('Metric Value')
                    if metric_value_str is None:
                        continue
                    
                    metric_value = float(metric_value_str.replace('%', '').strip())
                    
                    if metric_name == 'sm__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['sm'] = metric_value
                    elif metric_name == 'dram__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['dram'] = metric_value
                    elif metric_name == 'l1tex__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l1'] = metric_value
                    elif metric_name == 'lts__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l2'] = metric_value
                        
                except (KeyError, ValueError, TypeError):
                    continue
            
            metrics = {'sm': [], 'dram': [], 'l1': [], 'l2': []}
            for invocation in kernel_invocations.values():
                if all(k in invocation for k in ['sm', 'dram', 'l1', 'l2']):
                    metrics['sm'].append(invocation['sm'])
                    metrics['dram'].append(invocation['dram'])
                    metrics['l1'].append(invocation['l1'])
                    metrics['l2'].append(invocation['l2'])
            
            if not metrics['sm']:
                return None
            
            return {
                'sm': sum(metrics['sm']) / len(metrics['sm']),
                'dram': sum(metrics['dram']) / len(metrics['dram']),
                'l1': sum(metrics['l1']) / len(metrics['l1']),
                'l2': sum(metrics['l2']) / len(metrics['l2'])
            }
            
        except Exception as e:
            print(f"    NCU parse error: {e}")
            return None
    
    def _simulate_metrics(
        self,
        config: Dict[str, int],
        num_tokens: int
    ) -> Dict[str, float]:
        """
        Simulate NCU metrics for testing without GPU.
        
        The simulated metrics are based on typical patterns observed:
        - Larger block sizes generally improve throughput
        - More stages improve memory latency hiding
        - There's a sweet spot for num_warps
        """
        M = config.get("BLOCK_SIZE_M", 64)
        N = config.get("BLOCK_SIZE_N", 64)
        K = config.get("BLOCK_SIZE_K", 32)
        warps = config.get("num_warps", 4)
        stages = config.get("num_stages", 4)
        
        # Base metrics
        base_sm = 25.0
        base_dram = 35.0
        base_l1 = 75.0
        base_l2 = 65.0
        
        # Block size effects
        size_factor = (M + N) / 256.0
        base_sm += size_factor * 15
        base_dram += size_factor * 10
        
        # K affects L1 utilization
        base_l1 += (K / 64.0) * 5
        
        # Warps affect occupancy
        if warps == 8:
            base_sm += 5
        elif warps == 16:
            base_sm += 3  # Diminishing returns
        
        # Stages affect latency hiding
        base_dram += stages * 2
        base_l2 += stages * 1.5
        
        # Token count effects
        token_factor = min(num_tokens / 1024.0, 1.0)
        base_sm += token_factor * 10
        base_dram += token_factor * 8
        
        # Add some randomness
        random.seed(hash((M, N, K, warps, stages, num_tokens)))
        base_sm += random.uniform(-3, 3)
        base_dram += random.uniform(-2, 2)
        base_l1 += random.uniform(-2, 2)
        base_l2 += random.uniform(-2, 2)
        
        return {
            'sm': min(max(base_sm, 0), 100),
            'dram': min(max(base_dram, 0), 100),
            'l1': min(max(base_l1, 0), 100),
            'l2': min(max(base_l2, 0), 100)
        }
    
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        """
        Calculate reward from profiling metrics.
        
        Args:
            metrics: Dictionary with sm, dram, l1, l2 values
            
        Returns:
            Weighted reward score
        """
        reward = (
            metrics['sm'] * self.objective_weights['R_sm_throughput'] +
            metrics['dram'] * self.objective_weights['R_dram_throughput'] +
            metrics['l1'] * self.objective_weights['R_l1_hit_rate'] +
            metrics['l2'] * self.objective_weights['R_l2_hit_rate']
        )
        return reward
    
    def optimize_for_token_count(
        self,
        num_tokens: int,
        configs: Optional[List[Dict[str, int]]] = None,
        max_configs: Optional[int] = None,
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best kernel configuration for a specific token count.
        
        Args:
            num_tokens: Number of tokens to optimize for
            configs: List of configs to test (None = generate all)
            max_configs: Maximum number of configs to test (None = all)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with best config, metrics, and reward
        """
        if configs is None:
            configs = self.generate_all_configs()
        
        if max_configs is not None and len(configs) > max_configs:
            # Randomly sample configs
            random.shuffle(configs)
            configs = configs[:max_configs]
        
        best_result = None
        best_reward = float('-inf')
        
        for i, config in enumerate(configs, 1):
            if verbose:
                config_str = f"M={config['BLOCK_SIZE_M']}, N={config['BLOCK_SIZE_N']}, K={config['BLOCK_SIZE_K']}, warps={config['num_warps']}, stages={config['num_stages']}"
                print(f"  Testing config {i}/{len(configs)}: {{{config_str}}}")
            
            metrics = self.run_ncu_profile(config, num_tokens)
            
            if metrics is None:
                if verbose:
                    print(f"    Skipped (profiling failed)")
                continue
            
            reward = self.calculate_reward(metrics)
            
            # Record experiment
            experiment = {
                'token_count': num_tokens,
                'config': config.copy(),
                'metrics': metrics.copy(),
                'reward': reward,
                'timestamp': datetime.now().isoformat()
            }
            self.all_experiments.append(experiment)
            
            if verbose:
                print(f"    Reward: {reward:.2f} (SM: {metrics['sm']:.1f}%, DRAM: {metrics['dram']:.1f}%)")
            
            if reward > best_reward:
                best_reward = reward
                best_result = {
                    'config': config.copy(),
                    'metrics': metrics.copy(),
                    'reward': reward
                }
                if verbose:
                    print(f"    [NEW BEST!]")
        
        # Update best configs
        if best_result is not None:
            self.best_configs[num_tokens] = best_result['config']
            self.best_rewards[num_tokens] = best_result['reward']
        
        return best_result
    
    def run_full_optimization(
        self,
        token_counts: Optional[List[int]] = None,
        max_configs: Optional[int] = None,
        quick: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run optimization for all specified token counts.
        
        Args:
            token_counts: List of token counts to test
            max_configs: Maximum configs to test per token count
            quick: Whether to use quick token count list
            
        Returns:
            Dictionary mapping token count to best result
        """
        if token_counts is None:
            token_counts = TOKEN_COUNTS_QUICK if quick else TOKEN_COUNTS_FULL
        
        configs = self.generate_all_configs()
        total_tokens = len(token_counts)
        
        print("")
        print("=" * 60)
        print(f"[DeepSeekOptimizer] Starting Full Optimization")
        print(f"  Model: {self.config.model_name}")
        print(f"  Token counts: {total_tokens}")
        print(f"  Quick mode: {quick}")
        print(f"  Total valid configs: {len(configs)}")
        if max_configs:
            print(f"  Max configs per token: {max_configs}")
        print("=" * 60)
        print("")
        
        results = {}
        
        for i, token_count in enumerate(token_counts, 1):
            print(f"[{i}/{total_tokens}] Token count: {token_count}")
            
            result = self.optimize_for_token_count(
                num_tokens=token_count,
                configs=configs.copy(),
                max_configs=max_configs,
                verbose=True
            )
            
            if result is not None:
                results[token_count] = result
            
            print("")
        
        return results
    
    def export_vllm_config(self, output_path: Optional[str] = None) -> str:
        """
        Export optimized configs in vLLM format.
        
        Args:
            output_path: Path to save the config (None = auto-generate)
            
        Returns:
            Path to the saved config file
        """
        if output_path is None:
            filename = f"E={self.config.total_experts},N={self.config.inter_size},device_name=NVIDIA_H100_80GB_HBM3.json"
            output_path = os.path.join(self.output_dir, filename)
        
        # Build vLLM format config
        vllm_config = {}
        for token_count in sorted(self.best_configs.keys()):
            vllm_config[str(token_count)] = self.best_configs[token_count]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(vllm_config, f, indent=2)
        
        print(f"[DeepSeekOptimizer] vLLM config saved to: {output_path}")
        return output_path
    
    def export_experiment_log(self, output_path: Optional[str] = None) -> str:
        """
        Export all experiment data for analysis.
        
        Args:
            output_path: Path to save the log (None = auto-generate)
            
        Returns:
            Path to the saved log file
        """
        if output_path is None:
            output_path = os.path.join(self.log_dir, "deepseek_experiments.json")
        
        log_data = {
            "model_config": asdict(self.config),
            "search_space": self.search_space,
            "objective_weights": self.objective_weights,
            "best_results": {
                str(tc): {
                    "config": self.best_configs.get(tc),
                    "reward": self.best_rewards.get(tc)
                }
                for tc in self.best_configs.keys()
            },
            "all_experiments": self.all_experiments,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"[DeepSeekOptimizer] Experiment log saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print optimization summary."""
        print("")
        print("=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Experts: {self.config.num_routed_experts} routed + {self.config.num_shared_experts} shared")
        print(f"Top-K: {self.config.top_k}")
        print(f"MoE Intermediate Size: {self.config.moe_intermediate_size}")
        print("")
        print("Best configurations found:")
        
        for token_count in sorted(self.best_configs.keys()):
            config = self.best_configs[token_count]
            reward = self.best_rewards[token_count]
            print(f"  {token_count:>5} tokens: reward={reward:.2f}, "
                  f"M={config['BLOCK_SIZE_M']}, N={config['BLOCK_SIZE_N']}, "
                  f"K={config['BLOCK_SIZE_K']}, warps={config['num_warps']}, "
                  f"stages={config['num_stages']}")
        
        print("=" * 60)
    
    def cleanup(self):
        """Clean up temporary files."""
        for path in [self.temp_config_path, self.ncu_log_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def run_vllm_benchmark(
    config_path: str,
    model_name: str = "deepseek-ai/deepseek-moe-16b-base",
    gpu_id: int = 7,
    num_prompts: int = 100
) -> Optional[float]:
    """
    Run vLLM throughput benchmark with optimized kernel config.
    
    Args:
        config_path: Path to vLLM config JSON
        model_name: Name of the model to benchmark
        gpu_id: GPU ID to use
        num_prompts: Number of prompts for benchmark
        
    Returns:
        Throughput in tokens/sec or None on failure
    """
    print(f"\n[Benchmark] Running vLLM throughput test...")
    print(f"  Model: {model_name}")
    print(f"  Config: {config_path}")
    print(f"  GPU: {gpu_id}")
    
    # Copy config to vLLM path if available
    try:
        import vllm
        vllm_config_dir = os.path.join(
            os.path.dirname(vllm.__file__),
            "model_executor/layers/fused_moe/configs/"
        )
        if os.path.exists(vllm_config_dir):
            import shutil
            dst = os.path.join(vllm_config_dir, os.path.basename(config_path))
            shutil.copy2(config_path, dst)
            print(f"  Copied config to: {dst}")
    except ImportError:
        print(f"  vLLM not installed, skipping config copy")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    command = [
        "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
        "--model", model_name,
        "--num-prompts", str(num_prompts),
        "--trust-remote-code",
        "--enforce-eager",
        "--tensor-parallel-size", "1",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
    ]
    
    try:
        print(f"  Running benchmark...")
        result = subprocess.run(
            command,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # Parse throughput from output
        for line in result.stdout.strip().split('\n'):
            if line.startswith("Throughput:"):
                try:
                    parts = line.split(',')
                    if len(parts) > 2:
                        output_tok_s = parts[2].strip().split(' ')[0]
                        throughput = float(output_tok_s)
                        print(f"  Throughput: {throughput:.2f} tokens/sec")
                        return throughput
                except (IndexError, ValueError):
                    pass
        
        print(f"  Could not parse throughput from output")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"  Benchmark failed: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-MoE-16B-Base kernel parameter optimization"
    )
    
    parser.add_argument(
        "--gpu", type=int, default=7,
        help="GPU ID for profiling (default: 7)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use quick token count list (6 values instead of 18)"
    )
    parser.add_argument(
        "--no-ncu", action="store_true",
        help="Disable NCU profiling (simulation mode)"
    )
    parser.add_argument(
        "--max-configs", type=int, default=None,
        help="Maximum configs to test per token count"
    )
    parser.add_argument(
        "--run-benchmark", action="store_true",
        help="Run vLLM throughput benchmark after optimization"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./optimized_configs",
        help="Directory for output configs"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory for experiment logs"
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = DeepSeekMoEOptimizer(
        config=DeepSeekMoEConfig(),
        gpu_id=args.gpu,
        use_ncu=not args.no_ncu,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )
    
    try:
        # Run optimization
        optimizer.run_full_optimization(
            quick=args.quick,
            max_configs=args.max_configs
        )
        
        # Print summary
        optimizer.print_summary()
        
        # Export results
        config_path = optimizer.export_vllm_config()
        optimizer.export_experiment_log()
        
        print(f"\nConfig exported to: {config_path}")
        
        # Run benchmark if requested
        if args.run_benchmark:
            run_vllm_benchmark(
                config_path=config_path,
                model_name=optimizer.config.model_name,
                gpu_id=args.gpu
            )
        
    finally:
        optimizer.cleanup()
    
    print("\n[DeepSeekOptimizer] Optimization complete.")


if __name__ == "__main__":
    main()
