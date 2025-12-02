"""
Server Configuration Exporter for vLLM

Exports optimized server configurations for vLLM serving.
Tracks best-performing configurations and generates:
- JSON config files (aggressive and sustained modes)
- Bash launch scripts for easy deployment
- Complete optimization results log

Output files:
- config_aggressive.json: Maximum throughput config
- config_sustained.json: Thermal-safe config for 24/7 operation
- optimization_results.json: Full results log
- launch_scripts/launch_aggressive.sh: vLLM server launch script
- launch_scripts/launch_sustained.sh: vLLM server launch script
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path for script execution
if __name__ == "__main__" or "." not in __name__:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with fallback for both module and script execution
try:
    from .thermal_monitor import ThermalSummary
except ImportError:
    from thermal_monitor import ThermalSummary


@dataclass
class ServerConfig:
    """Optimized server configuration."""
    mode: str  # 'aggressive' or 'sustained'
    max_num_seqs: int
    max_num_batched_tokens: int
    throughput: float  # tokens/sec
    thermal_summary: Optional[Dict[str, Any]]  # Thermal stats during benchmark
    model: str
    gpu: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)


# Default target configuration
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GPU = "NVIDIA A100 40GB"


class ServerConfigExporter:
    """Exports optimized server configurations and generates launch scripts.
    
    Tracks two optimization modes:
    - Aggressive: Maximum throughput (may cause thermal throttling)
    - Sustained: Thermally-safe for 24/7 operation
    
    Example:
        exporter = ServerConfigExporter(output_dir="./server_configs")
        
        # Update with benchmark results
        exporter.update_best_configs(benchmark_result)
        
        # Save configs and generate scripts
        exporter.save_configs()
        exporter.print_summary()
    """
    
    def __init__(
        self,
        output_dir: str = "./server_configs",
        model: str = DEFAULT_MODEL,
        gpu: str = DEFAULT_GPU
    ):
        """Initialize the config exporter.
        
        Args:
            output_dir: Directory to save config files and scripts
            model: Model name for launch scripts
            gpu: GPU name for config metadata
        """
        self.output_dir = output_dir
        self.model = model
        self.gpu = gpu
        
        # Best configurations found
        self.best_aggressive: Optional[ServerConfig] = None
        self.best_sustained: Optional[ServerConfig] = None
        
        # All results for logging
        self.all_results: List[Dict[str, Any]] = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "launch_scripts"), exist_ok=True)
        
        print(f"[ServerConfigExporter] Output directory: {output_dir}")
    
    def update_best_configs(self, result) -> bool:
        """Update best configurations based on benchmark result.
        
        Args:
            result: BenchmarkResult from ServerProfilingWorker
            
        Returns:
            True if any best config was updated
        """
        updated = False
        timestamp = datetime.now().isoformat()
        
        # Extract thermal summary dict
        thermal_summary_dict = None
        if hasattr(result, 'thermal_summary') and result.thermal_summary:
            if hasattr(result.thermal_summary, 'to_dict'):
                thermal_summary_dict = result.thermal_summary.to_dict()
            elif isinstance(result.thermal_summary, dict):
                thermal_summary_dict = result.thermal_summary
        
        # Extract error info for record
        error = result.error if hasattr(result, 'error') else ''
        error_type = result.error_type.value if hasattr(result, 'error_type') and result.error_type else 'none'
        penalty = result.penalty if hasattr(result, 'penalty') else 0.0
        
        # Record result
        result_record = {
            'config': result.config if hasattr(result, 'config') else {},
            'throughput': result.throughput if hasattr(result, 'throughput') else 0.0,
            'is_thermally_safe': result.is_thermally_safe if hasattr(result, 'is_thermally_safe') else False,
            'thermal_summary': thermal_summary_dict,
            'duration': result.duration if hasattr(result, 'duration') else 0.0,
            'error': error,
            'error_type': error_type,
            'penalty': penalty,
            'timestamp': timestamp
        }
        self.all_results.append(result_record)
        
        # Skip failed benchmarks entirely - use is_successful if available, fallback to error check
        is_successful = True
        if hasattr(result, 'is_successful'):
            is_successful = result.is_successful
        elif hasattr(result, 'error') and result.error:
            is_successful = False
        elif hasattr(result, 'throughput') and result.throughput <= 0.0:
            is_successful = False
            
        if not is_successful:
            penalty_str = f" (penalty: {penalty})" if penalty != 0.0 else ""
            print(f"[ServerConfigExporter] Skipping failed benchmark{penalty_str}")
            return False
        
        throughput = result.throughput if hasattr(result, 'throughput') else 0.0
        config = result.config if hasattr(result, 'config') else {}
        
        # Update aggressive (max throughput regardless of thermals)
        if self.best_aggressive is None or throughput > self.best_aggressive.throughput:
            self.best_aggressive = ServerConfig(
                mode='aggressive',
                max_num_seqs=config.get('max_num_seqs', 256),
                max_num_batched_tokens=config.get('max_num_batched_tokens', 8192),
                throughput=throughput,
                thermal_summary=thermal_summary_dict,
                model=self.model,
                gpu=self.gpu,
                timestamp=timestamp
            )
            print(f"[ServerConfigExporter] New best AGGRESSIVE: {throughput:.2f} tokens/sec")
            updated = True
        
        # Update sustained (thermally safe only)
        is_thermally_safe = result.is_thermally_safe if hasattr(result, 'is_thermally_safe') else False
        if is_thermally_safe:
            if self.best_sustained is None or throughput > self.best_sustained.throughput:
                self.best_sustained = ServerConfig(
                    mode='sustained',
                    max_num_seqs=config.get('max_num_seqs', 256),
                    max_num_batched_tokens=config.get('max_num_batched_tokens', 8192),
                    throughput=throughput,
                    thermal_summary=thermal_summary_dict,
                    model=self.model,
                    gpu=self.gpu,
                    timestamp=timestamp
                )
                print(f"[ServerConfigExporter] New best SUSTAINED: {throughput:.2f} tokens/sec (thermally safe)")
                updated = True
        
        return updated
    
    def save_configs(self) -> None:
        """Save all config files and generate launch scripts."""
        self._save_aggressive_config()
        self._save_sustained_config()
        self._save_all_results()
        self._generate_launch_scripts()
        print(f"[ServerConfigExporter] Saved configs to: {self.output_dir}")
    
    def _save_aggressive_config(self) -> None:
        """Save aggressive (max throughput) configuration."""
        if self.best_aggressive is None:
            print("[ServerConfigExporter] No aggressive config to save")
            return
        
        config_data = {
            'mode': 'aggressive',
            'warning': 'This config maximizes throughput but may cause thermal throttling. '
                       'Monitor GPU temperature during operation.',
            'config': {
                'max_num_seqs': self.best_aggressive.max_num_seqs,
                'max_num_batched_tokens': self.best_aggressive.max_num_batched_tokens,
            },
            'performance': {
                'throughput_tokens_per_sec': self.best_aggressive.throughput,
            },
            'thermal_summary': self.best_aggressive.thermal_summary,
            'metadata': {
                'model': self.best_aggressive.model,
                'gpu': self.best_aggressive.gpu,
                'timestamp': self.best_aggressive.timestamp
            }
        }
        
        config_path = os.path.join(self.output_dir, "config_aggressive.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"[ServerConfigExporter] Saved aggressive config: {config_path}")
    
    def _save_sustained_config(self) -> None:
        """Save sustained (thermally-safe) configuration."""
        if self.best_sustained is None:
            print("[ServerConfigExporter] No sustained config to save")
            return
        
        config_data = {
            'mode': 'sustained',
            'description': 'This config is thermally safe for 24/7 continuous operation. '
                           'GPU temperature stays below target sustained threshold.',
            'config': {
                'max_num_seqs': self.best_sustained.max_num_seqs,
                'max_num_batched_tokens': self.best_sustained.max_num_batched_tokens,
            },
            'performance': {
                'throughput_tokens_per_sec': self.best_sustained.throughput,
            },
            'thermal_summary': self.best_sustained.thermal_summary,
            'metadata': {
                'model': self.best_sustained.model,
                'gpu': self.best_sustained.gpu,
                'timestamp': self.best_sustained.timestamp
            }
        }
        
        config_path = os.path.join(self.output_dir, "config_sustained.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"[ServerConfigExporter] Saved sustained config: {config_path}")
    
    def _save_all_results(self) -> None:
        """Save complete optimization results log."""
        results_data = {
            'optimization_summary': {
                'total_benchmarks': len(self.all_results),
                'best_aggressive_throughput': self.best_aggressive.throughput if self.best_aggressive else None,
                'best_sustained_throughput': self.best_sustained.throughput if self.best_sustained else None,
                'model': self.model,
                'gpu': self.gpu,
                'generated_at': datetime.now().isoformat()
            },
            'all_results': self.all_results
        }
        
        results_path = os.path.join(self.output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"[ServerConfigExporter] Saved optimization results: {results_path}")
    
    def _generate_launch_scripts(self) -> None:
        """Generate bash launch scripts for vLLM server."""
        scripts_dir = os.path.join(self.output_dir, "launch_scripts")
        
        # Generate aggressive launch script
        if self.best_aggressive:
            self._write_launch_script(
                config=self.best_aggressive,
                output_path=os.path.join(scripts_dir, "launch_aggressive.sh"),
                mode="aggressive"
            )
        
        # Generate sustained launch script
        if self.best_sustained:
            self._write_launch_script(
                config=self.best_sustained,
                output_path=os.path.join(scripts_dir, "launch_sustained.sh"),
                mode="sustained"
            )
    
    def _write_launch_script(
        self,
        config: ServerConfig,
        output_path: str,
        mode: str
    ) -> None:
        """Write a vLLM server launch script.
        
        Args:
            config: Server configuration
            output_path: Path to write the script
            mode: 'aggressive' or 'sustained'
        """
        if mode == "aggressive":
            header_comment = """# vLLM Server Launch Script - AGGRESSIVE MODE
# WARNING: This configuration maximizes throughput but may cause thermal throttling.
# Monitor GPU temperature during operation with: nvidia-smi -l 1
"""
        else:
            header_comment = """# vLLM Server Launch Script - SUSTAINED MODE
# This configuration is thermally safe for 24/7 continuous operation.
# GPU temperature will stay below the target sustained threshold.
"""
        
        script_content = f'''#!/bin/bash
{header_comment}
# Generated: {datetime.now().isoformat()}
# Model: {config.model}
# GPU: {config.gpu}
# Throughput: {config.throughput:.2f} tokens/sec

set -e

MODEL="{config.model}"
MAX_NUM_SEQS={config.max_num_seqs}
MAX_NUM_BATCHED_TOKENS={config.max_num_batched_tokens}

echo "Starting vLLM server in {mode.upper()} mode..."
echo "Model: $MODEL"
echo "Max sequences: $MAX_NUM_SEQS"
echo "Max batched tokens: $MAX_NUM_BATCHED_TOKENS"

python -m vllm.entrypoints.openai.api_server \\
    --model "$MODEL" \\
    --max-num-seqs $MAX_NUM_SEQS \\
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \\
    --trust-remote-code \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.90 \\
    --host 0.0.0.0 \\
    --port 8000
'''
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_path, 0o755)
        print(f"[ServerConfigExporter] Generated launch script: {output_path}")
    
    def print_summary(self) -> None:
        """Print optimization summary to console."""
        print("\n" + "=" * 70)
        print("              SERVER PARAMETER OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        print(f"\nModel: {self.model}")
        print(f"GPU: {self.gpu}")
        print(f"Total benchmarks: {len(self.all_results)}")
        
        print("\n--- AGGRESSIVE MODE (Max Throughput) ---")
        if self.best_aggressive:
            print(f"  max_num_seqs: {self.best_aggressive.max_num_seqs}")
            print(f"  max_num_batched_tokens: {self.best_aggressive.max_num_batched_tokens}")
            print(f"  Throughput: {self.best_aggressive.throughput:.2f} tokens/sec")
            if self.best_aggressive.thermal_summary:
                ts = self.best_aggressive.thermal_summary
                print(f"  Temp: avg={ts.get('temp_avg', 'N/A')}°C, max={ts.get('temp_max', 'N/A')}°C")
        else:
            print("  No aggressive config found")
        
        print("\n--- SUSTAINED MODE (Thermally Safe) ---")
        if self.best_sustained:
            print(f"  max_num_seqs: {self.best_sustained.max_num_seqs}")
            print(f"  max_num_batched_tokens: {self.best_sustained.max_num_batched_tokens}")
            print(f"  Throughput: {self.best_sustained.throughput:.2f} tokens/sec")
            if self.best_sustained.thermal_summary:
                ts = self.best_sustained.thermal_summary
                print(f"  Temp: avg={ts.get('temp_avg', 'N/A')}°C, max={ts.get('temp_max', 'N/A')}°C")
        else:
            print("  No thermally-safe config found")
        
        print(f"\nConfigs saved to: {self.output_dir}")
        print("=" * 70 + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary as dictionary.
        
        Returns:
            Dictionary with optimization summary
        """
        return {
            'total_benchmarks': len(self.all_results),
            'best_aggressive': self.best_aggressive.to_dict() if self.best_aggressive else None,
            'best_sustained': self.best_sustained.to_dict() if self.best_sustained else None,
            'model': self.model,
            'gpu': self.gpu,
            'output_dir': self.output_dir
        }
