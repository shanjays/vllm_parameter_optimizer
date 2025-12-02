"""
Server Parameter Optimizer - Main Entry Point

Main optimization script for vLLM server parameters.
Coordinates all components (LLM, profiling worker, config exporter, feedback collector, visualizer)
to find optimal --max-num-seqs and --max-num-batched-tokens configurations.

Target: NVIDIA A100 40GB with meta-llama/Llama-3.1-8B-Instruct
Benchmark Duration: 20 minutes per configuration
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from .thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSample
from .visualization import ThermalVisualizer
from .server_profiling_worker import (
    ServerProfilingWorkerLocal,
    BenchmarkResult,
    RAY_AVAILABLE
)
from .server_config_exporter import ServerConfigExporter
from .server_feedback_collector import ServerFeedbackCollector
from .server_meta_controller import ServerMetaController, PARAM_SPACE

# Try to import Ray
if RAY_AVAILABLE:
    import ray
    from .server_profiling_worker import ServerProfilingWorker


# Default configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GPU_TYPE = "NVIDIA A100 40GB"
BENCHMARK_DURATION_MINUTES = 20
NUM_ITERATIONS = 8
OUTPUT_DIR = "./server_optimization_results"

# Thermal thresholds for A100 40GB
THERMAL_CONFIG = ThermalConfig(
    max_safe_temp=83.0,
    target_sustained_temp=75.0,
    warning_temp=80.0,
    max_power=400.0,
    total_memory_gb=40.0,
    gpu_name=GPU_TYPE
)


class ServerParameterOptimizer:
    """Main optimizer for vLLM server parameters.
    
    Coordinates all components to find optimal configurations:
    - LLM meta-controller generates configurations
    - Profiling worker runs benchmarks
    - Thermal monitor tracks GPU temperature/power
    - Visualizer creates thermal plots
    - Feedback collector tracks results for LLM learning
    - Config exporter saves results and launch scripts
    
    Example:
        optimizer = ServerParameterOptimizer()
        optimizer.run_optimization()
        optimizer.print_final_summary()
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        gpu_type: str = GPU_TYPE,
        benchmark_duration_minutes: int = BENCHMARK_DURATION_MINUTES,
        num_iterations: int = NUM_ITERATIONS,
        output_dir: str = OUTPUT_DIR,
        gpu_id: int = 0,
        use_ray: bool = False,
        thermal_config: Optional[ThermalConfig] = None
    ):
        """Initialize the server parameter optimizer.
        
        Args:
            model_name: Model to benchmark
            gpu_type: GPU name for metadata
            benchmark_duration_minutes: Duration of each benchmark run
            num_iterations: Number of optimization iterations
            output_dir: Directory to save results
            gpu_id: GPU device index
            use_ray: Whether to use Ray for distributed profiling
            thermal_config: Custom thermal configuration
        """
        self.model_name = model_name
        self.gpu_type = gpu_type
        self.benchmark_duration_minutes = benchmark_duration_minutes
        self.num_iterations = num_iterations
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.use_ray = use_ray and RAY_AVAILABLE
        self.thermal_config = thermal_config or THERMAL_CONFIG
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "thermal_plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "launch_scripts"), exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Track optimization state
        self.current_iteration = 0
        self.total_benchmarks = 0
        self.start_time: Optional[float] = None
        
        self._print_header()
    
    def _init_components(self) -> None:
        """Initialize all optimizer components."""
        print("[ServerOptimizer] Initializing components...")
        
        # LLM meta-controller
        self.meta_controller = ServerMetaController()
        
        # Profiling worker
        if self.use_ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False)
            self.worker = ServerProfilingWorker.remote(
                gpu_id=self.gpu_id,
                model=self.model_name,
                thermal_config=self.thermal_config
            )
            print(f"[ServerOptimizer] Using Ray profiling worker on GPU {self.gpu_id}")
        else:
            self.worker = ServerProfilingWorkerLocal(
                gpu_id=self.gpu_id,
                model=self.model_name,
                thermal_config=self.thermal_config
            )
            print(f"[ServerOptimizer] Using local profiling worker on GPU {self.gpu_id}")
        
        # Config exporter
        self.config_exporter = ServerConfigExporter(
            output_dir=self.output_dir,
            model=self.model_name,
            gpu=self.gpu_type
        )
        
        # Feedback collector
        self.feedback_collector = ServerFeedbackCollector(
            state_file=os.path.join(self.output_dir, "feedback_state.json")
        )
        
        # Thermal visualizer
        self.visualizer = ThermalVisualizer()
        
        print("[ServerOptimizer] Components initialized")
    
    def _print_header(self) -> None:
        """Print optimization header."""
        print("\n" + "═" * 70)
        print(f"Server Parameter Optimization for {self.model_name}")
        print(f"GPU: {self.gpu_type}")
        print(f"Benchmark Duration: {self.benchmark_duration_minutes} minutes per config")
        print("═" * 70 + "\n")
    
    def run_optimization(self, num_iterations: Optional[int] = None) -> None:
        """Run the full optimization loop.
        
        Args:
            num_iterations: Override default number of iterations
        """
        iterations = num_iterations or self.num_iterations
        self.start_time = time.time()
        
        print(f"[ServerOptimizer] Starting optimization ({iterations} iterations)")
        
        for i in range(1, iterations + 1):
            self._run_iteration(i)
        
        # Save final results
        self._save_final_results()
        
        # Print summary
        self.print_final_summary()
    
    def _run_iteration(self, iteration_num: int) -> None:
        """Run a single optimization iteration.
        
        Args:
            iteration_num: Current iteration number
        """
        self.current_iteration = iteration_num
        
        print(f"\n--- [ServerOptimizer] Iteration {iteration_num}/{self.num_iterations} ---")
        
        # Generate configurations using LLM
        print("[ServerOptimizer] LLM generating configurations...")
        configs = self.meta_controller.generate_configs(self.feedback_collector)
        print(f"[ServerOptimizer] LLM suggested {len(configs)} configurations")
        
        # Benchmark each configuration
        iteration_configs = []
        iteration_results = []
        
        for config in configs:
            result = self._benchmark_config(config)
            iteration_configs.append({
                'max_num_seqs': config.get('max_num_seqs'),
                'max_num_batched_tokens': config.get('max_num_batched_tokens'),
                'name': config.get('name', 'unnamed')
            })
            iteration_results.append(result)
            
            # Update best configs
            self.config_exporter.update_best_configs(result)
            
            self.total_benchmarks += 1
        
        # Update feedback collector
        self.feedback_collector.add_iteration(iteration_configs, iteration_results)
        
        print(f"[ServerOptimizer] Iteration {iteration_num} complete")
    
    def _benchmark_config(self, config: Dict[str, Any]) -> BenchmarkResult:
        """Run benchmark for a single configuration.
        
        Args:
            config: Configuration dictionary with max_num_seqs and max_num_batched_tokens
            
        Returns:
            BenchmarkResult with throughput and thermal data
        """
        max_num_seqs = config.get('max_num_seqs', 64)
        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        config_name = config.get('name', f'seqs{max_num_seqs}_tokens{max_num_batched_tokens}')
        
        print(f"\n[ServerOptimizer] Testing: seqs={max_num_seqs}, tokens={max_num_batched_tokens}")
        print("[ThermalMonitor] Started monitoring (sampling every 1s)")
        
        # Run benchmark
        print(f"[Benchmark] Running {self.benchmark_duration_minutes}-minute throughput test...")
        
        if self.use_ray:
            result = ray.get(self.worker.run_benchmark.remote(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                duration_minutes=self.benchmark_duration_minutes
            ))
        else:
            result = self.worker.run_benchmark(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                duration_minutes=self.benchmark_duration_minutes
            )
        
        # Print results
        if result.error:
            print(f"[Benchmark] Error: {result.error}")
        else:
            print(f"[Benchmark] Throughput: {result.throughput:.1f} tokens/sec")
        
        # Print thermal summary
        if result.thermal_summary:
            ts = result.thermal_summary
            temp_min = ts.temp_min if hasattr(ts, 'temp_min') else ts.get('temp_min', 0)
            temp_max = ts.temp_max if hasattr(ts, 'temp_max') else ts.get('temp_max', 0)
            temp_avg = ts.temp_avg if hasattr(ts, 'temp_avg') else ts.get('temp_avg', 0)
            print(f"[ThermalMonitor] Temp: min={temp_min:.0f}°C, max={temp_max:.0f}°C, avg={temp_avg:.1f}°C")
        
        # Save thermal plot
        self._save_thermal_plot(config, result)
        
        # Print thermal status
        if result.is_thermally_safe:
            temp_max = 0
            if result.thermal_summary:
                ts = result.thermal_summary
                temp_max = ts.temp_max if hasattr(ts, 'temp_max') else ts.get('temp_max', 0)
            print(f"[ServerOptimizer] ✓ Thermally safe (max {temp_max:.0f}°C < {self.thermal_config.target_sustained_temp}°C target)")
            if self._is_new_best_sustained(result):
                print("[ServerOptimizer] → New best SUSTAINED config!")
        else:
            print(f"[ServerOptimizer] ⚠️ Above thermal target")
        
        if self._is_new_best_aggressive(result):
            print("[ServerOptimizer] → New best AGGRESSIVE config!")
        
        return result
    
    def _save_thermal_plot(
        self,
        config: Dict[str, Any],
        result: BenchmarkResult
    ) -> None:
        """Save thermal visualization plot.
        
        Args:
            config: Configuration dictionary
            result: Benchmark result with thermal data
        """
        if not self.visualizer.is_available():
            return
        
        # Get thermal samples from worker
        if self.use_ray:
            samples_dict = ray.get(self.worker.get_thermal_samples.remote())
        else:
            samples_dict = self.worker.get_thermal_samples()
        
        if not samples_dict:
            return
        
        # Convert dicts back to ThermalSample objects
        samples = [
            ThermalSample(**s) for s in samples_dict
        ]
        
        # Generate plot filename
        max_num_seqs = config.get('max_num_seqs', 0)
        max_num_batched_tokens = config.get('max_num_batched_tokens', 0)
        plot_filename = f"seqs{max_num_seqs}_tokens{max_num_batched_tokens}.png"
        plot_path = os.path.join(self.output_dir, "thermal_plots", plot_filename)
        
        # Create title
        title = f"Thermal Profile: seqs={max_num_seqs}, tokens={max_num_batched_tokens}"
        
        # Get thermal summary
        if result.thermal_summary:
            if hasattr(result.thermal_summary, 'to_dict'):
                summary = result.thermal_summary
            else:
                # Create a mock summary object for visualization
                from dataclasses import dataclass
                @dataclass
                class TempSummary:
                    duration_seconds: float = 0
                    sample_count: int = 0
                    temp_min: float = 0
                    temp_max: float = 0
                    temp_avg: float = 0
                    temp_final: float = 0
                    power_min: float = 0
                    power_max: float = 0
                    power_avg: float = 0
                    memory_max_used_mb: float = 0
                    memory_max_used_pct: float = 0
                    gpu_util_avg: float = 0
                    memory_util_avg: float = 0
                    is_thermally_safe: bool = True
                    max_temp_exceeded: bool = False
                    throttling_detected: bool = False
                    time_above_target: float = 0
                    
                    def to_dict(self):
                        return self.__dict__
                
                ts = result.thermal_summary
                summary = TempSummary(
                    duration_seconds=ts.get('duration_seconds', 0),
                    sample_count=ts.get('sample_count', 0),
                    temp_min=ts.get('temp_min', 0),
                    temp_max=ts.get('temp_max', 0),
                    temp_avg=ts.get('temp_avg', 0),
                    temp_final=ts.get('temp_final', 0),
                    power_min=ts.get('power_min', 0),
                    power_max=ts.get('power_max', 0),
                    power_avg=ts.get('power_avg', 0),
                    memory_max_used_mb=ts.get('memory_max_used_mb', 0),
                    memory_max_used_pct=ts.get('memory_max_used_pct', 0),
                    gpu_util_avg=ts.get('gpu_util_avg', 0),
                    memory_util_avg=ts.get('memory_util_avg', 0),
                    is_thermally_safe=ts.get('is_thermally_safe', True),
                    max_temp_exceeded=ts.get('max_temp_exceeded', False),
                    throttling_detected=ts.get('throttling_detected', False),
                    time_above_target=ts.get('time_above_target', 0)
                )
        else:
            return
        
        # Save plot
        success = self.visualizer.save_thermal_plot(
            samples=samples,
            summary=summary,
            output_path=plot_path,
            title=title,
            thermal_config=self.thermal_config
        )
        
        if success:
            print(f"[ThermalMonitor] Saved plot: thermal_plots/{plot_filename}")
    
    def _is_new_best_aggressive(self, result: BenchmarkResult) -> bool:
        """Check if result is a new best aggressive config.
        
        Args:
            result: Benchmark result
            
        Returns:
            True if this is a new best aggressive config
        """
        if self.config_exporter.best_aggressive is None:
            return True
        return result.throughput > self.config_exporter.best_aggressive.throughput
    
    def _is_new_best_sustained(self, result: BenchmarkResult) -> bool:
        """Check if result is a new best sustained config.
        
        Args:
            result: Benchmark result
            
        Returns:
            True if this is a new best sustained config
        """
        if not result.is_thermally_safe:
            return False
        if self.config_exporter.best_sustained is None:
            return True
        return result.throughput > self.config_exporter.best_sustained.throughput
    
    def _save_final_results(self) -> None:
        """Save all final configuration files."""
        print("\n[ServerOptimizer] Saving final results...")
        
        # Save configs and launch scripts
        self.config_exporter.save_configs()
        
        # Save complete optimization results
        results_data = {
            'optimization_summary': {
                'model': self.model_name,
                'gpu': self.gpu_type,
                'iterations_completed': self.current_iteration,
                'total_benchmarks': self.total_benchmarks,
                'benchmark_duration_minutes': self.benchmark_duration_minutes,
                'total_duration_seconds': time.time() - self.start_time if self.start_time else 0,
                'completed_at': datetime.now().isoformat()
            },
            'best_aggressive': self.config_exporter.best_aggressive.to_dict() if self.config_exporter.best_aggressive else None,
            'best_sustained': self.config_exporter.best_sustained.to_dict() if self.config_exporter.best_sustained else None,
            'feedback_summary': self.feedback_collector.get_summary()
        }
        
        results_path = os.path.join(self.output_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"[ServerOptimizer] Results saved to: {self.output_dir}")
    
    def print_final_summary(self) -> None:
        """Print comprehensive final summary."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "═" * 70)
        print("OPTIMIZATION COMPLETE - FINAL SUMMARY")
        print("═" * 70)
        
        print(f"\nModel: {self.model_name}")
        print(f"GPU: {self.gpu_type}")
        print(f"Total iterations: {self.current_iteration}")
        print(f"Total benchmarks: {self.total_benchmarks}")
        print(f"Total duration: {duration/3600:.1f} hours")
        
        # Best aggressive config
        print("\n🚀 BEST AGGRESSIVE CONFIG (Maximum Throughput):")
        if self.config_exporter.best_aggressive:
            ba = self.config_exporter.best_aggressive
            print(f"   --max-num-seqs {ba.max_num_seqs}")
            print(f"   --max-num-batched-tokens {ba.max_num_batched_tokens}")
            print(f"   Throughput: {ba.throughput:.1f} tokens/sec")
            if ba.thermal_summary:
                ts = ba.thermal_summary
                temp_max = ts.get('temp_max', 0) if isinstance(ts, dict) else ts.temp_max
                print(f"   Max Temp: {temp_max:.0f}°C")
            print("   ⚠️  May cause thermal throttling in long runs!")
        else:
            print("   No aggressive config found")
        
        # Best sustained config
        print("\n🌡️  BEST SUSTAINED CONFIG (Optimal for Long-Running):")
        if self.config_exporter.best_sustained:
            bs = self.config_exporter.best_sustained
            print(f"   --max-num-seqs {bs.max_num_seqs}")
            print(f"   --max-num-batched-tokens {bs.max_num_batched_tokens}")
            print(f"   Throughput: {bs.throughput:.1f} tokens/sec")
            if bs.thermal_summary:
                ts = bs.thermal_summary
                temp_max = ts.get('temp_max', 0) if isinstance(ts, dict) else ts.temp_max
                print(f"   Max Temp: {temp_max:.0f}°C")
            print("   ✓ Safe for continuous operation")
        else:
            print("   No thermally-safe config found")
        
        # Output files
        print(f"\n📊 Thermal plots saved to: {self.output_dir}/thermal_plots/")
        print(f"📈 Full results saved to: {self.output_dir}/optimization_results.json")
        print(f"🚀 Launch scripts saved to: {self.output_dir}/launch_scripts/")
        
        print("═" * 70 + "\n")


def main():
    """Main entry point for the server parameter optimizer."""
    print("\n" + "═" * 70)
    print("          SERVER PARAMETER OPTIMIZER FOR VLLM")
    print("═" * 70)
    
    optimizer = ServerParameterOptimizer(
        model_name=MODEL_NAME,
        gpu_type=GPU_TYPE,
        benchmark_duration_minutes=BENCHMARK_DURATION_MINUTES,
        num_iterations=NUM_ITERATIONS,
        output_dir=OUTPUT_DIR
    )
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
