"""
Server Parameter Optimizer - Main Entry Point

Main optimization script for vLLM server parameters.
Coordinates all components (LLM, profiling worker, config exporter, feedback collector, visualizer)
to find optimal --max-num-seqs and --max-num-batched-tokens configurations.

Target: NVIDIA H100 80GB with meta-llama/Llama-3.1-8B-Instruct
Benchmark Duration: 20 minutes per configuration
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path for script execution
if __name__ == "__main__" or "." not in __name__:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with fallback for both module and script execution
try:
    from .thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSample
except ImportError:
    from thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSample

try:
    from .visualization import ThermalVisualizer
except ImportError:
    from visualization import ThermalVisualizer

try:
    from .server_profiling_worker import (
        ServerProfilingWorkerLocal,
        BenchmarkResult,
        RAY_AVAILABLE,
        log_error_details
    )
except ImportError:
    from server_profiling_worker import (
        ServerProfilingWorkerLocal,
        BenchmarkResult,
        RAY_AVAILABLE,
        log_error_details
    )

try:
    from .server_config_exporter import ServerConfigExporter
except ImportError:
    from server_config_exporter import ServerConfigExporter

try:
    from .server_feedback_collector import ServerFeedbackCollector
except ImportError:
    from server_feedback_collector import ServerFeedbackCollector

try:
    from .server_meta_controller import ServerMetaController, PARAM_SPACE
except ImportError:
    from server_meta_controller import ServerMetaController, PARAM_SPACE

# Try to import Ray
if RAY_AVAILABLE:
    import ray
    try:
        from .server_profiling_worker import ServerProfilingWorker
    except ImportError:
        from server_profiling_worker import ServerProfilingWorker


# Default configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GPU_TYPE = "NVIDIA H100 80GB"
BENCHMARK_DURATION_MINUTES = 20
NUM_ITERATIONS = 8
OUTPUT_DIR = "./server_optimization_results"

# Logging separator width for consistent terminal output
SEPARATOR_WIDTH = 70

# GPU assignment (for multi-GPU systems)
LLM_GPU_ID = 0        # GPU for LLM meta-controller
BENCHMARK_GPU_ID = 1  # GPU for vLLM benchmarks

# Thermal thresholds for H100 80GB
THERMAL_CONFIG = ThermalConfig(
    max_safe_temp=85.0,
    target_sustained_temp=75.0,
    warning_temp=80.0,
    max_power=350.0,
    total_memory_gb=80.0,
    gpu_name=GPU_TYPE
)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices.
    
    Returns:
        List of GPU indices (e.g., [0, 1, 2]).
        Falls back to [0] if nvidia-smi fails (assumes at least one GPU).
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            if gpus:
                return gpus
    except Exception:
        pass
    # Default to GPU 0 if nvidia-smi is not available or fails.
    # This is intentional as most GPU systems have at least GPU 0.
    return [0]


def validate_gpu_assignment(llm_gpu: int, benchmark_gpu: int) -> None:
    """Validate GPU assignment and warn if using same GPU.
    
    Args:
        llm_gpu: GPU ID for LLM meta-controller
        benchmark_gpu: GPU ID for vLLM benchmarks
    """
    available = get_available_gpus()
    
    if llm_gpu not in available:
        print(f"[WARNING] LLM GPU {llm_gpu} not available. Available: {available}")
    if benchmark_gpu not in available:
        print(f"[WARNING] Benchmark GPU {benchmark_gpu} not available. Available: {available}")
    
    if llm_gpu == benchmark_gpu:
        print(f"\n{'='*60}")
        print(f"[WARNING] LLM and Benchmark using SAME GPU ({llm_gpu})!")
        print(f"This may cause VRAM conflicts and failures.")
        print(f"Consider using --llm-gpu and --benchmark-gpu with different values.")
        print(f"Available GPUs: {available}")
        print(f"{'='*60}\n")


def _get_thermal_value(thermal_summary: Any, key: str, default: float = 0.0) -> float:
    """Helper to safely extract a value from thermal summary.
    
    Args:
        thermal_summary: ThermalSummary object or dict
        key: Key to extract (e.g., 'temp_max', 'temp_avg')
        default: Default value if key not found
        
    Returns:
        Extracted value or default
    """
    if thermal_summary is None:
        return default
    if hasattr(thermal_summary, key):
        return getattr(thermal_summary, key)
    if isinstance(thermal_summary, dict):
        return thermal_summary.get(key, default)
    return default


class _ThermalSummaryProxy:
    """Proxy class to provide ThermalSummary-like interface from dict data.
    
    Used when thermal_summary is a dict (e.g., after JSON serialization)
    but we need an object with to_dict() method for visualization.
    """
    def __init__(self, data: Dict[str, Any]):
        self.duration_seconds = data.get('duration_seconds', 0.0)
        self.sample_count = data.get('sample_count', 0)
        self.temp_min = data.get('temp_min', 0.0)
        self.temp_max = data.get('temp_max', 0.0)
        self.temp_avg = data.get('temp_avg', 0.0)
        self.temp_final = data.get('temp_final', 0.0)
        self.power_min = data.get('power_min', 0.0)
        self.power_max = data.get('power_max', 0.0)
        self.power_avg = data.get('power_avg', 0.0)
        self.memory_max_used_mb = data.get('memory_max_used_mb', 0.0)
        self.memory_max_used_pct = data.get('memory_max_used_pct', 0.0)
        self.gpu_util_avg = data.get('gpu_util_avg', 0.0)
        self.memory_util_avg = data.get('memory_util_avg', 0.0)
        self.is_thermally_safe = data.get('is_thermally_safe', True)
        self.max_temp_exceeded = data.get('max_temp_exceeded', False)
        self.throttling_detected = data.get('throttling_detected', False)
        self.time_above_target = data.get('time_above_target', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'duration_seconds': self.duration_seconds,
            'sample_count': self.sample_count,
            'temp_min': self.temp_min,
            'temp_max': self.temp_max,
            'temp_avg': self.temp_avg,
            'temp_final': self.temp_final,
            'power_min': self.power_min,
            'power_max': self.power_max,
            'power_avg': self.power_avg,
            'memory_max_used_mb': self.memory_max_used_mb,
            'memory_max_used_pct': self.memory_max_used_pct,
            'gpu_util_avg': self.gpu_util_avg,
            'memory_util_avg': self.memory_util_avg,
            'is_thermally_safe': self.is_thermally_safe,
            'max_temp_exceeded': self.max_temp_exceeded,
            'throttling_detected': self.throttling_detected,
            'time_above_target': self.time_above_target,
        }


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
        llm_gpu_id: int = LLM_GPU_ID,
        benchmark_gpu_id: int = BENCHMARK_GPU_ID,
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
            llm_gpu_id: GPU device index for LLM meta-controller
            benchmark_gpu_id: GPU device index for vLLM benchmarks
            use_ray: Whether to use Ray for distributed profiling
            thermal_config: Custom thermal configuration
        """
        self.model_name = model_name
        self.gpu_type = gpu_type
        self.benchmark_duration_minutes = benchmark_duration_minutes
        self.num_iterations = num_iterations
        self.output_dir = output_dir
        self.llm_gpu_id = llm_gpu_id
        self.benchmark_gpu_id = benchmark_gpu_id
        self.use_ray = use_ray and RAY_AVAILABLE
        self.thermal_config = thermal_config or THERMAL_CONFIG
        
        # Validate GPU assignment
        validate_gpu_assignment(llm_gpu_id, benchmark_gpu_id)
        
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
        print(f"[ServerOptimizer] LLM GPU: {self.llm_gpu_id}, Benchmark GPU: {self.benchmark_gpu_id}")
        
        # LLM meta-controller (with explicit GPU)
        self.meta_controller = ServerMetaController(gpu_id=self.llm_gpu_id)
        
        # Profiling worker (on separate GPU)
        if self.use_ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, include_dashboard=False)
            self.worker = ServerProfilingWorker.remote(
                gpu_id=self.benchmark_gpu_id,
                model=self.model_name,
                thermal_config=self.thermal_config
            )
            print(f"[ServerOptimizer] Using Ray profiling worker on GPU {self.benchmark_gpu_id}")
        else:
            self.worker = ServerProfilingWorkerLocal(
                gpu_id=self.benchmark_gpu_id,
                model=self.model_name,
                thermal_config=self.thermal_config
            )
            print(f"[ServerOptimizer] Using local profiling worker on GPU {self.benchmark_gpu_id}")
        
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
        print("\n" + "═" * SEPARATOR_WIDTH)
        print(f"Server Parameter Optimization for {self.model_name}")
        print(f"GPU: {self.gpu_type}")
        print(f"LLM GPU: {self.llm_gpu_id} | Benchmark GPU: {self.benchmark_gpu_id}")
        print(f"Benchmark Duration: {self.benchmark_duration_minutes} minutes per config")
        print("═" * SEPARATOR_WIDTH + "\n")
    
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
        
        print("\n" + "═" * SEPARATOR_WIDTH)
        print(f"[ServerOptimizer] ITERATION {iteration_num}/{self.num_iterations}")
        print("═" * SEPARATOR_WIDTH)
        
        # Generate configurations using LLM
        print("\n[ServerOptimizer] LLM generating configurations...")
        configs = self.meta_controller.generate_configs(self.feedback_collector)
        print(f"[ServerOptimizer] LLM suggested {len(configs)} configurations")
        
        # Print parameter configs being tested
        print("\n" + "-" * SEPARATOR_WIDTH)
        print("[ServerOptimizer] PARAMETER CONFIGS TO BE TESTED THIS ITERATION:")
        print("-" * SEPARATOR_WIDTH)
        for i, config in enumerate(configs, 1):
            print(f"\n  Config {i}: {config.get('name', 'unnamed')}")
            print(f"    max_num_seqs: {config.get('max_num_seqs')}")
            print(f"    max_num_batched_tokens: {config.get('max_num_batched_tokens')}")
            if config.get('rationale'):
                print(f"    rationale: {config.get('rationale')}")
        print("-" * SEPARATOR_WIDTH + "\n")
        
        # Benchmark each configuration
        iteration_configs = []
        iteration_results = []
        
        for idx, config in enumerate(configs, 1):
            print(f"\n[ServerOptimizer] Testing config {idx}/{len(configs)}...")
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
        
        # Print iteration summary with detailed feedback
        print("\n" + "=" * SEPARATOR_WIDTH)
        print(f"[ServerOptimizer] ITERATION {iteration_num} COMPLETE - FEEDBACK SUMMARY:")
        print("=" * SEPARATOR_WIDTH)
        self._print_iteration_feedback(iteration_configs, iteration_results)
        print("=" * SEPARATOR_WIDTH + "\n")
    
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
        
        # Print detailed config info
        print("\n" + "-" * 50)
        print(f"[ServerOptimizer] TESTING CONFIGURATION: {config_name}")
        print("-" * 50)
        print(f"  Parameter Settings:")
        print(f"    --max-num-seqs: {max_num_seqs}")
        print(f"    --max-num-batched-tokens: {max_num_batched_tokens}")
        if config.get('rationale'):
            print(f"  Rationale: {config.get('rationale')}")
        print("-" * 50)
        
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
        
        # Print detailed results
        print("\n[Benchmark] RESULTS:")
        print("-" * 40)
        
        if result.error:
            # Log detailed error info for debugging
            log_error_details(result.error, result.config)
            print(f"[ServerOptimizer] ❌ Benchmark FAILED - penalty: {result.penalty}")
        else:
            print(f"  Throughput: {result.throughput:.1f} tokens/sec")
            if result.output_throughput:
                print(f"  Output Throughput: {result.output_throughput:.1f} tokens/sec")
            if result.latency:
                print(f"  Latency: {result.latency:.2f} ms")
            
            # Print thermal summary using helper function
            if result.thermal_summary:
                temp_min = _get_thermal_value(result.thermal_summary, 'temp_min')
                temp_max = _get_thermal_value(result.thermal_summary, 'temp_max')
                temp_avg = _get_thermal_value(result.thermal_summary, 'temp_avg')
                power_avg = _get_thermal_value(result.thermal_summary, 'power_avg')
                power_max = _get_thermal_value(result.thermal_summary, 'power_max')
                gpu_util = _get_thermal_value(result.thermal_summary, 'gpu_util_avg')
                mem_util = _get_thermal_value(result.thermal_summary, 'memory_util_avg')
                
                print(f"  Thermal Summary:")
                print(f"    Temperature: min={temp_min:.1f}°C, max={temp_max:.1f}°C, avg={temp_avg:.1f}°C")
                print(f"    Power: avg={power_avg:.1f}W, max={power_max:.1f}W")
                print(f"    GPU Utilization: {gpu_util:.1f}%")
                print(f"    Memory Utilization: {mem_util:.1f}%")
            
            # Save thermal plot (only for successful benchmarks)
            self._save_thermal_plot(config, result)
            
            # Print thermal status - only check for "best" if successful
            if result.is_thermally_safe:
                temp_max = _get_thermal_value(result.thermal_summary, 'temp_max')
                print(f"[ServerOptimizer] ✓ Thermally safe (max {temp_max:.0f}°C < {self.thermal_config.target_sustained_temp}°C target)")
                if self._is_new_best_sustained(result):
                    print("[ServerOptimizer] → New best SUSTAINED config!")
            else:
                print(f"[ServerOptimizer] ⚠️ Above thermal target")
            
            if self._is_new_best_aggressive(result):
                print("[ServerOptimizer] → New best AGGRESSIVE config!")
        
        print("-" * 40)
        
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
                # Use module-level proxy class to wrap dict data
                summary = _ThermalSummaryProxy(result.thermal_summary)
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
        # Skip failed benchmarks
        if not result.is_successful:
            return False
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
        # Skip failed benchmarks
        if not result.is_successful:
            return False
        if not result.is_thermally_safe:
            return False
        if self.config_exporter.best_sustained is None:
            return True
        return result.throughput > self.config_exporter.best_sustained.throughput
    
    def _print_iteration_feedback(
        self,
        configs: List[Dict[str, Any]],
        results: List[BenchmarkResult]
    ) -> None:
        """Print detailed feedback from benchmark results.
        
        Args:
            configs: List of configurations tested
            results: List of BenchmarkResult objects
        """
        print("\nBenchmark Results:")
        print("-" * 60)
        
        for i, (config, result) in enumerate(zip(configs, results), 1):
            config_name = config.get('name', 'unnamed')
            max_num_seqs = config.get('max_num_seqs', 'N/A')
            max_num_batched_tokens = config.get('max_num_batched_tokens', 'N/A')
            
            print(f"\n  [{i}] Config: {config_name}")
            print(f"      Parameters:")
            print(f"        max_num_seqs: {max_num_seqs}")
            print(f"        max_num_batched_tokens: {max_num_batched_tokens}")
            
            if result.error:
                print(f"      Status: ❌ FAILED")
                print(f"      Error: {result.error}")
                print(f"      Penalty: {result.penalty}")
            else:
                print(f"      Status: ✓ SUCCESS")
                print(f"      Throughput: {result.throughput:.2f} tokens/sec")
                if result.output_throughput:
                    print(f"      Output Throughput: {result.output_throughput:.2f} tokens/sec")
                
                # Print thermal data
                if result.thermal_summary:
                    print(f"      Thermal Data:")
                    temp_min = _get_thermal_value(result.thermal_summary, 'temp_min')
                    temp_max = _get_thermal_value(result.thermal_summary, 'temp_max')
                    temp_avg = _get_thermal_value(result.thermal_summary, 'temp_avg')
                    power_avg = _get_thermal_value(result.thermal_summary, 'power_avg')
                    gpu_util_avg = _get_thermal_value(result.thermal_summary, 'gpu_util_avg')
                    
                    print(f"        Temperature: min={temp_min:.1f}°C, max={temp_max:.1f}°C, avg={temp_avg:.1f}°C")
                    print(f"        Power (avg): {power_avg:.1f}W")
                    print(f"        GPU Utilization (avg): {gpu_util_avg:.1f}%")
                    print(f"        Thermally Safe: {'Yes' if result.is_thermally_safe else 'No'}")
        
        # Print current best configurations
        print("\n" + "-" * 60)
        print("Current Best Configurations:")
        
        if self.config_exporter.best_aggressive:
            ba = self.config_exporter.best_aggressive
            print(f"\n  🚀 Best Aggressive (Max Throughput):")
            print(f"      max_num_seqs: {ba.max_num_seqs}")
            print(f"      max_num_batched_tokens: {ba.max_num_batched_tokens}")
            print(f"      Throughput: {ba.throughput:.2f} tokens/sec")
        else:
            print("\n  🚀 Best Aggressive: Not yet found")
        
        if self.config_exporter.best_sustained:
            bs = self.config_exporter.best_sustained
            print(f"\n  🌡️  Best Sustained (Thermally Safe):")
            print(f"      max_num_seqs: {bs.max_num_seqs}")
            print(f"      max_num_batched_tokens: {bs.max_num_batched_tokens}")
            print(f"      Throughput: {bs.throughput:.2f} tokens/sec")
        else:
            print("\n  🌡️  Best Sustained: Not yet found")
    
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
        
        print("\n" + "═" * SEPARATOR_WIDTH)
        print("OPTIMIZATION COMPLETE - FINAL SUMMARY")
        print("═" * SEPARATOR_WIDTH)
        
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
            temp_max = _get_thermal_value(ba.thermal_summary, 'temp_max')
            if temp_max > 0:
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
            temp_max = _get_thermal_value(bs.thermal_summary, 'temp_max')
            if temp_max > 0:
                print(f"   Max Temp: {temp_max:.0f}°C")
            print("   ✓ Safe for continuous operation")
        else:
            print("   No thermally-safe config found")
        
        # Output files
        print(f"\n📊 Thermal plots saved to: {self.output_dir}/thermal_plots/")
        print(f"📈 Full results saved to: {self.output_dir}/optimization_results.json")
        print(f"🚀 Launch scripts saved to: {self.output_dir}/launch_scripts/")
        
        print("═" * SEPARATOR_WIDTH + "\n")


def main():
    """Main entry point for the server parameter optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Server Parameter Optimizer for vLLM")
    parser.add_argument("--llm-gpu", type=int, default=LLM_GPU_ID,
                        help=f"GPU ID for LLM meta-controller (default: {LLM_GPU_ID})")
    parser.add_argument("--benchmark-gpu", type=int, default=BENCHMARK_GPU_ID,
                        help=f"GPU ID for vLLM benchmarks (default: {BENCHMARK_GPU_ID})")
    parser.add_argument("--duration", type=int, default=BENCHMARK_DURATION_MINUTES,
                        help=f"Benchmark duration in minutes (default: {BENCHMARK_DURATION_MINUTES})")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS,
                        help=f"Number of optimization iterations (default: {NUM_ITERATIONS})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    print("\n" + "═" * SEPARATOR_WIDTH)
    print("          SERVER PARAMETER OPTIMIZER FOR VLLM")
    print("═" * SEPARATOR_WIDTH)
    
    optimizer = ServerParameterOptimizer(
        model_name=MODEL_NAME,
        gpu_type=GPU_TYPE,
        benchmark_duration_minutes=args.duration,
        num_iterations=args.iterations,
        output_dir=args.output_dir,
        llm_gpu_id=args.llm_gpu,
        benchmark_gpu_id=args.benchmark_gpu,
    )
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
