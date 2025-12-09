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
    from .server_meta_controller import ServerMetaController, PARAM_SPACE, MIN_TOKENS_PER_SEQUENCE
except ImportError:
    from server_meta_controller import ServerMetaController, PARAM_SPACE, MIN_TOKENS_PER_SEQUENCE

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
    
    def run_thermal_boundary_search(
        self,
        target_peak_temp: float,
        peak_tol: float = 1.0,
        peak_reduction: float = 5.0,
        duration_minutes: int = 10,
        repeat_count: int = 2
    ) -> None:
        """Run thermal-boundary search to find configs at target temperatures.
        
        This search mode finds the largest parameter configurations that produce
        a target peak GPU temperature, then finds reduced-temperature alternatives.
        
        Args:
            target_peak_temp: Target peak GPU temperature in °C
            peak_tol: Temperature tolerance in °C (configs within target ± tol are acceptable)
            peak_reduction: Temperature reduction for safer config (°C below target)
            duration_minutes: Benchmark duration in minutes
            repeat_count: Number of times to repeat each benchmark (uses worst-case peak)
        """
        self.start_time = time.time()
        
        print("\n" + "═" * SEPARATOR_WIDTH)
        print(f"[ServerOptimizer] THERMAL-BOUNDARY SEARCH MODE")
        print("═" * SEPARATOR_WIDTH)
        print(f"Target peak temp: {target_peak_temp}°C (tolerance: ±{peak_tol}°C)")
        print(f"Reduced-temp target: {target_peak_temp - peak_reduction}°C")
        print(f"Benchmark: {duration_minutes} minutes × {repeat_count} repeats")
        print(f"Absolute max safe temp: {self.thermal_config.max_safe_temp}°C")
        print("═" * SEPARATOR_WIDTH + "\n")
        
        # Create iteration output directory
        iteration_dir = os.path.join(self.output_dir, "iteration_thermal_boundary")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Phase A: Find boundary config at target temperature
        print("\n" + "=" * SEPARATOR_WIDTH)
        print("[ServerOptimizer] PHASE A: Finding boundary config at target temperature")
        print("=" * SEPARATOR_WIDTH + "\n")
        
        boundary_config, boundary_result = self.find_boundary_config_for_target_temp(
            target_temp=target_peak_temp,
            tol=peak_tol,
            duration_minutes=duration_minutes,
            repeat_count=repeat_count,
            iteration_dir=iteration_dir
        )
        
        if boundary_config is None:
            print("\n[ServerOptimizer] ❌ Failed to find boundary config at target temperature")
            print("[ServerOptimizer] Try adjusting --target-peak-temp or --peak-tol")
            return
        
        # Update best configs with boundary result
        self.config_exporter.update_best_configs(boundary_result)
        
        print(f"\n[ServerOptimizer] ✓ Boundary config found:")
        print(f"  max_num_seqs: {boundary_config.get('max_num_seqs')}")
        print(f"  max_num_batched_tokens: {boundary_config.get('max_num_batched_tokens')}")
        peak_temp = _get_thermal_value(boundary_result.thermal_summary, 'temp_max')
        print(f"  Peak temp: {peak_temp:.1f}°C")
        print(f"  Throughput: {boundary_result.throughput:.1f} tokens/sec")
        
        # Phase B: Find reduced-temperature config
        print("\n" + "=" * SEPARATOR_WIDTH)
        print(f"[ServerOptimizer] PHASE B: Finding reduced-temp config (target: {target_peak_temp - peak_reduction}°C)")
        print("=" * SEPARATOR_WIDTH + "\n")
        
        reduced_config, reduced_result = self.find_boundary_config_for_target_temp(
            target_temp=target_peak_temp - peak_reduction,
            tol=peak_tol,
            duration_minutes=duration_minutes,
            repeat_count=repeat_count,
            iteration_dir=iteration_dir,
            phase_name="Phase_B"
        )
        
        if reduced_config is None:
            print(f"\n[ServerOptimizer] ⚠️ Could not find reduced-temp config at {target_peak_temp - peak_reduction}°C")
            print("[ServerOptimizer] Using boundary config as both aggressive and sustained")
        else:
            # Update best configs with reduced-temp result
            self.config_exporter.update_best_configs(reduced_result)
            
            print(f"\n[ServerOptimizer] ✓ Reduced-temp config found:")
            print(f"  max_num_seqs: {reduced_config.get('max_num_seqs')}")
            print(f"  max_num_batched_tokens: {reduced_config.get('max_num_batched_tokens')}")
            peak_temp = _get_thermal_value(reduced_result.thermal_summary, 'temp_max')
            print(f"  Peak temp: {peak_temp:.1f}°C")
            print(f"  Throughput: {reduced_result.throughput:.1f} tokens/sec")
        
        # Save final results
        self._save_final_results()
        
        # Print summary
        self.print_final_summary()
    
    def find_boundary_config_for_target_temp(
        self,
        target_temp: float,
        tol: float,
        duration_minutes: int,
        repeat_count: int,
        iteration_dir: str,
        phase_name: str = "Phase_A"
    ) -> tuple:
        """Find the largest config that produces target peak temperature.
        
        Searches through LLM-generated configs and parameter space to find
        the config with the largest footprint (max_num_seqs * max_num_batched_tokens)
        that produces a peak temperature within target ± tol.
        
        Args:
            target_temp: Target peak temperature in °C
            tol: Temperature tolerance in °C
            duration_minutes: Benchmark duration in minutes
            repeat_count: Number of benchmark repeats
            iteration_dir: Directory to save iteration results
            phase_name: Phase identifier for logging
            
        Returns:
            Tuple of (config_dict, BenchmarkResult) or (None, None) if not found
        """
        print(f"\n[{phase_name}] Generating candidate configs from LLM...")
        
        # Get LLM-generated configs with thermal target
        llm_configs = self.meta_controller.generate_configs(
            feedback_collector=self.feedback_collector,
            target_peak_temp=target_temp,
            peak_tol=tol
        )
        
        # Save LLM raw output
        llm_output_path = os.path.join(iteration_dir, f"{phase_name}_llm_raw.txt")
        with open(llm_output_path, 'w') as f:
            f.write(f"Target peak temp: {target_temp}°C ± {tol}°C\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.write("LLM Generated Configs:\n")
            for i, cfg in enumerate(llm_configs, 1):
                f.write(f"\nConfig {i}:\n")
                f.write(json.dumps(cfg, indent=2))
                f.write("\n")
        print(f"[{phase_name}] Saved LLM output to {llm_output_path}")
        
        # Build candidate list: LLM configs + fallback enumeration
        candidates = []
        
        # Add LLM configs
        for cfg in llm_configs:
            footprint = cfg.get('max_num_seqs', 0) * cfg.get('max_num_batched_tokens', 0)
            candidates.append((footprint, cfg))
        
        # Add fallback candidates from parameter space (top 20 by footprint)
        print(f"[{phase_name}] Adding fallback candidates from parameter space...")
        param_space = self.meta_controller.get_param_space()
        fallback_candidates = []
        
        for seqs in param_space['max_num_seqs']:
            for tokens in param_space['max_num_batched_tokens']:
                # Check constraint
                if tokens >= seqs * MIN_TOKENS_PER_SEQUENCE:
                    footprint = seqs * tokens
                    fallback_candidates.append((footprint, {
                        'max_num_seqs': seqs,
                        'max_num_batched_tokens': tokens,
                        'name': f'fallback_seqs{seqs}_tokens{tokens}'
                    }))
        
        # Sort fallback by footprint (largest first) and take top 20
        fallback_candidates.sort(reverse=True, key=lambda x: x[0])
        candidates.extend(fallback_candidates[:20])
        
        # Remove duplicates and sort by footprint (largest first)
        seen = set()
        unique_candidates = []
        for footprint, cfg in candidates:
            key = (cfg.get('max_num_seqs'), cfg.get('max_num_batched_tokens'))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((footprint, cfg))
        
        unique_candidates.sort(reverse=True, key=lambda x: x[0])
        
        print(f"[{phase_name}] Testing {len(unique_candidates)} candidate configs (largest-first)...")
        
        # Search largest-first for acceptable config
        best_config = None
        best_result = None
        
        for idx, (footprint, cfg) in enumerate(unique_candidates, 1):
            print(f"\n[{phase_name}] Candidate {idx}/{len(unique_candidates)}: "
                  f"seqs={cfg.get('max_num_seqs')}, tokens={cfg.get('max_num_batched_tokens')} "
                  f"(footprint={footprint})")
            
            # Run benchmark with repeats
            peak_temps = []
            results = []
            
            for repeat_num in range(1, repeat_count + 1):
                print(f"[{phase_name}] Repeat {repeat_num}/{repeat_count}...")
                
                result = self._benchmark_config_for_thermal_search(
                    config=cfg,
                    duration_minutes=duration_minutes,
                    phase_name=phase_name,
                    repeat_num=repeat_num
                )
                
                results.append(result)
                
                # Check for abort conditions
                if result.error:
                    print(f"[{phase_name}] ❌ Benchmark failed: {result.error}")
                    break
                
                # Get peak temp
                peak_temp = _get_thermal_value(result.thermal_summary, 'temp_max')
                peak_temps.append(peak_temp)
                print(f"[{phase_name}] Peak temp: {peak_temp:.1f}°C")
                
                # Check absolute max safe temp
                if peak_temp >= self.thermal_config.max_safe_temp:
                    print(f"[{phase_name}] ⚠️ ABORT: Peak temp {peak_temp:.1f}°C >= max safe {self.thermal_config.max_safe_temp}°C")
                    break
            
            # Skip failed benchmarks
            if results and results[0].error:
                continue
            
            # Use worst-case (highest) peak temp
            if peak_temps:
                worst_peak = max(peak_temps)
                print(f"[{phase_name}] Worst-case peak temp across {len(peak_temps)} repeats: {worst_peak:.1f}°C")
                
                # Check if within target ± tol
                if abs(worst_peak - target_temp) <= tol:
                    print(f"[{phase_name}] ✓ ACCEPTABLE: {worst_peak:.1f}°C within {target_temp}°C ± {tol}°C")
                    best_config = cfg
                    best_result = results[0]  # Use first result for throughput
                    
                    # Save thermal summary with worst-case peak
                    thermal_summary_path = os.path.join(iteration_dir, f"{phase_name}_thermal_summary.json")
                    with open(thermal_summary_path, 'w') as f:
                        json.dump({
                            'config': cfg,
                            'target_temp': target_temp,
                            'tolerance': tol,
                            'worst_peak_temp': worst_peak,
                            'all_peak_temps': peak_temps,
                            'throughput': best_result.throughput,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                    
                    break  # Stop at first acceptable config (largest footprint)
                else:
                    print(f"[{phase_name}] ✗ Outside tolerance: {worst_peak:.1f}°C not in [{target_temp - tol:.1f}, {target_temp + tol:.1f}]°C")
        
        return best_config, best_result
    
    def _benchmark_config_for_thermal_search(
        self,
        config: Dict[str, Any],
        duration_minutes: int,
        phase_name: str,
        repeat_num: int
    ) -> BenchmarkResult:
        """Run a single benchmark for thermal search.
        
        Args:
            config: Config dict with max_num_seqs and max_num_batched_tokens
            duration_minutes: Benchmark duration
            phase_name: Phase identifier
            repeat_num: Repeat number
            
        Returns:
            BenchmarkResult
        """
        max_num_seqs = config.get('max_num_seqs', 64)
        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        
        print(f"\n[{phase_name}] Benchmarking: seqs={max_num_seqs}, tokens={max_num_batched_tokens} (repeat {repeat_num})")
        
        if self.use_ray:
            result = ray.get(self.worker.run_benchmark.remote(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                duration_minutes=duration_minutes
            ))
        else:
            result = self.worker.run_benchmark(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                duration_minutes=duration_minutes
            )
        
        # Print result
        if result.error:
            print(f"[{phase_name}] ❌ Failed: {result.error}")
        else:
            print(f"[{phase_name}] Throughput: {result.throughput:.1f} tokens/sec")
            if result.thermal_summary:
                temp_max = _get_thermal_value(result.thermal_summary, 'temp_max')
                temp_avg = _get_thermal_value(result.thermal_summary, 'temp_avg')
                print(f"[{phase_name}] Thermal: max={temp_max:.1f}°C, avg={temp_avg:.1f}°C")
        
        self.total_benchmarks += 1
        return result
    
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
    
    # Thermal-boundary search mode arguments
    parser.add_argument("--search-mode", type=str, default="default",
                        choices=["default", "thermal-boundary"],
                        help="Search mode: 'default' (standard iterative) or 'thermal-boundary' (target temperature-based)")
    parser.add_argument("--target-peak-temp", type=float, default=None,
                        help="Target peak GPU temperature in °C for thermal-boundary mode")
    parser.add_argument("--peak-tol", type=float, default=1.0,
                        help="Temperature tolerance in °C for thermal-boundary mode (default: 1.0)")
    parser.add_argument("--peak-reduction", type=float, default=5.0,
                        help="Temperature reduction in °C for reduced-temp config (default: 5.0)")
    parser.add_argument("--benchmark-duration", type=int, default=10,
                        help="Benchmark duration in minutes for thermal-boundary mode (default: 10)")
    parser.add_argument("--repeat-count", type=int, default=2,
                        help="Number of times to repeat each benchmark for thermal-boundary mode (default: 2)")
    parser.add_argument("--absolute-max-temp", type=float, default=None,
                        help="Absolute maximum temperature override (default: uses THERMAL_CONFIG.max_safe_temp)")
    
    args = parser.parse_args()
    
    print("\n" + "═" * SEPARATOR_WIDTH)
    print("          SERVER PARAMETER OPTIMIZER FOR VLLM")
    print("═" * SEPARATOR_WIDTH)
    
    # Override thermal config if absolute-max-temp provided
    thermal_config = THERMAL_CONFIG
    if args.absolute_max_temp:
        thermal_config = ThermalConfig(
            max_safe_temp=args.absolute_max_temp,
            target_sustained_temp=THERMAL_CONFIG.target_sustained_temp,
            warning_temp=THERMAL_CONFIG.warning_temp,
            max_power=THERMAL_CONFIG.max_power,
            total_memory_gb=THERMAL_CONFIG.total_memory_gb,
            gpu_name=THERMAL_CONFIG.gpu_name
        )
        print(f"[ServerOptimizer] Using custom absolute max temp: {args.absolute_max_temp}°C")
    
    optimizer = ServerParameterOptimizer(
        model_name=MODEL_NAME,
        gpu_type=GPU_TYPE,
        benchmark_duration_minutes=args.duration,
        num_iterations=args.iterations,
        output_dir=args.output_dir,
        llm_gpu_id=args.llm_gpu,
        benchmark_gpu_id=args.benchmark_gpu,
        thermal_config=thermal_config
    )
    
    if args.search_mode == "thermal-boundary":
        if args.target_peak_temp is None:
            print("[ERROR] --target-peak-temp is required for thermal-boundary mode")
            sys.exit(1)
        
        print(f"\n[ServerOptimizer] Running thermal-boundary search mode")
        print(f"  Target peak temp: {args.target_peak_temp}°C")
        print(f"  Peak tolerance: ±{args.peak_tol}°C")
        print(f"  Peak reduction: {args.peak_reduction}°C")
        print(f"  Benchmark duration: {args.benchmark_duration} minutes")
        print(f"  Repeat count: {args.repeat_count}\n")
        
        optimizer.run_thermal_boundary_search(
            target_peak_temp=args.target_peak_temp,
            peak_tol=args.peak_tol,
            peak_reduction=args.peak_reduction,
            duration_minutes=args.benchmark_duration,
            repeat_count=args.repeat_count
        )
    else:
        optimizer.run_optimization()


if __name__ == "__main__":
    main()
