"""
Server Profiling Worker for vLLM Benchmarks

Ray-based profiling worker for running vLLM throughput benchmarks with thermal monitoring.
Optimizes --max-num-seqs and --max-num-batched-tokens for H100 80GB GPU.

Features:
- Ray actor that runs on a dedicated GPU
- Runs vLLM throughput benchmarks with specified parameters
- Integrates with ThermalMonitor for temperature/power monitoring during benchmarks
- Optionally runs nsys profiling for detailed GPU metrics
- Parses benchmark output to extract throughput (tokens/sec)
- Error classification and VRAM penalty detection for RL learning
"""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any

# Add parent directory to path for script execution
if __name__ == "__main__" or "." not in __name__:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with fallback for both module and script execution
try:
    from .thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSummary, get_gpu_info
except ImportError:
    from thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSummary, get_gpu_info

try:
    from .nsys_metrics_extractor import NsysMetricsExtractor, NsysProfile, NsysMetrics
except ImportError:
    from nsys_metrics_extractor import NsysMetricsExtractor, NsysProfile, NsysMetrics

# Check if ray is available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def get_physical_gpu_id(visible_index: int) -> str:
    """Map a visible GPU index to the physical GPU ID.
    
    When CUDA_VISIBLE_DEVICES is set in the parent process (e.g., "5,6"),
    the gpu_id parameter represents a visible index (0 or 1), not the physical
    GPU ID. This function returns the physical GPU ID that should be passed
    to subprocess CUDA_VISIBLE_DEVICES.
    
    For example:
    - Parent has CUDA_VISIBLE_DEVICES="5,6", gpu_id=0 -> returns "5"
    - Parent has CUDA_VISIBLE_DEVICES="5,6", gpu_id=1 -> returns "6"
    - Parent has no CUDA_VISIBLE_DEVICES, gpu_id=1 -> returns "1"
    
    Args:
        visible_index: The visible GPU index (0, 1, 2, etc.)
        
    Returns:
        String representation of the physical GPU ID to use
    """
    parent_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    
    if not parent_cuda_devices:
        # No CUDA_VISIBLE_DEVICES set, use the index directly
        return str(visible_index)
    
    # Parse the parent's CUDA_VISIBLE_DEVICES list
    gpu_list = [g.strip() for g in parent_cuda_devices.split(",") if g.strip()]
    
    if not gpu_list:
        # Empty or invalid CUDA_VISIBLE_DEVICES, use index directly
        return str(visible_index)
    
    if visible_index < 0 or visible_index >= len(gpu_list):
        # Index out of range, log warning and use index directly
        print(f"[WARNING] GPU index {visible_index} out of range for "
              f"CUDA_VISIBLE_DEVICES={parent_cuda_devices}. Using index directly.")
        return str(visible_index)
    
    # Map visible index to physical GPU ID
    physical_gpu = gpu_list[visible_index]
    return physical_gpu


class BenchmarkErrorType(Enum):
    """Classification of benchmark errors for appropriate handling."""
    NONE = "none"
    VRAM_OOM = "vram_oom"  # Out of memory
    CUDA_ERROR = "cuda_error"  # CUDA runtime error
    ENGINE_INIT_FAILED = "engine_init_failed"  # vLLM engine failed to start
    TIMEOUT = "timeout"  # Benchmark timed out
    INVALID_CONFIG = "invalid_config"  # Invalid parameter combination
    UNKNOWN = "unknown"  # Unknown error


def classify_error(error_message: str) -> BenchmarkErrorType:
    """Classify benchmark error for appropriate handling.
    
    Args:
        error_message: The error message to classify
        
    Returns:
        BenchmarkErrorType indicating the category of error
    """
    if not error_message:
        return BenchmarkErrorType.NONE
    
    error_lower = error_message.lower()
    
    # VRAM/OOM errors
    if any(phrase in error_lower for phrase in [
        'out of memory', 'oom', 'cuda out of memory',
        'allocat', 'memory', 'vram',
        'torch.cuda.outofmemoryerror',
        'cuda error: out of memory'
    ]):
        return BenchmarkErrorType.VRAM_OOM
    
    # Engine initialization failures (often VRAM related)
    if any(phrase in error_lower for phrase in [
        'engine core initialization failed',
        'failed to initialize',
        'runtimeerror: engine',
        'failed core proc'
    ]):
        return BenchmarkErrorType.ENGINE_INIT_FAILED
    
    # CUDA errors
    if any(phrase in error_lower for phrase in [
        'cuda error', 'cudnn error', 'cublas error',
        'cuda runtime error', 'device-side assert'
    ]):
        return BenchmarkErrorType.CUDA_ERROR
    
    # Timeout
    if 'timeout' in error_lower or 'timed out' in error_lower:
        return BenchmarkErrorType.TIMEOUT
    
    # Invalid config
    if any(phrase in error_lower for phrase in [
        'invalid', 'must be', 'should be', 'constraint',
        'max_num_batched_tokens', 'max_num_seqs'
    ]):
        return BenchmarkErrorType.INVALID_CONFIG
    
    return BenchmarkErrorType.UNKNOWN


def get_error_penalty(error_type: BenchmarkErrorType) -> float:
    """Get throughput penalty for error type (negative reward for RL).
    
    Args:
        error_type: The type of error that occurred
        
    Returns:
        Negative float representing the penalty for this error type
    """
    penalties = {
        BenchmarkErrorType.NONE: 0.0,
        BenchmarkErrorType.VRAM_OOM: -100.0,  # Heavy penalty for OOM
        BenchmarkErrorType.ENGINE_INIT_FAILED: -80.0,  # Often VRAM related
        BenchmarkErrorType.CUDA_ERROR: -50.0,
        BenchmarkErrorType.TIMEOUT: -30.0,  # Might be too aggressive config
        BenchmarkErrorType.INVALID_CONFIG: -20.0,
        BenchmarkErrorType.UNKNOWN: -10.0,
    }
    return penalties.get(error_type, -10.0)


def log_error_details(error: str, config: Dict[str, Any]) -> None:
    """Log detailed error information for debugging.
    
    Args:
        error: The error message
        config: The configuration that caused the error
    """
    error_type = classify_error(error)
    penalty = get_error_penalty(error_type)
    
    print(f"\n{'='*60}")
    print(f"[BENCHMARK ERROR] Configuration Failed")
    print(f"{'='*60}")
    print(f"Config: max_num_seqs={config.get('max_num_seqs')}, "
          f"max_num_batched_tokens={config.get('max_num_batched_tokens')}")
    print(f"Error Type: {error_type.value}")
    print(f"Penalty: {penalty}")
    print(f"\nFull Error:\n{error}")
    
    # Provide debugging hints
    if error_type == BenchmarkErrorType.VRAM_OOM:
        print(f"\n[DEBUG HINT] VRAM exceeded! Try reducing:")
        print(f"  - max_num_seqs (current: {config.get('max_num_seqs')})")
        print(f"  - max_num_batched_tokens (current: {config.get('max_num_batched_tokens')})")
        print(f"  - For H100 80GB, try max_num_seqs <= 128, max_num_batched_tokens <= 32768")
    elif error_type == BenchmarkErrorType.ENGINE_INIT_FAILED:
        print(f"\n[DEBUG HINT] Engine init failed - likely VRAM or config issue")
        print(f"  - Check if another process is using GPU memory")
        print(f"  - Try smaller batch sizes")
    elif error_type == BenchmarkErrorType.TIMEOUT:
        print(f"\n[DEBUG HINT] Benchmark timed out")
        print(f"  - Configuration may be too aggressive")
        print(f"  - Try reducing max_num_seqs or max_num_batched_tokens")
    elif error_type == BenchmarkErrorType.CUDA_ERROR:
        print(f"\n[DEBUG HINT] CUDA error occurred")
        print(f"  - Check GPU health and driver status")
        print(f"  - Try restarting the GPU or reducing workload")
    print(f"{'='*60}\n")


def is_benchmark_successful(result) -> bool:
    """Check if a benchmark result was successful.
    
    Works with both BenchmarkResult objects and dictionaries.
    This is the single source of truth for determining success.
    
    Args:
        result: BenchmarkResult object or dictionary
        
    Returns:
        True if benchmark completed successfully
    """
    # If it's a BenchmarkResult with is_successful property, use that
    if hasattr(result, 'is_successful'):
        return result.is_successful
    
    # For dictionaries, check the explicit is_successful field first
    if isinstance(result, dict):
        if 'is_successful' in result:
            return result['is_successful']
        # Fallback logic for dicts without explicit is_successful
        has_error = bool(result.get('error', ''))
        has_throughput = result.get('throughput', 0.0) > 0.0
        return not has_error and has_throughput
    
    # Fallback for other types - check attributes
    error = getattr(result, 'error', '') or ''
    throughput = getattr(result, 'throughput', 0.0) or 0.0
    return error == "" and throughput > 0.0


@dataclass
class BenchmarkResult:
    """Result from a vLLM benchmark run."""
    config: Dict[str, Any]  # max_num_seqs, max_num_batched_tokens
    throughput: float  # Total tokens/sec (input + output)
    output_throughput: float  # Output tokens/sec
    latency: Optional[float]  # Average latency if available
    thermal_summary: Optional[ThermalSummary]  # Thermal stats during benchmark
    is_thermally_safe: bool  # Whether run stayed below target sustained temp
    nsys_metrics: Optional[Dict[str, Any]]  # Optional nsys profiling metrics
    duration: float  # Total benchmark duration in seconds
    error: str = ""  # Error message if benchmark failed
    error_type: BenchmarkErrorType = BenchmarkErrorType.NONE  # Classified error type
    penalty: float = 0.0  # Negative value for errors (for RL reward)
    
    @property
    def is_successful(self) -> bool:
        """Check if benchmark completed successfully."""
        return self.error == "" and self.throughput > 0.0
    
    @property
    def effective_throughput(self) -> float:
        """Get throughput with penalty applied (for RL reward).
        
        Returns:
            Throughput if successful, otherwise the negative penalty value
        """
        if self.is_successful:
            return self.throughput
        return self.penalty  # Return negative penalty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            'config': self.config,
            'throughput': self.throughput,
            'output_throughput': self.output_throughput,
            'latency': self.latency,
            'thermal_summary': self.thermal_summary.to_dict() if self.thermal_summary else None,
            'is_thermally_safe': self.is_thermally_safe,
            'nsys_metrics': self.nsys_metrics,
            'duration': self.duration,
            'error': self.error,
            'error_type': self.error_type.value if self.error_type else 'none',
            'penalty': self.penalty,
            'is_successful': self.is_successful,
            'effective_throughput': self.effective_throughput
        }


# Default configuration for the target GPU and model
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GPU = "NVIDIA H100 80GB"
DEFAULT_BENCHMARK_DURATION_MINUTES = 20
DEFAULT_NUM_PROMPTS = 500


def _create_worker_class():
    """Create the Ray actor class if Ray is available."""
    if not RAY_AVAILABLE:
        return None
    
    @ray.remote(num_gpus=1)
    class _ServerProfilingWorker:
        """Ray actor for vLLM benchmark profiling with thermal monitoring.
        
        This worker runs on a dedicated GPU and executes vLLM throughput
        benchmarks while monitoring thermal behavior.
        
        Example:
            worker = ServerProfilingWorker.remote(gpu_id=0)
            result = ray.get(worker.run_benchmark.remote(
                max_num_seqs=256,
                max_num_batched_tokens=8192,
                duration_minutes=20
            ))
        """
        
        def __init__(
            self,
            gpu_id: int = 0,
            model: str = DEFAULT_MODEL,
            thermal_config: Optional[ThermalConfig] = None
        ):
            """Initialize the profiling worker.
            
            Args:
                gpu_id: GPU device index to use
                model: Model name to benchmark
                thermal_config: Custom thermal thresholds (default: A100 40GB settings)
            """
            self.gpu_id = gpu_id
            self.model = model
            self.thermal_config = thermal_config or ThermalConfig()
            
            # Initialize thermal monitor
            self.thermal_monitor = ThermalMonitor(
                gpu_index=gpu_id,
                sample_interval=1.0,
                config=self.thermal_config
            )
            
            # Initialize nsys extractor (may not be available)
            self.nsys_extractor = NsysMetricsExtractor(gpu_index=gpu_id)
            
            # Store results
            self._last_result: Optional[BenchmarkResult] = None
            
            pid = os.getpid()
            print(f"[ServerProfilingWorker] Initialized on GPU {gpu_id} (PID: {pid})")
            print(f"[ServerProfilingWorker] Model: {model}")
            print(f"[ServerProfilingWorker] Thermal target: {self.thermal_config.target_sustained_temp}°C")
        
        def get_gpu_info(self) -> Optional[Dict[str, Any]]:
            """Get GPU information.
            
            Returns:
                Dictionary with GPU info or None if nvidia-smi fails
            """
            return get_gpu_info(self.gpu_id)
        
        def get_thermal_samples(self) -> List[Dict[str, Any]]:
            """Get thermal samples from the last benchmark run.
            
            Returns:
                List of thermal sample dictionaries for visualization
            """
            return self.thermal_monitor.get_samples_as_dict()
        
        def run_benchmark(
            self,
            max_num_seqs: int,
            max_num_batched_tokens: int,
            num_prompts: int = DEFAULT_NUM_PROMPTS,
            duration_minutes: int = DEFAULT_BENCHMARK_DURATION_MINUTES,
            dataset_path: str = "ShareGPT_Vicuna_unfiltered.json",
            use_nsys: bool = False
        ) -> BenchmarkResult:
            """Run vLLM benchmark with specified configuration.
            
            Args:
                max_num_seqs: Maximum number of sequences per batch
                max_num_batched_tokens: Maximum tokens per batch
                num_prompts: Number of prompts to process
                duration_minutes: Expected duration for long-running benchmarks
                dataset_path: Path to benchmark dataset
                use_nsys: Whether to run with nsys profiling
                
            Returns:
                BenchmarkResult with throughput and thermal data
            """
            config = {
                'max_num_seqs': max_num_seqs,
                'max_num_batched_tokens': max_num_batched_tokens,
                'model': self.model,
                'num_prompts': num_prompts
            }
            
            print(f"[ServerProfilingWorker] Running benchmark with config: {config}")
            
            # Build benchmark command
            command = self._build_benchmark_command(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                num_prompts=num_prompts,
                dataset_path=dataset_path
            )
            
            # Set up environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = get_physical_gpu_id(self.gpu_id)
            
            # Start thermal monitoring
            self.thermal_monitor.start_monitoring()
            
            start_time = time.time()
            throughput = 0.0
            output_throughput = 0.0
            latency = None
            nsys_metrics = None
            error = ""
            
            try:
                if use_nsys and self.nsys_extractor.is_available():
                    # Run with nsys profiling
                    profile = self.nsys_extractor.profile_command(
                        command=" ".join(command),
                        output_dir="/tmp/nsys_profiles",
                        profile_name=f"server_benchmark_{max_num_seqs}_{max_num_batched_tokens}",
                        timeout=duration_minutes * 60 + 120
                    )
                    if profile.success:
                        nsys_metrics = self.nsys_extractor.get_metrics_summary(profile)
                        # Parse throughput from nsys output
                        throughput, output_throughput, latency = self._parse_benchmark_output(
                            profile.raw_stats
                        )
                    else:
                        error = f"nsys profiling failed: {profile.error_message}"
                else:
                    # Run benchmark directly
                    timeout = duration_minutes * 60 + 120  # Add 2 minutes buffer
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env
                    )
                    
                    if result.returncode != 0:
                        error = f"Benchmark failed with code {result.returncode}: {result.stderr[-500:]}"
                        print(f"[ServerProfilingWorker] {error}")
                    else:
                        # Parse throughput from output
                        throughput, output_throughput, latency = self._parse_benchmark_output(
                            result.stdout
                        )
                        print(f"[ServerProfilingWorker] Throughput: {throughput:.2f} tokens/sec")
                        
            except subprocess.TimeoutExpired:
                error = f"Benchmark timed out after {duration_minutes + 2} minutes"
                print(f"[ServerProfilingWorker] {error}")
            except Exception as e:
                error = f"Benchmark error: {str(e)}"
                print(f"[ServerProfilingWorker] {error}")
            finally:
                # Stop thermal monitoring
                self.thermal_monitor.stop_monitoring()
            
            duration = time.time() - start_time
            
            # Get thermal summary
            thermal_summary = self.thermal_monitor.get_thermal_summary()
            is_thermally_safe = self.thermal_monitor.is_thermally_safe()
            
            if thermal_summary:
                print(f"[ServerProfilingWorker] Thermal: avg={thermal_summary.temp_avg:.1f}°C, "
                      f"max={thermal_summary.temp_max:.1f}°C, safe={is_thermally_safe}")
            
            # Classify error and calculate penalty
            error_type = classify_error(error)
            penalty = get_error_penalty(error_type)
            
            result = BenchmarkResult(
                config=config,
                throughput=throughput,
                output_throughput=output_throughput,
                latency=latency,
                thermal_summary=thermal_summary,
                is_thermally_safe=is_thermally_safe,
                nsys_metrics=nsys_metrics,
                duration=duration,
                error=error,
                error_type=error_type,
                penalty=penalty
            )
            
            self._last_result = result
            return result
        
        def _build_benchmark_command(
            self,
            max_num_seqs: int,
            max_num_batched_tokens: int,
            num_prompts: int,
            dataset_path: str
        ) -> List[str]:
            """Build the vLLM benchmark command.
            
            Args:
                max_num_seqs: Maximum sequences per batch
                max_num_batched_tokens: Maximum tokens per batch
                num_prompts: Number of prompts to process
                dataset_path: Path to dataset file
                
            Returns:
                List of command arguments
            """
            command = [
                "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
                "--model", self.model,
                "--max-num-seqs", str(max_num_seqs),
                "--max-num-batched-tokens", str(max_num_batched_tokens),
                "--num-prompts", str(num_prompts),
                "--trust-remote-code",
                "--enforce-eager",
                "--tensor-parallel-size", "1",
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.90",
            ]
            
            # Add dataset if it exists
            if os.path.exists(dataset_path):
                command.extend(["--dataset-path", dataset_path])
            
            return command
        
        def _parse_benchmark_output(
            self,
            output: str
        ) -> tuple:
            """Parse benchmark output to extract throughput metrics.
            
            Args:
                output: Raw stdout from benchmark command
                
            Returns:
                Tuple of (total_throughput, output_throughput, latency)
            """
            throughput = 0.0
            output_throughput = 0.0
            latency = None
            
            if not output:
                return throughput, output_throughput, latency
            
            # Try various patterns for throughput parsing
            # Pattern 1: "Throughput: X.XX requests/s, Y.YY tokens/s"
            match = re.search(
                r"Throughput:\s*([0-9.]+)\s*requests/s,\s*([0-9.]+)\s*tokens/s",
                output
            )
            if match:
                throughput = float(match.group(2))
            
            # Pattern 2: "X.XX output tokens/s" or similar
            match = re.search(
                r"([0-9.]+)\s*(?:output\s+)?tokens/s",
                output,
                re.IGNORECASE
            )
            if match:
                output_throughput = float(match.group(1))
                if throughput == 0.0:
                    throughput = output_throughput
            
            # Pattern 3: Look for latency
            match = re.search(
                r"(?:mean|avg|average)\s*latency[:\s]+([0-9.]+)\s*(?:ms|s)",
                output,
                re.IGNORECASE
            )
            if match:
                latency = float(match.group(1))
            
            return throughput, output_throughput, latency
    
    return _ServerProfilingWorker


# Create the worker class
ServerProfilingWorker = _create_worker_class()


class ServerProfilingWorkerLocal:
    """Local (non-Ray) version of ServerProfilingWorker for testing.
    
    This class provides the same interface as the Ray actor but runs
    locally without Ray. Useful for testing and development.
    """
    
    def __init__(
        self,
        gpu_id: int = 0,
        model: str = DEFAULT_MODEL,
        thermal_config: Optional[ThermalConfig] = None
    ):
        """Initialize the local profiling worker.
        
        Args:
            gpu_id: GPU device index to use
            model: Model name to benchmark
            thermal_config: Custom thermal thresholds
        """
        self.gpu_id = gpu_id
        self.model = model
        self.thermal_config = thermal_config or ThermalConfig()
        
        self.thermal_monitor = ThermalMonitor(
            gpu_index=gpu_id,
            sample_interval=1.0,
            config=self.thermal_config
        )
        
        self.nsys_extractor = NsysMetricsExtractor(gpu_index=gpu_id)
        self._last_result: Optional[BenchmarkResult] = None
        
        print(f"[ServerProfilingWorkerLocal] Initialized on GPU {gpu_id}")
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information."""
        return get_gpu_info(self.gpu_id)
    
    def get_thermal_samples(self) -> List[Dict[str, Any]]:
        """Get thermal samples from the last benchmark run."""
        return self.thermal_monitor.get_samples_as_dict()
    
    def run_benchmark(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_prompts: int = DEFAULT_NUM_PROMPTS,
        duration_minutes: int = DEFAULT_BENCHMARK_DURATION_MINUTES,
        dataset_path: str = "ShareGPT_Vicuna_unfiltered.json",
        use_nsys: bool = False
    ) -> BenchmarkResult:
        """Run vLLM benchmark with specified configuration.
        
        Same interface as the Ray actor version.
        """
        config = {
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens,
            'model': self.model,
            'num_prompts': num_prompts
        }
        
        print(f"[ServerProfilingWorkerLocal] Running benchmark with config: {config}")
        
        command = self._build_benchmark_command(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            num_prompts=num_prompts,
            dataset_path=dataset_path
        )
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = get_physical_gpu_id(self.gpu_id)
        
        self.thermal_monitor.start_monitoring()
        
        start_time = time.time()
        throughput = 0.0
        output_throughput = 0.0
        latency = None
        nsys_metrics = None
        error = ""
        
        try:
            if use_nsys and self.nsys_extractor.is_available():
                profile = self.nsys_extractor.profile_command(
                    command=" ".join(command),
                    output_dir="/tmp/nsys_profiles",
                    profile_name=f"server_benchmark_{max_num_seqs}_{max_num_batched_tokens}",
                    timeout=duration_minutes * 60 + 120
                )
                if profile.success:
                    nsys_metrics = self.nsys_extractor.get_metrics_summary(profile)
                    throughput, output_throughput, latency = self._parse_benchmark_output(
                        profile.raw_stats
                    )
                else:
                    error = f"nsys profiling failed: {profile.error_message}"
            else:
                timeout = duration_minutes * 60 + 120
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env
                )
                
                if result.returncode != 0:
                    error = f"Benchmark failed with code {result.returncode}: {result.stderr[-500:]}"
                else:
                    throughput, output_throughput, latency = self._parse_benchmark_output(
                        result.stdout
                    )
                    
        except subprocess.TimeoutExpired:
            error = f"Benchmark timed out after {duration_minutes + 2} minutes"
        except Exception as e:
            error = f"Benchmark error: {str(e)}"
        finally:
            self.thermal_monitor.stop_monitoring()
        
        duration = time.time() - start_time
        thermal_summary = self.thermal_monitor.get_thermal_summary()
        is_thermally_safe = self.thermal_monitor.is_thermally_safe()
        
        # Classify error and calculate penalty
        error_type = classify_error(error)
        penalty = get_error_penalty(error_type)
        
        result = BenchmarkResult(
            config=config,
            throughput=throughput,
            output_throughput=output_throughput,
            latency=latency,
            thermal_summary=thermal_summary,
            is_thermally_safe=is_thermally_safe,
            nsys_metrics=nsys_metrics,
            duration=duration,
            error=error,
            error_type=error_type,
            penalty=penalty
        )
        
        self._last_result = result
        return result
    
    def _build_benchmark_command(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        num_prompts: int,
        dataset_path: str
    ) -> List[str]:
        """Build the vLLM benchmark command."""
        command = [
            "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", self.model,
            "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--num-prompts", str(num_prompts),
            "--trust-remote-code",
            "--enforce-eager",
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",
        ]
        
        if os.path.exists(dataset_path):
            command.extend(["--dataset-path", dataset_path])
        
        return command
    
    def _parse_benchmark_output(self, output: str) -> tuple:
        """Parse benchmark output to extract throughput metrics."""
        throughput = 0.0
        output_throughput = 0.0
        latency = None
        
        if not output:
            return throughput, output_throughput, latency
        
        match = re.search(
            r"Throughput:\s*([0-9.]+)\s*requests/s,\s*([0-9.]+)\s*tokens/s",
            output
        )
        if match:
            throughput = float(match.group(2))
        
        match = re.search(
            r"([0-9.]+)\s*(?:output\s+)?tokens/s",
            output,
            re.IGNORECASE
        )
        if match:
            output_throughput = float(match.group(1))
            if throughput == 0.0:
                throughput = output_throughput
        
        match = re.search(
            r"(?:mean|avg|average)\s*latency[:\s]+([0-9.]+)\s*(?:ms|s)",
            output,
            re.IGNORECASE
        )
        if match:
            latency = float(match.group(1))
        
        return throughput, output_throughput, latency
