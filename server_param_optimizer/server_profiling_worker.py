"""
Server Profiling Worker for vLLM Benchmarks

Ray-based profiling worker for running vLLM throughput benchmarks with thermal monitoring.
Optimizes --max-num-seqs and --max-num-batched-tokens for A100 40GB GPU.

Features:
- Ray actor that runs on a dedicated GPU
- Runs vLLM throughput benchmarks with specified parameters
- Integrates with ThermalMonitor for temperature/power monitoring during benchmarks
- Optionally runs nsys profiling for detailed GPU metrics
- Parses benchmark output to extract throughput (tokens/sec)
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from .thermal_monitor import ThermalMonitor, ThermalConfig, ThermalSummary, get_gpu_info
from .nsys_metrics_extractor import NsysMetricsExtractor, NsysProfile, NsysMetrics

# Check if ray is available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


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
            'error': self.error
        }


# Default configuration for the target GPU and model
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GPU = "NVIDIA A100 40GB"
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
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            
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
            
            result = BenchmarkResult(
                config=config,
                throughput=throughput,
                output_throughput=output_throughput,
                latency=latency,
                thermal_summary=thermal_summary,
                is_thermally_safe=is_thermally_safe,
                nsys_metrics=nsys_metrics,
                duration=duration,
                error=error
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
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
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
        
        result = BenchmarkResult(
            config=config,
            throughput=throughput,
            output_throughput=output_throughput,
            latency=latency,
            thermal_summary=thermal_summary,
            is_thermally_safe=is_thermally_safe,
            nsys_metrics=nsys_metrics,
            duration=duration,
            error=error
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
