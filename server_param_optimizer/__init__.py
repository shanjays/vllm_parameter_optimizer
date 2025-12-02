"""
Server Parameter Optimizer for vLLM

Optimizes --max-num-seqs and --max-num-batched-tokens for:
1. Maximum throughput (aggressive mode)
2. Sustained throughput (thermal-safe mode)

Target: NVIDIA A100 40GB with meta-llama/Llama-3.1-8B-Instruct
"""

from .thermal_monitor import ThermalMonitor, ThermalSample, ThermalConfig, get_gpu_info
from .visualization import ThermalVisualizer, PlotConfig
from .nsys_metrics_extractor import NsysMetricsExtractor, NsysProfile, check_nsys_available
from .server_profiling_worker import (
    ServerProfilingWorker,
    ServerProfilingWorkerLocal,
    BenchmarkResult,
    BenchmarkErrorType,
    classify_error,
    get_error_penalty,
    log_error_details,
    is_benchmark_successful,
    RAY_AVAILABLE
)
from .server_config_exporter import ServerConfigExporter, ServerConfig
from .server_feedback_collector import ServerFeedbackCollector, IterationFeedback
from .server_meta_controller import ServerMetaController, PARAM_SPACE
from .server_optimizer import ServerParameterOptimizer

__all__ = [
    # Thermal monitoring (PR 1)
    'ThermalMonitor',
    'ThermalSample',
    'ThermalConfig',
    'get_gpu_info',
    # Visualization (PR 1)
    'ThermalVisualizer',
    'PlotConfig',
    # Nsys profiling (PR 1)
    'NsysMetricsExtractor',
    'NsysProfile',
    'check_nsys_available',
    # Server profiling worker (PR 2)
    'ServerProfilingWorker',
    'ServerProfilingWorkerLocal',
    'BenchmarkResult',
    'BenchmarkErrorType',
    'classify_error',
    'get_error_penalty',
    'log_error_details',
    'is_benchmark_successful',
    'RAY_AVAILABLE',
    # Config exporter (PR 2)
    'ServerConfigExporter',
    'ServerConfig',
    # Feedback collector (PR 2)
    'ServerFeedbackCollector',
    'IterationFeedback',
    # Main optimizer (PR 3)
    'ServerMetaController',
    'PARAM_SPACE',
    'ServerParameterOptimizer',
]
