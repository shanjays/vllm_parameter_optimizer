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
    RAY_AVAILABLE
)
from .server_config_exporter import ServerConfigExporter, ServerConfig
from .server_feedback_collector import ServerFeedbackCollector, IterationFeedback

__all__ = [
    # Thermal monitoring
    'ThermalMonitor',
    'ThermalSample',
    'ThermalConfig',
    'get_gpu_info',
    # Visualization
    'ThermalVisualizer',
    'PlotConfig',
    # Nsys profiling
    'NsysMetricsExtractor',
    'NsysProfile',
    'check_nsys_available',
    # Server profiling worker
    'ServerProfilingWorker',
    'ServerProfilingWorkerLocal',
    'BenchmarkResult',
    'RAY_AVAILABLE',
    # Config exporter
    'ServerConfigExporter',
    'ServerConfig',
    # Feedback collector
    'ServerFeedbackCollector',
    'IterationFeedback',
]
