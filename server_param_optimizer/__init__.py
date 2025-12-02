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

__all__ = [
    'ThermalMonitor',
    'ThermalSample',
    'ThermalConfig',
    'get_gpu_info',
    'ThermalVisualizer',
    'PlotConfig',
    'NsysMetricsExtractor',
    'NsysProfile',
    'check_nsys_available',
]
