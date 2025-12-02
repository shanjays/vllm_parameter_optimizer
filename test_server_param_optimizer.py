"""
Tests for Server Parameter Optimizer Core Infrastructure

These tests verify:
1. ThermalSample and ThermalConfig dataclasses
2. ThermalMonitor methods (without actual nvidia-smi)
3. ThermalVisualizer availability checking
4. NsysMetricsExtractor availability checking and stats parsing
5. Module imports and exports
6. ServerProfilingWorker and BenchmarkResult
7. ServerConfigExporter and ServerConfig
8. ServerFeedbackCollector and IterationFeedback
"""

import json
import os
import sys
import tempfile
import shutil
import time
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_param_optimizer.thermal_monitor import (
    ThermalSample,
    ThermalConfig,
    ThermalSummary,
    ThermalMonitor,
    get_gpu_info
)
from server_param_optimizer.visualization import (
    PlotConfig,
    ThermalVisualizer,
    MATPLOTLIB_AVAILABLE
)
from server_param_optimizer.nsys_metrics_extractor import (
    NsysMetrics,
    NsysProfile,
    NsysMetricsExtractor,
    check_nsys_available
)


# ============================================================
# ThermalSample Tests
# ============================================================

def test_thermal_sample_creation():
    """Test ThermalSample dataclass creation."""
    sample = ThermalSample(
        timestamp=1700000000.0,
        temperature=65.0,
        power=250.0,
        memory_used=20000.0,
        memory_total=40000.0,
        gpu_utilization=85.0,
        memory_utilization=50.0
    )
    
    assert sample.timestamp == 1700000000.0
    assert sample.temperature == 65.0
    assert sample.power == 250.0
    assert sample.memory_used == 20000.0
    assert sample.memory_total == 40000.0
    assert sample.gpu_utilization == 85.0
    assert sample.memory_utilization == 50.0
    
    print("✅ test_thermal_sample_creation PASSED")


def test_thermal_sample_to_dict():
    """Test ThermalSample serialization to dict."""
    sample = ThermalSample(
        timestamp=1700000000.0,
        temperature=65.0,
        power=250.0,
        memory_used=20000.0,
        memory_total=40000.0,
        gpu_utilization=85.0,
        memory_utilization=50.0
    )
    
    d = sample.to_dict()
    
    assert isinstance(d, dict)
    assert d['timestamp'] == 1700000000.0
    assert d['temperature'] == 65.0
    assert 'power' in d
    assert 'memory_used' in d
    
    print("✅ test_thermal_sample_to_dict PASSED")


# ============================================================
# ThermalConfig Tests
# ============================================================

def test_thermal_config_defaults():
    """Test ThermalConfig has correct H100 80GB defaults."""
    config = ThermalConfig()
    
    assert config.max_safe_temp == 85.0
    assert config.target_sustained_temp == 75.0
    assert config.warning_temp == 80.0
    assert config.max_power == 350.0
    assert config.total_memory_gb == 80.0
    assert "H100" in config.gpu_name
    
    print("✅ test_thermal_config_defaults PASSED")


def test_thermal_config_custom():
    """Test ThermalConfig with custom values."""
    config = ThermalConfig(
        max_safe_temp=85.0,
        target_sustained_temp=70.0,
        max_power=350.0
    )
    
    assert config.max_safe_temp == 85.0
    assert config.target_sustained_temp == 70.0
    assert config.max_power == 350.0
    
    print("✅ test_thermal_config_custom PASSED")


def test_thermal_config_to_dict():
    """Test ThermalConfig serialization."""
    config = ThermalConfig()
    d = config.to_dict()
    
    assert isinstance(d, dict)
    assert 'max_safe_temp' in d
    assert 'target_sustained_temp' in d
    assert 'max_power' in d
    
    print("✅ test_thermal_config_to_dict PASSED")


# ============================================================
# ThermalMonitor Tests
# ============================================================

def test_thermal_monitor_init():
    """Test ThermalMonitor initialization."""
    monitor = ThermalMonitor(gpu_index=0, sample_interval=1.0)
    
    assert monitor.gpu_index == 0
    assert monitor.sample_interval == 1.0
    assert monitor.config.max_safe_temp == 85.0
    assert monitor._monitoring == False
    assert len(monitor._samples) == 0
    
    print("✅ test_thermal_monitor_init PASSED")


def test_thermal_monitor_with_custom_config():
    """Test ThermalMonitor with custom ThermalConfig."""
    custom_config = ThermalConfig(max_safe_temp=80.0, target_sustained_temp=70.0)
    monitor = ThermalMonitor(config=custom_config)
    
    assert monitor.config.max_safe_temp == 80.0
    assert monitor.config.target_sustained_temp == 70.0
    
    print("✅ test_thermal_monitor_with_custom_config PASSED")


def test_thermal_monitor_get_samples_empty():
    """Test get_samples returns empty list when no samples."""
    monitor = ThermalMonitor()
    
    samples = monitor.get_samples()
    
    assert samples == []
    assert monitor.get_sample_count() == 0
    
    print("✅ test_thermal_monitor_get_samples_empty PASSED")


def test_thermal_monitor_get_thermal_summary_empty():
    """Test get_thermal_summary returns None when no samples."""
    monitor = ThermalMonitor()
    
    summary = monitor.get_thermal_summary()
    
    assert summary is None
    
    print("✅ test_thermal_monitor_get_thermal_summary_empty PASSED")


def test_thermal_monitor_is_thermally_safe_empty():
    """Test is_thermally_safe returns True when no samples (safe default)."""
    monitor = ThermalMonitor()
    
    assert monitor.is_thermally_safe() == True
    
    print("✅ test_thermal_monitor_is_thermally_safe_empty PASSED")


def test_thermal_monitor_get_samples_as_dict():
    """Test get_samples_as_dict returns list of dicts."""
    monitor = ThermalMonitor()
    
    # Manually add a sample for testing
    sample = ThermalSample(
        timestamp=time.time(),
        temperature=70.0,
        power=300.0,
        memory_used=25000.0,
        memory_total=40000.0,
        gpu_utilization=90.0,
        memory_utilization=62.5
    )
    monitor._samples.append(sample)
    
    samples_dict = monitor.get_samples_as_dict()
    
    assert isinstance(samples_dict, list)
    assert len(samples_dict) == 1
    assert isinstance(samples_dict[0], dict)
    assert samples_dict[0]['temperature'] == 70.0
    
    print("✅ test_thermal_monitor_get_samples_as_dict PASSED")


def test_thermal_summary_calculation():
    """Test ThermalSummary calculation from samples."""
    monitor = ThermalMonitor()
    
    # Add mock samples
    base_time = time.time()
    for i in range(5):
        sample = ThermalSample(
            timestamp=base_time + i,
            temperature=70.0 + i,  # 70, 71, 72, 73, 74
            power=300.0 + i * 10,  # 300, 310, 320, 330, 340
            memory_used=25000.0,
            memory_total=40000.0,
            gpu_utilization=90.0,
            memory_utilization=62.5
        )
        monitor._samples.append(sample)
    
    summary = monitor.get_thermal_summary()
    
    assert summary is not None
    assert summary.sample_count == 5
    assert summary.temp_min == 70.0
    assert summary.temp_max == 74.0
    assert summary.temp_avg == 72.0
    assert summary.temp_final == 74.0
    assert summary.power_min == 300.0
    assert summary.power_max == 340.0
    assert summary.is_thermally_safe == True  # All below 75°C target
    
    print("✅ test_thermal_summary_calculation PASSED")


def test_thermal_summary_not_safe():
    """Test ThermalSummary detects unsafe temperatures."""
    monitor = ThermalMonitor()
    
    base_time = time.time()
    # Add sample above target temperature (75°C)
    sample = ThermalSample(
        timestamp=base_time,
        temperature=80.0,  # Above 75°C target
        power=350.0,
        memory_used=30000.0,
        memory_total=40000.0,
        gpu_utilization=95.0,
        memory_utilization=75.0
    )
    monitor._samples.append(sample)
    
    summary = monitor.get_thermal_summary()
    
    assert summary is not None
    assert summary.is_thermally_safe == False
    assert monitor.is_thermally_safe() == False
    
    print("✅ test_thermal_summary_not_safe PASSED")


def test_thermal_summary_throttling_detected():
    """Test ThermalSummary detects throttling temperatures."""
    monitor = ThermalMonitor()
    
    base_time = time.time()
    # Add sample at throttle temperature (83°C)
    sample = ThermalSample(
        timestamp=base_time,
        temperature=85.0,  # Above 83°C throttle threshold
        power=380.0,
        memory_used=35000.0,
        memory_total=40000.0,
        gpu_utilization=98.0,
        memory_utilization=87.5
    )
    monitor._samples.append(sample)
    
    summary = monitor.get_thermal_summary()
    
    assert summary is not None
    assert summary.throttling_detected == True
    assert summary.max_temp_exceeded == True
    
    print("✅ test_thermal_summary_throttling_detected PASSED")


def test_thermal_summary_to_dict():
    """Test ThermalSummary serialization."""
    monitor = ThermalMonitor()
    
    sample = ThermalSample(
        timestamp=time.time(),
        temperature=70.0,
        power=300.0,
        memory_used=25000.0,
        memory_total=40000.0,
        gpu_utilization=90.0,
        memory_utilization=62.5
    )
    monitor._samples.append(sample)
    
    summary = monitor.get_thermal_summary()
    d = summary.to_dict()
    
    assert isinstance(d, dict)
    assert 'temp_min' in d
    assert 'temp_max' in d
    assert 'is_thermally_safe' in d
    
    print("✅ test_thermal_summary_to_dict PASSED")


# ============================================================
# PlotConfig Tests
# ============================================================

def test_plot_config_defaults():
    """Test PlotConfig has reasonable defaults."""
    config = PlotConfig()
    
    assert config.figure_width == 14.0
    assert config.figure_height == 10.0
    assert config.dpi == 150
    assert config.line_width == 2.0
    
    print("✅ test_plot_config_defaults PASSED")


def test_plot_config_custom():
    """Test PlotConfig with custom values."""
    config = PlotConfig(figure_width=12.0, dpi=200)
    
    assert config.figure_width == 12.0
    assert config.dpi == 200
    
    print("✅ test_plot_config_custom PASSED")


# ============================================================
# ThermalVisualizer Tests
# ============================================================

def test_thermal_visualizer_init():
    """Test ThermalVisualizer initialization."""
    visualizer = ThermalVisualizer()
    
    assert visualizer.config is not None
    assert isinstance(visualizer.config, PlotConfig)
    
    print("✅ test_thermal_visualizer_init PASSED")


def test_thermal_visualizer_is_available():
    """Test ThermalVisualizer reports matplotlib availability correctly."""
    visualizer = ThermalVisualizer()
    
    # Should match module-level constant
    assert visualizer.is_available() == MATPLOTLIB_AVAILABLE
    
    print("✅ test_thermal_visualizer_is_available PASSED")


def test_thermal_visualizer_save_empty_samples():
    """Test save_thermal_plot returns False for empty samples."""
    visualizer = ThermalVisualizer()
    
    result = visualizer.save_thermal_plot(
        samples=[],
        summary=None,
        output_path="/tmp/test_empty.png"
    )
    
    assert result == False
    
    print("✅ test_thermal_visualizer_save_empty_samples PASSED")


def test_thermal_visualizer_save_plot():
    """Test save_thermal_plot creates file when matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        print("⏭️  test_thermal_visualizer_save_plot SKIPPED (matplotlib not installed)")
        return
    
    visualizer = ThermalVisualizer()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create mock samples
        monitor = ThermalMonitor()
        base_time = time.time()
        for i in range(10):
            sample = ThermalSample(
                timestamp=base_time + i,
                temperature=65.0 + i * 0.5,
                power=280.0 + i * 5,
                memory_used=20000.0 + i * 500,
                memory_total=40000.0,
                gpu_utilization=80.0 + i,
                memory_utilization=50.0 + i
            )
            monitor._samples.append(sample)
        
        samples = monitor.get_samples()
        summary = monitor.get_thermal_summary()
        
        output_path = os.path.join(temp_dir, "thermal_test.png")
        result = visualizer.save_thermal_plot(
            samples=samples,
            summary=summary,
            output_path=output_path,
            title="Test Thermal Plot"
        )
        
        assert result == True
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        print("✅ test_thermal_visualizer_save_plot PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# NsysMetrics Tests
# ============================================================

def test_nsys_metrics_defaults():
    """Test NsysMetrics default values."""
    metrics = NsysMetrics()
    
    assert metrics.kernel_count == 0
    assert metrics.total_kernel_time_ms == 0.0
    assert metrics.avg_kernel_time_us == 0.0
    assert metrics.memcpy_count == 0
    assert metrics.top_kernels == []
    
    print("✅ test_nsys_metrics_defaults PASSED")


def test_nsys_metrics_to_dict():
    """Test NsysMetrics serialization."""
    metrics = NsysMetrics(
        kernel_count=1000,
        total_kernel_time_ms=50.5
    )
    
    d = metrics.to_dict()
    
    assert isinstance(d, dict)
    assert d['kernel_count'] == 1000
    assert d['total_kernel_time_ms'] == 50.5
    
    print("✅ test_nsys_metrics_to_dict PASSED")


# ============================================================
# NsysProfile Tests
# ============================================================

def test_nsys_profile_creation():
    """Test NsysProfile dataclass creation."""
    profile = NsysProfile(
        output_path="/tmp/profile.nsys-rep",
        duration_seconds=30.5,
        metrics=NsysMetrics(kernel_count=500),
        raw_stats="test stats output"
    )
    
    assert profile.output_path == "/tmp/profile.nsys-rep"
    assert profile.duration_seconds == 30.5
    assert profile.metrics.kernel_count == 500
    assert profile.success == True
    
    print("✅ test_nsys_profile_creation PASSED")


def test_nsys_profile_failure():
    """Test NsysProfile with failure state."""
    profile = NsysProfile(
        output_path="",
        duration_seconds=0.0,
        metrics=NsysMetrics(),
        raw_stats="",
        success=False,
        error_message="nsys not installed"
    )
    
    assert profile.success == False
    assert profile.error_message == "nsys not installed"
    
    print("✅ test_nsys_profile_failure PASSED")


def test_nsys_profile_to_dict():
    """Test NsysProfile serialization."""
    profile = NsysProfile(
        output_path="/tmp/test.nsys-rep",
        duration_seconds=10.0,
        metrics=NsysMetrics(),
        raw_stats="stats"
    )
    
    d = profile.to_dict()
    
    assert isinstance(d, dict)
    assert d['output_path'] == "/tmp/test.nsys-rep"
    assert d['success'] == True
    assert 'metrics' in d
    
    print("✅ test_nsys_profile_to_dict PASSED")


# ============================================================
# NsysMetricsExtractor Tests
# ============================================================

def test_nsys_extractor_init():
    """Test NsysMetricsExtractor initialization."""
    extractor = NsysMetricsExtractor(gpu_index=0)
    
    assert extractor.gpu_index == 0
    # is_available() should return False if nsys not installed
    # We can't assert the value, but it should not raise
    _ = extractor.is_available()
    
    print("✅ test_nsys_extractor_init PASSED")


def test_check_nsys_available():
    """Test check_nsys_available function."""
    result = check_nsys_available()
    
    # Result should be boolean
    assert isinstance(result, bool)
    
    print("✅ test_check_nsys_available PASSED")


def test_nsys_extractor_profile_unavailable():
    """Test profile_command when nsys is not available."""
    extractor = NsysMetricsExtractor()
    
    if extractor.is_available():
        print("⏭️  test_nsys_extractor_profile_unavailable SKIPPED (nsys is available)")
        return
    
    profile = extractor.profile_command(
        command="echo test",
        output_dir="/tmp"
    )
    
    assert profile.success == False
    assert "not installed" in profile.error_message
    
    print("✅ test_nsys_extractor_profile_unavailable PASSED")


def test_nsys_extractor_parse_empty_stats():
    """Test _parse_stats_output with empty input."""
    extractor = NsysMetricsExtractor()
    
    metrics = extractor._parse_stats_output("")
    
    assert metrics.kernel_count == 0
    assert metrics.total_kernel_time_ms == 0.0
    
    print("✅ test_nsys_extractor_parse_empty_stats PASSED")


def test_nsys_extractor_get_metrics_summary_failure():
    """Test get_metrics_summary with failed profile."""
    extractor = NsysMetricsExtractor()
    
    profile = NsysProfile(
        output_path="",
        duration_seconds=0.0,
        metrics=NsysMetrics(),
        raw_stats="",
        success=False,
        error_message="test error"
    )
    
    summary = extractor.get_metrics_summary(profile)
    
    assert summary['success'] == False
    assert summary['error'] == "test error"
    
    print("✅ test_nsys_extractor_get_metrics_summary_failure PASSED")


def test_nsys_extractor_get_metrics_summary_success():
    """Test get_metrics_summary with successful profile."""
    extractor = NsysMetricsExtractor()
    
    metrics = NsysMetrics(
        kernel_count=100,
        total_kernel_time_ms=25.5,
        avg_kernel_time_us=255.0
    )
    
    profile = NsysProfile(
        output_path="/tmp/test.nsys-rep",
        duration_seconds=30.0,
        metrics=metrics,
        raw_stats="test",
        success=True
    )
    
    summary = extractor.get_metrics_summary(profile)
    
    assert summary['success'] == True
    assert summary['kernel_count'] == 100
    assert summary['total_kernel_time_ms'] == 25.5
    
    print("✅ test_nsys_extractor_get_metrics_summary_success PASSED")


# ============================================================
# Module Import Tests
# ============================================================

def test_module_imports():
    """Test that all expected classes are exported from the package."""
    from server_param_optimizer import (
        ThermalMonitor,
        ThermalSample,
        ThermalConfig,
        get_gpu_info,
        ThermalVisualizer,
        PlotConfig,
        NsysMetricsExtractor,
        NsysProfile,
        check_nsys_available
    )
    
    # All imports should work
    assert ThermalMonitor is not None
    assert ThermalSample is not None
    assert ThermalConfig is not None
    assert get_gpu_info is not None
    assert ThermalVisualizer is not None
    assert PlotConfig is not None
    assert NsysMetricsExtractor is not None
    assert NsysProfile is not None
    assert check_nsys_available is not None
    
    print("✅ test_module_imports PASSED")


# ============================================================
# BenchmarkResult Tests
# ============================================================

def test_benchmark_result_creation():
    """Test BenchmarkResult dataclass creation."""
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    result = BenchmarkResult(
        config={'max_num_seqs': 256, 'max_num_batched_tokens': 8192},
        throughput=1500.0,
        output_throughput=1200.0,
        latency=50.0,
        thermal_summary=None,
        is_thermally_safe=True,
        nsys_metrics=None,
        duration=120.0,
        error=""
    )
    
    assert result.config['max_num_seqs'] == 256
    assert result.throughput == 1500.0
    assert result.output_throughput == 1200.0
    assert result.is_thermally_safe == True
    assert result.duration == 120.0
    assert result.error == ""
    
    print("✅ test_benchmark_result_creation PASSED")


def test_benchmark_result_to_dict():
    """Test BenchmarkResult serialization to dict."""
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    result = BenchmarkResult(
        config={'max_num_seqs': 128},
        throughput=1000.0,
        output_throughput=800.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=False,
        nsys_metrics={'kernel_count': 100},
        duration=60.0
    )
    
    d = result.to_dict()
    
    assert isinstance(d, dict)
    assert d['config']['max_num_seqs'] == 128
    assert d['throughput'] == 1000.0
    assert d['is_thermally_safe'] == False
    assert d['nsys_metrics']['kernel_count'] == 100
    
    print("✅ test_benchmark_result_to_dict PASSED")


def test_benchmark_result_with_thermal_summary():
    """Test BenchmarkResult with ThermalSummary."""
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    from server_param_optimizer.thermal_monitor import ThermalSummary
    
    thermal_summary = ThermalSummary(
        duration_seconds=120.0,
        sample_count=120,
        temp_min=60.0,
        temp_max=72.0,
        temp_avg=68.0,
        temp_final=70.0,
        power_min=250.0,
        power_max=350.0,
        power_avg=300.0,
        memory_max_used_mb=30000.0,
        memory_max_used_pct=75.0,
        gpu_util_avg=90.0,
        memory_util_avg=60.0,
        is_thermally_safe=True,
        max_temp_exceeded=False,
        throttling_detected=False,
        time_above_target=0.0
    )
    
    result = BenchmarkResult(
        config={'max_num_seqs': 256},
        throughput=1500.0,
        output_throughput=1200.0,
        latency=45.0,
        thermal_summary=thermal_summary,
        is_thermally_safe=True,
        nsys_metrics=None,
        duration=120.0
    )
    
    d = result.to_dict()
    
    assert d['thermal_summary'] is not None
    assert d['thermal_summary']['temp_max'] == 72.0
    assert d['thermal_summary']['is_thermally_safe'] == True
    
    print("✅ test_benchmark_result_with_thermal_summary PASSED")


# ============================================================
# Error Classification and Penalty Tests
# ============================================================

def test_benchmark_error_type_enum():
    """Test BenchmarkErrorType enum values."""
    from server_param_optimizer.server_profiling_worker import BenchmarkErrorType
    
    assert BenchmarkErrorType.NONE.value == "none"
    assert BenchmarkErrorType.VRAM_OOM.value == "vram_oom"
    assert BenchmarkErrorType.CUDA_ERROR.value == "cuda_error"
    assert BenchmarkErrorType.ENGINE_INIT_FAILED.value == "engine_init_failed"
    assert BenchmarkErrorType.TIMEOUT.value == "timeout"
    assert BenchmarkErrorType.INVALID_CONFIG.value == "invalid_config"
    assert BenchmarkErrorType.UNKNOWN.value == "unknown"
    
    print("✅ test_benchmark_error_type_enum PASSED")


def test_classify_error_vram_oom():
    """Test classify_error for VRAM/OOM errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    # Various OOM error patterns
    assert classify_error("CUDA out of memory") == BenchmarkErrorType.VRAM_OOM
    assert classify_error("torch.cuda.OutOfMemoryError") == BenchmarkErrorType.VRAM_OOM
    assert classify_error("RuntimeError: Out of memory") == BenchmarkErrorType.VRAM_OOM
    assert classify_error("Cannot allocate VRAM") == BenchmarkErrorType.VRAM_OOM
    assert classify_error("Memory allocation failed") == BenchmarkErrorType.VRAM_OOM
    
    print("✅ test_classify_error_vram_oom PASSED")


def test_classify_error_engine_init():
    """Test classify_error for engine initialization failures."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    # Engine init error patterns
    assert classify_error("RuntimeError: Engine core initialization failed. See root cause above.") == BenchmarkErrorType.ENGINE_INIT_FAILED
    assert classify_error("Failed to initialize vLLM engine") == BenchmarkErrorType.ENGINE_INIT_FAILED
    assert classify_error("Failed core proc(s): {}") == BenchmarkErrorType.ENGINE_INIT_FAILED
    
    print("✅ test_classify_error_engine_init PASSED")


def test_classify_error_cuda():
    """Test classify_error for CUDA errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    # CUDA error patterns
    assert classify_error("CUDA error: device-side assert triggered") == BenchmarkErrorType.CUDA_ERROR
    assert classify_error("cudnn error") == BenchmarkErrorType.CUDA_ERROR
    assert classify_error("cublas error") == BenchmarkErrorType.CUDA_ERROR
    
    print("✅ test_classify_error_cuda PASSED")


def test_classify_error_timeout():
    """Test classify_error for timeout errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    # Timeout patterns
    assert classify_error("Benchmark timed out after 22 minutes") == BenchmarkErrorType.TIMEOUT
    assert classify_error("Process timeout exceeded") == BenchmarkErrorType.TIMEOUT
    
    print("✅ test_classify_error_timeout PASSED")


def test_classify_error_invalid_config():
    """Test classify_error for invalid config errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    # Invalid config patterns
    assert classify_error("max_num_batched_tokens must be >= max_num_seqs") == BenchmarkErrorType.INVALID_CONFIG
    assert classify_error("Invalid value for max_num_seqs") == BenchmarkErrorType.INVALID_CONFIG
    
    print("✅ test_classify_error_invalid_config PASSED")


def test_classify_error_none():
    """Test classify_error returns NONE for empty errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    assert classify_error("") == BenchmarkErrorType.NONE
    assert classify_error(None) == BenchmarkErrorType.NONE
    
    print("✅ test_classify_error_none PASSED")


def test_classify_error_unknown():
    """Test classify_error returns UNKNOWN for unrecognized errors."""
    from server_param_optimizer.server_profiling_worker import classify_error, BenchmarkErrorType
    
    assert classify_error("Some random error message that doesn't match") == BenchmarkErrorType.UNKNOWN
    
    print("✅ test_classify_error_unknown PASSED")


def test_get_error_penalty():
    """Test get_error_penalty returns correct penalties."""
    from server_param_optimizer.server_profiling_worker import get_error_penalty, BenchmarkErrorType
    
    assert get_error_penalty(BenchmarkErrorType.NONE) == 0.0
    assert get_error_penalty(BenchmarkErrorType.VRAM_OOM) == -100.0
    assert get_error_penalty(BenchmarkErrorType.ENGINE_INIT_FAILED) == -80.0
    assert get_error_penalty(BenchmarkErrorType.CUDA_ERROR) == -50.0
    assert get_error_penalty(BenchmarkErrorType.TIMEOUT) == -30.0
    assert get_error_penalty(BenchmarkErrorType.INVALID_CONFIG) == -20.0
    assert get_error_penalty(BenchmarkErrorType.UNKNOWN) == -10.0
    
    print("✅ test_get_error_penalty PASSED")


def test_benchmark_result_is_successful():
    """Test BenchmarkResult.is_successful property."""
    from server_param_optimizer.server_profiling_worker import BenchmarkResult, BenchmarkErrorType
    
    # Successful result
    success_result = BenchmarkResult(
        config={'max_num_seqs': 64},
        throughput=1500.0,
        output_throughput=1200.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=True,
        nsys_metrics=None,
        duration=60.0,
        error=""
    )
    assert success_result.is_successful == True
    assert success_result.effective_throughput == 1500.0
    
    # Failed result with error
    failed_result = BenchmarkResult(
        config={'max_num_seqs': 128},
        throughput=0.0,
        output_throughput=0.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=False,
        nsys_metrics=None,
        duration=10.0,
        error="CUDA out of memory",
        error_type=BenchmarkErrorType.VRAM_OOM,
        penalty=-100.0
    )
    assert failed_result.is_successful == False
    assert failed_result.effective_throughput == -100.0
    
    print("✅ test_benchmark_result_is_successful PASSED")


def test_benchmark_result_with_error_type():
    """Test BenchmarkResult with error classification."""
    from server_param_optimizer.server_profiling_worker import BenchmarkResult, BenchmarkErrorType
    
    result = BenchmarkResult(
        config={'max_num_seqs': 256, 'max_num_batched_tokens': 32768},
        throughput=0.0,
        output_throughput=0.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=False,
        nsys_metrics=None,
        duration=5.0,
        error="RuntimeError: Engine core initialization failed",
        error_type=BenchmarkErrorType.ENGINE_INIT_FAILED,
        penalty=-80.0
    )
    
    d = result.to_dict()
    
    assert d['error_type'] == 'engine_init_failed'
    assert d['penalty'] == -80.0
    assert d['is_successful'] == False
    assert d['effective_throughput'] == -80.0
    
    print("✅ test_benchmark_result_with_error_type PASSED")


def test_log_error_details():
    """Test log_error_details function."""
    from server_param_optimizer.server_profiling_worker import log_error_details
    import io
    import sys
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        log_error_details(
            "CUDA out of memory when allocating tensor",
            {'max_num_seqs': 128, 'max_num_batched_tokens': 32768}
        )
    finally:
        sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    assert "[BENCHMARK ERROR]" in output
    assert "vram_oom" in output
    assert "-100.0" in output
    assert "max_num_seqs=128" in output
    assert "[DEBUG HINT]" in output
    
    print("✅ test_log_error_details PASSED")


def test_is_benchmark_successful():
    """Test is_benchmark_successful utility function."""
    from server_param_optimizer.server_profiling_worker import (
        is_benchmark_successful, 
        BenchmarkResult, 
        BenchmarkErrorType
    )
    
    # Test with BenchmarkResult object (successful)
    success_result = BenchmarkResult(
        config={'max_num_seqs': 64},
        throughput=1500.0,
        output_throughput=1200.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=True,
        nsys_metrics=None,
        duration=60.0,
        error=""
    )
    assert is_benchmark_successful(success_result) == True
    
    # Test with BenchmarkResult object (failed)
    failed_result = BenchmarkResult(
        config={'max_num_seqs': 256},
        throughput=0.0,
        output_throughput=0.0,
        latency=None,
        thermal_summary=None,
        is_thermally_safe=False,
        nsys_metrics=None,
        duration=5.0,
        error="OOM error",
        error_type=BenchmarkErrorType.VRAM_OOM,
        penalty=-100.0
    )
    assert is_benchmark_successful(failed_result) == False
    
    # Test with dictionary (successful)
    success_dict = {'throughput': 1500.0, 'error': '', 'is_successful': True}
    assert is_benchmark_successful(success_dict) == True
    
    # Test with dictionary (failed, explicit is_successful)
    failed_dict_explicit = {'throughput': 0.0, 'error': 'error', 'is_successful': False}
    assert is_benchmark_successful(failed_dict_explicit) == False
    
    # Test with dictionary (failed, inferred from fields)
    failed_dict_inferred = {'throughput': 0.0, 'error': 'OOM'}
    assert is_benchmark_successful(failed_dict_inferred) == False
    
    # Test with dictionary (successful, inferred from fields)
    success_dict_inferred = {'throughput': 1000.0, 'error': ''}
    assert is_benchmark_successful(success_dict_inferred) == True
    
    print("✅ test_is_benchmark_successful PASSED")


# ============================================================
# GPU ID Mapping Tests (CUDA_VISIBLE_DEVICES handling)
# ============================================================

def test_get_physical_gpu_id_no_cuda_visible_devices():
    """Test get_physical_gpu_id when CUDA_VISIBLE_DEVICES is not set."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Remove CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Should return the index directly as a string
        assert get_physical_gpu_id(0) == "0"
        assert get_physical_gpu_id(1) == "1"
        assert get_physical_gpu_id(5) == "5"
    finally:
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_no_cuda_visible_devices PASSED")


def test_get_physical_gpu_id_with_cuda_visible_devices():
    """Test get_physical_gpu_id when CUDA_VISIBLE_DEVICES is set in parent."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Set CUDA_VISIBLE_DEVICES to a specific list
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        
        # Should map visible index to physical GPU
        assert get_physical_gpu_id(0) == "5"
        assert get_physical_gpu_id(1) == "6"
        
        # Test with different configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,7"
        assert get_physical_gpu_id(0) == "2"
        assert get_physical_gpu_id(1) == "3"
        assert get_physical_gpu_id(2) == "7"
        
        # Test single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        assert get_physical_gpu_id(0) == "4"
    finally:
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_with_cuda_visible_devices PASSED")


def test_get_physical_gpu_id_with_spaces():
    """Test get_physical_gpu_id handles spaces in CUDA_VISIBLE_DEVICES."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Test with spaces
        os.environ["CUDA_VISIBLE_DEVICES"] = " 5 , 6 , 7 "
        assert get_physical_gpu_id(0) == "5"
        assert get_physical_gpu_id(1) == "6"
        assert get_physical_gpu_id(2) == "7"
    finally:
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_with_spaces PASSED")


def test_get_physical_gpu_id_out_of_range():
    """Test get_physical_gpu_id when index is out of range."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    import io
    import sys
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        
        # Capture warning output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Index out of range should return the index directly with a warning
        result = get_physical_gpu_id(3)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert result == "3"
        assert "[WARNING]" in output
        assert "out of range" in output
    finally:
        sys.stdout = sys.__stdout__
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_out_of_range PASSED")


def test_get_physical_gpu_id_empty_cuda_visible_devices():
    """Test get_physical_gpu_id when CUDA_VISIBLE_DEVICES is empty."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Empty string
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        assert get_physical_gpu_id(0) == "0"
        assert get_physical_gpu_id(1) == "1"
        
        # Just whitespace
        os.environ["CUDA_VISIBLE_DEVICES"] = "   "
        assert get_physical_gpu_id(0) == "0"
    finally:
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_empty_cuda_visible_devices PASSED")


def test_get_physical_gpu_id_negative_index():
    """Test get_physical_gpu_id handles negative indices gracefully."""
    from server_param_optimizer.server_profiling_worker import get_physical_gpu_id
    import os
    import io
    import sys
    
    # Save original value
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    try:
        # Test with no CUDA_VISIBLE_DEVICES (should return "0")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Capture warning output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        result = get_physical_gpu_id(-1)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should warn and default to 0
        assert result == "0"
        assert "[WARNING]" in output
        assert "negative" in output.lower()
        
        # Test with CUDA_VISIBLE_DEVICES set
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        result = get_physical_gpu_id(-2)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should warn and map 0 -> "5"
        assert result == "5"
        assert "[WARNING]" in output
    finally:
        sys.stdout = sys.__stdout__
        # Restore original value
        if original is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
    
    print("✅ test_get_physical_gpu_id_negative_index PASSED")


# ============================================================
# ServerProfilingWorkerLocal Tests
# ============================================================

def test_server_profiling_worker_local_init():
    """Test ServerProfilingWorkerLocal initialization."""
    from server_param_optimizer.server_profiling_worker import ServerProfilingWorkerLocal
    
    worker = ServerProfilingWorkerLocal(gpu_id=0, model="test-model")
    
    assert worker.gpu_id == 0
    assert worker.model == "test-model"
    assert worker.thermal_config is not None
    assert worker.thermal_monitor is not None
    
    print("✅ test_server_profiling_worker_local_init PASSED")


def test_server_profiling_worker_local_parse_benchmark_output():
    """Test ServerProfilingWorkerLocal benchmark output parsing."""
    from server_param_optimizer.server_profiling_worker import ServerProfilingWorkerLocal
    
    worker = ServerProfilingWorkerLocal()
    
    # Test pattern 1: "Throughput: X.XX requests/s, Y.YY tokens/s"
    output1 = "Throughput: 25.50 requests/s, 1850.75 tokens/s"
    t1, o1, l1 = worker._parse_benchmark_output(output1)
    assert t1 == 1850.75
    
    # Test pattern 2: "X.XX output tokens/s"
    output2 = "Output: 1500.5 output tokens/s"
    t2, o2, l2 = worker._parse_benchmark_output(output2)
    assert o2 == 1500.5
    
    # Test empty output
    t3, o3, l3 = worker._parse_benchmark_output("")
    assert t3 == 0.0
    assert o3 == 0.0
    
    print("✅ test_server_profiling_worker_local_parse_benchmark_output PASSED")


def test_server_profiling_worker_local_build_command():
    """Test ServerProfilingWorkerLocal command building."""
    from server_param_optimizer.server_profiling_worker import ServerProfilingWorkerLocal
    
    worker = ServerProfilingWorkerLocal(model="test-model")
    
    command = worker._build_benchmark_command(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        num_prompts=100,
        dataset_path="/nonexistent/path.json"
    )
    
    assert "python" in command
    assert "--max-num-seqs" in command
    assert "256" in command
    assert "--max-num-batched-tokens" in command
    assert "8192" in command
    assert "--model" in command
    assert "test-model" in command
    
    print("✅ test_server_profiling_worker_local_build_command PASSED")


# ============================================================
# ServerConfig Tests
# ============================================================

def test_server_config_creation():
    """Test ServerConfig dataclass creation."""
    from server_param_optimizer.server_config_exporter import ServerConfig
    
    config = ServerConfig(
        mode='aggressive',
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        throughput=1500.0,
        thermal_summary={'temp_max': 75.0},
        model='test-model',
        gpu='NVIDIA A100',
        timestamp='2024-01-01T00:00:00'
    )
    
    assert config.mode == 'aggressive'
    assert config.max_num_seqs == 256
    assert config.max_num_batched_tokens == 8192
    assert config.throughput == 1500.0
    
    print("✅ test_server_config_creation PASSED")


def test_server_config_to_dict():
    """Test ServerConfig serialization."""
    from server_param_optimizer.server_config_exporter import ServerConfig
    
    config = ServerConfig(
        mode='sustained',
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        throughput=1200.0,
        thermal_summary=None,
        model='test-model',
        gpu='NVIDIA A100',
        timestamp='2024-01-01T00:00:00'
    )
    
    d = config.to_dict()
    
    assert isinstance(d, dict)
    assert d['mode'] == 'sustained'
    assert d['max_num_seqs'] == 128
    
    print("✅ test_server_config_to_dict PASSED")


# ============================================================
# ServerConfigExporter Tests
# ============================================================

def test_server_config_exporter_init():
    """Test ServerConfigExporter initialization."""
    from server_param_optimizer.server_config_exporter import ServerConfigExporter
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ServerConfigExporter(
            output_dir=temp_dir,
            model="test-model",
            gpu="Test GPU"
        )
        
        assert exporter.output_dir == temp_dir
        assert exporter.model == "test-model"
        assert exporter.gpu == "Test GPU"
        assert exporter.best_aggressive is None
        assert exporter.best_sustained is None
        
        print("✅ test_server_config_exporter_init PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_config_exporter_update_best_configs():
    """Test ServerConfigExporter update_best_configs method."""
    from server_param_optimizer.server_config_exporter import ServerConfigExporter
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ServerConfigExporter(output_dir=temp_dir)
        
        # Create a thermally-safe result
        result1 = BenchmarkResult(
            config={'max_num_seqs': 128, 'max_num_batched_tokens': 4096},
            throughput=1200.0,
            output_throughput=1000.0,
            latency=50.0,
            thermal_summary=None,
            is_thermally_safe=True,
            nsys_metrics=None,
            duration=60.0
        )
        
        updated = exporter.update_best_configs(result1)
        assert updated == True
        assert exporter.best_aggressive is not None
        assert exporter.best_sustained is not None
        assert exporter.best_aggressive.throughput == 1200.0
        assert exporter.best_sustained.throughput == 1200.0
        
        # Create a higher throughput but not thermally safe result
        result2 = BenchmarkResult(
            config={'max_num_seqs': 256, 'max_num_batched_tokens': 8192},
            throughput=1500.0,
            output_throughput=1200.0,
            latency=45.0,
            thermal_summary=None,
            is_thermally_safe=False,
            nsys_metrics=None,
            duration=60.0
        )
        
        updated = exporter.update_best_configs(result2)
        assert updated == True
        assert exporter.best_aggressive.throughput == 1500.0
        assert exporter.best_sustained.throughput == 1200.0  # Should not change
        
        print("✅ test_server_config_exporter_update_best_configs PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_config_exporter_skips_failed_benchmarks():
    """Test ServerConfigExporter skips failed benchmarks."""
    from server_param_optimizer.server_config_exporter import ServerConfigExporter
    from server_param_optimizer.server_profiling_worker import BenchmarkResult, BenchmarkErrorType
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ServerConfigExporter(output_dir=temp_dir)
        
        # Create a failed result (simulating OOM)
        failed_result = BenchmarkResult(
            config={'max_num_seqs': 256, 'max_num_batched_tokens': 32768},
            throughput=0.0,
            output_throughput=0.0,
            latency=None,
            thermal_summary=None,
            is_thermally_safe=False,
            nsys_metrics=None,
            duration=5.0,
            error="CUDA out of memory",
            error_type=BenchmarkErrorType.VRAM_OOM,
            penalty=-100.0
        )
        
        # Failed benchmark should not update best configs
        updated = exporter.update_best_configs(failed_result)
        assert updated == False
        assert exporter.best_aggressive is None
        assert exporter.best_sustained is None
        
        # Now add a successful result
        success_result = BenchmarkResult(
            config={'max_num_seqs': 64, 'max_num_batched_tokens': 8192},
            throughput=1500.0,
            output_throughput=1200.0,
            latency=50.0,
            thermal_summary=None,
            is_thermally_safe=True,
            nsys_metrics=None,
            duration=60.0
        )
        
        updated = exporter.update_best_configs(success_result)
        assert updated == True
        assert exporter.best_aggressive.throughput == 1500.0
        assert exporter.best_sustained.throughput == 1500.0
        
        # Verify the failed result was still recorded in all_results
        assert len(exporter.all_results) == 2
        assert exporter.all_results[0]['error'] == "CUDA out of memory"
        assert exporter.all_results[0]['error_type'] == 'vram_oom'
        assert exporter.all_results[0]['penalty'] == -100.0
        
        print("✅ test_server_config_exporter_skips_failed_benchmarks PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_config_exporter_save_configs():
    """Test ServerConfigExporter save_configs method."""
    from server_param_optimizer.server_config_exporter import ServerConfigExporter
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ServerConfigExporter(output_dir=temp_dir)
        
        result = BenchmarkResult(
            config={'max_num_seqs': 256, 'max_num_batched_tokens': 8192},
            throughput=1500.0,
            output_throughput=1200.0,
            latency=None,
            thermal_summary=None,
            is_thermally_safe=True,
            nsys_metrics=None,
            duration=120.0
        )
        
        exporter.update_best_configs(result)
        exporter.save_configs()
        
        # Check files were created
        assert os.path.exists(os.path.join(temp_dir, "config_aggressive.json"))
        assert os.path.exists(os.path.join(temp_dir, "config_sustained.json"))
        assert os.path.exists(os.path.join(temp_dir, "optimization_results.json"))
        
        # Verify content
        with open(os.path.join(temp_dir, "config_aggressive.json"), 'r') as f:
            aggressive_config = json.load(f)
        assert aggressive_config['config']['max_num_seqs'] == 256
        
        print("✅ test_server_config_exporter_save_configs PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_config_exporter_generate_launch_scripts():
    """Test ServerConfigExporter launch script generation."""
    from server_param_optimizer.server_config_exporter import ServerConfigExporter
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = ServerConfigExporter(output_dir=temp_dir)
        
        result = BenchmarkResult(
            config={'max_num_seqs': 256, 'max_num_batched_tokens': 8192},
            throughput=1500.0,
            output_throughput=1200.0,
            latency=None,
            thermal_summary=None,
            is_thermally_safe=True,
            nsys_metrics=None,
            duration=120.0
        )
        
        exporter.update_best_configs(result)
        exporter.save_configs()
        
        # Check launch scripts were created
        aggressive_script = os.path.join(temp_dir, "launch_scripts", "launch_aggressive.sh")
        sustained_script = os.path.join(temp_dir, "launch_scripts", "launch_sustained.sh")
        
        assert os.path.exists(aggressive_script)
        assert os.path.exists(sustained_script)
        
        # Check scripts are executable
        assert os.access(aggressive_script, os.X_OK)
        assert os.access(sustained_script, os.X_OK)
        
        # Check script content
        with open(aggressive_script, 'r') as f:
            content = f.read()
        assert "MAX_NUM_SEQS=256" in content
        assert "MAX_NUM_BATCHED_TOKENS=8192" in content
        assert "AGGRESSIVE" in content.upper()
        
        print("✅ test_server_config_exporter_generate_launch_scripts PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# IterationFeedback Tests
# ============================================================

def test_iteration_feedback_creation():
    """Test IterationFeedback dataclass creation."""
    from server_param_optimizer.server_feedback_collector import IterationFeedback
    
    feedback = IterationFeedback(
        iteration=1,
        configs_tested=[{'max_num_seqs': 256}],
        results=[{'throughput': 1500.0}],
        best_aggressive_throughput=1500.0,
        best_sustained_throughput=1200.0,
        timestamp='2024-01-01T00:00:00'
    )
    
    assert feedback.iteration == 1
    assert len(feedback.configs_tested) == 1
    assert feedback.best_aggressive_throughput == 1500.0
    
    print("✅ test_iteration_feedback_creation PASSED")


def test_iteration_feedback_to_dict():
    """Test IterationFeedback serialization."""
    from server_param_optimizer.server_feedback_collector import IterationFeedback
    
    feedback = IterationFeedback(
        iteration=2,
        configs_tested=[],
        results=[],
        best_aggressive_throughput=0.0,
        best_sustained_throughput=0.0,
        timestamp='2024-01-01T00:00:00'
    )
    
    d = feedback.to_dict()
    
    assert isinstance(d, dict)
    assert d['iteration'] == 2
    
    print("✅ test_iteration_feedback_to_dict PASSED")


# ============================================================
# ServerFeedbackCollector Tests
# ============================================================

def test_server_feedback_collector_init():
    """Test ServerFeedbackCollector initialization."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        assert collector.state_file == temp_file
        assert collector.iterations == []
        assert collector.best_aggressive_throughput == 0.0
        assert collector.best_sustained_throughput == 0.0
        
        print("✅ test_server_feedback_collector_init PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(os.path.dirname(temp_file))


def test_server_feedback_collector_add_iteration():
    """Test ServerFeedbackCollector add_iteration method."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        configs = [
            {'max_num_seqs': 128, 'max_num_batched_tokens': 4096},
            {'max_num_seqs': 256, 'max_num_batched_tokens': 8192}
        ]
        
        results = [
            BenchmarkResult(
                config=configs[0],
                throughput=1200.0,
                output_throughput=1000.0,
                latency=50.0,
                thermal_summary=None,
                is_thermally_safe=True,
                nsys_metrics=None,
                duration=60.0
            ),
            BenchmarkResult(
                config=configs[1],
                throughput=1500.0,
                output_throughput=1200.0,
                latency=45.0,
                thermal_summary=None,
                is_thermally_safe=False,
                nsys_metrics=None,
                duration=60.0
            )
        ]
        
        collector.add_iteration(configs, results)
        
        assert len(collector.iterations) == 1
        assert collector.best_aggressive_throughput == 1500.0
        assert collector.best_sustained_throughput == 1200.0
        assert len(collector.all_configs_tested) == 2
        
        print("✅ test_server_feedback_collector_add_iteration PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(os.path.dirname(temp_file))


def test_server_feedback_collector_get_feedback_for_prompt():
    """Test ServerFeedbackCollector get_feedback_for_prompt method."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Empty collector should return empty string
        feedback = collector.get_feedback_for_prompt()
        assert feedback == ""
        
        # Add some data
        configs = [{'max_num_seqs': 256, 'max_num_batched_tokens': 8192}]
        results = [{'throughput': 1500.0, 'is_thermally_safe': True}]
        collector.add_iteration(configs, results)
        
        feedback = collector.get_feedback_for_prompt()
        
        assert "SERVER PARAMETER OPTIMIZATION FEEDBACK" in feedback
        assert "Iterations Completed: 1" in feedback
        assert "BEST CONFIGURATIONS FOUND" in feedback
        
        print("✅ test_server_feedback_collector_get_feedback_for_prompt PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(os.path.dirname(temp_file))


def test_server_feedback_collector_get_untested_configs():
    """Test ServerFeedbackCollector get_untested_configs method."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Add some tested configs
        configs = [
            {'max_num_seqs': 128, 'max_num_batched_tokens': 4096},
            {'max_num_seqs': 256, 'max_num_batched_tokens': 4096}
        ]
        results = [{'throughput': 1000.0}, {'throughput': 1200.0}]
        collector.add_iteration(configs, results)
        
        # Define parameter space
        param_space = {
            'max_num_seqs': [128, 256, 512],
            'max_num_batched_tokens': [4096, 8192]
        }
        
        untested = collector.get_untested_configs(param_space)
        
        # Should have 4 untested configs (6 total - 2 tested)
        assert len(untested) == 4
        
        # Verify specific untested configs
        untested_set = {(c['max_num_seqs'], c['max_num_batched_tokens']) for c in untested}
        assert (128, 8192) in untested_set
        assert (256, 8192) in untested_set
        assert (512, 4096) in untested_set
        assert (512, 8192) in untested_set
        
        print("✅ test_server_feedback_collector_get_untested_configs PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(os.path.dirname(temp_file))


def test_server_feedback_collector_reset():
    """Test ServerFeedbackCollector reset method."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Add some data
        configs = [{'max_num_seqs': 256}]
        results = [{'throughput': 1500.0, 'is_thermally_safe': True}]
        collector.add_iteration(configs, results)
        
        assert len(collector.iterations) == 1
        
        # Reset
        collector.reset()
        
        assert len(collector.iterations) == 0
        assert collector.best_aggressive_throughput == 0.0
        assert collector.best_sustained_throughput == 0.0
        assert not os.path.exists(temp_file)
        
        print("✅ test_server_feedback_collector_reset PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(os.path.dirname(temp_file)):
            shutil.rmtree(os.path.dirname(temp_file))


def test_server_feedback_collector_persistence():
    """Test ServerFeedbackCollector state persistence."""
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_state.json")
    try:
        # Create collector and add data
        collector1 = ServerFeedbackCollector(state_file=temp_file)
        configs = [{'max_num_seqs': 256, 'max_num_batched_tokens': 8192}]
        results = [{'throughput': 1500.0, 'is_thermally_safe': True}]
        collector1.add_iteration(configs, results)
        
        # Verify state file was created
        assert os.path.exists(temp_file)
        
        # Create new collector that should load the saved state
        collector2 = ServerFeedbackCollector(state_file=temp_file)
        
        assert len(collector2.iterations) == 1
        assert collector2.best_aggressive_throughput == 1500.0
        assert len(collector2.all_configs_tested) == 1
        
        print("✅ test_server_feedback_collector_persistence PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(temp_dir)


# ============================================================
# New Module Import Tests
# ============================================================

def test_new_module_imports():
    """Test that all new classes are exported from the package."""
    from server_param_optimizer import (
        ServerProfilingWorker,
        ServerProfilingWorkerLocal,
        BenchmarkResult,
        RAY_AVAILABLE,
        ServerConfigExporter,
        ServerConfig,
        ServerFeedbackCollector,
        IterationFeedback,
    )
    
    # All imports should work (ServerProfilingWorker may be None if Ray not available)
    assert ServerProfilingWorkerLocal is not None
    assert BenchmarkResult is not None
    assert isinstance(RAY_AVAILABLE, bool)
    assert ServerConfigExporter is not None
    assert ServerConfig is not None
    assert ServerFeedbackCollector is not None
    assert IterationFeedback is not None
    
    print("✅ test_new_module_imports PASSED")


# ============================================================
# ServerMetaController Tests
# ============================================================

def test_server_meta_controller_init():
    """Test ServerMetaController initialization."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    controller = ServerMetaController()
    
    assert controller.model_name is not None
    assert controller.device is not None
    assert controller._initialized == False
    
    print("✅ test_server_meta_controller_init PASSED")


def test_server_meta_controller_get_param_space():
    """Test ServerMetaController get_param_space method."""
    from server_param_optimizer.server_meta_controller import ServerMetaController, PARAM_SPACE
    
    controller = ServerMetaController()
    
    param_space = controller.get_param_space()
    
    assert 'max_num_seqs' in param_space
    assert 'max_num_batched_tokens' in param_space
    assert param_space == PARAM_SPACE
    
    print("✅ test_server_meta_controller_get_param_space PASSED")


def test_server_meta_controller_validate_config():
    """Test ServerMetaController config validation."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    controller = ServerMetaController()
    
    # Valid config
    valid_config = {
        'max_num_seqs': 64,
        'max_num_batched_tokens': 16384
    }
    assert controller._validate_config(valid_config) == True
    
    # Invalid config - violates constraint (tokens < seqs * 128)
    invalid_config = {
        'max_num_seqs': 256,
        'max_num_batched_tokens': 2048  # 2048 < 256 * 128 = 32768
    }
    assert controller._validate_config(invalid_config) == False
    
    # Missing required fields
    missing_config = {'max_num_seqs': 64}
    assert controller._validate_config(missing_config) == False
    
    # Not a dict
    assert controller._validate_config("not a dict") == False
    
    print("✅ test_server_meta_controller_validate_config PASSED")


def test_server_meta_controller_generate_default_configs():
    """Test ServerMetaController default config generation."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    controller = ServerMetaController()
    
    configs = controller._generate_default_configs()
    
    assert isinstance(configs, list)
    assert len(configs) >= 2  # At least aggressive and conservative
    
    # Check all configs have required fields
    for config in configs:
        assert 'max_num_seqs' in config
        assert 'max_num_batched_tokens' in config
        assert 'name' in config
        # Check values are valid
        assert controller._validate_config(config)
    
    print("✅ test_server_meta_controller_generate_default_configs PASSED")


def test_server_meta_controller_parse_configs():
    """Test ServerMetaController JSON parsing."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    controller = ServerMetaController()
    
    # Valid JSON with <param> tags
    valid_output = '''
<param>
{
  "configs": [
    {
      "name": "test_config",
      "max_num_seqs": 128,
      "max_num_batched_tokens": 32768,
      "rationale": "Test config"
    }
  ]
}
</param>
'''
    configs = controller._parse_configs(valid_output)
    assert len(configs) == 1
    assert configs[0]['name'] == 'test_config'
    assert configs[0]['max_num_seqs'] == 128
    
    # JSON without tags (fallback)
    json_only = '{"configs": [{"name": "test", "max_num_seqs": 64, "max_num_batched_tokens": 8192}]}'
    configs = controller._parse_configs(json_only)
    assert len(configs) == 1
    
    # Invalid output
    invalid_output = "no json here"
    configs = controller._parse_configs(invalid_output)
    assert configs == []
    
    print("✅ test_server_meta_controller_parse_configs PASSED")


def test_server_meta_controller_build_prompt():
    """Test ServerMetaController prompt building."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    controller = ServerMetaController()
    
    feedback_str = "Previous iteration: seqs=64, tokens=8192 -> 1500 tokens/sec"
    prompt = controller._build_prompt(feedback_str)
    
    assert "meta-llama/Llama-3.1-8B-Instruct" in prompt
    assert "NVIDIA H100 80GB" in prompt
    assert "max-num-seqs" in prompt
    assert "max-num-batched-tokens" in prompt
    assert feedback_str in prompt
    assert "<param>" in prompt
    
    print("✅ test_server_meta_controller_build_prompt PASSED")


# ============================================================
# ServerParameterOptimizer Tests
# ============================================================

def test_server_optimizer_init():
    """Test ServerParameterOptimizer initialization."""
    from server_param_optimizer.server_optimizer import ServerParameterOptimizer
    
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        assert optimizer.model_name is not None
        assert optimizer.gpu_type is not None
        assert optimizer.benchmark_duration_minutes == 1
        assert optimizer.num_iterations == 1
        assert optimizer.meta_controller is not None
        assert optimizer.worker is not None
        assert optimizer.config_exporter is not None
        assert optimizer.feedback_collector is not None
        assert optimizer.visualizer is not None
        
        print("✅ test_server_optimizer_init PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_optimizer_module_import():
    """Test that ServerParameterOptimizer is exported correctly."""
    from server_param_optimizer import (
        ServerParameterOptimizer,
        ServerMetaController,
        PARAM_SPACE
    )
    
    assert ServerParameterOptimizer is not None
    assert ServerMetaController is not None
    assert PARAM_SPACE is not None
    assert 'max_num_seqs' in PARAM_SPACE
    assert 'max_num_batched_tokens' in PARAM_SPACE
    
    print("✅ test_server_optimizer_module_import PASSED")


# ============================================================
# Multi-GPU Support Tests
# ============================================================

def test_gpu_id_defaults():
    """Test default GPU ID constants."""
    from server_param_optimizer import LLM_GPU_ID, BENCHMARK_GPU_ID
    
    assert LLM_GPU_ID == 0
    assert BENCHMARK_GPU_ID == 1
    
    print("✅ test_gpu_id_defaults PASSED")


def test_get_available_gpus():
    """Test get_available_gpus function."""
    from server_param_optimizer import get_available_gpus
    
    gpus = get_available_gpus()
    
    assert isinstance(gpus, list)
    assert len(gpus) >= 1
    assert all(isinstance(g, int) for g in gpus)
    # GPU 0 should always be in the list (at minimum)
    assert 0 in gpus
    
    print("✅ test_get_available_gpus PASSED")


def test_validate_gpu_assignment():
    """Test validate_gpu_assignment warning function."""
    from server_param_optimizer import validate_gpu_assignment
    import io
    import sys
    
    # Test same GPU warning
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        validate_gpu_assignment(0, 0)
    finally:
        sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "SAME GPU" in output
    assert "VRAM conflicts" in output
    
    print("✅ test_validate_gpu_assignment PASSED")


def test_server_meta_controller_gpu_id():
    """Test ServerMetaController with gpu_id parameter."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    
    # Test with explicit gpu_id
    controller = ServerMetaController(gpu_id=2)
    
    assert controller.gpu_id == 2
    # Device should include the GPU ID
    assert "2" in controller.device or controller.device == "cpu"
    
    print("✅ test_server_meta_controller_gpu_id PASSED")


def test_server_optimizer_gpu_separation():
    """Test ServerParameterOptimizer with separate GPU IDs."""
    from server_param_optimizer.server_optimizer import ServerParameterOptimizer
    
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1,
            llm_gpu_id=0,
            benchmark_gpu_id=2
        )
        
        assert optimizer.llm_gpu_id == 0
        assert optimizer.benchmark_gpu_id == 2
        assert optimizer.meta_controller.gpu_id == 0
        assert optimizer.worker.gpu_id == 2
        
        print("✅ test_server_optimizer_gpu_separation PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_optimizer_same_gpu_warning():
    """Test ServerParameterOptimizer warns when using same GPU."""
    from server_param_optimizer.server_optimizer import ServerParameterOptimizer
    import io
    import sys
    
    temp_dir = tempfile.mkdtemp()
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1,
            llm_gpu_id=0,
            benchmark_gpu_id=0  # Same GPU
        )
        
        output = captured_output.getvalue()
        
        # Should warn about same GPU usage
        assert "SAME GPU" in output
    finally:
        sys.stdout = sys.__stdout__
        shutil.rmtree(temp_dir)
    
    print("✅ test_server_optimizer_same_gpu_warning PASSED")


def test_module_exports_gpu_utilities():
    """Test that GPU utility functions are exported from the module."""
    from server_param_optimizer import (
        LLM_GPU_ID,
        BENCHMARK_GPU_ID,
        get_available_gpus,
        validate_gpu_assignment
    )
    
    assert LLM_GPU_ID is not None
    assert BENCHMARK_GPU_ID is not None
    assert callable(get_available_gpus)
    assert callable(validate_gpu_assignment)
    
    print("✅ test_module_exports_gpu_utilities PASSED")


# ============================================================
# Enhanced Logging Tests
# ============================================================

def test_server_meta_controller_generate_configs_logging():
    """Test that generate_configs prints feedback and validated configs."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
    import io
    import sys
    
    controller = ServerMetaController()
    
    # Create a mock feedback collector with some data
    temp_file = os.path.join(tempfile.mkdtemp(), "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Add some test data
        configs = [{'max_num_seqs': 64, 'max_num_batched_tokens': 8192}]
        results = [{'throughput': 1200.0, 'is_thermally_safe': True}]
        collector.add_iteration(configs, results)
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            generated = controller.generate_configs(collector)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Verify feedback is printed
        assert "FEEDBACK FOR LLM PROMPT" in output
        assert "SERVER PARAMETER OPTIMIZATION FEEDBACK" in output
        
        # Verify validated configs are printed
        assert "VALIDATED CONFIGURATIONS TO TEST" in output
        assert "max_num_seqs" in output
        assert "max_num_batched_tokens" in output
        
        # Verify default configs message is printed (since LLM is not available)
        assert "Using DEFAULT configurations" in output or "LLM not available" in output
        
        print("✅ test_server_meta_controller_generate_configs_logging PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(os.path.dirname(temp_file))


def test_server_meta_controller_default_configs_logging():
    """Test that default configs generation prints config details."""
    from server_param_optimizer.server_meta_controller import ServerMetaController
    import io
    import sys
    
    controller = ServerMetaController()
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        configs = controller._generate_default_configs()
    finally:
        sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    # Verify default config logging
    assert "Using DEFAULT configurations" in output
    assert "aggressive_high" in output
    assert "balanced" in output
    assert "conservative" in output
    assert "low_concurrency" in output
    assert "seqs=128" in output  # aggressive config
    assert "seqs=64" in output   # balanced config
    
    print("✅ test_server_meta_controller_default_configs_logging PASSED")


def test_server_optimizer_iteration_feedback_logging():
    """Test that _print_iteration_feedback prints detailed benchmark results."""
    from server_param_optimizer.server_optimizer import ServerParameterOptimizer
    from server_param_optimizer.server_profiling_worker import BenchmarkResult
    from server_param_optimizer.thermal_monitor import ThermalSummary
    import io
    import sys
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Capture stdout during init
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        sys.stdout = sys.__stdout__
        
        # Create mock configs and results
        configs = [
            {'max_num_seqs': 64, 'max_num_batched_tokens': 8192, 'name': 'test_config'}
        ]
        
        thermal_summary = ThermalSummary(
            duration_seconds=60.0,
            sample_count=60,
            temp_min=60.0,
            temp_max=72.0,
            temp_avg=68.0,
            temp_final=70.0,
            power_min=250.0,
            power_max=350.0,
            power_avg=300.0,
            memory_max_used_mb=30000.0,
            memory_max_used_pct=75.0,
            gpu_util_avg=90.0,
            memory_util_avg=60.0,
            is_thermally_safe=True,
            max_temp_exceeded=False,
            throttling_detected=False,
            time_above_target=0.0
        )
        
        results = [
            BenchmarkResult(
                config=configs[0],
                throughput=1500.0,
                output_throughput=1200.0,
                latency=50.0,
                thermal_summary=thermal_summary,
                is_thermally_safe=True,
                nsys_metrics=None,
                duration=60.0
            )
        ]
        
        # First update best configs
        optimizer.config_exporter.update_best_configs(results[0])
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            optimizer._print_iteration_feedback(configs, results)
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        
        # Verify feedback contains expected information
        assert "Benchmark Results" in output
        assert "test_config" in output
        assert "max_num_seqs: 64" in output
        assert "max_num_batched_tokens: 8192" in output
        assert "Throughput: 1500.00" in output
        assert "Thermal Data" in output
        assert "Temperature:" in output
        assert "Current Best Configurations" in output
        assert "Best Aggressive" in output
        
        print("✅ test_server_optimizer_iteration_feedback_logging PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_server_optimizer_benchmark_config_logging():
    """Test that _benchmark_config prints detailed config info."""
    from server_param_optimizer.server_optimizer import ServerParameterOptimizer
    import io
    import sys
    
    # Check that the code has the expected logging by inspecting the source
    temp_dir = tempfile.mkdtemp()
    try:
        # Capture stdout during init
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        sys.stdout = sys.__stdout__
        
        # Verify the _benchmark_config method exists and has the logging code
        import inspect
        source = inspect.getsource(optimizer._benchmark_config)
        
        # Check that the source code contains the expected logging statements
        assert "TESTING CONFIGURATION" in source
        assert "Parameter Settings" in source
        assert "--max-num-seqs" in source
        assert "--max-num-batched-tokens" in source
        assert "Rationale:" in source
        assert "RESULTS:" in source
        assert "Thermal Summary:" in source
        
        print("✅ test_server_optimizer_benchmark_config_logging PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# Main Test Runner
# ============================================================

if __name__ == "__main__":
    print("--- SERVER PARAMETER OPTIMIZER TESTS ---\n")
    
    print("=== ThermalSample Tests ===")
    test_thermal_sample_creation()
    test_thermal_sample_to_dict()
    
    print("\n=== ThermalConfig Tests ===")
    test_thermal_config_defaults()
    test_thermal_config_custom()
    test_thermal_config_to_dict()
    
    print("\n=== ThermalMonitor Tests ===")
    test_thermal_monitor_init()
    test_thermal_monitor_with_custom_config()
    test_thermal_monitor_get_samples_empty()
    test_thermal_monitor_get_thermal_summary_empty()
    test_thermal_monitor_is_thermally_safe_empty()
    test_thermal_monitor_get_samples_as_dict()
    test_thermal_summary_calculation()
    test_thermal_summary_not_safe()
    test_thermal_summary_throttling_detected()
    test_thermal_summary_to_dict()
    
    print("\n=== PlotConfig Tests ===")
    test_plot_config_defaults()
    test_plot_config_custom()
    
    print("\n=== ThermalVisualizer Tests ===")
    test_thermal_visualizer_init()
    test_thermal_visualizer_is_available()
    test_thermal_visualizer_save_empty_samples()
    test_thermal_visualizer_save_plot()
    
    print("\n=== NsysMetrics Tests ===")
    test_nsys_metrics_defaults()
    test_nsys_metrics_to_dict()
    
    print("\n=== NsysProfile Tests ===")
    test_nsys_profile_creation()
    test_nsys_profile_failure()
    test_nsys_profile_to_dict()
    
    print("\n=== NsysMetricsExtractor Tests ===")
    test_nsys_extractor_init()
    test_check_nsys_available()
    test_nsys_extractor_profile_unavailable()
    test_nsys_extractor_parse_empty_stats()
    test_nsys_extractor_get_metrics_summary_failure()
    test_nsys_extractor_get_metrics_summary_success()
    
    print("\n=== Module Import Tests ===")
    test_module_imports()
    
    print("\n=== BenchmarkResult Tests ===")
    test_benchmark_result_creation()
    test_benchmark_result_to_dict()
    test_benchmark_result_with_thermal_summary()
    
    print("\n=== Error Classification Tests ===")
    test_benchmark_error_type_enum()
    test_classify_error_vram_oom()
    test_classify_error_engine_init()
    test_classify_error_cuda()
    test_classify_error_timeout()
    test_classify_error_invalid_config()
    test_classify_error_none()
    test_classify_error_unknown()
    test_get_error_penalty()
    test_benchmark_result_is_successful()
    test_benchmark_result_with_error_type()
    test_log_error_details()
    test_is_benchmark_successful()
    
    print("\n=== GPU ID Mapping Tests ===")
    test_get_physical_gpu_id_no_cuda_visible_devices()
    test_get_physical_gpu_id_with_cuda_visible_devices()
    test_get_physical_gpu_id_with_spaces()
    test_get_physical_gpu_id_out_of_range()
    test_get_physical_gpu_id_empty_cuda_visible_devices()
    test_get_physical_gpu_id_negative_index()
    
    print("\n=== ServerProfilingWorkerLocal Tests ===")
    test_server_profiling_worker_local_init()
    test_server_profiling_worker_local_parse_benchmark_output()
    test_server_profiling_worker_local_build_command()
    
    print("\n=== ServerConfig Tests ===")
    test_server_config_creation()
    test_server_config_to_dict()
    
    print("\n=== ServerConfigExporter Tests ===")
    test_server_config_exporter_init()
    test_server_config_exporter_update_best_configs()
    test_server_config_exporter_skips_failed_benchmarks()
    test_server_config_exporter_save_configs()
    test_server_config_exporter_generate_launch_scripts()
    
    print("\n=== IterationFeedback Tests ===")
    test_iteration_feedback_creation()
    test_iteration_feedback_to_dict()
    
    print("\n=== ServerFeedbackCollector Tests ===")
    test_server_feedback_collector_init()
    test_server_feedback_collector_add_iteration()
    test_server_feedback_collector_get_feedback_for_prompt()
    test_server_feedback_collector_get_untested_configs()
    test_server_feedback_collector_reset()
    test_server_feedback_collector_persistence()
    
    print("\n=== New Module Import Tests ===")
    test_new_module_imports()
    
    print("\n=== ServerMetaController Tests ===")
    test_server_meta_controller_init()
    test_server_meta_controller_get_param_space()
    test_server_meta_controller_validate_config()
    test_server_meta_controller_generate_default_configs()
    test_server_meta_controller_parse_configs()
    test_server_meta_controller_build_prompt()
    
    print("\n=== ServerParameterOptimizer Tests ===")
    test_server_optimizer_init()
    test_server_optimizer_module_import()
    
    print("\n=== Multi-GPU Support Tests ===")
    test_gpu_id_defaults()
    test_get_available_gpus()
    test_validate_gpu_assignment()
    test_server_meta_controller_gpu_id()
    test_server_optimizer_gpu_separation()
    test_server_optimizer_same_gpu_warning()
    test_module_exports_gpu_utilities()
    
    print("\n=== Enhanced Logging Tests ===")
    test_server_meta_controller_generate_configs_logging()
    test_server_meta_controller_default_configs_logging()
    test_server_optimizer_iteration_feedback_logging()
    test_server_optimizer_benchmark_config_logging()
    
    print("\n" + "="*60)
    print("✅ ALL SERVER PARAMETER OPTIMIZER TESTS PASSED")
    print("="*60)
