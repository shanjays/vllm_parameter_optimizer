"""
Tests for Server Parameter Optimizer Core Infrastructure

These tests verify:
1. ThermalSample and ThermalConfig dataclasses
2. ThermalMonitor methods (without actual nvidia-smi)
3. ThermalVisualizer availability checking
4. NsysMetricsExtractor availability checking and stats parsing
5. Module imports and exports
"""

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
    """Test ThermalConfig has correct A100 40GB defaults."""
    config = ThermalConfig()
    
    assert config.max_safe_temp == 83.0
    assert config.target_sustained_temp == 75.0
    assert config.warning_temp == 80.0
    assert config.max_power == 400.0
    assert config.total_memory_gb == 40.0
    assert "A100" in config.gpu_name
    
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
    assert monitor.config.max_safe_temp == 83.0
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
    
    print("\n" + "="*60)
    print("✅ ALL SERVER PARAMETER OPTIMIZER TESTS PASSED")
    print("="*60)
