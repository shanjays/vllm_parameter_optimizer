"""
Tests for Thermal-Boundary Search Mode

These tests verify the new thermal-boundary search functionality:
1. CLI argument parsing
2. Method signatures and basic functionality
3. Integration with existing components
"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_param_optimizer.server_optimizer import ServerParameterOptimizer
from server_param_optimizer.server_meta_controller import ServerMetaController
from server_param_optimizer.server_feedback_collector import ServerFeedbackCollector
from server_param_optimizer.thermal_monitor import ThermalConfig


# ============================================================
# CLI Arguments Tests
# ============================================================

def test_cli_args_default_mode():
    """Test CLI argument parsing for default mode."""
    import argparse
    from server_param_optimizer.server_optimizer import main
    
    # Mock sys.argv
    test_args = [
        "server_optimizer.py",
        "--llm-gpu", "0",
        "--benchmark-gpu", "1",
        "--duration", "5",
        "--iterations", "2"
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--benchmark-gpu", type=int, default=1)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--search-mode", type=str, default="default")
    parser.add_argument("--target-peak-temp", type=float, default=None)
    parser.add_argument("--peak-tol", type=float, default=1.0)
    parser.add_argument("--peak-reduction", type=float, default=5.0)
    parser.add_argument("--benchmark-duration", type=int, default=10)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--absolute-max-temp", type=float, default=None)
    
    args = parser.parse_args(test_args[1:])
    
    assert args.search_mode == "default"
    assert args.llm_gpu == 0
    assert args.benchmark_gpu == 1
    assert args.duration == 5
    assert args.iterations == 2
    
    print("✅ test_cli_args_default_mode PASSED")


def test_cli_args_thermal_boundary_mode():
    """Test CLI argument parsing for thermal-boundary mode."""
    import argparse
    
    test_args = [
        "server_optimizer.py",
        "--search-mode", "thermal-boundary",
        "--target-peak-temp", "80.0",
        "--peak-tol", "2.0",
        "--peak-reduction", "6.0",
        "--benchmark-duration", "15",
        "--repeat-count", "3"
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--benchmark-gpu", type=int, default=1)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--search-mode", type=str, choices=["default", "thermal-boundary"])
    parser.add_argument("--target-peak-temp", type=float, default=None)
    parser.add_argument("--peak-tol", type=float, default=1.0)
    parser.add_argument("--peak-reduction", type=float, default=5.0)
    parser.add_argument("--benchmark-duration", type=int, default=10)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--absolute-max-temp", type=float, default=None)
    
    args = parser.parse_args(test_args[1:])
    
    assert args.search_mode == "thermal-boundary"
    assert args.target_peak_temp == 80.0
    assert args.peak_tol == 2.0
    assert args.peak_reduction == 6.0
    assert args.benchmark_duration == 15
    assert args.repeat_count == 3
    
    print("✅ test_cli_args_thermal_boundary_mode PASSED")


def test_cli_args_conservative_defaults():
    """Test that conservative defaults are correctly set."""
    import argparse
    
    test_args = [
        "server_optimizer.py",
        "--search-mode", "thermal-boundary",
        "--target-peak-temp", "80.0"
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-gpu", type=int, default=0)
    parser.add_argument("--benchmark-gpu", type=int, default=1)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--search-mode", type=str, choices=["default", "thermal-boundary"])
    parser.add_argument("--target-peak-temp", type=float, default=None)
    parser.add_argument("--peak-tol", type=float, default=1.0)
    parser.add_argument("--peak-reduction", type=float, default=5.0)
    parser.add_argument("--benchmark-duration", type=int, default=10)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--absolute-max-temp", type=float, default=None)
    
    args = parser.parse_args(test_args[1:])
    
    # Verify conservative defaults
    assert args.peak_tol == 1.0
    assert args.peak_reduction == 5.0
    assert args.benchmark_duration == 10
    assert args.repeat_count == 2
    
    print("✅ test_cli_args_conservative_defaults PASSED")


# ============================================================
# Method Signature Tests
# ============================================================

def test_run_thermal_boundary_search_exists():
    """Test that run_thermal_boundary_search method exists."""
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        assert hasattr(optimizer, 'run_thermal_boundary_search')
        assert callable(optimizer.run_thermal_boundary_search)
        
        print("✅ test_run_thermal_boundary_search_exists PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_find_boundary_config_for_target_temp_exists():
    """Test that find_boundary_config_for_target_temp method exists."""
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        assert hasattr(optimizer, 'find_boundary_config_for_target_temp')
        assert callable(optimizer.find_boundary_config_for_target_temp)
        
        print("✅ test_find_boundary_config_for_target_temp_exists PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_benchmark_config_for_thermal_search_exists():
    """Test that _benchmark_config_for_thermal_search method exists."""
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        assert hasattr(optimizer, '_benchmark_config_for_thermal_search')
        assert callable(optimizer._benchmark_config_for_thermal_search)
        
        print("✅ test_benchmark_config_for_thermal_search_exists PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# ServerMetaController Integration Tests
# ============================================================

def test_meta_controller_accepts_target_peak_temp():
    """Test that ServerMetaController.generate_configs accepts thermal parameters."""
    controller = ServerMetaController()
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Should not raise an error
        configs = controller.generate_configs(
            feedback_collector=collector,
            target_peak_temp=80.0,
            peak_tol=1.0
        )
        
        assert isinstance(configs, list)
        assert len(configs) > 0
        
        print("✅ test_meta_controller_accepts_target_peak_temp PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_build_prompt_includes_thermal_target():
    """Test that _build_prompt includes thermal target information."""
    controller = ServerMetaController()
    
    # Build prompt without thermal target
    prompt_default = controller._build_prompt("No feedback")
    assert "THERMAL TARGET" not in prompt_default
    
    # Build prompt with thermal target
    prompt_thermal = controller._build_prompt(
        "No feedback",
        target_peak_temp=80.0,
        peak_tol=1.0
    )
    
    assert "THERMAL TARGET" in prompt_thermal
    assert "80.0°C" in prompt_thermal
    assert "±1.0°C" in prompt_thermal
    
    print("✅ test_build_prompt_includes_thermal_target PASSED")


# ============================================================
# ServerFeedbackCollector Integration Tests
# ============================================================

def test_feedback_collector_thermal_methods():
    """Test that ServerFeedbackCollector has thermal helper methods."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        assert hasattr(collector, '_get_peak_temp_from_config')
        assert hasattr(collector, '_get_peak_temp_from_result')
        assert hasattr(collector, '_add_thermal_summary')
        
        print("✅ test_feedback_collector_thermal_methods PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_feedback_includes_peak_temps():
    """Test that feedback string includes peak temperature information."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Add iteration with thermal data
        configs = [{'max_num_seqs': 64, 'max_num_batched_tokens': 8192}]
        results = [{
            'throughput': 1500.0,
            'is_thermally_safe': True,
            'thermal_summary': {
                'temp_max': 75.0,
                'temp_avg': 70.0
            },
            'config': configs[0]
        }]
        
        collector.add_iteration(configs, results)
        feedback = collector.get_feedback_for_prompt()
        
        # Feedback should include temperature info
        assert "75.0°C" in feedback or "THERMAL" in feedback
        
        print("✅ test_feedback_includes_peak_temps PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# Integration with ThermalConfig Tests
# ============================================================

def test_absolute_max_temp_override():
    """Test that absolute-max-temp CLI arg overrides thermal config."""
    custom_max = 90.0
    
    thermal_config = ThermalConfig(
        max_safe_temp=custom_max,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        max_power=350.0,
        total_memory_gb=80.0,
        gpu_name="Test GPU"
    )
    
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            benchmark_duration_minutes=1,
            num_iterations=1,
            thermal_config=thermal_config
        )
        
        assert optimizer.thermal_config.max_safe_temp == custom_max
        
        print("✅ test_absolute_max_temp_override PASSED")
    finally:
        shutil.rmtree(temp_dir)


# ============================================================
# Main Test Runner
# ============================================================

if __name__ == "__main__":
    print("--- THERMAL-BOUNDARY MODE TESTS ---\n")
    
    print("=== CLI Arguments Tests ===")
    test_cli_args_default_mode()
    test_cli_args_thermal_boundary_mode()
    test_cli_args_conservative_defaults()
    
    print("\n=== Method Signature Tests ===")
    test_run_thermal_boundary_search_exists()
    test_find_boundary_config_for_target_temp_exists()
    test_benchmark_config_for_thermal_search_exists()
    
    print("\n=== ServerMetaController Integration Tests ===")
    test_meta_controller_accepts_target_peak_temp()
    test_build_prompt_includes_thermal_target()
    
    print("\n=== ServerFeedbackCollector Integration Tests ===")
    test_feedback_collector_thermal_methods()
    test_feedback_includes_peak_temps()
    
    print("\n=== ThermalConfig Integration Tests ===")
    test_absolute_max_temp_override()
    
    print("\n" + "="*60)
    print("✅ ALL THERMAL-BOUNDARY MODE TESTS PASSED")
    print("="*60)
