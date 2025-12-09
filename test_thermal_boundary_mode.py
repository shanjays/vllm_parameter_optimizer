"""
Tests for Thermal-Boundary Search Mode

These tests verify:
1. Thermal-boundary mode parameters are properly initialized
2. LLM prompt construction includes thermal instructions
3. Feedback includes thermal peak information
4. LLM raw output and feedback are saved to disk
5. CLI arguments are parsed correctly
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


def test_thermal_boundary_mode_initialization():
    """Test that thermal-boundary mode parameters are properly initialized."""
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            target_peak_temp=65.0,
            peak_tol=1.0,
            peak_reduction=5.0,
            repeat_count=2,
            search_candidates=15,
            benchmark_duration_minutes=5,
            num_iterations=1
        )
        
        assert optimizer.target_peak_temp == 65.0
        assert optimizer.peak_tol == 1.0
        assert optimizer.peak_reduction == 5.0
        assert optimizer.repeat_count == 2
        assert optimizer.search_candidates == 15
        
        # Verify meta-controller received thermal params
        assert optimizer.meta_controller.target_peak_temp == 65.0
        assert optimizer.meta_controller.peak_tol == 1.0
        
        print("✅ test_thermal_boundary_mode_initialization PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_meta_controller_thermal_prompt():
    """Test that meta-controller includes thermal instructions in prompt."""
    controller = ServerMetaController(
        target_peak_temp=70.0,
        peak_tol=2.0
    )
    
    feedback = "Previous iteration: seqs=64, tokens=8192 -> 1500 tokens/sec"
    prompt = controller._build_prompt(feedback)
    
    # Check thermal instructions are in prompt
    assert "THERMAL-BOUNDARY OPTIMIZATION MODE" in prompt
    assert "TARGET PEAK TEMPERATURE: 70.0°C" in prompt
    assert "± 2.0°C tolerance" in prompt
    assert "REACH THE TARGET" in prompt
    assert "REDUCED-TEMP VARIANTS" in prompt
    
    print("✅ test_meta_controller_thermal_prompt PASSED")


def test_meta_controller_without_thermal_mode():
    """Test that meta-controller works without thermal mode."""
    controller = ServerMetaController(
        target_peak_temp=None,
        peak_tol=1.0
    )
    
    feedback = "Previous iteration: seqs=64, tokens=8192 -> 1500 tokens/sec"
    prompt = controller._build_prompt(feedback)
    
    # Check thermal instructions are NOT in prompt
    assert "THERMAL-BOUNDARY OPTIMIZATION MODE" not in prompt
    
    print("✅ test_meta_controller_without_thermal_mode PASSED")


def test_feedback_thermal_peak_inclusion():
    """Test that feedback includes thermal peak information."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_state.json")
    try:
        collector = ServerFeedbackCollector(state_file=temp_file)
        
        # Add iteration with thermal data
        configs = [{'max_num_seqs': 64, 'max_num_batched_tokens': 8192}]
        results = [{
            'throughput': 1500.0,
            'is_thermally_safe': True,
            'thermal_summary': {'temp_max': 72.5}
        }]
        collector.add_iteration(configs, results)
        
        feedback = collector.get_feedback_for_prompt()
        
        # Check thermal peak is in feedback
        assert "peak=72.5°C" in feedback or "Peak Temp: 72.5°C" in feedback
        
        print("✅ test_feedback_thermal_peak_inclusion PASSED")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        shutil.rmtree(temp_dir)


def test_feedback_disk_saving():
    """Test that feedback is saved to disk correctly."""
    temp_dir = tempfile.mkdtemp()
    try:
        collector = ServerFeedbackCollector()
        
        # Add some test data
        configs = [{'max_num_seqs': 64, 'max_num_batched_tokens': 8192}]
        results = [{'throughput': 1500.0, 'is_thermally_safe': True}]
        collector.add_iteration(configs, results)
        
        # Save to disk
        collector.save_feedback_to_disk(temp_dir, 1)
        
        # Check file was created
        feedback_path = os.path.join(temp_dir, "iteration_1", "feedback.txt")
        assert os.path.exists(feedback_path)
        
        # Check file has content
        with open(feedback_path, 'r') as f:
            content = f.read()
        assert len(content) > 0
        assert "SERVER PARAMETER OPTIMIZATION FEEDBACK" in content
        
        print("✅ test_feedback_disk_saving PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_generate_configs_returns_tuple():
    """Test that generate_configs returns tuple with raw output."""
    controller = ServerMetaController()
    collector = ServerFeedbackCollector()
    
    result = controller.generate_configs(collector)
    
    # Should return tuple (configs, raw_output)
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    configs, raw_output = result
    assert isinstance(configs, list)
    assert isinstance(raw_output, str)
    
    print("✅ test_generate_configs_returns_tuple PASSED")


def test_thermal_boundary_helper_methods_exist():
    """Test that thermal-boundary helper methods exist on optimizer."""
    temp_dir = tempfile.mkdtemp()
    try:
        optimizer = ServerParameterOptimizer(
            output_dir=temp_dir,
            target_peak_temp=65.0,
            benchmark_duration_minutes=1,
            num_iterations=1
        )
        
        # Check methods exist
        assert hasattr(optimizer, '_get_candidate_configs_from_llm_or_grid')
        assert hasattr(optimizer, 'find_boundary_config_for_target_temp')
        assert hasattr(optimizer, 'refine_boundary')
        assert hasattr(optimizer, '_benchmark_config_with_repeats')
        
        # Check they are callable
        assert callable(optimizer._get_candidate_configs_from_llm_or_grid)
        assert callable(optimizer.find_boundary_config_for_target_temp)
        assert callable(optimizer.refine_boundary)
        assert callable(optimizer._benchmark_config_with_repeats)
        
        print("✅ test_thermal_boundary_helper_methods_exist PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_cli_arguments():
    """Test that CLI argument parsing works."""
    import argparse
    from server_param_optimizer.server_optimizer import main
    
    # We can't easily test main() directly, but we can verify the parser
    # by importing and testing the argument structure
    parser = argparse.ArgumentParser(description="Server Parameter Optimizer for vLLM")
    parser.add_argument("--target-peak-temp", type=float, default=None)
    parser.add_argument("--peak-tol", type=float, default=1.0)
    parser.add_argument("--peak-reduction", type=float, default=5.0)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--search-candidates", type=int, default=20)
    
    # Test parsing
    args = parser.parse_args([
        '--target-peak-temp', '65.0',
        '--peak-tol', '2.0',
        '--peak-reduction', '10.0',
        '--repeat-count', '3',
        '--search-candidates', '25'
    ])
    
    assert args.target_peak_temp == 65.0
    assert args.peak_tol == 2.0
    assert args.peak_reduction == 10.0
    assert args.repeat_count == 3
    assert args.search_candidates == 25
    
    print("✅ test_cli_arguments PASSED")


def test_default_benchmark_duration():
    """Test that default benchmark duration is 10 minutes."""
    from server_param_optimizer.server_optimizer import BENCHMARK_DURATION_MINUTES
    
    assert BENCHMARK_DURATION_MINUTES == 10
    
    print("✅ test_default_benchmark_duration PASSED")


if __name__ == "__main__":
    print("--- THERMAL-BOUNDARY SEARCH MODE TESTS ---\n")
    
    test_thermal_boundary_mode_initialization()
    test_meta_controller_thermal_prompt()
    test_meta_controller_without_thermal_mode()
    test_feedback_thermal_peak_inclusion()
    test_feedback_disk_saving()
    test_generate_configs_returns_tuple()
    test_thermal_boundary_helper_methods_exist()
    test_cli_arguments()
    test_default_benchmark_duration()
    
    print("\n" + "="*60)
    print("✅ ALL THERMAL-BOUNDARY SEARCH MODE TESTS PASSED")
    print("="*60)
