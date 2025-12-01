"""
Tests for MetaController PPO model persistence functionality.

Tests the following features:
1. training_logger parameter in __init__
2. _run_exploration_phase passes load_existing=True to ExplorationAgent
3. _run_exploration_phase tracks best configs from environment
4. _run_exploration_phase saves best model when reward improves

Note: These tests verify the changes by parsing the meta_controller.py source code
directly, following the pattern used in test_training_fixes.py to avoid heavy 
dependencies like ray, vllm, etc.
"""

import re
import ast
import inspect


def get_meta_controller_source():
    """Read the meta_controller.py source file."""
    with open('meta_controller.py', 'r') as f:
        return f.read()


def test_training_logger_parameter_in_init():
    """Test that __init__ signature includes training_logger parameter."""
    print("\n=== Test: training_logger parameter in __init__ ===")
    
    source = get_meta_controller_source()
    
    # Find the __init__ method signature
    # Looking for: def __init__(self, ..., training_logger=None):
    init_pattern = r'def __init__\s*\([^)]*training_logger[^)]*\)'
    match = re.search(init_pattern, source)
    
    if match:
        print(f"✅ Found training_logger parameter in __init__ signature")
        return True
    else:
        print(f"❌ training_logger parameter NOT found in __init__ signature")
        return False


def test_training_logger_default_none():
    """Test that training_logger defaults to None."""
    print("\n=== Test: training_logger defaults to None ===")
    
    source = get_meta_controller_source()
    
    # Check for training_logger=None in __init__ signature
    pattern = r'training_logger\s*=\s*None'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ training_logger defaults to None")
        return True
    else:
        print(f"❌ training_logger does NOT default to None")
        return False


def test_training_logger_stored_in_init():
    """Test that training_logger is stored as self.training_logger in __init__."""
    print("\n=== Test: training_logger stored as instance attribute ===")
    
    source = get_meta_controller_source()
    
    # Check for self.training_logger = training_logger in __init__
    pattern = r'self\.training_logger\s*=\s*training_logger'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ training_logger is stored as self.training_logger")
        return True
    else:
        print(f"❌ training_logger is NOT stored as self.training_logger")
        return False


def test_exploration_agent_load_existing_true():
    """Test that ExplorationAgent is created with load_existing=True."""
    print("\n=== Test: ExplorationAgent created with load_existing=True ===")
    
    source = get_meta_controller_source()
    
    # Find the _run_exploration_phase method and check for load_existing=True
    # Looking for: ExplorationAgent(..., load_existing=True, ...)
    pattern = r'ExplorationAgent\s*\([^)]*load_existing\s*=\s*True[^)]*\)'
    match = re.search(pattern, source, re.DOTALL)
    
    if match:
        print(f"✅ ExplorationAgent is created with load_existing=True")
        return True
    else:
        print(f"❌ ExplorationAgent is NOT created with load_existing=True")
        return False


def test_exploration_agent_receives_training_logger():
    """Test that ExplorationAgent receives training_logger from controller."""
    print("\n=== Test: ExplorationAgent receives training_logger ===")
    
    source = get_meta_controller_source()
    
    # Check for training_logger parameter being passed to ExplorationAgent
    # Looking for: ExplorationAgent(..., training_logger=..., ...)
    pattern = r'ExplorationAgent\s*\([^)]*training_logger\s*=[^)]*\)'
    match = re.search(pattern, source, re.DOTALL)
    
    if match:
        print(f"✅ ExplorationAgent receives training_logger parameter")
        return True
    else:
        print(f"❌ ExplorationAgent does NOT receive training_logger parameter")
        return False


def test_persistent_log_dir_format():
    """Test that log_dir uses persistent format without timestamp."""
    print("\n=== Test: log_dir uses persistent format ===")
    
    source = get_meta_controller_source()
    
    # Check that the new log_dir format is used: ./logs/exploration_agent/tokens_{token_count}/
    # This should NOT have a timestamp pattern like: exploration_{timestamp}_tokens
    persistent_pattern = r'log_dir\s*=\s*f["\']\.?/?logs/exploration_agent/tokens_'
    match = re.search(persistent_pattern, source)
    
    if match:
        print(f"✅ log_dir uses persistent format without timestamp")
        return True
    else:
        print(f"❌ log_dir does NOT use persistent format")
        return False


def test_save_best_if_improved_called():
    """Test that save_best_if_improved is called in _run_exploration_phase."""
    print("\n=== Test: save_best_if_improved is called ===")
    
    source = get_meta_controller_source()
    
    # Check for agent.save_best_if_improved call in _run_exploration_phase
    pattern = r'agent\.save_best_if_improved\s*\('
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ save_best_if_improved is called")
        return True
    else:
        print(f"❌ save_best_if_improved is NOT called")
        return False


def test_best_configs_tracked():
    """Test that best_configs dict is created to track configs."""
    print("\n=== Test: best_configs dict is tracked ===")
    
    source = get_meta_controller_source()
    
    # Check for best_configs = {} initialization in _run_exploration_phase
    pattern = r'best_configs\s*=\s*\{\}'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ best_configs dict is initialized")
        return True
    else:
        print(f"❌ best_configs dict is NOT initialized")
        return False


def test_config_exporter_updated():
    """Test that config_exporter.update_best_config is called with correct params."""
    print("\n=== Test: config_exporter.update_best_config called correctly ===")
    
    source = get_meta_controller_source()
    
    # Check for config_exporter.update_best_config call with token_count, config, reward
    # The call should be in the _run_exploration_phase method with best config from results
    pattern = r'self\.config_exporter\.update_best_config\s*\(\s*token_count\s*=\s*token_count'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ config_exporter.update_best_config called with correct parameters")
        return True
    else:
        print(f"❌ config_exporter.update_best_config NOT called with correct parameters")
        return False


def test_backward_compatibility_init_signature():
    """Test that __init__ maintains backward compatibility."""
    print("\n=== Test: __init__ backward compatibility ===")
    
    source = get_meta_controller_source()
    
    # Original required parameters: user_goal, model_name, exploration_steps, profiling_gpu_id, static_args
    required_params = ['user_goal', 'model_name', 'exploration_steps', 'profiling_gpu_id', 'static_args']
    optional_params = ['config_exporter', 'token_counts', 'training_logger', 'feedback_collector']
    
    all_found = True
    
    for param in required_params:
        if param not in source:
            print(f"❌ Required parameter '{param}' not found")
            all_found = False
    
    # Check optional params have defaults
    for param in optional_params:
        pattern = rf'{param}\s*=\s*None'
        if not re.search(pattern, source):
            print(f"❌ Optional parameter '{param}' should default to None")
            all_found = False
    
    if all_found:
        print(f"✅ All parameters present with correct defaults")
        return True
    return False


def test_feedback_collector_parameter_in_init():
    """Test that __init__ signature includes feedback_collector parameter."""
    print("\n=== Test: feedback_collector parameter in __init__ ===")
    
    source = get_meta_controller_source()
    
    # Find the __init__ method signature
    # Looking for: def __init__(self, ..., feedback_collector=None):
    init_pattern = r'def __init__\s*\([^)]*feedback_collector[^)]*\)'
    match = re.search(init_pattern, source)
    
    if match:
        print(f"✅ Found feedback_collector parameter in __init__ signature")
        return True
    else:
        print(f"❌ feedback_collector parameter NOT found in __init__ signature")
        return False


def test_feedback_collector_defaults_none():
    """Test that feedback_collector defaults to None."""
    print("\n=== Test: feedback_collector defaults to None ===")
    
    source = get_meta_controller_source()
    
    # Check for feedback_collector=None in __init__ signature
    pattern = r'feedback_collector\s*=\s*None'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ feedback_collector defaults to None")
        return True
    else:
        print(f"❌ feedback_collector does NOT default to None")
        return False


def test_feedback_collector_stored_in_init():
    """Test that feedback_collector is stored as self.feedback_collector in __init__."""
    print("\n=== Test: feedback_collector stored as instance attribute ===")
    
    source = get_meta_controller_source()
    
    # Check for self.feedback_collector = feedback_collector in __init__
    pattern = r'self\.feedback_collector\s*=\s*feedback_collector'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ feedback_collector is stored as self.feedback_collector")
        return True
    else:
        print(f"❌ feedback_collector is NOT stored as self.feedback_collector")
        return False


def test_feedback_collector_record_policy_result():
    """Test that feedback_collector.record_policy_result is called."""
    print("\n=== Test: feedback_collector.record_policy_result is called ===")
    
    source = get_meta_controller_source()
    
    # Check for self.feedback_collector.record_policy_result call
    pattern = r'self\.feedback_collector\.record_policy_result\s*\('
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ feedback_collector.record_policy_result is called")
        return True
    else:
        print(f"❌ feedback_collector.record_policy_result is NOT called")
        return False


def test_exploration_phase_returns_best_configs():
    """Test that _run_exploration_phase returns best_configs."""
    print("\n=== Test: _run_exploration_phase returns best_configs ===")
    
    source = get_meta_controller_source()
    
    # Check for return statement with best_configs (flexible pattern for any slice)
    pattern = r'return\s+sorted_results\[[^\]]+\]\s*,\s*best_configs'
    match = re.search(pattern, source)
    
    if match:
        print(f"✅ _run_exploration_phase returns best_configs")
        return True
    else:
        print(f"❌ _run_exploration_phase does NOT return best_configs")
        return False


if __name__ == "__main__":
    print("--- META CONTROLLER LOAD EXISTING TESTS ---")
    
    all_passed = True
    
    all_passed &= test_training_logger_parameter_in_init()
    all_passed &= test_training_logger_default_none()
    all_passed &= test_training_logger_stored_in_init()
    all_passed &= test_exploration_agent_load_existing_true()
    all_passed &= test_exploration_agent_receives_training_logger()
    all_passed &= test_persistent_log_dir_format()
    all_passed &= test_save_best_if_improved_called()
    all_passed &= test_best_configs_tracked()
    all_passed &= test_config_exporter_updated()
    all_passed &= test_backward_compatibility_init_signature()
    
    # New tests for feedback_collector
    all_passed &= test_feedback_collector_parameter_in_init()
    all_passed &= test_feedback_collector_defaults_none()
    all_passed &= test_feedback_collector_stored_in_init()
    all_passed &= test_feedback_collector_record_policy_result()
    all_passed &= test_exploration_phase_returns_best_configs()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL META CONTROLLER LOAD EXISTING TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("="*60)
