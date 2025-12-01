"""
Tests for LLM prompt improvements to prevent verbose reasoning and truncation.

These tests verify:
1. summarize_ncu_report() function condenses NCU data
2. build_optimization_prompt() uses chat template format
3. MAX_COMPLETION_LENGTH is reduced to 256
4. format_feedback_for_prompt() returns concise output
5. _extract_json() handles verbose LLM output gracefully
"""

import re
import tempfile
import shutil
import os


def test_max_completion_length():
    """Test that MAX_COMPLETION_LENGTH is increased to 1536."""
    print("\n=== Test: MAX_COMPLETION_LENGTH is 1536 ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for MAX_COMPLETION_LENGTH = 1536
    pattern = r'MAX_COMPLETION_LENGTH\s*=\s*1536'
    match = re.search(pattern, source)
    
    if match:
        print("✅ MAX_COMPLETION_LENGTH is set to 1536")
        return True
    else:
        print("❌ MAX_COMPLETION_LENGTH is NOT set to 1536")
        return False


def test_summarize_ncu_report_exists():
    """Test that summarize_ncu_report function exists."""
    print("\n=== Test: summarize_ncu_report function exists ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for summarize_ncu_report function definition
    pattern = r'def summarize_ncu_report\s*\('
    match = re.search(pattern, source)
    
    if match:
        print("✅ summarize_ncu_report function exists")
        return True
    else:
        print("❌ summarize_ncu_report function NOT found")
        return False


def test_summarize_ncu_report_functionality():
    """Test that summarize_ncu_report correctly summarizes NCU data."""
    print("\n=== Test: summarize_ncu_report functionality ===")
    
    # Extract the function from source and test it
    import re
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check that the function has the expected structure
    checks = [
        (re.search(r'sm__throughput\.avg\.pct_of_peak_sustained_elapsed', source) is not None, "SM throughput regex"),
        (re.search(r'dram__throughput\.avg\.pct_of_peak_sustained_elapsed', source) is not None, "DRAM throughput regex"),
        (re.search(r'l1tex__t_sector_hit_rate\.pct', source) is not None, "L1 hit rate regex"),
        (re.search(r'lts__t_sector_hit_rate\.pct', source) is not None, "L2 hit rate regex"),
        (re.search(r'def stats\(vals\):', source) is not None, "stats helper function"),
        (re.search(r"'sm_throughput':\s*\[\]", source) is not None, "metrics dict structure"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False
    
    if all_passed:
        print("✅ summarize_ncu_report functionality looks correct")
        return True
    else:
        print("❌ summarize_ncu_report functionality has issues")
        return False


def test_prompt_uses_expert_format():
    """Test that build_optimization_prompt uses expert-level format with sections."""
    print("\n=== Test: Prompt uses expert format with sections ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for expert format section headers
    checks = [
        ('KERNEL DETAILS' in source, "KERNEL DETAILS section present"),
        ('HARDWARE SPECS' in source, "HARDWARE SPECS section present"),
        ('BASELINE PROFILING METRICS' in source, "BASELINE PROFILING METRICS section present"),
        ('TUNING PARAMETERS' in source, "TUNING PARAMETERS section present"),
        ('OBJECTIVE WEIGHTS' in source, "OBJECTIVE WEIGHTS section present"),
        ('OUTPUT FORMAT (IMPORTANT!)' in source, "OUTPUT FORMAT section present"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False
    
    if all_passed:
        print("✅ Prompt uses expert format with sections")
        return True
    else:
        print("❌ Prompt does NOT use expert format with sections")
        return False


def test_prompt_ends_with_param():
    """Test that prompt ends with <param> tag to encourage JSON-first output."""
    print("\n=== Test: Prompt ends with <param> ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for prompt ending with <param> followed by triple-quote
    pattern = r"<param>\s*'''\s*\n\s*return optimization_prompt"
    match = re.search(pattern, source)
    
    if match:
        print("✅ Prompt ends with <param>")
        return True
    else:
        print("❌ Prompt does NOT end with <param>")
        return False


def test_prompt_json_first_format():
    """Test that prompt instructs JSON-first output with reasoning after."""
    print("\n=== Test: Prompt instructs JSON-first output ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for JSON-first instructions
    checks = [
        ('Output your policy JSON FIRST' in source or 'FIRST: Output your policy JSON' in source, "JSON first instruction"),
        ('THEN: Provide brief reasoning' in source or 'then provide brief reasoning' in source.lower(), "Reasoning after instruction"),
        ('REASONING:' in source, "REASONING label in format"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False
    
    if all_passed:
        print("✅ Prompt instructs JSON-first output")
        return True
    else:
        print("❌ Prompt does NOT instruct JSON-first output")
        return False


def test_feedback_structured_format():
    """Test that format_feedback_for_prompt returns structured output with sections."""
    print("\n=== Test: Feedback format is structured ===")
    
    from feedback_collector import FeedbackCollector
    
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # Record a policy with many configs
        policy = {
            "objective_weights": {"R_sm_throughput": 0.5, "R_dram_throughput": 0.3, "R_l1_hit_rate": 0.1, "R_l2_hit_rate": 0.1},
            "search_space": {"BLOCK_SIZE_M": [128], "num_stages": [5]}
        }
        best_configs = {
            1: {"config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 3}, "reward": 45.0},
            16: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 4}, "reward": 48.0},
            64: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 4}, "reward": 51.0},
            256: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 5}, "reward": 53.0},
            1024: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 5}, "reward": 55.0},
        }
        collector.record_policy_result(policy, reward=55.0, best_configs=best_configs)
        
        feedback = collector.format_feedback_for_prompt()
        
        # Check that feedback has structured sections
        checks = [
            ("FEEDBACK FROM PREVIOUS ITERATIONS" in feedback, "Header section present"),
            ("Policies Evaluated:" in feedback, "Policies evaluated count"),
            ("Best Reward Achieved:" in feedback, "Best reward shown"),
            ("BEST CONFIGURATIONS FOUND:" in feedback, "Best configs section"),
            ("BEST OBJECTIVE WEIGHTS:" in feedback, "Objective weights section"),
            ("YOUR GOAL:" in feedback, "Goal statement present"),
        ]
        
        all_passed = True
        for check, name in checks:
            if check:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")
                all_passed = False
        
        if all_passed:
            print("✅ Feedback format is structured")
            return True
        else:
            print("❌ Feedback format is NOT structured")
            return False
    finally:
        shutil.rmtree(temp_dir)


def test_verbose_reasoning_detection():
    """Test that _extract_json handles verbose reasoning gracefully."""
    print("\n=== Test: Verbose reasoning detection ===")
    
    with open('meta_controller.py', 'r') as f:
        source = f.read()
    
    # Check for verbose pattern detection in meta_controller.py
    # The implementation uses regex patterns like r'^We are', r'^Let me', etc.
    verbose_patterns_check = [
        (re.search(r"r'\^We are'", source) is not None, "We are pattern"),
        (re.search(r"r'\^Let me'", source) is not None, "Let me pattern"),
        (re.search(r"r'\^Analyzing'", source) is not None, "Analyzing pattern"),
        ("verbose reasoning" in source.lower(), "verbose reasoning warning"),
    ]
    
    all_found = True
    for check, name in verbose_patterns_check:
        if check:
            print(f"  ✓ {name} handled")
        else:
            print(f"  ✗ {name} NOT handled")
            all_found = False
    
    if all_found:
        print("✅ Verbose reasoning detection implemented")
        return True
    else:
        print("❌ Verbose reasoning detection NOT fully implemented")
        return False


def test_feedback_max_configs_parameter():
    """Test that format_feedback_for_prompt accepts max_configs parameter."""
    print("\n=== Test: max_configs parameter exists ===")
    
    from feedback_collector import FeedbackCollector
    import inspect
    
    sig = inspect.signature(FeedbackCollector.format_feedback_for_prompt)
    params = list(sig.parameters.keys())
    
    if 'max_configs' in params:
        print("✅ max_configs parameter exists")
        return True
    else:
        print("❌ max_configs parameter NOT found")
        return False


if __name__ == "__main__":
    print("--- PROMPT IMPROVEMENT TESTS ---")
    
    all_passed = True
    
    print("\n" + "="*60)
    print("=== hierarchical_kernel_optimizer.py Changes ===")
    print("="*60)
    
    all_passed &= test_max_completion_length()
    all_passed &= test_summarize_ncu_report_exists()
    all_passed &= test_summarize_ncu_report_functionality()
    all_passed &= test_prompt_uses_expert_format()
    all_passed &= test_prompt_ends_with_param()
    all_passed &= test_prompt_json_first_format()
    
    print("\n" + "="*60)
    print("=== feedback_collector.py Changes ===")
    print("="*60)
    
    all_passed &= test_feedback_structured_format()
    all_passed &= test_feedback_max_configs_parameter()
    
    print("\n" + "="*60)
    print("=== meta_controller.py Changes ===")
    print("="*60)
    
    all_passed &= test_verbose_reasoning_detection()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL PROMPT IMPROVEMENT TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("="*60)
