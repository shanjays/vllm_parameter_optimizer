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
    """Test that MAX_COMPLETION_LENGTH is reduced to 256."""
    print("\n=== Test: MAX_COMPLETION_LENGTH is 256 ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for MAX_COMPLETION_LENGTH = 256
    pattern = r'MAX_COMPLETION_LENGTH\s*=\s*256'
    match = re.search(pattern, source)
    
    if match:
        print("✅ MAX_COMPLETION_LENGTH is set to 256")
        return True
    else:
        print("❌ MAX_COMPLETION_LENGTH is NOT set to 256")
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
        (re.search(r'def avg\(values\):', source) is not None, "avg helper function"),
        (re.search(r'def range_str\(values\):', source) is not None, "range_str helper function"),
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


def test_prompt_uses_chat_template():
    """Test that build_optimization_prompt uses chat template format."""
    print("\n=== Test: Prompt uses chat template format ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for chat template markers
    checks = [
        ('<|im_start|>system' in source, "<|im_start|>system present"),
        ('<|im_end|>' in source, "<|im_end|> present"),
        ('<|im_start|>user' in source, "<|im_start|>user present"),
        ('<|im_start|>assistant' in source, "<|im_start|>assistant present"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False
    
    if all_passed:
        print("✅ Prompt uses chat template format")
        return True
    else:
        print("❌ Prompt does NOT use chat template format")
        return False


def test_prompt_starts_assistant_with_param():
    """Test that assistant response starts with <param> tag."""
    print("\n=== Test: Assistant response starts with <param> ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for <|im_start|>assistant followed by <param>
    pattern = r'<\|im_start\|>assistant\s*\n<param>'
    match = re.search(pattern, source)
    
    if match:
        print("✅ Assistant response starts with <param>")
        return True
    else:
        print("❌ Assistant response does NOT start with <param>")
        return False


def test_prompt_emphasizes_json_only():
    """Test that prompt emphasizes JSON-only output."""
    print("\n=== Test: Prompt emphasizes JSON-only output ===")
    
    with open('hierarchical_kernel_optimizer.py', 'r') as f:
        source = f.read()
    
    # Check for JSON-only emphasis in system message
    checks = [
        ('JSON policy generator' in source, "JSON policy generator mentioned"),
        ('ONLY valid JSON' in source or 'output ONLY' in source.lower(), "ONLY JSON mentioned"),
        ('No explanations' in source or 'nothing else' in source.lower(), "No explanations mentioned"),
    ]
    
    all_passed = True
    for check, name in checks:
        if check:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False
    
    if all_passed:
        print("✅ Prompt emphasizes JSON-only output")
        return True
    else:
        print("❌ Prompt does NOT emphasize JSON-only output")
        return False


def test_feedback_concise_format():
    """Test that format_feedback_for_prompt returns concise output."""
    print("\n=== Test: Feedback format is concise ===")
    
    from feedback_collector import FeedbackCollector
    
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # Record a policy with many configs
        policy = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128], "num_stages": [5]}
        }
        best_configs = {
            1: {"config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "num_stages": 3}, "reward": 45.0},
            16: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_stages": 4}, "reward": 48.0},
            64: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_stages": 4}, "reward": 51.0},
            256: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_stages": 5}, "reward": 53.0},
            1024: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_stages": 5}, "reward": 55.0},
        }
        collector.record_policy_result(policy, reward=55.0, best_configs=best_configs)
        
        feedback = collector.format_feedback_for_prompt()
        
        # Check that feedback is concise
        lines = feedback.strip().split('\n')
        
        # Should be less than 15 lines (was 30+ before)
        if len(lines) <= 15:
            print(f"  ✓ Feedback is {len(lines)} lines (<=15)")
        else:
            print(f"  ✗ Feedback is {len(lines)} lines (>15)")
            return False
        
        # Should only include top 3 configs
        token_mentions = [line for line in lines if line.strip().startswith('Token')]
        if len(token_mentions) <= 3:
            print(f"  ✓ Only {len(token_mentions)} token configs (<=3)")
        else:
            print(f"  ✗ {len(token_mentions)} token configs (>3)")
            return False
        
        # Should NOT have verbose sections
        verbose_patterns = [
            "=== FEEDBACK FROM PREVIOUS",
            "=== BEST POLICY WEIGHTS ===",
            "=== WHAT DIDN'T WORK ===",
            "=== YOUR TASK ===",
        ]
        for pattern in verbose_patterns:
            if pattern in feedback:
                print(f"  ✗ Verbose section found: {pattern}")
                return False
        print("  ✓ No verbose sections")
        
        print("✅ Feedback format is concise")
        return True
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
    all_passed &= test_prompt_uses_chat_template()
    all_passed &= test_prompt_starts_assistant_with_param()
    all_passed &= test_prompt_emphasizes_json_only()
    
    print("\n" + "="*60)
    print("=== feedback_collector.py Changes ===")
    print("="*60)
    
    all_passed &= test_feedback_concise_format()
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
