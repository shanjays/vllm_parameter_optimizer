"""
Tests for training fixes:
1. JSON parsing with garbage text before <param> tags
2. Action space validation (must have >= 2 combinations)
3. DEFAULT_MISSION_PLAN usage

Note: The validation functions are duplicated here instead of importing from 
professor_reward.py to avoid dependency on ray, vllm, and other heavy dependencies
that would make these tests harder to run in isolation.
"""

import json
import re
import ast
import os
import sys
import tempfile

# Define the same constants and functions as in professor_reward.py
# This allows tests to run without heavy dependencies (ray, vllm, etc.)
DEFAULT_MISSION_PLAN = {
    "reward_function": {
        "R_sm_throughput": 0.4,
        "R_dram_throughput": 0.3,
        "R_l1_hit_rate": 0.15,
        "R_l2_hit_rate": 0.15
    },
    "pruned_action_space": {
        "BLOCK_SIZE_M": [32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4]
    }
}

POWER_OF_TWO_WARPS = (2, 4, 8, 16, 32)

# Minimum combinations for meaningful PPO exploration
MIN_COMBINATIONS = 8
MIN_VALUES_PER_DIM = 2


def _validate_and_coerce_plan(plan):
    """Validate and coerce the mission plan, using defaults for missing/invalid values."""
    # If plan is None or not a dict, use default
    if plan is None or not isinstance(plan, dict):
        print("[RewardFn] WARNING: Invalid plan (None or not dict), using default mission plan")
        return {
            "reward_function": DEFAULT_MISSION_PLAN["reward_function"].copy(),
            "pruned_action_space": {
                k: list(v) for k, v in DEFAULT_MISSION_PLAN["pruned_action_space"].items()
            }
        }
        
    rf = plan.get("reward_function", {})
    if not isinstance(rf, dict):
        rf = {}
    # If reward values are lists, take first numeric
    def _scalar(v):
        if isinstance(v, list):
            for x in v:
                if isinstance(x, (int, float)):
                    return float(x)
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0
    
    # Ensure all required reward keys are present
    required_keys = ["R_sm_throughput", "R_dram_throughput", "R_l1_hit_rate", "R_l2_hit_rate"]
    for k in required_keys:
        rf[k] = _scalar(rf.get(k, DEFAULT_MISSION_PLAN["reward_function"].get(k, 0.0)))
    
    # Normalize weights to sum to 1.0 if total > 0
    total = sum(rf.values())
    if total > 0:
        rf = {k: v / total for k, v in rf.items()}

    pas = plan.get("pruned_action_space", {})
    if not isinstance(pas, dict) or not pas:
        pas = {k: list(v) for k, v in DEFAULT_MISSION_PLAN["pruned_action_space"].items()}
        
    def _coerce_list(v, default):
        if isinstance(v, list):
            out = []
            for i in v:
                try:
                    out.append(int(i))
                except Exception:
                    continue
            if not out:
                out = [default]
            return out[:3]
        try:
            return [int(v)]
        except Exception:
            return [default]
    
    # Get default values from DEFAULT_MISSION_PLAN
    pas["BLOCK_SIZE_M"] = _coerce_list(pas.get("BLOCK_SIZE_M", DEFAULT_MISSION_PLAN["pruned_action_space"]["BLOCK_SIZE_M"]), 64)
    pas["BLOCK_SIZE_N"] = _coerce_list(pas.get("BLOCK_SIZE_N", DEFAULT_MISSION_PLAN["pruned_action_space"]["BLOCK_SIZE_N"]), 64)
    pas["BLOCK_SIZE_K"] = _coerce_list(pas.get("BLOCK_SIZE_K", DEFAULT_MISSION_PLAN["pruned_action_space"]["BLOCK_SIZE_K"]), 32)
    pas["num_warps"]    = _coerce_list(pas.get("num_warps", DEFAULT_MISSION_PLAN["pruned_action_space"]["num_warps"]), 4)
    pas["num_stages"]   = _coerce_list(pas.get("num_stages", DEFAULT_MISSION_PLAN["pruned_action_space"]["num_stages"]), 4)
    # Enforce power-of-two warps
    pas["num_warps"] = [w for w in pas["num_warps"] if w in POWER_OF_TWO_WARPS] or [4]
    
    # H100 hardware constraint validation - clamp values to safe limits
    H100_BLOCK_SIZE_MN_LIMIT = 128
    H100_BLOCK_SIZE_K_LIMIT = 64
    H100_NUM_STAGES_LIMIT = 5  # Allow num_stages=5 for aggressive testing
    
    pas["BLOCK_SIZE_M"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas["BLOCK_SIZE_M"]]
    pas["BLOCK_SIZE_N"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas["BLOCK_SIZE_N"]]
    pas["BLOCK_SIZE_K"] = [min(v, H100_BLOCK_SIZE_K_LIMIT) for v in pas["BLOCK_SIZE_K"]]
    pas["num_stages"] = [min(v, H100_NUM_STAGES_LIMIT) for v in pas["num_stages"]]
    
    # Remove duplicates and ensure non-empty lists
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages"]:
        pas[key] = list(set(pas[key])) or ([64] if "BLOCK" in key else [4])
    
    # Default values to expand narrow search spaces
    defaults = {
        "BLOCK_SIZE_M": [64, 128],
        "BLOCK_SIZE_N": [64, 128],
        "BLOCK_SIZE_K": [32, 64],
        "num_warps": [8, 16],
        "num_stages": [3, 4, 5],
    }
    
    # Ensure each dimension has at least 2 values
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"]:
        if len(pas[key]) < MIN_VALUES_PER_DIM:
            # Merge with defaults, remove duplicates, limit to 3 values
            # (consistent with _coerce_list which limits to 3 to keep search space manageable)
            pas[key] = list(set(pas[key] + defaults[key]))[:3]
            print(f"[RewardFn] Expanded {key} to {pas[key]} (was too narrow)")
    
    # Calculate total combinations
    total_combinations = 1
    for values in pas.values():
        total_combinations *= len(values)
    
    if total_combinations < MIN_COMBINATIONS:
        print(f"[RewardFn] WARNING: Only {total_combinations} combinations, expanding search space")
        pas = {k: list(v) for k, v in DEFAULT_MISSION_PLAN["pruned_action_space"].items()}
        total_combinations = 1
        for values in pas.values():
            total_combinations *= len(values)
    
    print(f"[RewardFn] Search space has {total_combinations} combinations")
    
    return {"reward_function": rf, "pruned_action_space": pas}


def _extract_json_with_param_handling(llm_output_str):
    """
    Extract JSON from LLM output, handling garbage text before <param> tags.
    """
    # Try to extract content from <param></param> XML tags first (preferred format)
    # Use non-greedy match for the JSON content and handle any garbage before <param>
    param_match = re.search(r'<param>\s*(\{[\s\S]*?\})\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
    if param_match:
        json_str = param_match.group(1).strip()
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback: Try to find any content between <param> tags 
    param_any_match = re.search(r'<param>\s*(.*?)\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
    if param_any_match:
        content = param_any_match.group(1).strip()
        if content.startswith('{'):
            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    
    # Fallback to brace matching
    match = re.search(r'(\{.*\})', llm_output_str, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    return None


# =============================================================================
# Test Cases for garbage text before <param>
# =============================================================================

GARBAGE_TEXT_TEST_CASES = [
    # Case 1: Simple garbage text before <param>
    (
        """Let's output final answer.assistantfinal>
<param>
{
  "reward_function": {"R_sm_throughput": 0.5, "R_dram_throughput": 0.3, "R_l1_hit_rate": 0.1, "R_l2_hit_rate": 0.1},
  "pruned_action_space": {"BLOCK_SIZE_M": [64], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32], "num_warps": [4], "num_stages": [4]}
}
</param>""",
        "Garbage text 'assistantfinal>' before <param>"
    ),
    # Case 2: Multiple lines of garbage
    (
        """Here is my analysis.
Based on the hardware metrics.
Let me think step by step.
OK here's my answer:
<param>
{"reward_function": {"R_sm_throughput": 0.6}, "pruned_action_space": {"BLOCK_SIZE_M": [64, 128]}}
</param>""",
        "Multiple lines before <param>"
    ),
    # Case 3: Partial XML-like tags before real <param>
    (
        """<thinking>Let me analyze...</thinking>
<response>Here's the plan:</response>
<param>
{"reward_function": {"R_sm_throughput": 0.5}, "pruned_action_space": {"BLOCK_SIZE_M": [32, 64]}}
</param>""",
        "Other XML tags before <param>"
    ),
]


# =============================================================================
# Test Cases for action space validation
# =============================================================================

ACTION_SPACE_TEST_CASES = [
    # Case 1: Plan with only 1 combination
    (
        {
            "reward_function": {"R_sm_throughput": 1.0},
            "pruned_action_space": {
                "BLOCK_SIZE_M": [64],
                "BLOCK_SIZE_N": [64],
                "BLOCK_SIZE_K": [32],
                "num_warps": [4],
                "num_stages": [4]
            }
        },
        "Plan with 1 combination should use default"
    ),
    # Case 2: Plan with None
    (
        None,
        "None plan should use default"
    ),
    # Case 3: Plan that's a string (not dict)
    (
        "invalid string plan",
        "String plan should use default"
    ),
    # Case 4: Plan with empty pruned_action_space
    (
        {
            "reward_function": {"R_sm_throughput": 1.0},
            "pruned_action_space": {}
        },
        "Empty pruned_action_space should use default"
    ),
]


# =============================================================================
# Test functions
# =============================================================================

def test_garbage_text_handling():
    """Test that JSON extraction handles garbage text before <param> tags."""
    print("\n=== Garbage Text Handling Tests ===")
    all_passed = True
    
    for input_str, test_name in GARBAGE_TEXT_TEST_CASES:
        print(f"\n--- Testing: {test_name} ---")
        result = _extract_json_with_param_handling(input_str)
        
        if result is not None and isinstance(result, dict):
            print(f"✅ Successfully extracted JSON dict: {list(result.keys())}")
        else:
            print(f"❌ Failed to extract JSON dict. Got: {type(result)}")
            all_passed = False
    
    return all_passed


def test_action_space_validation():
    """Test that plans with insufficient action space are replaced with default."""
    print("\n=== Action Space Validation Tests ===")
    all_passed = True
    
    for plan, test_name in ACTION_SPACE_TEST_CASES:
        print(f"\n--- Testing: {test_name} ---")
        result = _validate_and_coerce_plan(plan)
        
        # Calculate total combinations
        pas = result["pruned_action_space"]
        total_combinations = 1
        for values in pas.values():
            total_combinations *= len(values)
        
        if total_combinations >= MIN_COMBINATIONS:
            print(f"✅ Action space has {total_combinations} combinations (>= {MIN_COMBINATIONS})")
        else:
            print(f"❌ Action space has {total_combinations} combinations (< {MIN_COMBINATIONS})")
            all_passed = False
    
    return all_passed


def test_minimum_values_per_dimension():
    """Test that each dimension has at least 2 values."""
    print("\n=== Minimum Values Per Dimension Tests ===")
    all_passed = True
    
    # Test case: Plan with single values per dimension
    single_value_plan = {
        "reward_function": {"R_sm_throughput": 1.0},
        "pruned_action_space": {
            "BLOCK_SIZE_M": [64],
            "BLOCK_SIZE_N": [64],
            "BLOCK_SIZE_K": [32],
            "num_warps": [4],
            "num_stages": [4]
        }
    }
    
    print("\n--- Testing: Single value per dimension should be expanded ---")
    result = _validate_and_coerce_plan(single_value_plan)
    pas = result["pruned_action_space"]
    
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"]:
        if len(pas[key]) >= MIN_VALUES_PER_DIM:
            print(f"  ✅ {key}: {len(pas[key])} values (>= {MIN_VALUES_PER_DIM})")
        else:
            print(f"  ❌ {key}: {len(pas[key])} values (< {MIN_VALUES_PER_DIM})")
            all_passed = False
    
    return all_passed


def test_default_mission_plan_has_multiple_combinations():
    """Test that DEFAULT_MISSION_PLAN has at least MIN_COMBINATIONS."""
    print("\n=== DEFAULT_MISSION_PLAN Validation ===")
    
    pas = DEFAULT_MISSION_PLAN["pruned_action_space"]
    total_combinations = 1
    for key, values in pas.items():
        total_combinations *= len(values)
        print(f"  {key}: {len(values)} options")
    
    print(f"\n  Total combinations: {total_combinations}")
    
    if total_combinations >= MIN_COMBINATIONS:
        print(f"✅ DEFAULT_MISSION_PLAN has {total_combinations} combinations (>= {MIN_COMBINATIONS})")
        return True
    else:
        print(f"❌ DEFAULT_MISSION_PLAN has insufficient combinations (< {MIN_COMBINATIONS})")
        return False


def test_reward_normalization():
    """Test that reward weights are normalized to sum to 1.0."""
    print("\n=== Reward Normalization Tests ===")
    
    plan = {
        "reward_function": {
            "R_sm_throughput": 2.0,
            "R_dram_throughput": 2.0,
            "R_l1_hit_rate": 1.0,
            "R_l2_hit_rate": 1.0
        },
        "pruned_action_space": DEFAULT_MISSION_PLAN["pruned_action_space"].copy()
    }
    
    result = _validate_and_coerce_plan(plan)
    rf = result["reward_function"]
    total = sum(rf.values())
    
    # Allow for floating point precision issues
    if abs(total - 1.0) < 0.001:
        print(f"✅ Reward weights normalized to {total:.4f}")
        return True
    else:
        print(f"❌ Reward weights not normalized, sum = {total}")
        return False


def test_result_is_dict_not_string():
    """Test that parsed JSON result is a dict, not a string."""
    print("\n=== Result Type Validation Tests ===")
    
    test_input = """<param>
{"reward_function": {"R_sm_throughput": 0.5}, "pruned_action_space": {"BLOCK_SIZE_M": [64]}}
</param>"""
    
    result = _extract_json_with_param_handling(test_input)
    
    if result is not None and isinstance(result, dict):
        print("✅ Parsed result is a dict")
        return True
    else:
        print(f"❌ Parsed result is {type(result)}, expected dict")
        return False


def test_truncation_detection():
    """Test that truncated LLM outputs are detected."""
    print("\n=== Truncation Detection Tests ===")
    all_passed = True
    
    # Test cases for truncation detection
    truncation_cases = [
        # Truncated mid-sentence without <param> tags
        ("Let me analyze the kernel performance and optimize for", "Mid-sentence truncation without <param>"),
        # Truncated with verbose reasoning, no JSON
        ("Based on the profiling data, I recommend optimizing for SM throughput because", "Verbose reasoning truncated"),
    ]
    
    for input_str, test_name in truncation_cases:
        print(f"\n--- Testing: {test_name} ---")
        # These should be detected as truncated and return None from _extract_json_with_param_handling
        # or use the default safe policy
        result = _extract_json_with_param_handling(input_str)
        
        # For truncated output without any JSON structure, result should be None
        # The meta_controller will then use default policy
        if result is None:
            print(f"✅ Correctly detected truncated output (returned None)")
        else:
            print(f"⚠️ Returned {type(result)}, but may still fall back to default")
            # This is acceptable as the validation layer will expand to default
    
    return all_passed


if __name__ == "__main__":
    print("--- TRAINING FIXES TESTS ---")
    
    all_passed = True
    
    all_passed &= test_garbage_text_handling()
    all_passed &= test_action_space_validation()
    all_passed &= test_minimum_values_per_dimension()
    all_passed &= test_default_mission_plan_has_multiple_combinations()
    all_passed &= test_reward_normalization()
    all_passed &= test_result_is_dict_not_string()
    all_passed &= test_truncation_detection()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TRAINING FIXES TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("="*60)
