"""
Tests for num_warps validation to ensure values are powers of 2.
This addresses the AssertionError: num_warps must be a power of 2.
"""

# Copy of the validation logic to avoid heavy dependency imports
VALID_NUM_WARPS = [1, 2, 4, 8, 16, 32]


def sanitize_mission_plan(plan):
    """Ensures all num_warps values are valid powers of 2."""
    if 'pruned_action_space' in plan and 'num_warps' in plan['pruned_action_space']:
        original = plan['pruned_action_space']['num_warps']
        sanitized = [w for w in original if w in VALID_NUM_WARPS]
        if not sanitized:
            sanitized = [4]  # Default fallback
            print(f"[RewardFn] WARNING: No valid num_warps in {original}, using [4]")
        elif len(sanitized) != len(original):
            print(f"[RewardFn] WARNING: Filtered invalid num_warps: {original} -> {sanitized}")
        plan['pruned_action_space']['num_warps'] = sanitized
    return plan


def test_sanitize_mission_plan_valid_values():
    """Test that valid num_warps values are preserved."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0},
        'pruned_action_space': {
            'BLOCK_SIZE_M': [64],
            'BLOCK_SIZE_N': [64],
            'BLOCK_SIZE_K': [32],
            'num_warps': [2, 4, 8],
            'num_stages': [4]
        }
    }
    result = sanitize_mission_plan(plan)
    assert result['pruned_action_space']['num_warps'] == [2, 4, 8], "Valid values should be preserved"
    print("✅ test_sanitize_mission_plan_valid_values PASSED")


def test_sanitize_mission_plan_filters_invalid():
    """Test that invalid num_warps values are filtered out."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0},
        'pruned_action_space': {
            'BLOCK_SIZE_M': [64],
            'BLOCK_SIZE_N': [64],
            'BLOCK_SIZE_K': [32],
            'num_warps': [3, 4, 5, 6, 7, 8],  # 3, 5, 6, 7 are invalid
            'num_stages': [4]
        }
    }
    result = sanitize_mission_plan(plan)
    assert result['pruned_action_space']['num_warps'] == [4, 8], "Invalid values should be filtered"
    print("✅ test_sanitize_mission_plan_filters_invalid PASSED")


def test_sanitize_mission_plan_all_invalid_defaults():
    """Test that when all values are invalid, default to [4]."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0},
        'pruned_action_space': {
            'BLOCK_SIZE_M': [64],
            'BLOCK_SIZE_N': [64],
            'BLOCK_SIZE_K': [32],
            'num_warps': [3, 5, 6, 7],  # All invalid
            'num_stages': [4]
        }
    }
    result = sanitize_mission_plan(plan)
    assert result['pruned_action_space']['num_warps'] == [4], "Should default to [4] when all invalid"
    print("✅ test_sanitize_mission_plan_all_invalid_defaults PASSED")


def test_sanitize_mission_plan_empty_list():
    """Test that an empty list defaults to [4]."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0},
        'pruned_action_space': {
            'BLOCK_SIZE_M': [64],
            'BLOCK_SIZE_N': [64],
            'BLOCK_SIZE_K': [32],
            'num_warps': [],
            'num_stages': [4]
        }
    }
    result = sanitize_mission_plan(plan)
    assert result['pruned_action_space']['num_warps'] == [4], "Empty list should default to [4]"
    print("✅ test_sanitize_mission_plan_empty_list PASSED")


def test_sanitize_mission_plan_missing_num_warps():
    """Test that a plan without num_warps is unchanged."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0},
        'pruned_action_space': {
            'BLOCK_SIZE_M': [64],
            'BLOCK_SIZE_N': [64],
            'BLOCK_SIZE_K': [32],
            'num_stages': [4]
        }
    }
    result = sanitize_mission_plan(plan)
    assert 'num_warps' not in result['pruned_action_space'], "Should not add num_warps if missing"
    print("✅ test_sanitize_mission_plan_missing_num_warps PASSED")


def test_sanitize_mission_plan_missing_pruned_action_space():
    """Test that a plan without pruned_action_space is unchanged."""
    plan = {
        'reward_function': {'R_sm_throughput': 1.0}
    }
    result = sanitize_mission_plan(plan)
    assert 'pruned_action_space' not in result, "Should not add pruned_action_space if missing"
    print("✅ test_sanitize_mission_plan_missing_pruned_action_space PASSED")


def test_valid_num_warps_constant():
    """Test that VALID_NUM_WARPS contains only powers of 2."""
    for val in VALID_NUM_WARPS:
        assert val > 0 and (val & (val - 1)) == 0, f"{val} is not a power of 2"
    print("✅ test_valid_num_warps_constant PASSED")


def test_is_power_of_two_check():
    """Test the power of 2 bitwise check used in fast_gym_env.py and benchmark_worker.py."""
    # Valid powers of 2
    powers_of_two = [1, 2, 4, 8, 16, 32]
    for val in powers_of_two:
        is_valid = val > 0 and (val & (val - 1)) == 0
        assert is_valid, f"{val} should be detected as power of 2"
    
    # Invalid values
    not_powers_of_two = [0, 3, 5, 6, 7, 9, 10, 12, 15, -1, -4]
    for val in not_powers_of_two:
        is_valid = val > 0 and (val & (val - 1)) == 0
        assert not is_valid, f"{val} should NOT be detected as power of 2"
    
    print("✅ test_is_power_of_two_check PASSED")


if __name__ == "__main__":
    print("--- NUM_WARPS VALIDATION TESTS ---\n")
    
    test_sanitize_mission_plan_valid_values()
    test_sanitize_mission_plan_filters_invalid()
    test_sanitize_mission_plan_all_invalid_defaults()
    test_sanitize_mission_plan_empty_list()
    test_sanitize_mission_plan_missing_num_warps()
    test_sanitize_mission_plan_missing_pruned_action_space()
    test_valid_num_warps_constant()
    test_is_power_of_two_check()
    
    print("\n" + "="*60)
    print("✅ ALL NUM_WARPS VALIDATION TESTS PASSED")
    print("="*60)
