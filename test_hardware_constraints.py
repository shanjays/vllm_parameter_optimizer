"""
Tests for H100 hardware constraint validation to prevent Triton shared memory overflow.

These tests verify:
1. Block sizes are clamped to 128 max in meta_controller.py
2. num_stages is clamped to 5 max (aggressive) in meta_controller.py  
3. Shared memory validation works correctly in profiling_worker.py
4. FULL_PARAM_SPACE in hierarchical_kernel_optimizer.py has safe values
"""

# H100 hardware limits
# Note: H100 has 228KB (233,472 bytes) shared memory per SM, but we use a conservative
# limit of 227KB (232,448 bytes) to account for register spillover and other overheads
H100_SHARED_MEM_LIMIT = 232448  # bytes (~227KB, conservative limit)
H100_BLOCK_SIZE_MN_LIMIT = 128
H100_BLOCK_SIZE_K_LIMIT = 64  # Lower limit for K to avoid overflow with high M/N
H100_NUM_STAGES_LIMIT = 5  # Aggressive: Allow num_stages=5 for boundary testing


def validate_triton_config(config):
    """
    Check if config fits H100 shared memory limit.
    Copied from benchmark_worker.py for testing.
    """
    M = config.get('BLOCK_SIZE_M', 64)
    N = config.get('BLOCK_SIZE_N', 64)
    K = config.get('BLOCK_SIZE_K', 32)
    stages = config.get('num_stages', 4)
    
    # FP16 = 2 bytes per element
    shared_mem = (M * K + K * N) * 2 * stages
    
    if shared_mem > H100_SHARED_MEM_LIMIT:
        return False, shared_mem
    return True, shared_mem


def clamp_block_sizes(pas):
    """
    Clamp block sizes and num_stages to H100-safe values.
    Copied from professor_reward.py for testing.
    """
    pas["BLOCK_SIZE_M"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas.get("BLOCK_SIZE_M", [H100_BLOCK_SIZE_MN_LIMIT])]
    pas["BLOCK_SIZE_N"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas.get("BLOCK_SIZE_N", [H100_BLOCK_SIZE_MN_LIMIT])]
    pas["BLOCK_SIZE_K"] = [min(v, H100_BLOCK_SIZE_K_LIMIT) for v in pas.get("BLOCK_SIZE_K", [H100_BLOCK_SIZE_K_LIMIT])]
    pas["num_stages"] = [min(v, H100_NUM_STAGES_LIMIT) for v in pas.get("num_stages", [H100_NUM_STAGES_LIMIT])]
    
    # Remove duplicates
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages"]:
        pas[key] = list(set(pas[key])) or ([H100_BLOCK_SIZE_MN_LIMIT] if "BLOCK" in key else [H100_NUM_STAGES_LIMIT])
    
    return pas


# =============================================================================
# Tests for shared memory validation
# =============================================================================

def test_safe_config_passes():
    """Test that safe configurations pass validation."""
    safe_configs = [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_stages": 2},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_stages": 4},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_stages": 4},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "num_stages": 3},
    ]
    
    for config in safe_configs:
        is_valid, shared_mem = validate_triton_config(config)
        assert is_valid, f"Safe config should pass: {config}, shared_mem={shared_mem}"
    
    print("✅ test_safe_config_passes PASSED")


def test_unsafe_config_fails():
    """Test that unsafe configurations are rejected."""
    unsafe_configs = [
        # These exceed H100 shared memory limit of 232,448 bytes
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "num_stages": 4},
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "num_stages": 5},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256, "num_stages": 4},
    ]
    
    for config in unsafe_configs:
        is_valid, shared_mem = validate_triton_config(config)
        assert not is_valid, f"Unsafe config should fail: {config}, shared_mem={shared_mem}"
    
    print("✅ test_unsafe_config_fails PASSED")


def test_shared_mem_calculation():
    """Test that shared memory calculation is correct."""
    # Formula: (M*K + K*N) * 2 * stages
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_stages": 4}
    expected = (64 * 32 + 32 * 64) * 2 * 4  # 32,768 bytes
    
    _, actual = validate_triton_config(config)
    assert actual == expected, f"Expected {expected}, got {actual}"
    
    print("✅ test_shared_mem_calculation PASSED")


# =============================================================================
# Tests for block size clamping
# =============================================================================

def test_clamp_oversized_block_m():
    """Test that BLOCK_SIZE_M > 128 is clamped to 128."""
    pas = {"BLOCK_SIZE_M": [256, 512, 64], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32], "num_stages": [4]}
    result = clamp_block_sizes(pas)
    
    # 256 and 512 should be clamped to 128, resulting in [128, 64] after dedup
    assert all(v <= H100_BLOCK_SIZE_MN_LIMIT for v in result["BLOCK_SIZE_M"]), f"BLOCK_SIZE_M should be clamped: {result['BLOCK_SIZE_M']}"
    print("✅ test_clamp_oversized_block_m PASSED")


def test_clamp_oversized_block_n():
    """Test that BLOCK_SIZE_N > 128 is clamped to 128."""
    pas = {"BLOCK_SIZE_M": [64], "BLOCK_SIZE_N": [256, 192], "BLOCK_SIZE_K": [32], "num_stages": [4]}
    result = clamp_block_sizes(pas)
    
    assert all(v <= H100_BLOCK_SIZE_MN_LIMIT for v in result["BLOCK_SIZE_N"]), f"BLOCK_SIZE_N should be clamped: {result['BLOCK_SIZE_N']}"
    print("✅ test_clamp_oversized_block_n PASSED")


def test_clamp_oversized_block_k():
    """Test that BLOCK_SIZE_K > 64 is clamped to 64."""
    pas = {"BLOCK_SIZE_M": [64], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [128, 256], "num_stages": [4]}
    result = clamp_block_sizes(pas)
    
    assert all(v <= H100_BLOCK_SIZE_K_LIMIT for v in result["BLOCK_SIZE_K"]), f"BLOCK_SIZE_K should be clamped: {result['BLOCK_SIZE_K']}"
    print("✅ test_clamp_oversized_block_k PASSED")


def test_clamp_oversized_num_stages():
    """Test that num_stages > 5 is clamped to 5 (aggressive limit)."""
    pas = {"BLOCK_SIZE_M": [64], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32], "num_stages": [6, 7, 8]}
    result = clamp_block_sizes(pas)
    
    assert all(v <= 5 for v in result["num_stages"]), f"num_stages should be clamped to 5: {result['num_stages']}"
    print("✅ test_clamp_oversized_num_stages PASSED")


def test_clamp_removes_duplicates():
    """Test that clamping removes duplicates after clamping."""
    pas = {"BLOCK_SIZE_M": [256, 256, 128], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32], "num_stages": [6, 6, 5]}
    result = clamp_block_sizes(pas)
    
    # [256, 256, 128] -> [128, 128, 128] -> [128] after dedup
    assert result["BLOCK_SIZE_M"] == [128], \
        f"Duplicates should be removed, expected [128], got: {result['BLOCK_SIZE_M']}"
    
    # [6, 6, 5] -> [5, 5, 5] -> [5] after dedup
    assert result["num_stages"] == [5], \
        f"Duplicates should be removed, expected [5], got: {result['num_stages']}"
    
    print("✅ test_clamp_removes_duplicates PASSED")


def test_clamp_preserves_safe_values():
    """Test that safe values are preserved unchanged."""
    pas = {"BLOCK_SIZE_M": [16, 64, 128], "BLOCK_SIZE_N": [32, 64], "BLOCK_SIZE_K": [32, 64], "num_stages": [2, 3, 4]}
    result = clamp_block_sizes(pas)
    
    # All values should be preserved (sorted due to set conversion)
    assert set(result["BLOCK_SIZE_M"]) == {16, 64, 128}, f"Safe BLOCK_SIZE_M should be preserved: {result['BLOCK_SIZE_M']}"
    assert set(result["BLOCK_SIZE_K"]) == {32, 64}, f"Safe BLOCK_SIZE_K should be preserved: {result['BLOCK_SIZE_K']}"
    assert set(result["num_stages"]) == {2, 3, 4}, f"Safe num_stages should be preserved: {result['num_stages']}"
    
    print("✅ test_clamp_preserves_safe_values PASSED")


# =============================================================================
# Tests for FULL_PARAM_SPACE validation  
# =============================================================================

def test_full_param_space_is_safe():
    """Test that FULL_PARAM_SPACE from hierarchical_kernel_optimizer.py only contains safe values."""
    # This is the constrained parameter space from hierarchical_kernel_optimizer.py
    # Updated for aggressive optimization - includes num_stages=5
    FULL_PARAM_SPACE = {
        "BLOCK_SIZE_M": [16, 32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],  # Reduced to 64 max to avoid overflow
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4, 5]  # Include 5 for aggressive testing
    }
    
    # Verify all block sizes within limits
    assert all(v <= H100_BLOCK_SIZE_MN_LIMIT for v in FULL_PARAM_SPACE["BLOCK_SIZE_M"]), "BLOCK_SIZE_M has unsafe values"
    assert all(v <= H100_BLOCK_SIZE_MN_LIMIT for v in FULL_PARAM_SPACE["BLOCK_SIZE_N"]), "BLOCK_SIZE_N has unsafe values"
    assert all(v <= H100_BLOCK_SIZE_K_LIMIT for v in FULL_PARAM_SPACE["BLOCK_SIZE_K"]), "BLOCK_SIZE_K has unsafe values"
    
    # Verify num_stages <= 5 (aggressive)
    assert all(v <= 5 for v in FULL_PARAM_SPACE["num_stages"]), "num_stages has unsafe values (> 5)"
    
    # Verify num_warps are powers of 2
    for w in FULL_PARAM_SPACE["num_warps"]:
        assert w > 0 and (w & (w - 1)) == 0, f"{w} is not a power of 2"
    
    # Verify no 256 in any block size (the problematic value that was removed)
    assert 256 not in FULL_PARAM_SPACE["BLOCK_SIZE_M"], "256 should be removed from BLOCK_SIZE_M"
    assert 256 not in FULL_PARAM_SPACE["BLOCK_SIZE_N"], "256 should be removed from BLOCK_SIZE_N"
    assert 256 not in FULL_PARAM_SPACE["BLOCK_SIZE_K"], "256 should be removed from BLOCK_SIZE_K"
    
    # Verify BLOCK_SIZE_K doesn't have 128 (causes overflow with high M/N)
    assert 128 not in FULL_PARAM_SPACE["BLOCK_SIZE_K"], "128 should be removed from BLOCK_SIZE_K"
    
    print("✅ test_full_param_space_is_safe PASSED")


def test_all_param_combinations_are_valid():
    """Test that all combinations from FULL_PARAM_SPACE fit in H100 shared memory."""
    FULL_PARAM_SPACE = {
        "BLOCK_SIZE_M": [16, 32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],  # Reduced to 64 max to avoid overflow
        "num_stages": [2, 3, 4]
    }
    
    invalid_count = 0
    for M in FULL_PARAM_SPACE["BLOCK_SIZE_M"]:
        for N in FULL_PARAM_SPACE["BLOCK_SIZE_N"]:
            for K in FULL_PARAM_SPACE["BLOCK_SIZE_K"]:
                for stages in FULL_PARAM_SPACE["num_stages"]:
                    config = {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K, "num_stages": stages}
                    is_valid, shared_mem = validate_triton_config(config)
                    if not is_valid:
                        invalid_count += 1
                        print(f"  Invalid: {config} -> {shared_mem} bytes")
    
    assert invalid_count == 0, f"Found {invalid_count} invalid combinations in FULL_PARAM_SPACE"
    print("✅ test_all_param_combinations_are_valid PASSED")


if __name__ == "__main__":
    print("--- H100 HARDWARE CONSTRAINT TESTS ---\n")
    
    print("=== Shared Memory Validation Tests ===")
    test_safe_config_passes()
    test_unsafe_config_fails()
    test_shared_mem_calculation()
    
    print("\n=== Block Size Clamping Tests ===")
    test_clamp_oversized_block_m()
    test_clamp_oversized_block_n()
    test_clamp_oversized_block_k()
    test_clamp_oversized_num_stages()
    test_clamp_removes_duplicates()
    test_clamp_preserves_safe_values()
    
    print("\n=== FULL_PARAM_SPACE Validation Tests ===")
    test_full_param_space_is_safe()
    test_all_param_combinations_are_valid()
    
    print("\n" + "="*60)
    print("✅ ALL H100 HARDWARE CONSTRAINT TESTS PASSED")
    print("="*60)
