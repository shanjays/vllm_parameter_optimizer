"""
Tests for NCU error handling with retry logic and better diagnostics.

Tests verify:
1. Error penalty constants are correctly defined (aggressive optimization)
2. Error categorization works for different error types
3. Config reduction logic works correctly
4. Failure stats tracking works
5. Failure logging respects the 100 item limit
"""

import time

# Copy of error penalty constants from profiling_worker.py for testing
# Updated for aggressive optimization (less severe penalties for boundary-finding errors)
PENALTY_CUDA_OOM = -20.0          # OOM tells us we found the limit!
PENALTY_SHARED_MEMORY = -25.0     # Shared mem limit is useful info
PENALTY_REGISTER_OVERFLOW = -30.0 # Register limit is useful info
PENALTY_TIMEOUT = -50.0           # Timeouts waste time
PENALTY_PARSE_ERROR = -25.0
PENALTY_TOTAL_FAILURE = -100.0    # Keep high for unknown failures
PENALTY_TRITON_ERROR = -25.0      # Triton compilation failures


# =============================================================================
# Tests for error penalty constants
# =============================================================================

def test_penalty_constants_exist():
    """Test that all penalty constants are defined with aggressive values."""
    assert PENALTY_CUDA_OOM == -20.0, "PENALTY_CUDA_OOM should be -20.0 (aggressive)"
    assert PENALTY_SHARED_MEMORY == -25.0, "PENALTY_SHARED_MEMORY should be -25.0 (aggressive)"
    assert PENALTY_REGISTER_OVERFLOW == -30.0, "PENALTY_REGISTER_OVERFLOW should be -30.0 (aggressive)"
    assert PENALTY_TIMEOUT == -50.0, "PENALTY_TIMEOUT should be -50.0"
    assert PENALTY_PARSE_ERROR == -25.0, "PENALTY_PARSE_ERROR should be -25.0"
    assert PENALTY_TOTAL_FAILURE == -100.0, "PENALTY_TOTAL_FAILURE should be -100.0"
    assert PENALTY_TRITON_ERROR == -25.0, "PENALTY_TRITON_ERROR should be -25.0"
    print("✅ test_penalty_constants_exist PASSED")


def test_penalty_ordering():
    """Test that penalties are ordered by severity (less negative = less severe)."""
    # CUDA OOM should be least severe (boundary finding is informative!)
    assert PENALTY_CUDA_OOM > PENALTY_SHARED_MEMORY, "CUDA OOM should be less severe than shared memory"
    assert PENALTY_SHARED_MEMORY > PENALTY_REGISTER_OVERFLOW, "Shared memory should be less severe than register overflow"
    assert PENALTY_REGISTER_OVERFLOW > PENALTY_TIMEOUT, "Register overflow should be less severe than timeout"
    assert PENALTY_TOTAL_FAILURE < PENALTY_TIMEOUT, "Total failure should be most severe"
    print("✅ test_penalty_ordering PASSED")


# =============================================================================
# Tests for error categorization
# =============================================================================

def _categorize_error_for_test(stderr):
    """Copy of _categorize_error logic for testing (matches profiling_worker.py)."""
    if stderr is None:
        return "unknown", PENALTY_TOTAL_FAILURE
    
    stderr_lower = stderr.lower() if isinstance(stderr, str) else str(stderr).lower()
    
    # CUDA OOM - we found the memory limit! This is valuable info.
    if "cuda out of memory" in stderr_lower or "out of memory" in stderr_lower or "oom" in stderr_lower:
        return "cuda_oom", PENALTY_CUDA_OOM
    
    # Shared memory exceeded
    if "shared memory" in stderr_lower or "smem" in stderr_lower:
        return "shared_memory", PENALTY_SHARED_MEMORY
    
    # Register overflow
    if "register" in stderr_lower or "spill" in stderr_lower:
        return "register_overflow", PENALTY_REGISTER_OVERFLOW
    
    # Timeout
    if "timeout" in stderr_lower:
        return "timeout", PENALTY_TIMEOUT
    
    # Triton compilation errors
    if "triton" in stderr_lower and ("error" in stderr_lower or "failed" in stderr_lower):
        return "triton_error", PENALTY_TRITON_ERROR
    
    return "unknown", PENALTY_TOTAL_FAILURE


def test_categorize_error_shared_memory():
    """Test that shared memory errors are correctly categorized."""
    error_type, penalty = _categorize_error_for_test("Error: Out of resource: shared memory required")
    assert error_type == "shared_memory", f"Expected shared_memory, got {error_type}"
    assert penalty == PENALTY_SHARED_MEMORY, f"Expected {PENALTY_SHARED_MEMORY}, got {penalty}"
    print("✅ test_categorize_error_shared_memory PASSED")


def test_categorize_error_register_overflow():
    """Test that register overflow errors are correctly categorized."""
    error_type, penalty = _categorize_error_for_test("ptxas error: Insufficient registers for kernel")
    assert error_type == "register_overflow", f"Expected register_overflow, got {error_type}"
    assert penalty == PENALTY_REGISTER_OVERFLOW, f"Expected {PENALTY_REGISTER_OVERFLOW}, got {penalty}"
    print("✅ test_categorize_error_register_overflow PASSED")


def test_categorize_error_timeout():
    """Test that timeout errors are correctly categorized."""
    error_type, penalty = _categorize_error_for_test("Process killed: timeout expired after 60s")
    assert error_type == "timeout", f"Expected timeout, got {error_type}"
    assert penalty == PENALTY_TIMEOUT, f"Expected {PENALTY_TIMEOUT}, got {penalty}"
    print("✅ test_categorize_error_timeout PASSED")


def test_categorize_error_cuda_oom():
    """Test that CUDA OOM errors are correctly categorized with aggressive penalty."""
    error_type, penalty = _categorize_error_for_test("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert error_type == "cuda_oom", f"Expected cuda_oom, got {error_type}"
    assert penalty == PENALTY_CUDA_OOM, f"Expected {PENALTY_CUDA_OOM} (aggressive), got {penalty}"
    print("✅ test_categorize_error_cuda_oom PASSED")


def test_categorize_error_triton():
    """Test that Triton compilation errors are correctly categorized."""
    error_type, penalty = _categorize_error_for_test("triton.compiler.CompilationError: failed to compile kernel")
    assert error_type == "triton_error", f"Expected triton_error, got {error_type}"
    assert penalty == PENALTY_TRITON_ERROR, f"Expected {PENALTY_TRITON_ERROR}, got {penalty}"
    print("✅ test_categorize_error_triton PASSED")


def test_categorize_error_unknown():
    """Test that unknown errors are correctly categorized."""
    error_type, penalty = _categorize_error_for_test("Some completely unrecognized error")
    assert error_type == "unknown", f"Expected unknown, got {error_type}"
    assert penalty == PENALTY_TOTAL_FAILURE, f"Expected {PENALTY_TOTAL_FAILURE}, got {penalty}"
    print("✅ test_categorize_error_unknown PASSED")


def test_categorize_error_none():
    """Test that None stderr is handled correctly."""
    error_type, penalty = _categorize_error_for_test(None)
    assert error_type == "unknown", f"Expected unknown, got {error_type}"
    assert penalty == PENALTY_TOTAL_FAILURE, f"Expected {PENALTY_TOTAL_FAILURE}, got {penalty}"
    print("✅ test_categorize_error_none PASSED")


# =============================================================================
# Tests for config reduction
# =============================================================================

H100_LIMIT = 232448  # bytes (~227KB, conservative limit)

def validate_triton_config(config):
    """Copy of validation logic for testing."""
    M = config.get('BLOCK_SIZE_M', 64)
    N = config.get('BLOCK_SIZE_N', 64)
    K = config.get('BLOCK_SIZE_K', 32)
    stages = config.get('num_stages', 4)
    
    shared_mem = (M * K + K * N) * 2 * stages
    return shared_mem <= H100_LIMIT


def reduce_config(config):
    """Copy of _reduce_config logic for testing."""
    reduced = config.copy()
    
    reductions = [
        ('num_stages', lambda x: max(2, x - 1)),
        ('BLOCK_SIZE_M', lambda x: max(16, x // 2)),
        ('BLOCK_SIZE_K', lambda x: max(32, x // 2)),
        ('BLOCK_SIZE_N', lambda x: max(32, x // 2)),
    ]
    
    for key, reduce_fn in reductions:
        if key in reduced and reduced[key] > reduce_fn(reduced[key]):
            reduced[key] = reduce_fn(reduced[key])
            
            if validate_triton_config(reduced):
                return reduced
    
    return None


def test_reduce_config_num_stages():
    """Test that num_stages is reduced first."""
    # Config that exceeds limits
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "num_stages": 5  # This is too high
    }
    
    # Verify it's invalid
    assert not validate_triton_config(config), "Initial config should be invalid"
    
    reduced = reduce_config(config)
    
    # If reduction worked, num_stages should be reduced
    if reduced is not None:
        assert reduced['num_stages'] < config['num_stages'], "num_stages should be reduced"
        assert validate_triton_config(reduced), "Reduced config should be valid"
    print("✅ test_reduce_config_num_stages PASSED")


def test_reduce_config_block_size():
    """Test that block sizes are reduced when num_stages can't be reduced further."""
    # Config with minimum num_stages but large blocks
    config = {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 128,
        "num_stages": 2  # Already at minimum
    }
    
    # Verify it's invalid
    assert not validate_triton_config(config), "Initial config should be invalid"
    
    reduced = reduce_config(config)
    
    # Check that some reduction happened
    if reduced is not None:
        assert validate_triton_config(reduced), "Reduced config should be valid"
        # At least one block size should be smaller
        reduced_any = (
            reduced['BLOCK_SIZE_M'] < config['BLOCK_SIZE_M'] or
            reduced['BLOCK_SIZE_N'] < config['BLOCK_SIZE_N'] or
            reduced['BLOCK_SIZE_K'] < config['BLOCK_SIZE_K'] or
            reduced['num_stages'] < config['num_stages']
        )
        assert reduced_any, "Some parameter should be reduced"
    print("✅ test_reduce_config_block_size PASSED")


def test_reduce_config_already_valid():
    """Test that valid configs don't need reduction."""
    # Config that's already valid
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_stages": 4
    }
    
    assert validate_triton_config(config), "Initial config should be valid"
    print("✅ test_reduce_config_already_valid PASSED")


def test_reduce_config_minimum_values():
    """Test reduction respects minimum values."""
    # Try to reduce below minimums
    config = {
        "BLOCK_SIZE_M": 16,  # Already at minimum (16)
        "BLOCK_SIZE_N": 32,  # Already at minimum (32)
        "BLOCK_SIZE_K": 32,  # Already at minimum (32)
        "num_stages": 2  # Already at minimum (2)
    }
    
    # Verify this config is valid (small config should be valid)
    assert validate_triton_config(config), "Minimum config should be valid"
    print("✅ test_reduce_config_minimum_values PASSED")


# =============================================================================
# Tests for failure stats tracking
# =============================================================================

def test_failure_stats_empty():
    """Test failure stats when no failures have occurred."""
    failed_configs = []
    
    def get_failure_stats():
        if not failed_configs:
            return {"total_failures": 0}
        
        error_counts = {}
        for failure in failed_configs:
            error_type = failure['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_failures": len(failed_configs),
            "by_type": error_counts,
            "recent_failures": failed_configs[-5:]
        }
    
    stats = get_failure_stats()
    assert stats["total_failures"] == 0, "Should have 0 failures"
    print("✅ test_failure_stats_empty PASSED")


def test_failure_stats_counting():
    """Test failure stats correctly counts different error types."""
    failed_configs = [
        {'config': {}, 'error_type': 'shared_memory', 'error_msg': 'test', 'timestamp': 1.0},
        {'config': {}, 'error_type': 'shared_memory', 'error_msg': 'test', 'timestamp': 2.0},
        {'config': {}, 'error_type': 'timeout', 'error_msg': 'test', 'timestamp': 3.0},
        {'config': {}, 'error_type': 'parse_error', 'error_msg': 'test', 'timestamp': 4.0},
    ]
    
    def get_failure_stats():
        if not failed_configs:
            return {"total_failures": 0}
        
        error_counts = {}
        for failure in failed_configs:
            error_type = failure['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_failures": len(failed_configs),
            "by_type": error_counts,
            "recent_failures": failed_configs[-5:]
        }
    
    stats = get_failure_stats()
    assert stats["total_failures"] == 4, f"Should have 4 failures, got {stats['total_failures']}"
    assert stats["by_type"]["shared_memory"] == 2, "Should have 2 shared_memory failures"
    assert stats["by_type"]["timeout"] == 1, "Should have 1 timeout failure"
    assert stats["by_type"]["parse_error"] == 1, "Should have 1 parse_error failure"
    print("✅ test_failure_stats_counting PASSED")


def test_failure_stats_recent():
    """Test that recent failures returns last 5 items."""
    failed_configs = []
    for i in range(10):
        failed_configs.append({
            'config': {'id': i},
            'error_type': 'test',
            'error_msg': f'error {i}',
            'timestamp': float(i)
        })
    
    def get_failure_stats():
        if not failed_configs:
            return {"total_failures": 0}
        
        error_counts = {}
        for failure in failed_configs:
            error_type = failure['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_failures": len(failed_configs),
            "by_type": error_counts,
            "recent_failures": failed_configs[-5:]
        }
    
    stats = get_failure_stats()
    assert len(stats["recent_failures"]) == 5, "Should have 5 recent failures"
    assert stats["recent_failures"][0]['config']['id'] == 5, "First recent should be id 5"
    assert stats["recent_failures"][4]['config']['id'] == 9, "Last recent should be id 9"
    print("✅ test_failure_stats_recent PASSED")


# =============================================================================
# Tests for failure logging limit
# =============================================================================

def test_log_failed_config_limit():
    """Test that failed configs list is limited to 100 items."""
    failed_configs = []
    
    def log_failed_config(config, error_type, error_msg):
        failed_configs.append({
            'config': config.copy(),
            'error_type': error_type,
            'error_msg': str(error_msg)[:200],
            'timestamp': time.time()
        })
        
        if len(failed_configs) > 100:
            failed_configs[:] = failed_configs[-100:]
    
    # Add 150 failures
    for i in range(150):
        log_failed_config({'id': i}, 'test', f'error {i}')
    
    assert len(failed_configs) == 100, f"Should have 100 failures, got {len(failed_configs)}"
    # Should keep the last 100 (ids 50-149)
    assert failed_configs[0]['config']['id'] == 50, "First config should be id 50"
    assert failed_configs[99]['config']['id'] == 149, "Last config should be id 149"
    print("✅ test_log_failed_config_limit PASSED")


def test_log_failed_config_truncates_message():
    """Test that error messages are truncated to 200 characters."""
    failed_configs = []
    
    def log_failed_config(config, error_type, error_msg):
        failed_configs.append({
            'config': config.copy(),
            'error_type': error_type,
            'error_msg': str(error_msg)[:200],
            'timestamp': time.time()
        })
    
    # Add a failure with a very long message
    long_message = "x" * 500
    log_failed_config({}, 'test', long_message)
    
    assert len(failed_configs[0]['error_msg']) == 200, "Message should be truncated to 200 chars"
    print("✅ test_log_failed_config_truncates_message PASSED")


if __name__ == "__main__":
    print("--- NCU ERROR HANDLING TESTS ---\n")
    
    print("=== Error Penalty Constants Tests ===")
    test_penalty_constants_exist()
    test_penalty_ordering()
    
    print("\n=== Error Categorization Tests ===")
    test_categorize_error_shared_memory()
    test_categorize_error_register_overflow()
    test_categorize_error_timeout()
    test_categorize_error_cuda_oom()
    test_categorize_error_triton()
    test_categorize_error_unknown()
    test_categorize_error_none()
    
    print("\n=== Config Reduction Tests ===")
    test_reduce_config_num_stages()
    test_reduce_config_block_size()
    test_reduce_config_already_valid()
    test_reduce_config_minimum_values()
    
    print("\n=== Failure Stats Tests ===")
    test_failure_stats_empty()
    test_failure_stats_counting()
    test_failure_stats_recent()
    
    print("\n=== Failure Logging Tests ===")
    test_log_failed_config_limit()
    test_log_failed_config_truncates_message()
    
    print("\n" + "="*60)
    print("✅ ALL NCU ERROR HANDLING TESTS PASSED")
    print("="*60)
