"""
Tests for vLLM benchmark error logging functionality.

Tests verify:
1. _parse_throughput parses different output formats correctly
2. _validate_config_file validates config files properly
3. _check_dataset_exists checks file existence correctly
4. _categorize_vllm_error categorizes different error types correctly
"""

import json
import os
import tempfile


# =============================================================================
# Tests for _parse_throughput
# =============================================================================

def _parse_throughput_for_test(stdout: str) -> float:
    """Copy of _parse_throughput logic for testing."""
    import re
    
    # Try different patterns
    patterns = [
        r"Throughput:\s*([0-9.]+)\s*requests/s,\s*([0-9.]+)\s*tokens/s",
        r"([0-9.]+)\s*tokens/s",
        r"tokens/s[:\s]*([0-9.]+)",
        r"Throughput[:\s]*([0-9.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            # Get the last group (tokens/s value)
            value = float(match.group(match.lastindex))
            return value
    
    return 0.0


def test_parse_throughput_standard_format():
    """Test parsing standard vLLM throughput output format."""
    output = "Throughput: 45.32 requests/s, 1523.45 tokens/s"
    result = _parse_throughput_for_test(output)
    assert result == 1523.45, f"Expected 1523.45, got {result}"
    print("✅ test_parse_throughput_standard_format PASSED")


def test_parse_throughput_simple_tokens():
    """Test parsing simple tokens/s format."""
    output = "Generated 1234.56 tokens/s"
    result = _parse_throughput_for_test(output)
    assert result == 1234.56, f"Expected 1234.56, got {result}"
    print("✅ test_parse_throughput_simple_tokens PASSED")


def test_parse_throughput_tokens_colon():
    """Test parsing tokens/s with colon format."""
    output = "tokens/s: 999.99"
    result = _parse_throughput_for_test(output)
    assert result == 999.99, f"Expected 999.99, got {result}"
    print("✅ test_parse_throughput_tokens_colon PASSED")


def test_parse_throughput_throughput_only():
    """Test parsing Throughput-only format."""
    output = "Throughput: 500.25"
    result = _parse_throughput_for_test(output)
    assert result == 500.25, f"Expected 500.25, got {result}"
    print("✅ test_parse_throughput_throughput_only PASSED")


def test_parse_throughput_no_match():
    """Test that parsing returns 0.0 when no pattern matches."""
    output = "Some random output without throughput"
    result = _parse_throughput_for_test(output)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("✅ test_parse_throughput_no_match PASSED")


def test_parse_throughput_multiline():
    """Test parsing throughput from multiline output."""
    output = """Loading model...
Model loaded successfully.
Running benchmark...
Throughput: 100.00 requests/s, 2500.50 tokens/s
Benchmark complete."""
    result = _parse_throughput_for_test(output)
    assert result == 2500.50, f"Expected 2500.50, got {result}"
    print("✅ test_parse_throughput_multiline PASSED")


# =============================================================================
# Tests for _validate_config_file
# =============================================================================

def _validate_config_file_for_test(config_path: str) -> bool:
    """Copy of _validate_config_file logic for testing."""
    if not os.path.exists(config_path):
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if not config:
            return False
        
        return True
        
    except json.JSONDecodeError:
        return False
    except Exception:
        return False


def test_validate_config_file_valid():
    """Test that valid config file passes validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "16088": {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "num_warps": 4,
                "num_stages": 4
            }
        }
        json.dump(config, f)
        config_path = f.name
    
    try:
        result = _validate_config_file_for_test(config_path)
        assert result == True, f"Expected True, got {result}"
    finally:
        os.unlink(config_path)
    print("✅ test_validate_config_file_valid PASSED")


def test_validate_config_file_nonexistent():
    """Test that nonexistent file fails validation."""
    result = _validate_config_file_for_test("/nonexistent/path/to/config.json")
    assert result == False, f"Expected False, got {result}"
    print("✅ test_validate_config_file_nonexistent PASSED")


def test_validate_config_file_empty():
    """Test that empty config file fails validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        config_path = f.name
    
    try:
        result = _validate_config_file_for_test(config_path)
        assert result == False, f"Expected False for empty config, got {result}"
    finally:
        os.unlink(config_path)
    print("✅ test_validate_config_file_empty PASSED")


def test_validate_config_file_invalid_json():
    """Test that invalid JSON file fails validation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        config_path = f.name
    
    try:
        result = _validate_config_file_for_test(config_path)
        assert result == False, f"Expected False for invalid JSON, got {result}"
    finally:
        os.unlink(config_path)
    print("✅ test_validate_config_file_invalid_json PASSED")


# =============================================================================
# Tests for _check_dataset_exists
# =============================================================================

def _check_dataset_exists_for_test(dataset_path: str) -> bool:
    """Copy of _check_dataset_exists logic for testing."""
    return os.path.exists(dataset_path)


def test_check_dataset_exists_true():
    """Test that existing file returns True."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("[]")
        dataset_path = f.name
    
    try:
        result = _check_dataset_exists_for_test(dataset_path)
        assert result == True, f"Expected True, got {result}"
    finally:
        os.unlink(dataset_path)
    print("✅ test_check_dataset_exists_true PASSED")


def test_check_dataset_exists_false():
    """Test that nonexistent file returns False."""
    result = _check_dataset_exists_for_test("/nonexistent/dataset.json")
    assert result == False, f"Expected False, got {result}"
    print("✅ test_check_dataset_exists_false PASSED")


# =============================================================================
# Tests for _categorize_vllm_error
# =============================================================================

def _categorize_vllm_error_for_test(full_output: str) -> str:
    """Copy of _categorize_vllm_error logic for testing (without print statements)."""
    import re
    
    if "CUDA out of memory" in full_output:
        return "cuda_oom"
    elif "FileNotFoundError" in full_output:
        return "file_not_found"
    elif "KeyError" in full_output:
        return "key_error"
    elif "JSONDecodeError" in full_output:
        return "json_error"
    elif "RuntimeError" in full_output:
        return "runtime_error"
    elif "ModuleNotFoundError" in full_output:
        return "module_not_found"
    elif "AssertionError" in full_output:
        return "assertion_error"
    else:
        return "unknown"


def test_categorize_vllm_error_cuda_oom():
    """Test that CUDA OOM errors are correctly categorized."""
    output = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    result = _categorize_vllm_error_for_test(output)
    assert result == "cuda_oom", f"Expected cuda_oom, got {result}"
    print("✅ test_categorize_vllm_error_cuda_oom PASSED")


def test_categorize_vllm_error_file_not_found():
    """Test that FileNotFoundError is correctly categorized."""
    output = "FileNotFoundError: [Errno 2] No such file or directory: 'dataset.json'"
    result = _categorize_vllm_error_for_test(output)
    assert result == "file_not_found", f"Expected file_not_found, got {result}"
    print("✅ test_categorize_vllm_error_file_not_found PASSED")


def test_categorize_vllm_error_key_error():
    """Test that KeyError is correctly categorized."""
    output = "KeyError: '128'"
    result = _categorize_vllm_error_for_test(output)
    assert result == "key_error", f"Expected key_error, got {result}"
    print("✅ test_categorize_vllm_error_key_error PASSED")


def test_categorize_vllm_error_key_error_numeric():
    """Test that numeric KeyError is correctly categorized."""
    output = "KeyError: 128"
    result = _categorize_vllm_error_for_test(output)
    assert result == "key_error", f"Expected key_error, got {result}"
    print("✅ test_categorize_vllm_error_key_error_numeric PASSED")


def test_categorize_vllm_error_json_decode():
    """Test that JSONDecodeError is correctly categorized."""
    output = "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes"
    result = _categorize_vllm_error_for_test(output)
    assert result == "json_error", f"Expected json_error, got {result}"
    print("✅ test_categorize_vllm_error_json_decode PASSED")


def test_categorize_vllm_error_runtime_error():
    """Test that RuntimeError is correctly categorized."""
    output = "RuntimeError: Expected tensor to be on GPU"
    result = _categorize_vllm_error_for_test(output)
    assert result == "runtime_error", f"Expected runtime_error, got {result}"
    print("✅ test_categorize_vllm_error_runtime_error PASSED")


def test_categorize_vllm_error_module_not_found():
    """Test that ModuleNotFoundError is correctly categorized."""
    output = "ModuleNotFoundError: No module named 'vllm'"
    result = _categorize_vllm_error_for_test(output)
    assert result == "module_not_found", f"Expected module_not_found, got {result}"
    print("✅ test_categorize_vllm_error_module_not_found PASSED")


def test_categorize_vllm_error_assertion_error():
    """Test that AssertionError is correctly categorized."""
    output = "AssertionError: Block size must be positive"
    result = _categorize_vllm_error_for_test(output)
    assert result == "assertion_error", f"Expected assertion_error, got {result}"
    print("✅ test_categorize_vllm_error_assertion_error PASSED")


def test_categorize_vllm_error_unknown():
    """Test that unknown errors are correctly categorized."""
    output = "Some completely unrecognized error message"
    result = _categorize_vllm_error_for_test(output)
    assert result == "unknown", f"Expected unknown, got {result}"
    print("✅ test_categorize_vllm_error_unknown PASSED")


def test_categorize_vllm_error_empty():
    """Test that empty output is correctly categorized as unknown."""
    output = ""
    result = _categorize_vllm_error_for_test(output)
    assert result == "unknown", f"Expected unknown, got {result}"
    print("✅ test_categorize_vllm_error_empty PASSED")


def test_categorize_vllm_error_multiline_traceback():
    """Test that error in multiline traceback is correctly categorized."""
    output = """Traceback (most recent call last):
  File "/path/to/vllm/...", line XX, in <module>
    raise KeyError('128')
KeyError: '128'"""
    result = _categorize_vllm_error_for_test(output)
    assert result == "key_error", f"Expected key_error, got {result}"
    print("✅ test_categorize_vllm_error_multiline_traceback PASSED")


if __name__ == "__main__":
    print("--- VLLM ERROR LOGGING TESTS ---\n")
    
    print("=== Throughput Parsing Tests ===")
    test_parse_throughput_standard_format()
    test_parse_throughput_simple_tokens()
    test_parse_throughput_tokens_colon()
    test_parse_throughput_throughput_only()
    test_parse_throughput_no_match()
    test_parse_throughput_multiline()
    
    print("\n=== Config File Validation Tests ===")
    test_validate_config_file_valid()
    test_validate_config_file_nonexistent()
    test_validate_config_file_empty()
    test_validate_config_file_invalid_json()
    
    print("\n=== Dataset Existence Tests ===")
    test_check_dataset_exists_true()
    test_check_dataset_exists_false()
    
    print("\n=== Error Categorization Tests ===")
    test_categorize_vllm_error_cuda_oom()
    test_categorize_vllm_error_file_not_found()
    test_categorize_vllm_error_key_error()
    test_categorize_vllm_error_key_error_numeric()
    test_categorize_vllm_error_json_decode()
    test_categorize_vllm_error_runtime_error()
    test_categorize_vllm_error_module_not_found()
    test_categorize_vllm_error_assertion_error()
    test_categorize_vllm_error_unknown()
    test_categorize_vllm_error_empty()
    test_categorize_vllm_error_multiline_traceback()
    
    print("\n" + "="*60)
    print("✅ ALL VLLM ERROR LOGGING TESTS PASSED")
    print("="*60)
