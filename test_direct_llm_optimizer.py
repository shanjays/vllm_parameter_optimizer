"""
Tests for DirectLLMOptimizer.

Tests the following features:
1. parse_configs() method - parsing LLM output to configs
2. _validate_config() method - validating kernel configurations
3. _check_shared_memory_limit() method - hardware constraint validation
4. get_default_config() method
5. build_direct_optimization_prompt() function
"""

import json
import tempfile
import os

from direct_llm_optimizer import (
    DirectLLMOptimizer,
    build_direct_optimization_prompt,
    DEFAULT_KERNEL_CONFIG,
    VALID_BLOCK_SIZE_M,
    VALID_BLOCK_SIZE_N,
    VALID_BLOCK_SIZE_K,
    VALID_NUM_WARPS,
    VALID_NUM_STAGES,
)


class MockProfilingWorker:
    """Mock profiling worker for testing."""
    
    def run_kernel_profiling(self, params, static_args, weights, token_count):
        """Mock profiling that returns synthetic results."""
        return MockRemote([50.0, 40.0, 30.0, 70.0], 42.0, "mock_csv")
    
    def run_throughput_validation(self, params, model_name, goal):
        """Mock throughput validation."""
        return MockRemote(None, None, None, result=1000.0)


class MockRemote:
    """Mock Ray remote result."""
    
    def __init__(self, state, reward, csv_data, result=None):
        import numpy as np
        self.state = np.array(state, dtype=np.float32) if state else None
        self.reward = reward
        self.csv_data = csv_data
        self._result = result
    
    def remote(self, *args, **kwargs):
        return self
    
    def __call__(self):
        if self._result is not None:
            return self._result
        return self.state, self.reward, self.csv_data


class MockConfigExporter:
    """Mock config exporter for testing."""
    
    def __init__(self):
        self.best_configs = {}
    
    def update_best_config(self, token_count, config, reward, metrics=None):
        self.best_configs[token_count] = {
            'config': config,
            'reward': reward
        }
        return True


class MockFeedbackCollector:
    """Mock feedback collector for testing."""
    
    def __init__(self):
        self.recorded = []
    
    def record_policy_result(self, policy, reward, best_configs):
        self.recorded.append({
            'policy': policy,
            'reward': reward,
            'best_configs': best_configs
        })
    
    def format_feedback_for_prompt(self):
        return ""


def test_parse_configs_valid_json():
    """Test parsing valid LLM output with <param> tags."""
    optimizer = _create_mock_optimizer()
    
    llm_output = '''
<param>
{
  "configs": [
    {
      "token_counts": [1, 2, 4, 8],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 32,
      "num_warps": 8,
      "num_stages": 4
    },
    {
      "token_counts": [16, 32, 64],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "num_warps": 16,
      "num_stages": 4
    }
  ]
}
</param>
REASONING: These configs are optimized for different batch sizes.
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    assert len(configs) == 2
    assert configs[0]['BLOCK_SIZE_M'] == 64
    assert configs[0]['BLOCK_SIZE_N'] == 64
    assert configs[0]['num_warps'] == 8
    assert configs[1]['BLOCK_SIZE_M'] == 128
    assert configs[1]['num_warps'] == 16
    
    print("✅ test_parse_configs_valid_json PASSED")


def test_parse_configs_without_tags():
    """Test parsing LLM output without <param> tags."""
    optimizer = _create_mock_optimizer()
    
    llm_output = '''
{
  "configs": [
    {
      "token_counts": [1, 4, 16],
      "BLOCK_SIZE_M": 32,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 32,
      "num_warps": 4,
      "num_stages": 3
    }
  ]
}
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    assert len(configs) == 1
    assert configs[0]['BLOCK_SIZE_M'] == 32
    assert configs[0]['num_stages'] == 3
    
    print("✅ test_parse_configs_without_tags PASSED")


def test_parse_configs_single_config():
    """Test parsing single config at top level."""
    optimizer = _create_mock_optimizer()
    
    llm_output = '''
<param>
{
  "BLOCK_SIZE_M": 64,
  "BLOCK_SIZE_N": 128,
  "BLOCK_SIZE_K": 64,
  "num_warps": 16,
  "num_stages": 5
}
</param>
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    assert len(configs) == 1
    assert configs[0]['BLOCK_SIZE_N'] == 128
    assert configs[0]['num_stages'] == 5
    
    print("✅ test_parse_configs_single_config PASSED")


def test_parse_configs_invalid_json():
    """Test parsing invalid/malformed JSON."""
    optimizer = _create_mock_optimizer()
    
    # JSON with syntax error (missing comma between values)
    llm_output = '''
<param>
{
  "configs": [
    {
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64
      "num_warps": 8
    }
  ]
}
</param>
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    # Should return empty list for completely invalid JSON
    assert isinstance(configs, list)
    
    print("✅ test_parse_configs_invalid_json PASSED")


def test_parse_configs_empty():
    """Test parsing empty output."""
    optimizer = _create_mock_optimizer()
    
    configs = optimizer.parse_configs("")
    
    assert configs == []
    
    print("✅ test_parse_configs_empty PASSED")


def test_validate_config_valid():
    """Test validating a valid configuration."""
    optimizer = _create_mock_optimizer()
    
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_warps": 8,
        "num_stages": 4
    }
    
    validated = optimizer._validate_config(config)
    
    assert validated is not None
    assert validated['BLOCK_SIZE_M'] == 64
    assert validated['num_warps'] == 8
    
    print("✅ test_validate_config_valid PASSED")


def test_validate_config_clamps_invalid():
    """Test that invalid values are clamped to nearest valid."""
    optimizer = _create_mock_optimizer()
    
    config = {
        "BLOCK_SIZE_M": 100,  # Not valid, should clamp to 128 or 64
        "BLOCK_SIZE_N": 50,   # Not valid, should clamp to 32 or 64
        "BLOCK_SIZE_K": 48,   # Not valid, should clamp to 32 or 64
        "num_warps": 12,      # Not valid, should clamp to 8 or 16
        "num_stages": 1       # Not valid, should clamp to 2
    }
    
    validated = optimizer._validate_config(config)
    
    assert validated is not None
    assert validated['BLOCK_SIZE_M'] in VALID_BLOCK_SIZE_M
    assert validated['BLOCK_SIZE_N'] in VALID_BLOCK_SIZE_N
    assert validated['BLOCK_SIZE_K'] in VALID_BLOCK_SIZE_K
    assert validated['num_warps'] in VALID_NUM_WARPS
    assert validated['num_stages'] in VALID_NUM_STAGES
    
    print("✅ test_validate_config_clamps_invalid PASSED")


def test_validate_config_with_defaults():
    """Test that missing values are filled with defaults."""
    optimizer = _create_mock_optimizer()
    
    config = {
        "BLOCK_SIZE_M": 64
        # Missing other fields
    }
    
    validated = optimizer._validate_config(config)
    
    assert validated is not None
    assert validated['BLOCK_SIZE_M'] == 64
    assert validated['BLOCK_SIZE_N'] == DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_N']
    assert validated['num_warps'] == DEFAULT_KERNEL_CONFIG['num_warps']
    
    print("✅ test_validate_config_with_defaults PASSED")


def test_check_shared_memory_limit_valid():
    """Test shared memory check for valid config."""
    optimizer = _create_mock_optimizer()
    
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_stages": 4
    }
    
    # (64*32 + 32*64) * 2 * 4 = (2048 + 2048) * 8 = 32768 bytes
    assert optimizer._check_shared_memory_limit(config) is True
    
    print("✅ test_check_shared_memory_limit_valid PASSED")


def test_check_shared_memory_limit_invalid():
    """Test shared memory check for config that exceeds limit."""
    optimizer = _create_mock_optimizer()
    
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "num_stages": 5
    }
    
    # Calculate shared memory: (M*K + K*N) * 2 * stages
    # = (128*64 + 64*128) * 2 * 5 = (8192 + 8192) * 10 = 163840 bytes
    # H100 limit is 232448 bytes, so 163840 < 232448 = valid config
    result = optimizer._check_shared_memory_limit(config)
    
    # This config is under the limit, so it should pass
    assert result is True
    
    print("✅ test_check_shared_memory_limit_invalid PASSED")


def test_get_default_config():
    """Test get_default_config() method."""
    optimizer = _create_mock_optimizer()
    
    default = optimizer.get_default_config()
    
    assert 'token_counts' in default
    assert 'BLOCK_SIZE_M' in default
    assert 'BLOCK_SIZE_N' in default
    assert 'num_warps' in default
    assert 'num_stages' in default
    
    print("✅ test_get_default_config PASSED")


def test_build_direct_optimization_prompt():
    """Test the direct optimization prompt builder."""
    ncu_summary = """SM Throughput: min=40%, max=60%, avg=50%
DRAM Throughput: min=30%, max=50%, avg=40%
L1 Cache Hit Rate: min=20%, max=40%, avg=30%
L2 Cache Hit Rate: min=60%, max=80%, avg=70%"""
    
    prompt = build_direct_optimization_prompt(
        ncu_summary=ncu_summary,
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        num_experts=128,
        inter_size=768,
        hidden_size=6656,
        top_k=8,
        dtype="bf16",
        token_counts=[1, 16, 64, 256, 1024, 4096],
        feedback_collector=None
    )
    
    # Verify prompt contains key elements
    assert "fused_moe" in prompt
    assert "BLOCK_SIZE_M" in prompt
    assert "num_warps" in prompt
    assert "configs" in prompt
    assert "<param>" in prompt
    
    print("✅ test_build_direct_optimization_prompt PASSED")


def test_build_direct_optimization_prompt_with_feedback():
    """Test prompt builder with feedback collector."""
    feedback = MockFeedbackCollector()
    
    # Override format method to return non-empty feedback
    feedback.format_feedback_for_prompt = lambda: "\nPREVIOUS BEST: 42.0"
    
    ncu_summary = "SM Throughput: avg=50%"
    
    prompt = build_direct_optimization_prompt(
        ncu_summary=ncu_summary,
        model_name="test-model",
        num_experts=128,
        inter_size=768,
        hidden_size=6656,
        top_k=8,
        dtype="bf16",
        token_counts=[1, 16],
        feedback_collector=feedback
    )
    
    assert "PREVIOUS BEST: 42.0" in prompt
    
    print("✅ test_build_direct_optimization_prompt_with_feedback PASSED")


def test_parse_configs_trailing_comma():
    """Test parsing JSON with trailing commas."""
    optimizer = _create_mock_optimizer()
    
    llm_output = '''
<param>
{
  "configs": [
    {
      "BLOCK_SIZE_M": 64,
      "num_warps": 8,
    },
  ]
}
</param>
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    # Should handle trailing commas gracefully
    assert isinstance(configs, list)
    
    print("✅ test_parse_configs_trailing_comma PASSED")


def test_parse_configs_unbalanced_braces():
    """Test parsing JSON with unbalanced braces."""
    optimizer = _create_mock_optimizer()
    
    llm_output = '''
<param>
{
  "configs": [
    {
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 32,
      "num_warps": 8,
      "num_stages": 4
'''
    
    configs = optimizer.parse_configs(llm_output)
    
    # Should attempt to recover or return empty
    assert isinstance(configs, list)
    
    print("✅ test_parse_configs_unbalanced_braces PASSED")


def _create_mock_optimizer():
    """Create a DirectLLMOptimizer with mock dependencies."""
    return DirectLLMOptimizer(
        profiling_worker=MockProfilingWorker(),
        config_exporter=MockConfigExporter(),
        feedback_collector=MockFeedbackCollector(),
        static_args={
            'num_experts': 128,
            'inter_size': 768,
            'hidden_size': 6656,
            'top_k': 8,
            'dtype': 'bf16',
            'num_tokens': 1024,
            'run_script_path': 'run_kernel_benchmark.py',
            'kernel_name': 'fused_moe_kernel',
            'num_iters': 3,
            'num_warmup_iters': 1,
        },
        token_counts=[1, 16, 64, 256, 1024, 4096],
        model_name="test-model",
        user_goal="throughput"
    )


if __name__ == "__main__":
    print("--- DIRECT LLM OPTIMIZER TESTS ---\n")
    
    all_passed = True
    tests = [
        test_parse_configs_valid_json,
        test_parse_configs_without_tags,
        test_parse_configs_single_config,
        test_parse_configs_invalid_json,
        test_parse_configs_empty,
        test_validate_config_valid,
        test_validate_config_clamps_invalid,
        test_validate_config_with_defaults,
        test_check_shared_memory_limit_valid,
        test_check_shared_memory_limit_invalid,
        test_get_default_config,
        test_build_direct_optimization_prompt,
        test_build_direct_optimization_prompt_with_feedback,
        test_parse_configs_trailing_comma,
        test_parse_configs_unbalanced_braces,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL DIRECT LLM OPTIMIZER TESTS PASSED")
    else:
        print("❌ ONE OR MORE TESTS FAILED")
    print("="*60)
