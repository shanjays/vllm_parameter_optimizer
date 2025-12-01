"""
Tests for VLLMConfigSaver - vLLM format config export functionality.

These tests verify:
1. Config filename generation matches vLLM's expected format
2. Best config tracking and updating works correctly
3. Config saving in vLLM format creates correct JSON structure
4. Multi-token count testing integration works
"""

import json
import os
import tempfile
import shutil
from config_saver import VLLMConfigSaver


def test_get_config_filename():
    """Test that config filename matches vLLM's expected format."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536, device_name="NVIDIA_H100_80GB_HBM3")
    expected = "E=128,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json"
    assert saver.get_config_filename() == expected, f"Expected {expected}, got {saver.get_config_filename()}"
    print("✅ test_get_config_filename PASSED")


def test_get_config_filename_different_values():
    """Test filename generation with different E and N values."""
    saver = VLLMConfigSaver(num_experts=64, inter_size=768, device_name="NVIDIA_A100")
    expected = "E=64,N=768,device_name=NVIDIA_A100.json"
    assert saver.get_config_filename() == expected, f"Expected {expected}, got {saver.get_config_filename()}"
    print("✅ test_get_config_filename_different_values PASSED")


def test_update_best_config_new_token_count():
    """Test that a new token count creates a new best config."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    
    result = saver.update_best_config(token_count=16, config=config, reward=50.0)
    
    assert result is True, "Should return True for new best config"
    assert "16" in saver.best_configs, "Token count 16 should be in best_configs"
    assert saver.best_rewards["16"] == 50.0, "Reward should be stored"
    assert len(saver.all_results) == 1, "Should have 1 result logged"
    print("✅ test_update_best_config_new_token_count PASSED")


def test_update_best_config_better_reward():
    """Test that a better reward updates the best config."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    config1 = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    config2 = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 3}
    
    saver.update_best_config(token_count=16, config=config1, reward=50.0)
    result = saver.update_best_config(token_count=16, config=config2, reward=75.0)
    
    assert result is True, "Should return True for better config"
    assert saver.best_configs["16"]["BLOCK_SIZE_M"] == 128, "BLOCK_SIZE_M should be updated"
    assert saver.best_rewards["16"] == 75.0, "Reward should be updated"
    assert len(saver.all_results) == 2, "Both results should be logged"
    print("✅ test_update_best_config_better_reward PASSED")


def test_update_best_config_worse_reward():
    """Test that a worse reward does not update the best config."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    config1 = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    config2 = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 2}
    
    saver.update_best_config(token_count=16, config=config1, reward=75.0)
    result = saver.update_best_config(token_count=16, config=config2, reward=50.0)
    
    assert result is False, "Should return False for worse config"
    assert saver.best_configs["16"]["BLOCK_SIZE_M"] == 64, "BLOCK_SIZE_M should not change"
    assert saver.best_rewards["16"] == 75.0, "Reward should not change"
    assert len(saver.all_results) == 2, "Both results should still be logged"
    print("✅ test_update_best_config_worse_reward PASSED")


def test_update_best_config_with_metrics():
    """Test that metrics are stored correctly."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    metrics = {"sm_throughput": 45.5, "dram_throughput": 60.2, "l1_hit_rate": 85.0, "l2_hit_rate": 70.0}
    
    saver.update_best_config(token_count=16, config=config, reward=50.0, metrics=metrics)
    
    assert saver.all_results[0]["metrics"] == metrics, "Metrics should be stored"
    print("✅ test_update_best_config_with_metrics PASSED")


def test_save_vllm_config():
    """Test that vLLM config is saved in correct format."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    
    # Add configs for multiple token counts
    configs = [
        (1, {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3}, 30.0),
        (16, {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}, 50.0),
        (128, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 4}, 70.0),
    ]
    for token_count, config, reward in configs:
        saver.update_best_config(token_count, config, reward)
    
    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        vllm_path = saver.save_vllm_config(output_dir=temp_dir)
        
        # Check vLLM format file exists and has correct structure
        assert os.path.exists(vllm_path), "vLLM config file should exist"
        with open(vllm_path, 'r') as f:
            vllm_config = json.load(f)
        
        assert "1" in vllm_config, "Token count 1 should be in config"
        assert "16" in vllm_config, "Token count 16 should be in config"
        assert "128" in vllm_config, "Token count 128 should be in config"
        
        # Check config structure
        assert vllm_config["1"]["BLOCK_SIZE_M"] == 16, "BLOCK_SIZE_M should be correct"
        assert vllm_config["16"]["BLOCK_SIZE_N"] == 64, "BLOCK_SIZE_N should be correct"
        assert "reward" not in vllm_config["1"], "vLLM format should not contain rewards"
        
        # Check detailed file exists
        detailed_path = os.path.join(temp_dir, "best_configs_detailed.json")
        assert os.path.exists(detailed_path), "Detailed config file should exist"
        with open(detailed_path, 'r') as f:
            detailed = json.load(f)
        assert "metadata" in detailed, "Detailed config should have metadata"
        assert detailed["metadata"]["num_experts"] == 128, "Metadata should have correct values"
        assert "1" in detailed["best_configs"], "Detailed config should have token counts"
        assert detailed["best_configs"]["1"]["reward"] == 30.0, "Detailed config should have rewards"
        
        # Check all results file exists
        all_results_path = os.path.join(temp_dir, "all_results.json")
        assert os.path.exists(all_results_path), "All results file should exist"
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)
        assert len(all_results) == 3, "Should have 3 results logged"
        
        print("✅ test_save_vllm_config PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_save_vllm_config_sorted_keys():
    """Test that token counts are sorted numerically in output."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    
    # Add in non-sorted order
    for token_count in [128, 1, 64, 16, 4]:
        saver.update_best_config(token_count, {"BLOCK_SIZE_M": 64}, reward=50.0)
    
    temp_dir = tempfile.mkdtemp()
    try:
        vllm_path = saver.save_vllm_config(output_dir=temp_dir)
        with open(vllm_path, 'r') as f:
            vllm_config = json.load(f)
        
        keys = list(vllm_config.keys())
        expected = ["1", "4", "16", "64", "128"]
        assert keys == expected, f"Keys should be numerically sorted: {keys}"
        print("✅ test_save_vllm_config_sorted_keys PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_get_summary():
    """Test that summary returns correct information."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    
    saver.update_best_config(1, {"BLOCK_SIZE_M": 64}, reward=30.0)
    saver.update_best_config(16, {"BLOCK_SIZE_M": 64}, reward=50.0)
    saver.update_best_config(16, {"BLOCK_SIZE_M": 128}, reward=60.0)  # Better config
    
    summary = saver.get_summary()
    
    assert summary["total_token_counts"] == 2, "Should have 2 token counts"
    assert summary["total_experiments"] == 3, "Should have 3 experiments"
    assert summary["best_rewards"]["1"] == 30.0, "Token 1 reward should be 30.0"
    assert summary["best_rewards"]["16"] == 60.0, "Token 16 reward should be 60.0"
    assert summary["config_filename"] == "E=128,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json"
    print("✅ test_get_summary PASSED")


def test_default_config_values():
    """Test that default config values are applied for missing keys."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    
    # Config with missing keys
    incomplete_config = {"BLOCK_SIZE_M": 64}
    saver.update_best_config(16, incomplete_config, reward=50.0)
    
    saved_config = saver.best_configs["16"]
    assert saved_config["BLOCK_SIZE_M"] == 64, "Provided value should be used"
    assert saved_config["BLOCK_SIZE_N"] == 64, "Default should be applied"
    assert saved_config["BLOCK_SIZE_K"] == 32, "Default should be applied"
    assert saved_config["GROUP_SIZE_M"] == 8, "Default should be applied"
    assert saved_config["num_warps"] == 4, "Default should be applied"
    assert saved_config["num_stages"] == 4, "Default should be applied"
    print("✅ test_default_config_values PASSED")


def test_multiple_token_counts():
    """Test handling of many token counts (simulating full run)."""
    saver = VLLMConfigSaver(num_experts=128, inter_size=1536)
    
    # Simulate vLLM token count values
    token_counts = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]
    
    for tc in token_counts:
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
        saver.update_best_config(tc, config, reward=50.0 + tc * 0.01)
    
    assert len(saver.best_configs) == len(token_counts), f"Should have {len(token_counts)} configs"
    assert len(saver.all_results) == len(token_counts), f"Should have {len(token_counts)} results"
    print("✅ test_multiple_token_counts PASSED")


if __name__ == "__main__":
    print("--- CONFIG SAVER TESTS ---\n")
    
    print("=== Filename Generation Tests ===")
    test_get_config_filename()
    test_get_config_filename_different_values()
    
    print("\n=== Best Config Update Tests ===")
    test_update_best_config_new_token_count()
    test_update_best_config_better_reward()
    test_update_best_config_worse_reward()
    test_update_best_config_with_metrics()
    test_default_config_values()
    
    print("\n=== Config Saving Tests ===")
    test_save_vllm_config()
    test_save_vllm_config_sorted_keys()
    
    print("\n=== Summary Tests ===")
    test_get_summary()
    test_multiple_token_counts()
    
    print("\n" + "="*60)
    print("✅ ALL CONFIG SAVER TESTS PASSED")
    print("="*60)
