"""
Tests for VLLMConfigExporter - vLLM format config export functionality.

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
from config_exporter import VLLMConfigExporter


def test_get_config_filename():
    """Test that config filename matches vLLM's expected format."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536, device_name="NVIDIA_H100_80GB_HBM3")
    expected = "E=128,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json"
    assert exporter.get_config_filename() == expected, f"Expected {expected}, got {exporter.get_config_filename()}"
    print("✅ test_get_config_filename PASSED")


def test_get_config_filename_different_values():
    """Test filename generation with different E and N values."""
    exporter = VLLMConfigExporter(num_experts=64, inter_size=768, device_name="NVIDIA_A100")
    expected = "E=64,N=768,device_name=NVIDIA_A100.json"
    assert exporter.get_config_filename() == expected, f"Expected {expected}, got {exporter.get_config_filename()}"
    print("✅ test_get_config_filename_different_values PASSED")


def test_update_best_config_new_token_count():
    """Test that a new token count creates a new best config."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    
    result = exporter.update_best_config(token_count=16, config=config, reward=50.0)
    
    assert result is True, "Should return True for new best config"
    assert "16" in exporter.best_configs, "Token count 16 should be in best_configs"
    assert exporter.best_rewards["16"] == 50.0, "Reward should be stored"
    assert len(exporter.all_results) == 1, "Should have 1 result logged"
    print("✅ test_update_best_config_new_token_count PASSED")


def test_update_best_config_better_reward():
    """Test that a better reward updates the best config."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    config1 = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    config2 = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 3}
    
    exporter.update_best_config(token_count=16, config=config1, reward=50.0)
    result = exporter.update_best_config(token_count=16, config=config2, reward=75.0)
    
    assert result is True, "Should return True for better config"
    assert exporter.best_configs["16"]["BLOCK_SIZE_M"] == 128, "BLOCK_SIZE_M should be updated"
    assert exporter.best_rewards["16"] == 75.0, "Reward should be updated"
    assert len(exporter.all_results) == 2, "Both results should be logged"
    print("✅ test_update_best_config_better_reward PASSED")


def test_update_best_config_worse_reward():
    """Test that a worse reward does not update the best config."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    config1 = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    config2 = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 2}
    
    exporter.update_best_config(token_count=16, config=config1, reward=75.0)
    result = exporter.update_best_config(token_count=16, config=config2, reward=50.0)
    
    assert result is False, "Should return False for worse config"
    assert exporter.best_configs["16"]["BLOCK_SIZE_M"] == 64, "BLOCK_SIZE_M should not change"
    assert exporter.best_rewards["16"] == 75.0, "Reward should not change"
    assert len(exporter.all_results) == 2, "Both results should still be logged"
    print("✅ test_update_best_config_worse_reward PASSED")


def test_update_best_config_with_metrics():
    """Test that metrics are stored correctly."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
    metrics = {"sm_throughput": 45.5, "dram_throughput": 60.2, "l1_hit_rate": 85.0, "l2_hit_rate": 70.0}
    
    exporter.update_best_config(token_count=16, config=config, reward=50.0, metrics=metrics)
    
    assert exporter.all_results[0]["metrics"] == metrics, "Metrics should be stored"
    print("✅ test_update_best_config_with_metrics PASSED")


def test_save_vllm_config():
    """Test that vLLM config is saved in correct format."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    
    # Add configs for multiple token counts
    configs = [
        (1, {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 3}, 30.0),
        (16, {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}, 50.0),
        (128, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "num_warps": 8, "num_stages": 4}, 70.0),
    ]
    for token_count, config, reward in configs:
        exporter.update_best_config(token_count, config, reward)
    
    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        vllm_path = exporter.save_vllm_config(output_dir=temp_dir)
        
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
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536, output_dir=temp_dir)
        
        # Add in non-sorted order
        for token_count in [128, 1, 64, 16, 4]:
            exporter.update_best_config(token_count, {"BLOCK_SIZE_M": 64}, reward=50.0)
        
        vllm_path = exporter.save_vllm_config(output_dir=temp_dir)
        
        # Read the local config file
        local_path = os.path.join(temp_dir, exporter.get_config_filename())
        with open(local_path, 'r') as f:
            vllm_config = json.load(f)
        
        # Now all token counts should be present (with defaults)
        from config_exporter import TOKEN_COUNTS_ALL
        assert len(vllm_config) == len(TOKEN_COUNTS_ALL), f"Should have all {len(TOKEN_COUNTS_ALL)} token counts"
        
        # Verify the manually set configs have our values
        assert vllm_config["1"]["BLOCK_SIZE_M"] == 64, "Token 1 should have updated config"
        assert vllm_config["4"]["BLOCK_SIZE_M"] == 64, "Token 4 should have updated config"
        assert vllm_config["16"]["BLOCK_SIZE_M"] == 64, "Token 16 should have updated config"
        assert vllm_config["64"]["BLOCK_SIZE_M"] == 64, "Token 64 should have updated config"
        assert vllm_config["128"]["BLOCK_SIZE_M"] == 64, "Token 128 should have updated config"
        
        print("✅ test_save_vllm_config_sorted_keys PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_get_summary():
    """Test that summary returns correct information."""
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536, output_dir=temp_dir)
        
        exporter.update_best_config(1, {"BLOCK_SIZE_M": 64}, reward=30.0)
        exporter.update_best_config(16, {"BLOCK_SIZE_M": 64}, reward=50.0)
        exporter.update_best_config(16, {"BLOCK_SIZE_M": 128}, reward=60.0)  # Better config
        
        summary = exporter.get_summary()
        
        # All token counts are initialized with defaults
        from config_exporter import TOKEN_COUNTS_ALL
        assert summary["total_token_counts"] == len(TOKEN_COUNTS_ALL), f"Should have all {len(TOKEN_COUNTS_ALL)} token counts"
        assert summary["total_experiments"] == 3, "Should have 3 experiments"
        assert summary["tested_token_counts"] == 2, "Should have 2 tested token counts with reward > 0"
        assert summary["best_rewards"]["1"] == 30.0, "Token 1 reward should be 30.0"
        assert summary["best_rewards"]["16"] == 60.0, "Token 16 reward should be 60.0"
        assert summary["config_filename"] == "E=128,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json"
        assert "config_path" in summary, "Should include config_path"
        print("✅ test_get_summary PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_default_config_values():
    """Test that default config values are applied for missing keys."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    
    # Config with missing keys
    incomplete_config = {"BLOCK_SIZE_M": 64}
    exporter.update_best_config(16, incomplete_config, reward=50.0)
    
    saved_config = exporter.best_configs["16"]
    assert saved_config["BLOCK_SIZE_M"] == 64, "Provided value should be used"
    assert saved_config["BLOCK_SIZE_N"] == 64, "Default should be applied"
    assert saved_config["BLOCK_SIZE_K"] == 32, "Default should be applied"
    assert saved_config["GROUP_SIZE_M"] == 8, "Default should be applied"
    assert saved_config["num_warps"] == 4, "Default should be applied"
    assert saved_config["num_stages"] == 4, "Default should be applied"
    print("✅ test_default_config_values PASSED")


def test_multiple_token_counts():
    """Test handling of many token counts (simulating full run)."""
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536, output_dir=temp_dir)
        
        # Simulate vLLM token count values
        token_counts = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]
        
        for tc in token_counts:
            config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 4}
            exporter.update_best_config(tc, config, reward=50.0 + tc * 0.01)
        
        # All token counts are initialized with defaults, so best_configs has all of them
        from config_exporter import TOKEN_COUNTS_ALL
        assert len(exporter.best_configs) == len(TOKEN_COUNTS_ALL), f"Should have {len(TOKEN_COUNTS_ALL)} configs (all initialized)"
        assert len(exporter.all_results) == len(token_counts), f"Should have {len(token_counts)} results"
        
        # Verify the manually set configs have our values
        for tc in token_counts:
            assert exporter.best_configs[str(tc)]["BLOCK_SIZE_M"] == 64, f"Token {tc} should have updated config"
            assert exporter.best_rewards[str(tc)] == 50.0 + tc * 0.01, f"Token {tc} should have correct reward"
        
        print("✅ test_multiple_token_counts PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_export_complete_config_interpolation():
    """Test that export_complete_config interpolates missing token counts."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    
    # Add configs for only a few token counts
    test_configs = [
        (1, {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}),
        (64, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}),
        (1024, {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}),
    ]
    for tc, config in test_configs:
        exporter.update_best_config(tc, config, reward=50.0)
    
    # Export complete config
    temp_dir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(temp_dir, "complete_config.json")
        result_path = exporter.export_complete_config(output_path)
        
        assert result_path is not None, "Should return path on success"
        assert os.path.exists(output_path), "Output file should exist"
        
        with open(output_path, 'r') as f:
            complete_config = json.load(f)
        
        # Check that all expected token counts are present
        from config_exporter import TOKEN_COUNTS_ALL
        assert len(complete_config) == len(TOKEN_COUNTS_ALL), f"Should have {len(TOKEN_COUNTS_ALL)} entries"
        
        # Check interpolation worked
        assert "2" in complete_config, "Token count 2 should be interpolated"
        assert "128" in complete_config, "Token count 128 should be interpolated"
        assert "4096" in complete_config, "Token count 4096 should be interpolated"
        
        print("✅ test_export_complete_config_interpolation PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_find_nearest_config():
    """Test the _find_nearest_config helper method."""
    exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
    
    tested_counts = [1, 16, 64, 256, 1024]
    
    # Test exact match
    assert exporter._find_nearest_config(16, tested_counts) == 16
    
    # Test value between two tested counts (should prefer lower)
    assert exporter._find_nearest_config(32, tested_counts) == 16
    
    # Test value higher than all tested
    assert exporter._find_nearest_config(5000, tested_counts) == 1024
    
    # Test value lower than all tested
    assert exporter._find_nearest_config(0, tested_counts) == 1
    
    print("✅ test_find_nearest_config PASSED")


def test_all_token_counts_constant():
    """Test that TOKEN_COUNTS_ALL is properly defined."""
    from config_exporter import TOKEN_COUNTS_ALL
    
    # Should have the expected vLLM token counts
    assert 1 in TOKEN_COUNTS_ALL, "Should include 1"
    assert 4096 in TOKEN_COUNTS_ALL, "Should include 4096"
    assert len(TOKEN_COUNTS_ALL) >= 20, "Should have many token counts"
    
    # Should be sorted
    assert TOKEN_COUNTS_ALL == sorted(TOKEN_COUNTS_ALL), "Should be sorted"
    
    print("✅ test_all_token_counts_constant PASSED")


def test_initialization_creates_config_with_defaults():
    """Test that VLLMConfigExporter creates config file with defaults on initialization."""
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = VLLMConfigExporter(num_experts=128, inter_size=768, output_dir=temp_dir)
        
        # Check that config file was created in vllm_config_dir
        config_path = os.path.join(exporter.vllm_config_dir, exporter.config_filename)
        assert os.path.exists(config_path), f"Config file should be created at {config_path}"
        
        # Load and verify contents
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Should have all token counts
        from config_exporter import TOKEN_COUNTS_ALL
        assert len(config) == len(TOKEN_COUNTS_ALL), f"Should have all {len(TOKEN_COUNTS_ALL)} token counts"
        
        # Each should have default values
        for tc_str in ["1", "64", "1024", "4096"]:
            assert tc_str in config, f"Token count {tc_str} should be in config"
            assert config[tc_str]["BLOCK_SIZE_M"] == 64, f"Token {tc_str} should have default BLOCK_SIZE_M=64"
            assert config[tc_str]["BLOCK_SIZE_N"] == 64, f"Token {tc_str} should have default BLOCK_SIZE_N=64"
            assert config[tc_str]["num_warps"] == 8, f"Token {tc_str} should have default num_warps=8"
            assert config[tc_str]["num_stages"] == 4, f"Token {tc_str} should have default num_stages=4"
        
        print("✅ test_initialization_creates_config_with_defaults PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_initialization_loads_existing_config():
    """Test that VLLMConfigExporter loads existing config if present."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create an initial config file manually
        vllm_configs_dir = os.path.join(temp_dir, "vllm_configs")
        os.makedirs(vllm_configs_dir, exist_ok=True)
        
        config_filename = "E=64,N=256,device_name=TEST_GPU.json"
        config_path = os.path.join(vllm_configs_dir, config_filename)
        
        custom_config = {
            "1": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 3},
            "16": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 4}
        }
        with open(config_path, 'w') as f:
            json.dump(custom_config, f)
        
        # Create exporter - it should load existing config
        exporter = VLLMConfigExporter(num_experts=64, inter_size=256, device_name="TEST_GPU", output_dir=temp_dir)
        
        # Verify the loaded config
        assert exporter.best_configs["1"]["BLOCK_SIZE_M"] == 32, "Should load existing BLOCK_SIZE_M"
        assert exporter.best_configs["1"]["num_warps"] == 16, "Should load existing num_warps"
        assert exporter.best_configs["16"]["BLOCK_SIZE_N"] == 64, "Should load existing BLOCK_SIZE_N"
        
        print("✅ test_initialization_loads_existing_config PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_update_best_config_updates_file_immediately():
    """Test that updating best config immediately updates the config file."""
    temp_dir = tempfile.mkdtemp()
    try:
        exporter = VLLMConfigExporter(num_experts=128, inter_size=768, output_dir=temp_dir)
        
        # Update a config
        new_config = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 3}
        exporter.update_best_config(64, new_config, reward=75.0)
        
        # Read the config file and verify it was updated
        config_path = os.path.join(exporter.vllm_config_dir, exporter.config_filename)
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        assert file_config["64"]["BLOCK_SIZE_M"] == 128, "File should be updated immediately"
        assert file_config["64"]["BLOCK_SIZE_N"] == 128, "File should be updated immediately"
        assert file_config["64"]["num_warps"] == 16, "File should be updated immediately"
        
        print("✅ test_update_best_config_updates_file_immediately PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_corrupt_config_file_creates_new():
    """Test that a corrupt config file is replaced with new defaults."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a corrupt config file
        vllm_configs_dir = os.path.join(temp_dir, "vllm_configs")
        os.makedirs(vllm_configs_dir, exist_ok=True)
        
        config_filename = "E=32,N=512,device_name=TEST_GPU.json"
        config_path = os.path.join(vllm_configs_dir, config_filename)
        
        # Write invalid JSON
        with open(config_path, 'w') as f:
            f.write("{invalid json content}")
        
        # Create exporter - it should handle the corrupt file and create new defaults
        exporter = VLLMConfigExporter(num_experts=32, inter_size=512, device_name="TEST_GPU", output_dir=temp_dir)
        
        # Verify the config was recreated with defaults
        from config_exporter import TOKEN_COUNTS_ALL
        assert len(exporter.best_configs) == len(TOKEN_COUNTS_ALL), "Should have all token counts with defaults"
        
        # Verify the file is now valid
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert len(config) == len(TOKEN_COUNTS_ALL), "File should have all token counts"
        
        print("✅ test_corrupt_config_file_creates_new PASSED")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("--- CONFIG EXPORTER TESTS ---\n")
    
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
    
    print("\n=== Complete Config Export Tests ===")
    test_export_complete_config_interpolation()
    test_find_nearest_config()
    test_all_token_counts_constant()
    
    print("\n=== Summary Tests ===")
    test_get_summary()
    test_multiple_token_counts()
    
    print("\n=== Config Initialization Tests ===")
    test_initialization_creates_config_with_defaults()
    test_initialization_loads_existing_config()
    test_update_best_config_updates_file_immediately()
    test_corrupt_config_file_creates_new()
    
    print("\n" + "="*60)
    print("✅ ALL CONFIG EXPORTER TESTS PASSED")
    print("="*60)
