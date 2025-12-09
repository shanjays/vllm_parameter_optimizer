"""
Tests for GPU Presets Module

These tests verify:
1. GPUPreset dataclass creation
2. GPU_PRESETS dictionary structure
3. get_gpu_preset() function
4. list_gpu_presets() function
5. Conversion to ThermalConfig
6. Conversion to GPU config dict
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_param_optimizer.gpu_presets import (
    GPUPreset,
    GPU_PRESETS,
    DEFAULT_GPU_PRESET,
    get_gpu_preset,
    list_gpu_presets,
    print_gpu_presets
)
from server_param_optimizer.thermal_monitor import ThermalConfig


# ============================================================
# GPUPreset Tests
# ============================================================

def test_gpu_preset_creation():
    """Test GPUPreset dataclass creation."""
    preset = GPUPreset(
        name="Test GPU",
        short_name="test-gpu",
        memory_gb=40.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=64,
        architecture="TestArch",
        memory_type="HBM2e"
    )
    
    assert preset.name == "Test GPU"
    assert preset.short_name == "test-gpu"
    assert preset.memory_gb == 40.0
    assert preset.tdp_watts == 400.0
    assert preset.max_safe_temp == 83.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 64
    assert preset.architecture == "TestArch"
    assert preset.memory_type == "HBM2e"
    
    print("✅ test_gpu_preset_creation PASSED")


def test_gpu_preset_to_thermal_config():
    """Test conversion from GPUPreset to ThermalConfig."""
    preset = GPUPreset(
        name="Test GPU",
        short_name="test-gpu",
        memory_gb=40.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=64,
        architecture="TestArch",
        memory_type="HBM2e"
    )
    
    thermal_config = preset.to_thermal_config()
    
    assert isinstance(thermal_config, ThermalConfig)
    assert thermal_config.max_safe_temp == 83.0
    assert thermal_config.target_sustained_temp == 75.0
    assert thermal_config.warning_temp == 80.0
    assert thermal_config.max_power == 400.0
    assert thermal_config.total_memory_gb == 40.0
    assert thermal_config.gpu_name == "Test GPU"
    
    print("✅ test_gpu_preset_to_thermal_config PASSED")


def test_gpu_preset_to_gpu_config():
    """Test conversion from GPUPreset to GPU config dict."""
    preset = GPUPreset(
        name="Test GPU",
        short_name="test-gpu",
        memory_gb=40.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=64,
        architecture="TestArch",
        memory_type="HBM2e"
    )
    
    gpu_config = preset.to_gpu_config()
    
    assert isinstance(gpu_config, dict)
    assert gpu_config['name'] == "Test GPU"
    assert gpu_config['memory_gb'] == 40.0
    assert gpu_config['tdp_watts'] == 400.0
    assert gpu_config['max_safe_temp'] == 83.0
    assert gpu_config['target_sustained_temp'] == 75.0
    assert gpu_config['recommended_max_seqs'] == 64
    assert gpu_config['architecture'] == "TestArch"
    assert gpu_config['memory_type'] == "HBM2e"
    
    print("✅ test_gpu_preset_to_gpu_config PASSED")


def test_gpu_preset_to_dict():
    """Test conversion from GPUPreset to dict."""
    preset = GPUPreset(
        name="Test GPU",
        short_name="test-gpu",
        memory_gb=40.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=64,
        architecture="TestArch",
        memory_type="HBM2e"
    )
    
    d = preset.to_dict()
    
    assert isinstance(d, dict)
    assert d['name'] == "Test GPU"
    assert d['short_name'] == "test-gpu"
    assert d['memory_gb'] == 40.0
    
    print("✅ test_gpu_preset_to_dict PASSED")


# ============================================================
# GPU_PRESETS Tests
# ============================================================

def test_gpu_presets_structure():
    """Test that GPU_PRESETS dictionary is properly structured."""
    assert isinstance(GPU_PRESETS, dict)
    assert len(GPU_PRESETS) > 0
    
    # Check required presets exist
    required_presets = ['a100-40gb', 'a100-80gb', 'h100-sxm', 'h100-pcie', 'h100-nvl']
    for preset_name in required_presets:
        assert preset_name in GPU_PRESETS, f"Missing required preset: {preset_name}"
        assert isinstance(GPU_PRESETS[preset_name], GPUPreset)
    
    # Check aliases
    assert 'a100' in GPU_PRESETS
    assert 'h100' in GPU_PRESETS
    assert GPU_PRESETS['a100'] == GPU_PRESETS['a100-40gb']
    assert GPU_PRESETS['h100'] == GPU_PRESETS['h100-pcie']
    
    print("✅ test_gpu_presets_structure PASSED")


def test_a100_40gb_preset():
    """Test A100 40GB preset has correct values."""
    preset = GPU_PRESETS['a100-40gb']
    
    assert preset.name == "NVIDIA A100 40GB"
    assert preset.short_name == "a100-40gb"
    assert preset.memory_gb == 40.0
    assert preset.tdp_watts == 400.0
    assert preset.max_safe_temp == 83.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 64
    assert preset.architecture == "Ampere"
    assert preset.memory_type == "HBM2e"
    
    print("✅ test_a100_40gb_preset PASSED")


def test_a100_80gb_preset():
    """Test A100 80GB preset has correct values."""
    preset = GPU_PRESETS['a100-80gb']
    
    assert preset.name == "NVIDIA A100 80GB"
    assert preset.short_name == "a100-80gb"
    assert preset.memory_gb == 80.0
    assert preset.tdp_watts == 400.0
    assert preset.max_safe_temp == 83.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 128
    assert preset.architecture == "Ampere"
    assert preset.memory_type == "HBM2e"
    
    print("✅ test_a100_80gb_preset PASSED")


def test_h100_sxm_preset():
    """Test H100 SXM preset has correct values."""
    preset = GPU_PRESETS['h100-sxm']
    
    assert preset.name == "NVIDIA H100 SXM"
    assert preset.short_name == "h100-sxm"
    assert preset.memory_gb == 80.0
    assert preset.tdp_watts == 700.0
    assert preset.max_safe_temp == 85.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 256
    assert preset.architecture == "Hopper"
    assert preset.memory_type == "HBM3"
    
    print("✅ test_h100_sxm_preset PASSED")


def test_h100_pcie_preset():
    """Test H100 PCIe preset has correct values."""
    preset = GPU_PRESETS['h100-pcie']
    
    assert preset.name == "NVIDIA H100 PCIe"
    assert preset.short_name == "h100-pcie"
    assert preset.memory_gb == 80.0
    assert preset.tdp_watts == 350.0
    assert preset.max_safe_temp == 85.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 256
    assert preset.architecture == "Hopper"
    assert preset.memory_type == "HBM3"
    
    print("✅ test_h100_pcie_preset PASSED")


def test_h100_nvl_preset():
    """Test H100 NVL preset has correct values."""
    preset = GPU_PRESETS['h100-nvl']
    
    assert preset.name == "NVIDIA H100 NVL"
    assert preset.short_name == "h100-nvl"
    assert preset.memory_gb == 94.0
    assert preset.tdp_watts == 400.0
    assert preset.max_safe_temp == 85.0
    assert preset.target_sustained_temp == 75.0
    assert preset.warning_temp == 80.0
    assert preset.recommended_max_seqs == 256
    assert preset.architecture == "Hopper"
    assert preset.memory_type == "HBM3"
    
    print("✅ test_h100_nvl_preset PASSED")


# ============================================================
# get_gpu_preset Tests
# ============================================================

def test_get_gpu_preset_valid():
    """Test get_gpu_preset with valid preset names."""
    # Test exact names
    preset = get_gpu_preset('a100-40gb')
    assert preset.short_name == 'a100-40gb'
    
    preset = get_gpu_preset('h100-sxm')
    assert preset.short_name == 'h100-sxm'
    
    # Test aliases
    preset = get_gpu_preset('a100')
    assert preset.short_name == 'a100-40gb'
    
    preset = get_gpu_preset('h100')
    assert preset.short_name == 'h100-pcie'
    
    # Test case insensitivity
    preset = get_gpu_preset('A100-40GB')
    assert preset.short_name == 'a100-40gb'
    
    preset = get_gpu_preset('H100-SXM')
    assert preset.short_name == 'h100-sxm'
    
    print("✅ test_get_gpu_preset_valid PASSED")


def test_get_gpu_preset_invalid():
    """Test get_gpu_preset with invalid preset names."""
    try:
        get_gpu_preset('invalid-gpu')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown GPU type" in str(e)
        assert "Available presets" in str(e)
    
    print("✅ test_get_gpu_preset_invalid PASSED")


# ============================================================
# list_gpu_presets Tests
# ============================================================

def test_list_gpu_presets():
    """Test list_gpu_presets returns unique presets."""
    presets = list_gpu_presets()
    
    assert isinstance(presets, dict)
    assert len(presets) >= 5  # At least 5 unique presets
    
    # Check that all values are GPUPreset instances
    for name, preset in presets.items():
        assert isinstance(preset, GPUPreset)
        assert preset.short_name == name
    
    # Check that aliases are not in the list (only unique presets)
    # The list should only contain unique short_names
    short_names = [p.short_name for p in presets.values()]
    assert len(short_names) == len(set(short_names))  # All unique
    
    print("✅ test_list_gpu_presets PASSED")


# ============================================================
# print_gpu_presets Tests
# ============================================================

def test_print_gpu_presets():
    """Test print_gpu_presets doesn't crash."""
    # This test just verifies it runs without error
    # Output is printed to stdout but we don't capture it
    try:
        print_gpu_presets()
        print("✅ test_print_gpu_presets PASSED")
    except Exception as e:
        print(f"❌ test_print_gpu_presets FAILED: {e}")
        raise


# ============================================================
# DEFAULT_GPU_PRESET Tests
# ============================================================

def test_default_gpu_preset():
    """Test DEFAULT_GPU_PRESET is valid."""
    assert DEFAULT_GPU_PRESET in GPU_PRESETS
    
    default_preset = GPU_PRESETS[DEFAULT_GPU_PRESET]
    assert isinstance(default_preset, GPUPreset)
    
    # Default should be A100 40GB for backward compatibility
    assert DEFAULT_GPU_PRESET == 'a100-40gb'
    assert default_preset.name == "NVIDIA A100 40GB"
    
    print("✅ test_default_gpu_preset PASSED")


# ============================================================
# Integration Tests
# ============================================================

def test_preset_thermal_config_integration():
    """Test that all presets can be converted to ThermalConfig."""
    for preset_name, preset in GPU_PRESETS.items():
        thermal_config = preset.to_thermal_config()
        
        assert isinstance(thermal_config, ThermalConfig)
        assert thermal_config.max_safe_temp > 0
        assert thermal_config.target_sustained_temp > 0
        assert thermal_config.warning_temp > 0
        assert thermal_config.max_power > 0
        assert thermal_config.total_memory_gb > 0
        assert len(thermal_config.gpu_name) > 0
    
    print("✅ test_preset_thermal_config_integration PASSED")


def test_preset_gpu_config_integration():
    """Test that all presets can be converted to GPU config dict."""
    for preset_name, preset in GPU_PRESETS.items():
        gpu_config = preset.to_gpu_config()
        
        assert isinstance(gpu_config, dict)
        assert 'name' in gpu_config
        assert 'memory_gb' in gpu_config
        assert 'tdp_watts' in gpu_config
        assert 'max_safe_temp' in gpu_config
        assert 'target_sustained_temp' in gpu_config
        assert 'recommended_max_seqs' in gpu_config
        assert 'architecture' in gpu_config
        assert 'memory_type' in gpu_config
    
    print("✅ test_preset_gpu_config_integration PASSED")


# ============================================================
# Main Test Runner
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GPU PRESETS MODULE TESTS")
    print("=" * 70 + "\n")
    
    print("=== GPUPreset Tests ===")
    test_gpu_preset_creation()
    test_gpu_preset_to_thermal_config()
    test_gpu_preset_to_gpu_config()
    test_gpu_preset_to_dict()
    
    print("\n=== GPU_PRESETS Tests ===")
    test_gpu_presets_structure()
    test_a100_40gb_preset()
    test_a100_80gb_preset()
    test_h100_sxm_preset()
    test_h100_pcie_preset()
    test_h100_nvl_preset()
    
    print("\n=== get_gpu_preset Tests ===")
    test_get_gpu_preset_valid()
    test_get_gpu_preset_invalid()
    
    print("\n=== list_gpu_presets Tests ===")
    test_list_gpu_presets()
    
    print("\n=== print_gpu_presets Tests ===")
    test_print_gpu_presets()
    
    print("\n=== DEFAULT_GPU_PRESET Tests ===")
    test_default_gpu_preset()
    
    print("\n=== Integration Tests ===")
    test_preset_thermal_config_integration()
    test_preset_gpu_config_integration()
    
    print("\n" + "=" * 70)
    print("ALL GPU PRESETS TESTS PASSED ✅")
    print("=" * 70 + "\n")
