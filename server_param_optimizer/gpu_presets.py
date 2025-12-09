"""
GPU Presets for Server Parameter Optimizer

Provides predefined GPU configurations for A100 and H100 variants,
making it easy to switch between GPU types without editing code.

Example:
    preset = get_gpu_preset('h100-sxm')
    thermal_config = preset.to_thermal_config()
    gpu_config = preset.to_gpu_config()
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

# Add parent directory to path for script execution
import sys
import os
if __name__ == "__main__" or "." not in __name__:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with fallback for both module and script execution
try:
    from .thermal_monitor import ThermalConfig
except ImportError:
    from thermal_monitor import ThermalConfig


@dataclass
class GPUPreset:
    """GPU preset with thermal and memory specifications.
    
    Attributes:
        name: Full GPU name (e.g., "NVIDIA H100 SXM")
        short_name: Short identifier (e.g., "h100-sxm")
        memory_gb: Total GPU memory in GB
        tdp_watts: Thermal Design Power in watts
        max_safe_temp: Maximum safe temperature before throttling (°C)
        target_sustained_temp: Target temperature for 24/7 operation (°C)
        warning_temp: Warning threshold temperature (°C)
        recommended_max_seqs: Recommended maximum concurrent sequences
        architecture: GPU architecture (e.g., "Hopper", "Ampere")
        memory_type: Memory type (e.g., "HBM3", "HBM2e")
    """
    name: str
    short_name: str
    memory_gb: float
    tdp_watts: float
    max_safe_temp: float
    target_sustained_temp: float
    warning_temp: float
    recommended_max_seqs: int
    architecture: str = "Unknown"
    memory_type: str = "Unknown"
    
    def to_thermal_config(self) -> ThermalConfig:
        """Convert GPU preset to ThermalConfig.
        
        Returns:
            ThermalConfig object with GPU-specific thresholds
        """
        return ThermalConfig(
            max_safe_temp=self.max_safe_temp,
            target_sustained_temp=self.target_sustained_temp,
            warning_temp=self.warning_temp,
            max_power=self.tdp_watts,
            total_memory_gb=self.memory_gb,
            gpu_name=self.name
        )
    
    def to_gpu_config(self) -> Dict[str, Any]:
        """Convert GPU preset to config dict for prompts.
        
        Returns:
            Dictionary with GPU specifications for LLM prompts
        """
        return {
            'name': self.name,
            'memory_gb': self.memory_gb,
            'tdp_watts': self.tdp_watts,
            'max_safe_temp': self.max_safe_temp,
            'target_sustained_temp': self.target_sustained_temp,
            'recommended_max_seqs': self.recommended_max_seqs,
            'architecture': self.architecture,
            'memory_type': self.memory_type
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary.
        
        Returns:
            Dictionary representation of the preset
        """
        return asdict(self)


# GPU Presets Dictionary
# Based on NVIDIA specifications and best practices
GPU_PRESETS: Dict[str, GPUPreset] = {
    # A100 Variants
    'a100-40gb': GPUPreset(
        name='NVIDIA A100 40GB',
        short_name='a100-40gb',
        memory_gb=40.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,  # A100 throttle threshold
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=64,
        architecture='Ampere',
        memory_type='HBM2e'
    ),
    'a100-80gb': GPUPreset(
        name='NVIDIA A100 80GB',
        short_name='a100-80gb',
        memory_gb=80.0,
        tdp_watts=400.0,
        max_safe_temp=83.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=128,
        architecture='Ampere',
        memory_type='HBM2e'
    ),
    
    # H100 Variants
    'h100-sxm': GPUPreset(
        name='NVIDIA H100 SXM',
        short_name='h100-sxm',
        memory_gb=80.0,
        tdp_watts=700.0,  # H100 SXM5 TDP
        max_safe_temp=85.0,  # H100 throttle threshold
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=256,
        architecture='Hopper',
        memory_type='HBM3'
    ),
    'h100-pcie': GPUPreset(
        name='NVIDIA H100 PCIe',
        short_name='h100-pcie',
        memory_gb=80.0,
        tdp_watts=350.0,  # H100 PCIe TDP
        max_safe_temp=85.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=256,
        architecture='Hopper',
        memory_type='HBM3'
    ),
    'h100-nvl': GPUPreset(
        name='NVIDIA H100 NVL',
        short_name='h100-nvl',
        memory_gb=94.0,  # H100 NVL has 94GB
        tdp_watts=400.0,
        max_safe_temp=85.0,
        target_sustained_temp=75.0,
        warning_temp=80.0,
        recommended_max_seqs=256,
        architecture='Hopper',
        memory_type='HBM3'
    ),
}

# Convenient aliases
GPU_PRESETS['a100'] = GPU_PRESETS['a100-40gb']
GPU_PRESETS['h100'] = GPU_PRESETS['h100-pcie']

# Default preset (maintaining backward compatibility with A100 40GB)
DEFAULT_GPU_PRESET = 'a100-40gb'


def get_gpu_preset(gpu_type: str) -> GPUPreset:
    """Get GPU preset by type name.
    
    Args:
        gpu_type: GPU type identifier (e.g., 'h100-sxm', 'a100', 'h100')
        
    Returns:
        GPUPreset object
        
    Raises:
        ValueError: If gpu_type is not found in GPU_PRESETS
        
    Example:
        >>> preset = get_gpu_preset('h100-sxm')
        >>> print(preset.name)
        NVIDIA H100 SXM
    """
    gpu_type_lower = gpu_type.lower()
    
    if gpu_type_lower not in GPU_PRESETS:
        available = ', '.join(sorted(set(p.short_name for p in GPU_PRESETS.values())))
        raise ValueError(
            f"Unknown GPU type: '{gpu_type}'. Available presets: {available}"
        )
    
    return GPU_PRESETS[gpu_type_lower]


def list_gpu_presets() -> Dict[str, GPUPreset]:
    """Get all available GPU presets.
    
    Returns:
        Dictionary mapping GPU type names to GPUPreset objects
        
    Example:
        >>> presets = list_gpu_presets()
        >>> for name, preset in presets.items():
        ...     print(f"{name}: {preset.name}")
    """
    # Return unique presets (excluding aliases)
    seen = set()
    unique_presets = {}
    
    for key, preset in GPU_PRESETS.items():
        if preset.short_name not in seen:
            seen.add(preset.short_name)
            unique_presets[preset.short_name] = preset
    
    return unique_presets


def print_gpu_presets() -> None:
    """Print a formatted table of all available GPU presets.
    
    Prints to stdout with aligned columns showing:
    - GPU Type
    - Name
    - Memory
    - TDP
    - Recommended max_seqs
    
    Example:
        >>> print_gpu_presets()
        Available GPU Presets:
        ═══════════════════════════════════════════════════════════════════════════════
        GPU Type      Name                    Memory    TDP      Max Seqs  Architecture
        ───────────────────────────────────────────────────────────────────────────────
        a100-40gb     NVIDIA A100 40GB        40 GB     400 W    64        Ampere
        a100-80gb     NVIDIA A100 80GB        80 GB     400 W    128       Ampere
        h100-sxm      NVIDIA H100 SXM         80 GB     700 W    256       Hopper
        ...
    """
    presets = list_gpu_presets()
    
    print("\nAvailable GPU Presets:")
    print("═" * 79)
    print(f"{'GPU Type':<13} {'Name':<23} {'Memory':<9} {'TDP':<8} {'Max Seqs':<9} {'Arch'}")
    print("─" * 79)
    
    # Sort by architecture and then by name
    sorted_presets = sorted(
        presets.items(),
        key=lambda x: (x[1].architecture, x[1].short_name)
    )
    
    for gpu_type, preset in sorted_presets:
        memory_str = f"{preset.memory_gb:.0f} GB"
        tdp_str = f"{preset.tdp_watts:.0f} W"
        print(f"{gpu_type:<13} {preset.name:<23} {memory_str:<9} {tdp_str:<8} "
              f"{preset.recommended_max_seqs:<9} {preset.architecture}")
    
    print("═" * 79)
    print(f"\nAliases: 'a100' -> 'a100-40gb', 'h100' -> 'h100-pcie'")
    print(f"Default: {DEFAULT_GPU_PRESET}")
    print()


def main():
    """CLI entry point for testing GPU presets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU Presets Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all GPU presets
  python gpu_presets.py --list
  
  # Get info about specific GPU
  python gpu_presets.py --info h100-sxm
  
  # Get thermal config for GPU
  python gpu_presets.py --thermal h100-pcie
        """
    )
    
    parser.add_argument('--list', action='store_true',
                        help='List all available GPU presets')
    parser.add_argument('--info', metavar='GPU_TYPE',
                        help='Show detailed info for a GPU preset')
    parser.add_argument('--thermal', metavar='GPU_TYPE',
                        help='Show thermal config for a GPU preset')
    
    args = parser.parse_args()
    
    if args.list:
        print_gpu_presets()
    elif args.info:
        try:
            preset = get_gpu_preset(args.info)
            print(f"\nGPU Preset: {preset.short_name}")
            print("─" * 60)
            print(f"Name:                     {preset.name}")
            print(f"Architecture:             {preset.architecture}")
            print(f"Memory:                   {preset.memory_gb} GB {preset.memory_type}")
            print(f"TDP:                      {preset.tdp_watts} W")
            print(f"Max Safe Temp:            {preset.max_safe_temp}°C")
            print(f"Target Sustained Temp:    {preset.target_sustained_temp}°C")
            print(f"Warning Temp:             {preset.warning_temp}°C")
            print(f"Recommended Max Seqs:     {preset.recommended_max_seqs}")
            print()
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    elif args.thermal:
        try:
            preset = get_gpu_preset(args.thermal)
            thermal_config = preset.to_thermal_config()
            print(f"\nThermal Config for {preset.name}:")
            print("─" * 60)
            print(f"max_safe_temp:            {thermal_config.max_safe_temp}°C")
            print(f"target_sustained_temp:    {thermal_config.target_sustained_temp}°C")
            print(f"warning_temp:             {thermal_config.warning_temp}°C")
            print(f"max_power:                {thermal_config.max_power} W")
            print(f"total_memory_gb:          {thermal_config.total_memory_gb} GB")
            print(f"gpu_name:                 {thermal_config.gpu_name}")
            print()
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
