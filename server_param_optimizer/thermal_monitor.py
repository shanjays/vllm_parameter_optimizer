"""
GPU Thermal and Power Monitoring Module

Provides real-time GPU thermal and power monitoring via nvidia-smi.
Used during benchmarks to track thermal behavior and determine if
a configuration is thermally safe for sustained operation.

Target: NVIDIA H100 80GB GPU
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ThermalSample:
    """Single sample of GPU thermal and power state."""
    timestamp: float  # Unix timestamp
    temperature: float  # Celsius
    power: float  # Watts
    memory_used: float  # MB
    memory_total: float  # MB
    gpu_utilization: float  # Percentage
    memory_utilization: float  # Percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ThermalConfig:
    """Configuration thresholds for H100 80GB GPU.
    
    These values are based on NVIDIA H100 specifications:
    - Max operating temp: 85°C
    - Throttle temp: 85°C (GPU starts throttling)
    - Target sustained temp: 75°C (recommended for 24/7 operation)
    - TDP: 350W (H100 PCIe) / 700W (H100 SXM5)
    - Memory: 80GB HBM3
    """
    max_safe_temp: float = 85.0  # Celsius - throttling threshold
    target_sustained_temp: float = 75.0  # Celsius - target for sustained operation
    warning_temp: float = 80.0  # Celsius - warning threshold
    max_power: float = 350.0  # Watts - TDP (H100 PCIe)
    total_memory_gb: float = 80.0  # GB - total GPU memory
    gpu_name: str = "NVIDIA H100 80GB"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ThermalSummary:
    """Summary statistics from a monitoring session."""
    duration_seconds: float
    sample_count: int
    
    # Temperature stats
    temp_min: float
    temp_max: float
    temp_avg: float
    temp_final: float
    
    # Power stats
    power_min: float
    power_max: float
    power_avg: float
    
    # Memory stats
    memory_max_used_mb: float
    memory_max_used_pct: float
    
    # Utilization stats
    gpu_util_avg: float
    memory_util_avg: float
    
    # Thermal status
    is_thermally_safe: bool
    max_temp_exceeded: bool
    throttling_detected: bool
    time_above_target: float  # seconds spent above target temp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


def get_gpu_info(gpu_index: int = 0) -> Optional[Dict[str, Any]]:
    """Get static GPU information.
    
    Args:
        gpu_index: GPU device index (default 0)
        
    Returns:
        Dictionary with GPU info or None if nvidia-smi fails
    """
    try:
        cmd = [
            'nvidia-smi',
            f'--id={gpu_index}',
            '--query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return None
            
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        if len(parts) >= 5:
            return {
                'name': parts[0],
                'memory_total_mb': float(parts[1]),
                'driver_version': parts[2],
                'pcie_gen': parts[3],
                'pcie_width': parts[4],
                'gpu_index': gpu_index
            }
        return None
        
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


class ThermalMonitor:
    """Background GPU thermal and power monitor.
    
    Collects GPU metrics at regular intervals during benchmarks.
    Runs in a background thread to avoid impacting benchmark performance.
    
    Example:
        monitor = ThermalMonitor(gpu_index=0)
        monitor.start_monitoring()
        # ... run benchmark ...
        monitor.stop_monitoring()
        summary = monitor.get_thermal_summary()
        if summary.is_thermally_safe:
            print("Config is thermally safe for sustained operation")
    """
    
    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval: float = 1.0,
        config: Optional[ThermalConfig] = None
    ):
        """Initialize the thermal monitor.
        
        Args:
            gpu_index: GPU device index to monitor
            sample_interval: Time between samples in seconds
            config: Thermal thresholds config (defaults to H100 80GB settings)
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.config = config or ThermalConfig()
        
        self._samples: List[ThermalSample] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring:
            return
            
        self._samples = []
        self._monitoring = True
        self._start_time = time.time()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop monitoring and wait for thread to finish."""
        self._monitoring = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            sample = self._collect_sample()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self.sample_interval)
            
    def _collect_sample(self) -> Optional[ThermalSample]:
        """Collect a single sample from nvidia-smi."""
        try:
            cmd = [
                'nvidia-smi',
                f'--id={self.gpu_index}',
                '--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
                
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            if len(parts) >= 6:
                return ThermalSample(
                    timestamp=time.time(),
                    temperature=float(parts[0]),
                    power=float(parts[1]),
                    memory_used=float(parts[2]),
                    memory_total=float(parts[3]),
                    gpu_utilization=float(parts[4]),
                    memory_utilization=float(parts[5])
                )
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            return None
            
    def get_samples(self) -> List[ThermalSample]:
        """Get a copy of all collected samples."""
        with self._lock:
            return list(self._samples)
            
    def get_samples_as_dict(self) -> List[Dict[str, Any]]:
        """Get samples as list of dictionaries for JSON serialization."""
        return [s.to_dict() for s in self.get_samples()]
        
    def get_thermal_summary(self) -> Optional[ThermalSummary]:
        """Calculate summary statistics from collected samples.
        
        Returns:
            ThermalSummary with statistics, or None if no samples collected
        """
        samples = self.get_samples()
        if not samples:
            return None
            
        # Calculate duration
        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0
        
        # Extract values
        temps = [s.temperature for s in samples]
        powers = [s.power for s in samples]
        gpu_utils = [s.gpu_utilization for s in samples]
        mem_utils = [s.memory_utilization for s in samples]
        mem_used = [s.memory_used for s in samples]
        mem_total = samples[0].memory_total if samples else 0.0
        
        # Temperature analysis
        temp_min = min(temps)
        temp_max = max(temps)
        temp_avg = sum(temps) / len(temps)
        temp_final = temps[-1]
        
        # Time above target temperature
        time_above_target = sum(
            1 for s in samples 
            if s.temperature > self.config.target_sustained_temp
        ) * self.sample_interval
        
        # Thermal status flags
        # On H100, throttling occurs at max_safe_temp (85°C)
        max_temp_exceeded = temp_max >= self.config.max_safe_temp
        # Throttling detected when any sample reached the throttle threshold
        throttling_detected = max_temp_exceeded
        # Thermally safe means staying below target sustained temp (75°C)
        is_thermally_safe = temp_max < self.config.target_sustained_temp
        
        return ThermalSummary(
            duration_seconds=duration,
            sample_count=len(samples),
            temp_min=temp_min,
            temp_max=temp_max,
            temp_avg=temp_avg,
            temp_final=temp_final,
            power_min=min(powers),
            power_max=max(powers),
            power_avg=sum(powers) / len(powers),
            memory_max_used_mb=max(mem_used),
            memory_max_used_pct=(max(mem_used) / mem_total * 100) if mem_total > 0 else 0.0,
            gpu_util_avg=sum(gpu_utils) / len(gpu_utils),
            memory_util_avg=sum(mem_utils) / len(mem_utils),
            is_thermally_safe=is_thermally_safe,
            max_temp_exceeded=max_temp_exceeded,
            throttling_detected=throttling_detected,
            time_above_target=time_above_target
        )
        
    def is_thermally_safe(self) -> bool:
        """Check if the run stayed below target sustained temperature.
        
        Returns:
            True if all samples were below target_sustained_temp (75°C for H100)
        """
        summary = self.get_thermal_summary()
        return summary.is_thermally_safe if summary else True
        
    def get_sample_count(self) -> int:
        """Get current number of samples collected."""
        with self._lock:
            return len(self._samples)
