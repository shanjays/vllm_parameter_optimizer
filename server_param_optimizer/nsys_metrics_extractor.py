"""
NVIDIA Nsight Systems Metrics Extractor

Extracts profiling metrics from nsys (NVIDIA Nsight Systems).
Parses stats text output and SQLite exports to extract:
- Kernel execution counts and times
- Memory copy times and bandwidth
- GPU utilization metrics

Handles nsys not being installed gracefully.
"""

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple


def check_nsys_available() -> bool:
    """Check if nsys (NVIDIA Nsight Systems) is available.
    
    Returns:
        True if nsys is installed and accessible
    """
    try:
        result = subprocess.run(
            ['nsys', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass
class NsysMetrics:
    """Extracted metrics from nsys profile."""
    # Kernel metrics
    kernel_count: int = 0
    total_kernel_time_ms: float = 0.0
    avg_kernel_time_us: float = 0.0
    max_kernel_time_us: float = 0.0
    
    # Memory metrics
    memcpy_count: int = 0
    total_memcpy_time_ms: float = 0.0
    memcpy_bandwidth_gbps: float = 0.0
    
    # Top kernels
    top_kernels: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class NsysProfile:
    """Result from an nsys profile run."""
    output_path: str  # Path to nsight-rep file
    duration_seconds: float  # Total profiling duration
    metrics: NsysMetrics  # Extracted metrics
    raw_stats: str  # Raw stats output
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'output_path': self.output_path,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics.to_dict(),
            'raw_stats': self.raw_stats[:5000] if self.raw_stats else "",  # Truncate for serialization
            'success': self.success,
            'error_message': self.error_message
        }


class NsysMetricsExtractor:
    """Extracts profiling metrics from NVIDIA Nsight Systems.
    
    Runs nsys profile on commands and extracts metrics from the
    stats output. Can also parse SQLite exports for more detailed metrics.
    
    Example:
        extractor = NsysMetricsExtractor()
        
        if extractor.is_available():
            profile = extractor.profile_command(
                command="python benchmark.py",
                output_dir="./profiles",
                profile_name="benchmark_v1"
            )
            
            if profile.success:
                print(f"Kernel count: {profile.metrics.kernel_count}")
                print(f"Total kernel time: {profile.metrics.total_kernel_time_ms}ms")
    """
    
    def __init__(self, gpu_index: int = 0):
        """Initialize the extractor.
        
        Args:
            gpu_index: GPU device index to profile
        """
        self.gpu_index = gpu_index
        self._nsys_available = check_nsys_available()
        
        if not self._nsys_available:
            print("[NsysMetricsExtractor] Warning: nsys not installed, profiling will be skipped")
    
    def is_available(self) -> bool:
        """Check if nsys is available for profiling."""
        return self._nsys_available
        
    def profile_command(
        self,
        command: str,
        output_dir: str = "./nsys_profiles",
        profile_name: str = "profile",
        timeout: int = 600,
        capture_range: str = "cudaProfilerApi",
        extra_args: Optional[List[str]] = None
    ) -> NsysProfile:
        """Run nsys profile on a command and extract metrics.
        
        Args:
            command: Shell command to profile
            output_dir: Directory for output files
            profile_name: Name for the profile output
            timeout: Maximum time for profiling in seconds
            capture_range: Capture range option (cudaProfilerApi, none, full)
            extra_args: Additional nsys arguments
            
        Returns:
            NsysProfile with extracted metrics
        """
        if not self._nsys_available:
            return NsysProfile(
                output_path="",
                duration_seconds=0.0,
                metrics=NsysMetrics(),
                raw_stats="",
                success=False,
                error_message="nsys not installed"
            )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        timestamp = int(time.time())
        output_base = os.path.join(output_dir, f"{profile_name}_{timestamp}")
        output_path = f"{output_base}.nsys-rep"
        
        # Build nsys command
        nsys_cmd = [
            'nsys', 'profile',
            '--output', output_base,
            '--force-overwrite', 'true',
            '--capture-range', capture_range,
            '--stats', 'true',
            '--gpu-metrics-device', str(self.gpu_index)
        ]
        
        if extra_args:
            nsys_cmd.extend(extra_args)
            
        # Add the command to profile
        nsys_cmd.extend(['--', 'bash', '-c', command])
        
        start_time = time.time()
        raw_stats = ""
        
        try:
            result = subprocess.run(
                nsys_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(self.gpu_index)}
            )
            
            duration = time.time() - start_time
            raw_stats = result.stdout + result.stderr
            
            if result.returncode != 0:
                return NsysProfile(
                    output_path=output_path if os.path.exists(output_path) else "",
                    duration_seconds=duration,
                    metrics=NsysMetrics(),
                    raw_stats=raw_stats,
                    success=False,
                    error_message=f"nsys exited with code {result.returncode}"
                )
            
            # Parse stats from output
            metrics = self._parse_stats_output(raw_stats)
            
            return NsysProfile(
                output_path=output_path if os.path.exists(output_path) else output_base,
                duration_seconds=duration,
                metrics=metrics,
                raw_stats=raw_stats,
                success=True
            )
            
        except subprocess.TimeoutExpired:
            return NsysProfile(
                output_path="",
                duration_seconds=timeout,
                metrics=NsysMetrics(),
                raw_stats="",
                success=False,
                error_message=f"Profiling timed out after {timeout}s"
            )
        except Exception as e:
            return NsysProfile(
                output_path="",
                duration_seconds=time.time() - start_time,
                metrics=NsysMetrics(),
                raw_stats=raw_stats,
                success=False,
                error_message=str(e)
            )
    
    def _parse_stats_output(self, stats_output: str) -> NsysMetrics:
        """Parse nsys stats text output to extract metrics.
        
        Args:
            stats_output: Raw text output from nsys --stats
            
        Returns:
            NsysMetrics with parsed values
        """
        metrics = NsysMetrics()
        
        if not stats_output:
            return metrics
        
        # Parse CUDA kernel summary
        # Format varies but typically looks like:
        # Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
        kernel_pattern = re.compile(
            r'^\s*(\d+\.?\d*)\s+(\d+)\s+(\d+)\s+(\d+\.?\d*)\s+.*?(\w+)',
            re.MULTILINE
        )
        
        # Look for kernel statistics section
        kernel_section = False
        memcpy_section = False
        
        lines = stats_output.split('\n')
        kernel_times = []
        memcpy_times = []
        
        for i, line in enumerate(lines):
            # Detect section headers
            if 'CUDA Kernel Statistics' in line or 'cuda_gpu_kern' in line.lower():
                kernel_section = True
                memcpy_section = False
                continue
            elif 'CUDA Memory Operation Statistics' in line or 'memcpy' in line.lower():
                memcpy_section = True
                kernel_section = False
                continue
            elif 'Summary' in line or '----' in line:
                continue
                
            # Parse kernel entries
            if kernel_section:
                # Try to extract time and count from tabular data
                # Common format: Time%  Time  Count  Avg  Min  Max  Name
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Try parsing as numeric data
                        time_ns = float(parts[1].replace(',', ''))
                        count = int(parts[2].replace(',', ''))
                        avg_ns = float(parts[3].replace(',', ''))
                        
                        kernel_times.append({
                            'total_time_ns': time_ns,
                            'count': count,
                            'avg_ns': avg_ns,
                            'name': parts[-1] if len(parts) > 4 else 'unknown'
                        })
                    except (ValueError, IndexError):
                        pass
                        
            # Parse memcpy entries
            if memcpy_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        time_ns = float(parts[1].replace(',', ''))
                        count = int(parts[2].replace(',', ''))
                        memcpy_times.append({
                            'total_time_ns': time_ns,
                            'count': count
                        })
                    except (ValueError, IndexError):
                        pass
        
        # Aggregate kernel metrics
        if kernel_times:
            metrics.kernel_count = sum(k['count'] for k in kernel_times)
            total_ns = sum(k['total_time_ns'] for k in kernel_times)
            metrics.total_kernel_time_ms = total_ns / 1_000_000
            if metrics.kernel_count > 0:
                metrics.avg_kernel_time_us = (total_ns / metrics.kernel_count) / 1_000
            
            # Sort by time and get top kernels
            kernel_times.sort(key=lambda x: x['total_time_ns'], reverse=True)
            metrics.top_kernels = kernel_times[:10]
            
            if kernel_times:
                metrics.max_kernel_time_us = max(k['avg_ns'] for k in kernel_times) / 1_000
        
        # Aggregate memcpy metrics
        if memcpy_times:
            metrics.memcpy_count = sum(m['count'] for m in memcpy_times)
            total_ns = sum(m['total_time_ns'] for m in memcpy_times)
            metrics.total_memcpy_time_ms = total_ns / 1_000_000
        
        # Try to extract summary statistics with regex patterns
        # Pattern for total kernel time
        kernel_time_match = re.search(
            r'Total\s+(?:GPU\s+)?Kernel\s+(?:Time|Duration)[:\s]+(\d+\.?\d*)\s*(ms|us|ns|s)',
            stats_output,
            re.IGNORECASE
        )
        if kernel_time_match:
            value = float(kernel_time_match.group(1))
            unit = kernel_time_match.group(2).lower()
            if unit == 'ns':
                metrics.total_kernel_time_ms = value / 1_000_000
            elif unit == 'us':
                metrics.total_kernel_time_ms = value / 1_000
            elif unit == 's':
                metrics.total_kernel_time_ms = value * 1_000
            else:  # ms
                metrics.total_kernel_time_ms = value
        
        return metrics
    
    def get_metrics_summary(self, profile: NsysProfile) -> Dict[str, Any]:
        """Get a summary of metrics suitable for display.
        
        Args:
            profile: NsysProfile to summarize
            
        Returns:
            Dictionary with formatted metrics summary
        """
        if not profile.success:
            return {
                'success': False,
                'error': profile.error_message
            }
            
        return {
            'success': True,
            'duration_seconds': round(profile.duration_seconds, 2),
            'kernel_count': profile.metrics.kernel_count,
            'total_kernel_time_ms': round(profile.metrics.total_kernel_time_ms, 3),
            'avg_kernel_time_us': round(profile.metrics.avg_kernel_time_us, 3),
            'memcpy_count': profile.metrics.memcpy_count,
            'total_memcpy_time_ms': round(profile.metrics.total_memcpy_time_ms, 3),
            'top_3_kernels': [
                {'name': k['name'], 'time_ms': k['total_time_ns'] / 1_000_000}
                for k in profile.metrics.top_kernels[:3]
            ]
        }
        
    def export_to_sqlite(self, profile_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """Export nsys profile to SQLite format for detailed analysis.
        
        Args:
            profile_path: Path to .nsys-rep file
            output_path: Output SQLite path (default: same as input with .sqlite)
            
        Returns:
            Path to SQLite file or None if export failed
        """
        if not self._nsys_available:
            return None
            
        if not os.path.exists(profile_path):
            return None
            
        if output_path is None:
            output_path = profile_path.replace('.nsys-rep', '.sqlite')
            
        try:
            result = subprocess.run(
                ['nsys', 'export', '--type', 'sqlite', '--output', output_path, profile_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
