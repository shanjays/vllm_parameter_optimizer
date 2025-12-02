"""
Visualization Module for Thermal Monitoring

Generates PNG thermal plots using matplotlib.
Creates 2x2 subplot figures showing temperature, power, utilization, and memory.
Includes thermal zone coloring and status indicators.

Handles matplotlib import gracefully - skips plots if not installed.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# Try to import matplotlib with Agg backend
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    # Figure size
    figure_width: float = 14.0
    figure_height: float = 10.0
    dpi: int = 150
    
    # Colors
    color_temperature: str = '#FF6B6B'  # Red
    color_power: str = '#4ECDC4'  # Teal
    color_utilization: str = '#45B7D1'  # Blue
    color_memory: str = '#96CEB4'  # Green
    
    # Thermal zone colors
    color_safe: str = '#D4EDDA'  # Light green
    color_warning: str = '#FFF3CD'  # Light yellow
    color_danger: str = '#F8D7DA'  # Light red
    
    # Threshold lines
    color_target_line: str = '#FFA500'  # Orange
    color_throttle_line: str = '#FF0000'  # Red
    color_tdp_line: str = '#FF4500'  # Orange-red
    
    # Text
    font_size_title: int = 14
    font_size_label: int = 11
    font_size_tick: int = 9
    font_size_summary: int = 10
    
    # Line width
    line_width: float = 2.0
    threshold_line_width: float = 1.5


class ThermalVisualizer:
    """Generates thermal monitoring visualization plots.
    
    Creates 2x2 subplot figures showing:
    1. Temperature over time with threshold lines
    2. Power consumption over time with TDP line
    3. GPU utilization over time
    4. Memory usage over time
    
    Example:
        from server_param_optimizer import ThermalVisualizer, ThermalMonitor
        
        monitor = ThermalMonitor()
        monitor.start_monitoring()
        # ... run benchmark ...
        monitor.stop_monitoring()
        
        samples = monitor.get_samples()
        summary = monitor.get_thermal_summary()
        
        visualizer = ThermalVisualizer()
        visualizer.save_thermal_plot(
            samples=samples,
            summary=summary,
            output_path='thermal_plot.png',
            title='Benchmark Thermal Profile'
        )
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize the visualizer.
        
        Args:
            config: Plot styling configuration
        """
        self.config = config or PlotConfig()
        
        if not MATPLOTLIB_AVAILABLE:
            print("[ThermalVisualizer] Warning: matplotlib not installed, plots will be skipped")
    
    def is_available(self) -> bool:
        """Check if matplotlib is available for plotting."""
        return MATPLOTLIB_AVAILABLE
        
    def save_thermal_plot(
        self,
        samples: List[Any],
        summary: Any,
        output_path: str,
        title: str = "GPU Thermal Profile",
        thermal_config: Optional[Any] = None
    ) -> bool:
        """Generate and save a 2x2 thermal monitoring plot.
        
        Args:
            samples: List of ThermalSample objects
            summary: ThermalSummary object with statistics
            output_path: Path to save PNG file
            title: Main title for the plot
            thermal_config: ThermalConfig for threshold values
            
        Returns:
            True if plot was saved successfully, False otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print(f"[ThermalVisualizer] Skipping plot (matplotlib not installed): {output_path}")
            return False
            
        if not samples:
            print(f"[ThermalVisualizer] No samples to plot")
            return False
        
        # Import ThermalConfig here to avoid circular import
        from .thermal_monitor import ThermalConfig
        config = thermal_config or ThermalConfig()
        
        try:
            # Extract time series data
            start_time = samples[0].timestamp
            times = [(s.timestamp - start_time) for s in samples]
            temps = [s.temperature for s in samples]
            powers = [s.power for s in samples]
            gpu_utils = [s.gpu_utilization for s in samples]
            memory_used = [s.memory_used / 1024 for s in samples]  # Convert to GB
            memory_total_gb = samples[0].memory_total / 1024 if samples else 40.0
            
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_width, self.config.figure_height))
            fig.suptitle(title, fontsize=self.config.font_size_title + 2, fontweight='bold')
            
            # 1. Temperature plot (top-left)
            ax_temp = axes[0, 0]
            self._plot_temperature(ax_temp, times, temps, config)
            
            # 2. Power plot (top-right)
            ax_power = axes[0, 1]
            self._plot_power(ax_power, times, powers, config)
            
            # 3. GPU Utilization plot (bottom-left)
            ax_util = axes[1, 0]
            self._plot_utilization(ax_util, times, gpu_utils)
            
            # 4. Memory Usage plot (bottom-right)
            ax_mem = axes[1, 1]
            self._plot_memory(ax_mem, times, memory_used, memory_total_gb)
            
            # Add summary text box
            self._add_summary_box(fig, summary, config)
            
            # Add thermal status indicator
            self._add_status_indicator(fig, summary, config)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            
            # Save plot
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"[ThermalVisualizer] Saved plot: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ThermalVisualizer] Error saving plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
            
    def _plot_temperature(self, ax, times: List[float], temps: List[float], config: Any) -> None:
        """Plot temperature over time with thermal zones."""
        # Add thermal zone coloring
        ax.axhspan(0, config.target_sustained_temp, alpha=0.3, color=self.config.color_safe, label='Safe Zone')
        ax.axhspan(config.target_sustained_temp, config.warning_temp, alpha=0.3, color=self.config.color_warning, label='Warning Zone')
        ax.axhspan(config.warning_temp, 100, alpha=0.3, color=self.config.color_danger, label='Danger Zone')
        
        # Plot temperature line
        ax.plot(times, temps, color=self.config.color_temperature, linewidth=self.config.line_width, label='Temperature')
        
        # Add threshold lines
        ax.axhline(y=config.target_sustained_temp, color=self.config.color_target_line, 
                   linestyle='--', linewidth=self.config.threshold_line_width, label=f'Target ({config.target_sustained_temp}°C)')
        ax.axhline(y=config.max_safe_temp, color=self.config.color_throttle_line, 
                   linestyle='--', linewidth=self.config.threshold_line_width, label=f'Throttle ({config.max_safe_temp}°C)')
        
        ax.set_xlabel('Time (seconds)', fontsize=self.config.font_size_label)
        ax.set_ylabel('Temperature (°C)', fontsize=self.config.font_size_label)
        ax.set_title('GPU Temperature', fontsize=self.config.font_size_title)
        ax.legend(loc='upper right', fontsize=self.config.font_size_tick)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(temps) - 5, max(max(temps) + 5, config.max_safe_temp + 5))
        
    def _plot_power(self, ax, times: List[float], powers: List[float], config: Any) -> None:
        """Plot power consumption over time."""
        ax.plot(times, powers, color=self.config.color_power, linewidth=self.config.line_width, label='Power')
        
        # Add TDP line
        ax.axhline(y=config.max_power, color=self.config.color_tdp_line, 
                   linestyle='--', linewidth=self.config.threshold_line_width, label=f'TDP ({config.max_power}W)')
        
        ax.set_xlabel('Time (seconds)', fontsize=self.config.font_size_label)
        ax.set_ylabel('Power (W)', fontsize=self.config.font_size_label)
        ax.set_title('Power Consumption', fontsize=self.config.font_size_title)
        ax.legend(loc='upper right', fontsize=self.config.font_size_tick)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(powers) + 50, config.max_power + 50))
        
    def _plot_utilization(self, ax, times: List[float], gpu_utils: List[float]) -> None:
        """Plot GPU utilization over time."""
        ax.plot(times, gpu_utils, color=self.config.color_utilization, linewidth=self.config.line_width)
        ax.fill_between(times, 0, gpu_utils, alpha=0.3, color=self.config.color_utilization)
        
        ax.set_xlabel('Time (seconds)', fontsize=self.config.font_size_label)
        ax.set_ylabel('Utilization (%)', fontsize=self.config.font_size_label)
        ax.set_title('GPU Utilization', fontsize=self.config.font_size_title)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
    def _plot_memory(self, ax, times: List[float], memory_gb: List[float], memory_total_gb: float) -> None:
        """Plot memory usage over time."""
        ax.plot(times, memory_gb, color=self.config.color_memory, linewidth=self.config.line_width, label='Used')
        ax.fill_between(times, 0, memory_gb, alpha=0.3, color=self.config.color_memory)
        
        # Add total memory line
        ax.axhline(y=memory_total_gb, color='gray', linestyle='--', 
                   linewidth=self.config.threshold_line_width, label=f'Total ({memory_total_gb:.0f} GB)')
        
        ax.set_xlabel('Time (seconds)', fontsize=self.config.font_size_label)
        ax.set_ylabel('Memory (GB)', fontsize=self.config.font_size_label)
        ax.set_title('GPU Memory Usage', fontsize=self.config.font_size_title)
        ax.legend(loc='upper right', fontsize=self.config.font_size_tick)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, memory_total_gb * 1.1)
        
    def _add_summary_box(self, fig, summary: Any, config: Any) -> None:
        """Add summary statistics text box."""
        summary_text = (
            f"Duration: {summary.duration_seconds:.1f}s | "
            f"Temp: {summary.temp_min:.0f}-{summary.temp_max:.0f}°C (avg {summary.temp_avg:.1f}°C) | "
            f"Power: {summary.power_min:.0f}-{summary.power_max:.0f}W (avg {summary.power_avg:.0f}W) | "
            f"Memory: {summary.memory_max_used_mb/1024:.1f} GB ({summary.memory_max_used_pct:.0f}%)"
        )
        
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=self.config.font_size_summary,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                 
    def _add_status_indicator(self, fig, summary: Any, config: Any) -> None:
        """Add thermal status indicator."""
        if summary.throttling_detected:
            status_text = "⚠️ THROTTLING DETECTED"
            status_color = '#FF0000'
        elif not summary.is_thermally_safe:
            status_text = "⚠️ ABOVE TARGET TEMPERATURE"
            status_color = '#FFA500'
        else:
            status_text = "✓ THERMALLY SAFE"
            status_color = '#28A745'
            
        fig.text(0.98, 0.98, status_text, ha='right', va='top', fontsize=12, fontweight='bold',
                 color=status_color, transform=fig.transFigure)
                 
    def save_comparison_plot(
        self,
        runs: List[Dict[str, Any]],
        output_path: str,
        title: str = "Configuration Comparison"
    ) -> bool:
        """Generate a comparison plot for multiple benchmark runs.
        
        Args:
            runs: List of dicts with 'name', 'samples', 'summary' keys
            output_path: Path to save PNG file
            title: Main title for the plot
            
        Returns:
            True if plot was saved successfully, False otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print(f"[ThermalVisualizer] Skipping plot (matplotlib not installed): {output_path}")
            return False
            
        if not runs:
            print(f"[ThermalVisualizer] No runs to compare")
            return False
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_width, self.config.figure_height))
            fig.suptitle(title, fontsize=self.config.font_size_title + 2, fontweight='bold')
            
            colors = plt.cm.Set2.colors  # Use colorblind-friendly palette
            
            for i, run in enumerate(runs):
                name = run.get('name', f'Run {i+1}')
                samples = run.get('samples', [])
                color = colors[i % len(colors)]
                
                if not samples:
                    continue
                    
                start_time = samples[0].timestamp
                times = [(s.timestamp - start_time) for s in samples]
                temps = [s.temperature for s in samples]
                powers = [s.power for s in samples]
                gpu_utils = [s.gpu_utilization for s in samples]
                memory_used = [s.memory_used / 1024 for s in samples]
                
                # Temperature
                axes[0, 0].plot(times, temps, color=color, linewidth=self.config.line_width, label=name)
                
                # Power
                axes[0, 1].plot(times, powers, color=color, linewidth=self.config.line_width, label=name)
                
                # Utilization
                axes[1, 0].plot(times, gpu_utils, color=color, linewidth=self.config.line_width, label=name)
                
                # Memory
                axes[1, 1].plot(times, memory_used, color=color, linewidth=self.config.line_width, label=name)
            
            # Configure axes
            axes[0, 0].set_title('Temperature', fontsize=self.config.font_size_title)
            axes[0, 0].set_ylabel('°C')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Power', fontsize=self.config.font_size_title)
            axes[0, 1].set_ylabel('Watts')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_title('GPU Utilization', fontsize=self.config.font_size_title)
            axes[1, 0].set_ylabel('%')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_title('Memory Usage', fontsize=self.config.font_size_title)
            axes[1, 1].set_ylabel('GB')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"[ThermalVisualizer] Saved comparison plot: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ThermalVisualizer] Error saving comparison plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
