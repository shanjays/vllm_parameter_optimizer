# Server Parameter Optimizer for vLLM

Optimizes vLLM server parameters `--max-num-seqs` and `--max-num-batched-tokens` for optimal performance on NVIDIA A100 40GB GPU with meta-llama/Llama-3.1-8B-Instruct model.

## Target Configuration

| Parameter | Value |
|-----------|-------|
| **GPU** | NVIDIA A100 40GB |
| **Model** | meta-llama/Llama-3.1-8B-Instruct |
| **Focus** | Server-level parameters (not kernel-level) |

## Features

### 1. Thermal Monitoring (`thermal_monitor.py`)
- Real-time GPU temperature, power, and memory monitoring via nvidia-smi
- Background thread collection (1-second intervals)
- Summary statistics (min/max/avg for all metrics)
- Thermal safety classification for sustained operation

### 2. Visualization (`visualization.py`)
- Generates PNG thermal profile plots
- 2x2 subplot layout: Temperature, Power, Utilization, Memory
- Thermal zone coloring (safe/warning/danger)
- Status indicators for thermal safety

### 3. Nsys Profiling (`nsys_metrics_extractor.py`)
- NVIDIA Nsight Systems integration
- Kernel execution count and timing extraction
- Memory copy bandwidth analysis
- Graceful handling when nsys is not installed

## Output Configurations

The optimizer produces two configurations:

| Config | Description | Temp Target |
|--------|-------------|-------------|
| **Aggressive** | Maximum throughput | Up to 83°C (throttle limit) |
| **Sustained** | 24/7 safe operation | Below 75°C |

## Installation Requirements

### Required
- Python 3.8+
- nvidia-smi (comes with NVIDIA drivers)

### Optional
- matplotlib (for visualization)
  ```bash
  pip install matplotlib
  ```
- nsys (NVIDIA Nsight Systems, for detailed profiling)
  - Download from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems)

## Module Structure

```
server_param_optimizer/
├── __init__.py              # Package exports
├── thermal_monitor.py       # GPU thermal/power monitoring
├── visualization.py         # PNG plot generation
├── nsys_metrics_extractor.py # Nsys profiling integration
└── README.md                # This file
```

## Quick Start

### Thermal Monitoring

```python
from server_param_optimizer import ThermalMonitor, ThermalConfig

# Create monitor with A100 40GB defaults
monitor = ThermalMonitor(gpu_index=0)

# Start monitoring
monitor.start_monitoring()

# ... run your benchmark here ...
import time
time.sleep(10)

# Stop and get results
monitor.stop_monitoring()

# Get summary statistics
summary = monitor.get_thermal_summary()
print(f"Temperature: {summary.temp_min:.1f}°C - {summary.temp_max:.1f}°C")
print(f"Power: {summary.power_avg:.0f}W")
print(f"Thermally safe: {summary.is_thermally_safe}")

# Get raw samples for plotting
samples = monitor.get_samples()
```

### Visualization

```python
from server_param_optimizer import ThermalVisualizer, ThermalMonitor

# After running thermal monitoring
monitor = ThermalMonitor()
monitor.start_monitoring()
# ... benchmark ...
monitor.stop_monitoring()

samples = monitor.get_samples()
summary = monitor.get_thermal_summary()

# Create plot
visualizer = ThermalVisualizer()
if visualizer.is_available():
    visualizer.save_thermal_plot(
        samples=samples,
        summary=summary,
        output_path='thermal_profile.png',
        title='Benchmark Thermal Profile'
    )
```

### Nsys Profiling

```python
from server_param_optimizer import NsysMetricsExtractor, check_nsys_available

# Check if nsys is available
if check_nsys_available():
    extractor = NsysMetricsExtractor(gpu_index=0)
    
    profile = extractor.profile_command(
        command="python benchmark.py --config test",
        output_dir="./profiles",
        profile_name="benchmark_v1"
    )
    
    if profile.success:
        print(f"Kernel count: {profile.metrics.kernel_count}")
        print(f"Total kernel time: {profile.metrics.total_kernel_time_ms:.2f}ms")
        
        # Get formatted summary
        summary = extractor.get_metrics_summary(profile)
        print(summary)
else:
    print("nsys not installed, skipping profiling")
```

## Thermal Thresholds (A100 40GB)

| Threshold | Value | Description |
|-----------|-------|-------------|
| `target_sustained_temp` | 75°C | Target for 24/7 operation |
| `warning_temp` | 80°C | Warning threshold |
| `max_safe_temp` | 83°C | Throttling threshold |
| `max_power` | 400W | TDP limit |
| `total_memory` | 40GB | HBM2e memory |

## Output File Structure

When running the full optimizer (coming in PR 2/3):

```
./server_optimization_results/
├── thermal/
│   ├── config_1_thermal.png
│   ├── config_2_thermal.png
│   └── comparison.png
├── profiles/
│   ├── config_1.nsys-rep
│   └── config_2.nsys-rep
├── results/
│   ├── aggressive_config.json
│   ├── sustained_config.json
│   └── full_results.json
└── logs/
    └── optimization.log
```

## API Reference

### ThermalMonitor

| Method | Description |
|--------|-------------|
| `start_monitoring()` | Begin background monitoring |
| `stop_monitoring()` | Stop monitoring thread |
| `get_samples()` | Get list of ThermalSample objects |
| `get_samples_as_dict()` | Get samples as JSON-serializable dicts |
| `get_thermal_summary()` | Get ThermalSummary with statistics |
| `is_thermally_safe()` | Check if run stayed below target temp |

### ThermalVisualizer

| Method | Description |
|--------|-------------|
| `is_available()` | Check if matplotlib is installed |
| `save_thermal_plot()` | Generate 2x2 thermal profile plot |
| `save_comparison_plot()` | Compare multiple runs |

### NsysMetricsExtractor

| Method | Description |
|--------|-------------|
| `is_available()` | Check if nsys is installed |
| `profile_command()` | Run nsys profile on a command |
| `get_metrics_summary()` | Get formatted metrics summary |
| `export_to_sqlite()` | Export profile to SQLite |

## Upcoming PRs

This is PR 1 of 3:

- **PR 1** (this): Core infrastructure (thermal monitoring, visualization, nsys extraction)
- **PR 2**: Benchmark runner and parameter search
- **PR 3**: Full optimization loop and result reporting

## License

MIT License - See repository root for details.
