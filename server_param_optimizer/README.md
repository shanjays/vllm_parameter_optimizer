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

### 4. Server Profiling Worker (`server_profiling_worker.py`)
- Ray-based and local profiling workers
- Runs vLLM throughput benchmarks with thermal monitoring
- Parses benchmark output for throughput metrics

### 5. Config Exporter (`server_config_exporter.py`)
- Exports aggressive and sustained configurations
- Generates bash launch scripts for easy deployment
- Saves complete optimization results

### 6. Feedback Collector (`server_feedback_collector.py`)
- Tracks optimization history for LLM learning
- Persists state across optimization runs
- Generates formatted feedback for LLM prompts

### 7. Meta-Controller (`server_meta_controller.py`)
- LLM-based configuration generation using gpt-oss-20b
- LoRA adapters for efficient fine-tuning
- Thermal-aware configuration suggestions

### 8. Main Optimizer (`server_optimizer.py`)
- Coordinates all components
- Runs iterative optimization loop
- Generates final results and launch scripts

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
- Ray (for distributed profiling)
  ```bash
  pip install ray
  ```
- Unsloth (for LLM meta-controller)
  ```bash
  pip install unsloth
  ```

## Usage

### Full Optimization Run

```bash
cd server_param_optimizer
python server_optimizer.py
```

This will:
1. Run 8 optimization iterations
2. Test 2-4 configs per iteration (LLM-guided)
3. Run 20-minute benchmarks with thermal monitoring
4. Save thermal plots and optimization results
5. Generate launch scripts for best configs

### Quick Test (Shorter Benchmarks)

```python
from server_param_optimizer import ServerParameterOptimizer

optimizer = ServerParameterOptimizer(
    benchmark_duration_minutes=5,  # Shorter for testing
    num_iterations=2
)
optimizer.run_optimization()
```

### Output Files

After optimization:
- `config_aggressive.json` - Max throughput config
- `config_sustained.json` - Thermal-safe config
- `optimization_results.json` - Full results
- `thermal_plots/*.png` - Thermal visualizations
- `launch_scripts/*.sh` - Ready-to-run commands

## Module Structure

```
server_param_optimizer/
├── __init__.py                    # Package exports
├── thermal_monitor.py             # GPU temp/power monitoring (PR 1)
├── visualization.py               # Thermal plot generation (PR 1)
├── nsys_metrics_extractor.py      # nsys profiling (PR 1)
├── server_profiling_worker.py     # Ray benchmark worker (PR 2)
├── server_config_exporter.py      # Config export + launch scripts (PR 2)
├── server_feedback_collector.py   # Feedback for LLM (PR 2)
├── server_meta_controller.py      # LLM config generation (PR 3)
├── server_optimizer.py            # Main entry point (PR 3)
└── README.md                      # Documentation
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

### LLM Meta-Controller

```python
from server_param_optimizer import ServerMetaController, ServerFeedbackCollector

# Create feedback collector (tracks optimization history)
feedback = ServerFeedbackCollector()

# Create meta-controller
controller = ServerMetaController()

# Generate configurations based on feedback
configs = controller.generate_configs(feedback)

for config in configs:
    print(f"Config: {config['name']}")
    print(f"  --max-num-seqs {config['max_num_seqs']}")
    print(f"  --max-num-batched-tokens {config['max_num_batched_tokens']}")
    print(f"  Rationale: {config.get('rationale', 'N/A')}")
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

After running the full optimizer:

```
./server_optimization_results/
├── thermal_plots/
│   ├── seqs32_tokens8192.png
│   ├── seqs64_tokens16384.png
│   └── ...
├── launch_scripts/
│   ├── launch_aggressive.sh
│   └── launch_sustained.sh
├── config_aggressive.json
├── config_sustained.json
├── optimization_results.json
└── feedback_state.json
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

### ServerMetaController

| Method | Description |
|--------|-------------|
| `generate_configs()` | Generate configs using LLM |
| `get_param_space()` | Get valid parameter values |
| `is_llm_available()` | Check if LLM is available |

### ServerParameterOptimizer

| Method | Description |
|--------|-------------|
| `run_optimization()` | Run full optimization loop |
| `print_final_summary()` | Print results summary |

## License

MIT License - See repository root for details.
