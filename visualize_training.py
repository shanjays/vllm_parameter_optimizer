#!/usr/bin/env python3
"""
Visualization script for RL training progress analysis.

This script parses TensorBoard event logs and JSON experiment logs to generate
visualizations for kernel parameter optimization training. It produces:
- Training curves (episode rewards, policy loss, value loss, entropy)
- Reward distribution by token count
- Configuration exploration heatmap
- NCU metrics progression

Usage:
    python visualize_training.py --log-dir ./logs --output-dir ./visualizations --show
    
    Options:
        --log-dir: Directory containing TensorBoard logs and training data
        --output-dir: Directory to save visualization PNG files
        --results-file: Path to all_results.json file (optional)
        --show: Display plots interactively
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np

# Import field type constants from training_logger
try:
    from training_logger import INTEGER_FIELDS, FLOAT_FIELDS
except ImportError:
    # Fallback if training_logger is not available
    INTEGER_FIELDS = ('step', 'token_count', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 
                      'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps', 'num_stages')
    FLOAT_FIELDS = ('reward', 'cumulative_reward', 'best_reward',
                    'sm_throughput', 'dram_throughput', 'l1_hit_rate', 'l2_hit_rate')


def load_tensorboard_scalars(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load scalar data from TensorBoard event files.
    
    Attempts to use tensorboard package if available, otherwise returns empty dict.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        Dict mapping metric names to lists of (step, value) tuples
    """
    scalars = defaultdict(list)
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find all event files recursively
        event_files = []
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                if f.startswith('events.out.tfevents'):
                    event_files.append(root)
                    break  # One per directory
        
        if not event_files:
            print(f"[Visualize] No TensorBoard event files found in {log_dir}")
            return scalars
            
        for event_dir in event_files:
            try:
                ea = EventAccumulator(event_dir)
                ea.Reload()
                
                for tag in ea.Tags().get('scalars', []):
                    for event in ea.Scalars(tag):
                        scalars[tag].append((event.step, event.value))
            except Exception as e:
                print(f"[Visualize] Warning: Could not load {event_dir}: {e}")
                continue
                
        print(f"[Visualize] Loaded {len(scalars)} scalar metrics from TensorBoard logs")
        
    except ImportError:
        print("[Visualize] TensorBoard not installed. Skipping TensorBoard log parsing.")
        print("[Visualize] Install with: pip install tensorboard")
        
    return scalars


def load_json_results(results_file: str) -> List[Dict[str, Any]]:
    """
    Load experiment results from JSON file.
    
    Args:
        results_file: Path to all_results.json file
        
    Returns:
        List of result dictionaries
    """
    if not os.path.exists(results_file):
        print(f"[Visualize] Results file not found: {results_file}")
        return []
        
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"[Visualize] Loaded {len(results)} results from {results_file}")
        return results
    except json.JSONDecodeError as e:
        print(f"[Visualize] Error parsing JSON file: {e}")
        return []


def load_csv_training_log(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load training log from CSV file.
    
    Args:
        csv_path: Path to training_log CSV file
        
    Returns:
        List of entry dictionaries
    """
    if not os.path.exists(csv_path):
        print(f"[Visualize] CSV file not found: {csv_path}")
        return []
        
    entries = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {}
                for key, value in row.items():
                    if value == '' or value is None:
                        entry[key] = None
                    elif key in INTEGER_FIELDS:
                        entry[key] = int(value) if value else None
                    elif key in FLOAT_FIELDS:
                        entry[key] = float(value) if value else None
                    else:
                        entry[key] = value
                entries.append(entry)
        print(f"[Visualize] Loaded {len(entries)} entries from {csv_path}")
    except Exception as e:
        print(f"[Visualize] Error loading CSV: {e}")
        
    return entries


def plot_training_curves(
    scalars: Dict[str, List[Tuple[int, float]]],
    results: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Generate training curves visualization.
    
    Creates a 2x2 grid showing:
    - Episode rewards
    - Policy loss
    - Value loss  
    - Entropy
    
    Args:
        scalars: TensorBoard scalar data
        results: JSON experiment results
        output_path: Path to save the figure
        
    Returns:
        True if figure was generated successfully
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    
    has_data = False
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    reward_data = None
    
    # Try TensorBoard data first
    for tag in ['rollout/ep_rew_mean', 'train/reward', 'episode_reward']:
        if tag in scalars and scalars[tag]:
            reward_data = scalars[tag]
            break
    
    # Fall back to JSON results
    if reward_data is None and results:
        reward_data = [(i, r.get('reward', 0)) for i, r in enumerate(results) 
                       if r.get('reward') is not None]
    
    if reward_data:
        steps, values = zip(*sorted(reward_data))
        ax.plot(steps, values, 'b-', linewidth=1.5, alpha=0.7)
        ax.fill_between(steps, 0, values, alpha=0.2)
        has_data = True
    ax.set_xlabel('Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Policy Loss
    ax = axes[0, 1]
    policy_loss_data = None
    for tag in ['train/policy_loss', 'policy_loss', 'loss/policy']:
        if tag in scalars and scalars[tag]:
            policy_loss_data = scalars[tag]
            break
            
    if policy_loss_data:
        steps, values = zip(*sorted(policy_loss_data))
        ax.plot(steps, values, 'r-', linewidth=1.5)
        has_data = True
    ax.set_xlabel('Steps')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Value Loss
    ax = axes[1, 0]
    value_loss_data = None
    for tag in ['train/value_loss', 'value_loss', 'loss/value']:
        if tag in scalars and scalars[tag]:
            value_loss_data = scalars[tag]
            break
            
    if value_loss_data:
        steps, values = zip(*sorted(value_loss_data))
        ax.plot(steps, values, 'g-', linewidth=1.5)
        has_data = True
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Entropy
    ax = axes[1, 1]
    entropy_data = None
    for tag in ['train/entropy_loss', 'entropy', 'train/entropy']:
        if tag in scalars and scalars[tag]:
            entropy_data = scalars[tag]
            break
            
    if entropy_data:
        steps, values = zip(*sorted(entropy_data))
        ax.plot(steps, values, 'm-', linewidth=1.5)
        has_data = True
    ax.set_xlabel('Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved training curves to: {output_path}")
    
    return has_data


def plot_reward_by_token_count(
    results: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Generate reward distribution by token count visualization.
    
    Creates box plots and bar charts showing reward distribution per token count.
    
    Args:
        results: JSON experiment results
        output_path: Path to save the figure
        
    Returns:
        True if figure was generated successfully
    """
    if not results:
        print("[Visualize] No results data for reward by token count plot")
        return False
    
    # Group rewards by token count
    token_rewards = defaultdict(list)
    for r in results:
        tc = r.get('token_count')
        reward = r.get('reward')
        if tc is not None and reward is not None:
            token_rewards[tc].append(reward)
    
    if not token_rewards:
        print("[Visualize] No valid token count data found")
        return False
    
    # Sort by token count
    sorted_counts = sorted(token_rewards.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Reward Distribution by Token Count', fontsize=14, fontweight='bold')
    
    # Plot 1: Box plots
    ax = axes[0]
    box_data = [token_rewards[tc] for tc in sorted_counts]
    positions = range(len(sorted_counts))
    
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([str(tc) for tc in sorted_counts], rotation=45, ha='right')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Best rewards bar chart
    ax = axes[1]
    best_rewards = [max(token_rewards[tc]) for tc in sorted_counts]
    mean_rewards = [np.mean(token_rewards[tc]) for tc in sorted_counts]
    
    x = np.arange(len(sorted_counts))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_rewards, width, label='Best', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, mean_rewards, width, label='Mean', color='blue', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in sorted_counts], rotation=45, ha='right')
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Reward')
    ax.set_title('Best vs Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved reward by token count to: {output_path}")
    
    return True


def plot_config_exploration(
    results: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Generate configuration exploration heatmap.
    
    Shows which kernel parameters were explored and their mean rewards.
    
    Args:
        results: JSON experiment results
        output_path: Path to save the figure
        
    Returns:
        True if figure was generated successfully
    """
    if not results:
        print("[Visualize] No results data for config exploration plot")
        return False
    
    # Collect unique values for each parameter
    param_values = {
        'BLOCK_SIZE_M': set(),
        'BLOCK_SIZE_N': set(),
        'BLOCK_SIZE_K': set(),
        'num_warps': set(),
        'num_stages': set(),
    }
    
    # Group rewards by parameter value
    param_rewards = {p: defaultdict(list) for p in param_values}
    
    for r in results:
        config = r.get('config', {})
        reward = r.get('reward')
        if reward is None:
            continue
            
        for param in param_values:
            val = config.get(param)
            if val is not None:
                param_values[param].add(val)
                param_rewards[param][val].append(reward)
    
    # Filter out empty parameters
    valid_params = [p for p in param_values if param_values[p]]
    if not valid_params:
        print("[Visualize] No configuration data found")
        return False
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle('Configuration Exploration Analysis', fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis
    
    for idx, param in enumerate(valid_params):
        ax = axes[idx]
        
        sorted_vals = sorted(param_values[param])
        mean_rewards = [np.mean(param_rewards[param][v]) for v in sorted_vals]
        counts = [len(param_rewards[param][v]) for v in sorted_vals]
        
        # Normalize colors by reward
        if mean_rewards:
            norm_rewards = np.array(mean_rewards)
            norm_rewards = (norm_rewards - norm_rewards.min()) / (norm_rewards.max() - norm_rewards.min() + 1e-8)
        else:
            norm_rewards = np.zeros(len(sorted_vals))
        
        bars = ax.bar(range(len(sorted_vals)), mean_rewards, 
                     color=[colors(r) for r in norm_rewards], alpha=0.8)
        
        # Add count annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.annotate(f'n={count}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(range(len(sorted_vals)))
        ax.set_xticklabels([str(v) for v in sorted_vals])
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'{param} Exploration')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(valid_params), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved config exploration to: {output_path}")
    
    return True


def plot_ncu_metrics(
    results: List[Dict[str, Any]],
    output_path: str
) -> bool:
    """
    Generate NCU metrics progression visualization.
    
    Creates a 2x2 grid showing SM throughput, DRAM throughput, L1 and L2 cache hit rates.
    
    Args:
        results: JSON experiment results
        output_path: Path to save the figure
        
    Returns:
        True if figure was generated successfully
    """
    if not results:
        print("[Visualize] No results data for NCU metrics plot")
        return False
    
    # Extract metrics
    steps = []
    sm_throughput = []
    dram_throughput = []
    l1_hit_rate = []
    l2_hit_rate = []
    
    for i, r in enumerate(results):
        metrics = r.get('metrics', {})
        if not metrics:
            continue
            
        steps.append(i)
        sm_throughput.append(metrics.get('sm_throughput'))
        dram_throughput.append(metrics.get('dram_throughput'))
        l1_hit_rate.append(metrics.get('l1_hit_rate'))
        l2_hit_rate.append(metrics.get('l2_hit_rate'))
    
    if not steps:
        print("[Visualize] No NCU metrics data found")
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('NCU Metrics Progression', fontsize=14, fontweight='bold')
    
    # Helper to plot with rolling average
    def plot_metric(ax, x, y, title, ylabel, color):
        # Filter None values
        valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if not valid:
            ax.set_title(f'{title} (No Data)')
            return
            
        x_valid, y_valid = zip(*valid)
        
        ax.plot(x_valid, y_valid, f'{color}o', alpha=0.3, markersize=4, label='Raw')
        
        # Rolling average
        if len(y_valid) >= 5:
            window = min(10, len(y_valid) // 2)
            y_smooth = np.convolve(y_valid, np.ones(window)/window, mode='valid')
            x_smooth = x_valid[window-1:]
            ax.plot(x_smooth, y_smooth, f'{color}-', linewidth=2, label=f'Rolling Avg (n={window})')
        
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plot_metric(axes[0, 0], steps, sm_throughput, 'SM Throughput', '% of Peak', 'b')
    plot_metric(axes[0, 1], steps, dram_throughput, 'DRAM Throughput', '% of Peak', 'g')
    plot_metric(axes[1, 0], steps, l1_hit_rate, 'L1 Cache Hit Rate', '%', 'r')
    plot_metric(axes[1, 1], steps, l2_hit_rate, 'L2 Cache Hit Rate', '%', 'm')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved NCU metrics to: {output_path}")
    
    return True


def find_results_file(log_dir: str) -> Optional[str]:
    """
    Find the all_results.json file in the log directory or optimized_configs.
    
    Args:
        log_dir: Base directory to search from
        
    Returns:
        Path to results file if found, None otherwise
    """
    # Try common locations
    candidates = [
        os.path.join(log_dir, 'all_results.json'),
        os.path.join(log_dir, '..', 'optimized_configs', 'all_results.json'),
        './optimized_configs/all_results.json',
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # Search recursively
    for root, dirs, files in os.walk(log_dir):
        if 'all_results.json' in files:
            return os.path.join(root, 'all_results.json')
    
    return None


def find_csv_log_files(log_dir: str) -> List[str]:
    """
    Find training log CSV files in the log directory.
    
    Args:
        log_dir: Base directory to search
        
    Returns:
        List of paths to CSV log files
    """
    csv_files = []
    
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('training_log') and f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    return csv_files


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations for RL training progress analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_training.py --log-dir ./logs --output-dir ./visualizations
    python visualize_training.py --results-file ./optimized_configs/all_results.json --show
    python visualize_training.py --log-dir ./logs --output-dir ./viz --show

Output Files:
    training_curves.png       - Episode rewards, policy loss, value loss, entropy
    reward_by_token_count.png - Token count analysis with box plots
    config_exploration.png    - Heatmap of explored configurations
    ncu_metrics_progression.png - NCU metric trends over time
        """
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory containing TensorBoard logs (default: ./logs)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualizations',
        help='Directory to save visualization PNG files (default: ./visualizations)'
    )
    
    parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Path to all_results.json file (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively after saving'
    )
    
    args = parser.parse_args()
    
    # Note: Backend switching after initial import is unreliable in most environments.
    # The plots are saved to files regardless, and --show may require TkAgg support.
    show_plots = args.show
    
    print("=" * 60)
    print("Training Visualization Tool")
    print("=" * 60)
    print(f"Log directory: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load TensorBoard scalars
    scalars = {}
    if os.path.exists(args.log_dir):
        scalars = load_tensorboard_scalars(args.log_dir)
    else:
        print(f"[Visualize] Log directory not found: {args.log_dir}")
    
    # Load JSON results
    results = []
    results_file = args.results_file
    if results_file is None:
        results_file = find_results_file(args.log_dir)
        
    if results_file:
        print(f"[Visualize] Using results file: {results_file}")
        results = load_json_results(results_file)
    else:
        print("[Visualize] No results file found")
    
    # Load CSV training logs if available
    csv_files = find_csv_log_files(args.log_dir)
    csv_entries = []
    for csv_file in csv_files:
        csv_entries.extend(load_csv_training_log(csv_file))
    
    # Merge CSV entries into results if we have them
    if csv_entries and not results:
        # Convert CSV entries to results format
        results = [
            {
                'token_count': e.get('token_count'),
                'reward': e.get('reward'),
                'config': {
                    'BLOCK_SIZE_M': e.get('BLOCK_SIZE_M'),
                    'BLOCK_SIZE_N': e.get('BLOCK_SIZE_N'),
                    'BLOCK_SIZE_K': e.get('BLOCK_SIZE_K'),
                    'num_warps': e.get('num_warps'),
                    'num_stages': e.get('num_stages'),
                },
                'metrics': {
                    'sm_throughput': e.get('sm_throughput'),
                    'dram_throughput': e.get('dram_throughput'),
                    'l1_hit_rate': e.get('l1_hit_rate'),
                    'l2_hit_rate': e.get('l2_hit_rate'),
                }
            }
            for e in csv_entries
        ]
    
    # Generate visualizations
    print("\n" + "-" * 40)
    print("Generating Visualizations")
    print("-" * 40)
    
    generated = 0
    
    # Training curves
    output_path = os.path.join(args.output_dir, 'training_curves.png')
    if plot_training_curves(scalars, results, output_path):
        generated += 1
    
    # Reward by token count
    output_path = os.path.join(args.output_dir, 'reward_by_token_count.png')
    if plot_reward_by_token_count(results, output_path):
        generated += 1
    
    # Config exploration
    output_path = os.path.join(args.output_dir, 'config_exploration.png')
    if plot_config_exploration(results, output_path):
        generated += 1
    
    # NCU metrics
    output_path = os.path.join(args.output_dir, 'ncu_metrics_progression.png')
    if plot_ncu_metrics(results, output_path):
        generated += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Generated {generated} visualization(s) in {args.output_dir}")
    print("=" * 60)
    
    if show_plots:
        print("\nDisplaying plots... (close windows to exit)")
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[Visualize] Could not display plots interactively: {e}")
            print("[Visualize] Plots have been saved to files.")
    
    return 0 if generated > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
