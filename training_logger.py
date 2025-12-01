"""
TrainingLogger - Structured logging module for RL training metrics.

This module provides the TrainingLogger class for CSV-based metrics logging
during kernel parameter optimization training. It logs per-step data including
step number, reward, configuration parameters, NCU metrics, and timestamps.

Usage:
    from training_logger import TrainingLogger
    
    logger = TrainingLogger(output_dir="./logs")
    logger.log_step(
        step=1,
        reward=50.0,
        config={"BLOCK_SIZE_M": 64, "num_warps": 4},
        metrics={"sm_throughput": 45.5}
    )
    logger.save()
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


# Field type constants for CSV parsing/loading
INTEGER_FIELDS = ('step', 'token_count', 'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 
                  'BLOCK_SIZE_K', 'GROUP_SIZE_M', 'num_warps', 'num_stages')
FLOAT_FIELDS = ('reward', 'cumulative_reward', 'best_reward',
                'sm_throughput', 'dram_throughput', 'l1_hit_rate', 'l2_hit_rate')


class TrainingLogger:
    """
    Structured logging class for RL training metrics.
    
    This class provides CSV-based metrics logging for per-step data during
    kernel parameter optimization training. It supports both CSV and JSON
    export formats and integrates with VLLMConfigExporter.
    
    Attributes:
        output_dir: Directory for saving log files
        session_id: Unique identifier for this training session
        entries: List of logged entries
    """
    
    # Column definitions for CSV export
    CSV_COLUMNS = [
        'step', 'timestamp', 'token_count', 'reward',
        'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K',
        'GROUP_SIZE_M', 'num_warps', 'num_stages',
        'sm_throughput', 'dram_throughput', 'l1_hit_rate', 'l2_hit_rate',
        'cumulative_reward', 'best_reward'
    ]
    
    def __init__(self, output_dir: str = "./logs", session_id: Optional[str] = None):
        """
        Initialize the training logger.
        
        Args:
            output_dir: Directory to save log files
            session_id: Optional unique identifier for this session.
                       If not provided, uses current timestamp.
        """
        self.output_dir = output_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.entries: List[Dict[str, Any]] = []
        self._cumulative_reward = 0.0
        self._best_reward = float('-inf')
        self._step_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
    def log_step(
        self,
        step: Optional[int] = None,
        reward: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        token_count: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a single training step.
        
        Args:
            step: Step number. If None, auto-increments from last step.
            reward: Reward value for this step
            config: Kernel configuration parameters (BLOCK_SIZE_M, etc.)
            metrics: NCU performance metrics (sm_throughput, etc.)
            token_count: Token count for this benchmark
            extra: Additional data to log
            
        Returns:
            Dict containing the logged entry
        """
        if step is None:
            step = self._step_counter
        self._step_counter = step + 1
        
        # Update cumulative statistics
        self._cumulative_reward += reward
        if reward > self._best_reward:
            self._best_reward = reward
        
        # Build entry
        config = config or {}
        metrics = metrics or {}
        
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'token_count': token_count,
            'reward': reward,
            'BLOCK_SIZE_M': config.get('BLOCK_SIZE_M'),
            'BLOCK_SIZE_N': config.get('BLOCK_SIZE_N'),
            'BLOCK_SIZE_K': config.get('BLOCK_SIZE_K'),
            'GROUP_SIZE_M': config.get('GROUP_SIZE_M'),
            'num_warps': config.get('num_warps'),
            'num_stages': config.get('num_stages'),
            'sm_throughput': metrics.get('sm_throughput'),
            'dram_throughput': metrics.get('dram_throughput'),
            'l1_hit_rate': metrics.get('l1_hit_rate'),
            'l2_hit_rate': metrics.get('l2_hit_rate'),
            'cumulative_reward': self._cumulative_reward,
            'best_reward': self._best_reward,
        }
        
        # Add extra data if provided
        if extra:
            entry['extra'] = extra
        
        self.entries.append(entry)
        return entry
    
    def get_csv_path(self) -> str:
        """Get the path to the CSV log file."""
        return os.path.join(self.output_dir, f"training_log_{self.session_id}.csv")
    
    def get_json_path(self) -> str:
        """Get the path to the JSON log file."""
        return os.path.join(self.output_dir, f"training_log_{self.session_id}.json")
    
    def save_csv(self, path: Optional[str] = None) -> str:
        """
        Save logged entries to CSV file.
        
        Args:
            path: Optional path to save to. If None, uses default path.
            
        Returns:
            Path to the saved CSV file
        """
        if path is None:
            path = self.get_csv_path()
            
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(entry)
                
        print(f"[TrainingLogger] Saved CSV log to: {path}")
        return path
    
    def save_json(self, path: Optional[str] = None) -> str:
        """
        Save logged entries to JSON file.
        
        Args:
            path: Optional path to save to. If None, uses default path.
            
        Returns:
            Path to the saved JSON file
        """
        if path is None:
            path = self.get_json_path()
            
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        data = {
            'session_id': self.session_id,
            'output_dir': self.output_dir,
            'total_steps': len(self.entries),
            'cumulative_reward': self._cumulative_reward,
            'best_reward': self._best_reward if self._best_reward != float('-inf') else None,
            'entries': self.entries
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        print(f"[TrainingLogger] Saved JSON log to: {path}")
        return path
    
    def save(self) -> tuple:
        """
        Save logged entries to both CSV and JSON files.
        
        Returns:
            Tuple of (csv_path, json_path)
        """
        csv_path = self.save_csv()
        json_path = self.save_json()
        return csv_path, json_path
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the training session.
        
        Returns:
            Dict containing summary statistics
        """
        if not self.entries:
            return {
                'session_id': self.session_id,
                'total_steps': 0,
                'cumulative_reward': 0.0,
                'best_reward': None,
                'mean_reward': None,
                'token_counts_covered': [],
            }
        
        rewards = [e['reward'] for e in self.entries if e.get('reward') is not None]
        token_counts = list(set(
            e['token_count'] for e in self.entries 
            if e.get('token_count') is not None
        ))
        
        return {
            'session_id': self.session_id,
            'total_steps': len(self.entries),
            'cumulative_reward': self._cumulative_reward,
            'best_reward': self._best_reward if self._best_reward != float('-inf') else None,
            'mean_reward': sum(rewards) / len(rewards) if rewards else None,
            'token_counts_covered': sorted(token_counts),
        }
    
    def get_entries_by_token_count(self, token_count: int) -> List[Dict[str, Any]]:
        """
        Get all entries for a specific token count.
        
        Args:
            token_count: Token count to filter by
            
        Returns:
            List of entries for the specified token count
        """
        return [e for e in self.entries if e.get('token_count') == token_count]
    
    def get_best_config_by_token_count(self, token_count: int) -> Optional[Dict[str, Any]]:
        """
        Get the best configuration for a specific token count.
        
        Args:
            token_count: Token count to find best config for
            
        Returns:
            Entry with highest reward for the token count, or None if no entries
        """
        entries = self.get_entries_by_token_count(token_count)
        if not entries:
            return None
        return max(entries, key=lambda e: e.get('reward', float('-inf')))
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """
        Export data in a format suitable for visualization.
        
        Returns:
            Dict containing visualization-friendly data structures
        """
        if not self.entries:
            return {
                'steps': [],
                'rewards': [],
                'cumulative_rewards': [],
                'best_rewards': [],
                'sm_throughputs': [],
                'dram_throughputs': [],
                'l1_hit_rates': [],
                'l2_hit_rates': [],
                'configs': [],
                'token_counts': [],
                'timestamps': [],
            }
        
        return {
            'steps': [e['step'] for e in self.entries],
            'rewards': [e.get('reward', 0) for e in self.entries],
            'cumulative_rewards': [e.get('cumulative_reward', 0) for e in self.entries],
            'best_rewards': [e.get('best_reward', 0) for e in self.entries],
            'sm_throughputs': [e.get('sm_throughput') for e in self.entries],
            'dram_throughputs': [e.get('dram_throughput') for e in self.entries],
            'l1_hit_rates': [e.get('l1_hit_rate') for e in self.entries],
            'l2_hit_rates': [e.get('l2_hit_rate') for e in self.entries],
            'configs': [
                {
                    'BLOCK_SIZE_M': e.get('BLOCK_SIZE_M'),
                    'BLOCK_SIZE_N': e.get('BLOCK_SIZE_N'),
                    'BLOCK_SIZE_K': e.get('BLOCK_SIZE_K'),
                    'num_warps': e.get('num_warps'),
                    'num_stages': e.get('num_stages'),
                }
                for e in self.entries
            ],
            'token_counts': [e.get('token_count') for e in self.entries],
            'timestamps': [e.get('timestamp') for e in self.entries],
        }
    
    @classmethod
    def load_from_csv(cls, path: str) -> 'TrainingLogger':
        """
        Load a TrainingLogger instance from a CSV file.
        
        Args:
            path: Path to the CSV file to load
            
        Returns:
            TrainingLogger instance with loaded entries
        """
        logger = cls(output_dir=os.path.dirname(path) or '.')
        # Reset cumulative stats before loading
        logger._cumulative_reward = 0.0
        logger._best_reward = float('-inf')
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values back to appropriate types
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
                
                logger.entries.append(entry)
                
                # Update cumulative stats
                if entry.get('reward') is not None:
                    logger._cumulative_reward += entry['reward']
                    if entry['reward'] > logger._best_reward:
                        logger._best_reward = entry['reward']
                        
        return logger
    
    @classmethod
    def load_from_json(cls, path: str) -> 'TrainingLogger':
        """
        Load a TrainingLogger instance from a JSON file.
        
        Args:
            path: Path to the JSON file to load
            
        Returns:
            TrainingLogger instance with loaded entries
        """
        with open(path, 'r') as f:
            data = json.load(f)
            
        logger = cls(
            output_dir=data.get('output_dir', os.path.dirname(path) or '.'),
            session_id=data.get('session_id')
        )
        logger.entries = data.get('entries', [])
        logger._cumulative_reward = data.get('cumulative_reward', 0.0)
        logger._best_reward = data.get('best_reward', float('-inf')) or float('-inf')
        
        return logger
