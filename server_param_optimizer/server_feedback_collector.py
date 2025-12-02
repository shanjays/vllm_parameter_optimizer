"""
Server Feedback Collector for LLM Meta-Controller

Collects and formats feedback from server parameter optimization runs
for use in LLM-guided optimization (to be used in PR 3).

Features:
- Tracks all configurations tested and their results
- Persists state to JSON file for resuming optimization
- Generates formatted feedback strings for LLM prompts
- Identifies untested configurations from parameter space
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple


@dataclass
class IterationFeedback:
    """Feedback from a single optimization iteration."""
    iteration: int
    configs_tested: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    best_aggressive_throughput: float
    best_sustained_throughput: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ServerFeedbackCollector:
    """Collects and formats feedback for LLM-guided server optimization.
    
    Tracks optimization history across iterations and generates formatted
    feedback for injection into LLM prompts, enabling the meta-controller
    to learn from previous results.
    
    Example:
        collector = ServerFeedbackCollector(state_file="./feedback_state.json")
        
        # After each iteration
        collector.add_iteration(configs_tested, results)
        
        # Get feedback for LLM prompt
        feedback = collector.get_feedback_for_prompt()
        
        # Find untested configurations
        untested = collector.get_untested_configs(param_space)
    """
    
    DEFAULT_STATE_FILE = "./server_feedback_state.json"
    
    def __init__(self, state_file: Optional[str] = None):
        """Initialize the feedback collector.
        
        Args:
            state_file: Path to persist feedback state across runs.
                       If None, uses DEFAULT_STATE_FILE.
        """
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        
        # Core tracking data
        self.iterations: List[IterationFeedback] = []
        self.all_configs_tested: List[Dict[str, Any]] = []
        self.all_results: List[Dict[str, Any]] = []
        
        # Best results tracking
        self.best_aggressive_throughput: float = 0.0
        self.best_aggressive_config: Optional[Dict[str, Any]] = None
        self.best_sustained_throughput: float = 0.0
        self.best_sustained_config: Optional[Dict[str, Any]] = None
        
        # Load existing state if available
        self._load_state()
    
    def add_iteration(
        self,
        configs: List[Dict[str, Any]],
        results: List[Any]
    ) -> None:
        """Add results from an optimization iteration.
        
        Args:
            configs: List of configurations tested (each with max_num_seqs, max_num_batched_tokens)
            results: List of BenchmarkResult objects or result dictionaries
        """
        iteration_num = len(self.iterations) + 1
        timestamp = datetime.now().isoformat()
        
        # Process results
        processed_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            # Handle both BenchmarkResult objects and dicts
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {'error': 'Unknown result format'}
            
            # Link config to result
            if i < len(configs):
                result_dict['config'] = configs[i]
            
            # Extract error information for RL learning
            is_successful = result_dict.get('is_successful', True)
            if not is_successful:
                # If is_successful not explicitly set, check for error
                if result_dict.get('error') or result_dict.get('throughput', 0.0) <= 0.0:
                    is_successful = False
            
            result_dict['is_successful'] = is_successful
            
            # Track effective throughput for RL reward
            if is_successful:
                result_dict['effective_throughput'] = result_dict.get('throughput', 0.0)
            else:
                result_dict['effective_throughput'] = result_dict.get('penalty', -10.0)
                failed_count += 1
            
            processed_results.append(result_dict)
            
            # Update best configs (only for successful benchmarks)
            if is_successful:
                throughput = result_dict.get('throughput', 0.0)
                is_thermally_safe = result_dict.get('is_thermally_safe', False)
                config = configs[i] if i < len(configs) else {}
                
                if throughput > self.best_aggressive_throughput:
                    self.best_aggressive_throughput = throughput
                    self.best_aggressive_config = config.copy()
                
                if is_thermally_safe and throughput > self.best_sustained_throughput:
                    self.best_sustained_throughput = throughput
                    self.best_sustained_config = config.copy()
        
        # Create iteration feedback
        iteration = IterationFeedback(
            iteration=iteration_num,
            configs_tested=configs,
            results=processed_results,
            best_aggressive_throughput=self.best_aggressive_throughput,
            best_sustained_throughput=self.best_sustained_throughput,
            timestamp=timestamp
        )
        
        self.iterations.append(iteration)
        self.all_configs_tested.extend(configs)
        self.all_results.extend(processed_results)
        
        print(f"[ServerFeedbackCollector] Iteration {iteration_num}: {len(configs)} configs tested ({failed_count} failed)")
        print(f"[ServerFeedbackCollector] Best aggressive: {self.best_aggressive_throughput:.2f} tokens/sec")
        print(f"[ServerFeedbackCollector] Best sustained: {self.best_sustained_throughput:.2f} tokens/sec")
        
        # Persist state
        self._save_state()
    
    def get_feedback_for_prompt(self, max_configs: int = 5) -> str:
        """Generate formatted feedback string for LLM prompts.
        
        Args:
            max_configs: Maximum number of best configs to include
            
        Returns:
            Formatted string for injection into LLM prompt
        """
        if not self.iterations:
            return ""
        
        lines = []
        lines.append("")
        lines.append("═══════════════════════════════════════════════════════════════════════════════")
        lines.append("                 SERVER PARAMETER OPTIMIZATION FEEDBACK")
        lines.append("═══════════════════════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append(f"Iterations Completed: {len(self.iterations)}")
        lines.append(f"Total Configs Tested: {len(self.all_configs_tested)}")
        lines.append("")
        
        # Best configs found
        lines.append("BEST CONFIGURATIONS FOUND:")
        lines.append("")
        
        if self.best_aggressive_config:
            lines.append("  AGGRESSIVE (Max Throughput):")
            lines.append(f"    max_num_seqs: {self.best_aggressive_config.get('max_num_seqs', 'N/A')}")
            lines.append(f"    max_num_batched_tokens: {self.best_aggressive_config.get('max_num_batched_tokens', 'N/A')}")
            lines.append(f"    Throughput: {self.best_aggressive_throughput:.2f} tokens/sec")
            lines.append("")
        
        if self.best_sustained_config:
            lines.append("  SUSTAINED (Thermally Safe):")
            lines.append(f"    max_num_seqs: {self.best_sustained_config.get('max_num_seqs', 'N/A')}")
            lines.append(f"    max_num_batched_tokens: {self.best_sustained_config.get('max_num_batched_tokens', 'N/A')}")
            lines.append(f"    Throughput: {self.best_sustained_throughput:.2f} tokens/sec")
            lines.append("")
        
        # Recent iteration results
        lines.append("RECENT ITERATION RESULTS:")
        recent_iterations = self.iterations[-3:]  # Last 3 iterations
        
        for iteration in recent_iterations:
            lines.append(f"\n  Iteration {iteration.iteration} ({iteration.timestamp}):")
            
            # Sort results by throughput
            sorted_results = sorted(
                iteration.results,
                key=lambda x: x.get('throughput', 0.0),
                reverse=True
            )[:max_configs]
            
            for result in sorted_results:
                config = result.get('config', {})
                throughput = result.get('throughput', 0.0)
                is_safe = result.get('is_thermally_safe', False)
                safe_marker = "✓" if is_safe else "✗"
                
                lines.append(
                    f"    seqs={config.get('max_num_seqs', '?'):4d}, "
                    f"tokens={config.get('max_num_batched_tokens', '?'):5d} → "
                    f"{throughput:8.2f} tok/s [{safe_marker}]"
                )
        
        # Patterns observed
        lines.append("")
        lines.append("PATTERNS OBSERVED:")
        self._add_pattern_observations(lines)
        
        # Add failure summary for LLM to learn from
        self._add_failure_summary(lines)
        
        lines.append("")
        lines.append(f"YOUR GOAL: Find configs that exceed {self.best_aggressive_throughput:.2f} tokens/sec")
        lines.append("           while maintaining thermal safety (<75°C for A100)")
        lines.append("")
        
        return "\n".join(lines)
    
    def _add_failure_summary(self, lines: List[str]) -> None:
        """Add summary of failed configurations for LLM learning."""
        failed_configs = [
            r for r in self.all_results 
            if not r.get('is_successful', True) or r.get('error')
        ]
        
        if not failed_configs:
            return
        
        lines.append("")
        lines.append(f"⚠️ FAILED CONFIGURATIONS ({len(failed_configs)} total):")
        
        # Show the last 5 failures
        recent_failures = failed_configs[-5:]
        for r in recent_failures:
            config = r.get('config', {})
            error_type = r.get('error_type', 'unknown')
            penalty = r.get('penalty', -10.0)
            lines.append(
                f"  ❌ seqs={config.get('max_num_seqs', '?')}, "
                f"tokens={config.get('max_num_batched_tokens', '?')} → "
                f"{error_type} (penalty: {penalty})"
            )
        
        lines.append("  → LLM should AVOID these parameter ranges!")
    
    def _add_pattern_observations(self, lines: List[str]) -> None:
        """Analyze results and add pattern observations."""
        if len(self.all_results) < 3:
            lines.append("  (Not enough data for pattern analysis)")
            return
        
        # Analyze correlation between parameters and throughput
        high_throughput_configs = [
            r for r in self.all_results
            if r.get('throughput', 0) > self.best_aggressive_throughput * 0.8
        ]
        
        thermally_safe_configs = [
            r for r in self.all_results
            if r.get('is_thermally_safe', False)
        ]
        
        if high_throughput_configs:
            avg_seqs = sum(
                r.get('config', {}).get('max_num_seqs', 0)
                for r in high_throughput_configs
            ) / len(high_throughput_configs)
            avg_tokens = sum(
                r.get('config', {}).get('max_num_batched_tokens', 0)
                for r in high_throughput_configs
            ) / len(high_throughput_configs)
            
            lines.append(f"  • High-throughput configs average: seqs≈{avg_seqs:.0f}, tokens≈{avg_tokens:.0f}")
        
        if thermally_safe_configs:
            max_safe_seqs = max(
                r.get('config', {}).get('max_num_seqs', 0)
                for r in thermally_safe_configs
            )
            lines.append(f"  • Thermally safe up to: max_num_seqs≈{max_safe_seqs}")
        
        # Count failures
        failures = [r for r in self.all_results if r.get('error')]
        if failures:
            lines.append(f"  • {len(failures)} configs failed (avoid similar settings)")
    
    def get_untested_configs(
        self,
        param_space: Dict[str, List[int]]
    ) -> List[Dict[str, int]]:
        """Get configurations not yet tested from parameter space.
        
        Args:
            param_space: Dictionary with 'max_num_seqs' and 'max_num_batched_tokens' lists
            
        Returns:
            List of untested configuration dictionaries
        """
        tested_set = set()
        for config in self.all_configs_tested:
            key = (
                config.get('max_num_seqs', 0),
                config.get('max_num_batched_tokens', 0)
            )
            tested_set.add(key)
        
        untested = []
        seqs_values = param_space.get('max_num_seqs', [])
        tokens_values = param_space.get('max_num_batched_tokens', [])
        
        for seqs in seqs_values:
            for tokens in tokens_values:
                if (seqs, tokens) not in tested_set:
                    untested.append({
                        'max_num_seqs': seqs,
                        'max_num_batched_tokens': tokens
                    })
        
        return untested
    
    def reset(self) -> None:
        """Reset all feedback state."""
        self.iterations = []
        self.all_configs_tested = []
        self.all_results = []
        self.best_aggressive_throughput = 0.0
        self.best_aggressive_config = None
        self.best_sustained_throughput = 0.0
        self.best_sustained_config = None
        
        # Remove state file
        if os.path.exists(self.state_file):
            try:
                os.remove(self.state_file)
            except IOError:
                pass
        
        print("[ServerFeedbackCollector] State reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feedback collector state.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'iterations_completed': len(self.iterations),
            'total_configs_tested': len(self.all_configs_tested),
            'best_aggressive_throughput': self.best_aggressive_throughput,
            'best_aggressive_config': self.best_aggressive_config,
            'best_sustained_throughput': self.best_sustained_throughput,
            'best_sustained_config': self.best_sustained_config,
        }
    
    def _load_state(self) -> None:
        """Load persisted state from file if it exists."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Restore iterations
            self.iterations = [
                IterationFeedback(**iter_data)
                for iter_data in data.get('iterations', [])
            ]
            
            self.all_configs_tested = data.get('all_configs_tested', [])
            self.all_results = data.get('all_results', [])
            
            self.best_aggressive_throughput = data.get('best_aggressive_throughput', 0.0)
            self.best_aggressive_config = data.get('best_aggressive_config')
            self.best_sustained_throughput = data.get('best_sustained_throughput', 0.0)
            self.best_sustained_config = data.get('best_sustained_config')
            
            print(f"[ServerFeedbackCollector] Loaded state from {self.state_file}")
            print(f"[ServerFeedbackCollector] Resuming with {len(self.iterations)} iterations")
            
        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"[ServerFeedbackCollector] Warning: Could not load state file: {e}")
    
    def _save_state(self) -> None:
        """Save current state to file for persistence."""
        data = {
            'iterations': [iter_fb.to_dict() for iter_fb in self.iterations],
            'all_configs_tested': self.all_configs_tested,
            'all_results': self.all_results,
            'best_aggressive_throughput': self.best_aggressive_throughput,
            'best_aggressive_config': self.best_aggressive_config,
            'best_sustained_throughput': self.best_sustained_throughput,
            'best_sustained_config': self.best_sustained_config,
            'last_updated': datetime.now().isoformat(),
        }
        
        try:
            # Ensure directory exists
            dir_path = os.path.dirname(self.state_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except IOError as e:
            print(f"[ServerFeedbackCollector] Warning: Could not save state file: {e}")
