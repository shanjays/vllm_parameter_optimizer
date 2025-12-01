"""
FeedbackCollector - Collects and formats training feedback for LLM prompts.

This module provides contextual feedback to the LLM during iterative training.
It tracks:
- Best configurations found per token count
- Policy performance history
- Successful and failed strategies

The feedback is formatted for injection into the LLM prompt to help
the model learn from previous iterations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


class FeedbackCollector:
    """
    Collects and formats training feedback for contextual learning.
    
    This class tracks optimization history and formats it for injection
    into the LLM prompt, enabling iterative learning from past results.
    
    Attributes:
        state_file: Path to the JSON file for persisting state
        policies_evaluated: Number of policies evaluated so far
        best_overall_reward: Best reward achieved across all policies
        best_configs_by_token: Dict mapping token_count -> (config, reward)
        best_policy_weights: Best performing policy objective weights
        successful_strategies: List of strategies that worked well
        failed_strategies: List of strategies that didn't work
    """
    
    DEFAULT_STATE_FILE = "./feedback_state.json"
    
    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize the feedback collector.
        
        Args:
            state_file: Path to persist feedback state across runs.
                       If None, uses DEFAULT_STATE_FILE.
        """
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        
        # Core tracking data
        self.policies_evaluated: int = 0
        self.best_overall_reward: float = 0.0
        self.best_configs_by_token: Dict[int, Dict[str, Any]] = {}
        self.best_policy_weights: Dict[str, float] = {}
        self.best_policy_reward: float = 0.0
        
        # Strategy analysis
        self.successful_strategies: List[str] = []
        self.failed_strategies: List[str] = []
        
        # History for analysis
        self.policy_history: List[Dict[str, Any]] = []
        
        # Load existing state if available
        self._load_state()
    
    def record_policy_result(
        self,
        policy: Dict[str, Any],
        reward: float,
        best_configs: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> None:
        """
        Record results after each policy evaluation.
        
        Args:
            policy: The optimization policy that was evaluated
                   (should contain 'objective_weights' and 'search_space')
            reward: The reward achieved by this policy
            best_configs: Optional dict mapping token_count -> {config, reward}
        """
        self.policies_evaluated += 1
        
        # Extract objective weights from policy
        weights = policy.get("objective_weights", {})
        
        # Record in history
        self.policy_history.append({
            "timestamp": datetime.now().isoformat(),
            "policy_num": self.policies_evaluated,
            "reward": reward,
            "objective_weights": weights.copy(),
            "search_space": policy.get("search_space", {}),
        })
        
        # Update best policy if this is the best so far
        if reward > self.best_policy_reward:
            self.best_policy_reward = reward
            self.best_policy_weights = weights.copy()
            self._analyze_successful_strategy(policy, reward)
        else:
            self._analyze_failed_strategy(policy, reward)
        
        # Update best overall reward
        if reward > self.best_overall_reward:
            self.best_overall_reward = reward
        
        # Update best configs by token count
        if best_configs:
            for token_count, config_data in best_configs.items():
                tc = int(token_count)
                config = config_data.get("config", config_data)
                config_reward = config_data.get("reward", reward)
                
                if tc not in self.best_configs_by_token or config_reward > self.best_configs_by_token[tc].get("reward", 0):
                    self.best_configs_by_token[tc] = {
                        "config": config,
                        "reward": config_reward
                    }
        
        # Persist state after each policy
        self._save_state()
    
    def _analyze_successful_strategy(self, policy: Dict[str, Any], reward: float) -> None:
        """
        Analyze and record what made this policy successful.
        
        Args:
            policy: The successful policy
            reward: The reward achieved
        """
        weights = policy.get("objective_weights", {})
        search_space = policy.get("search_space", {})
        
        new_strategies = []
        
        # Analyze weight patterns
        sm_weight = weights.get("R_sm_throughput", 0)
        if sm_weight >= 0.4:
            strategy = f"High SM throughput weight ({sm_weight:.2f}) was effective"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        
        dram_weight = weights.get("R_dram_throughput", 0)
        if dram_weight >= 0.3:
            strategy = f"Strong DRAM throughput weight ({dram_weight:.2f}) was effective"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        
        # Analyze search space patterns
        block_m = search_space.get("BLOCK_SIZE_M", [])
        if isinstance(block_m, list) and 128 in block_m:
            strategy = "Including BLOCK_SIZE_M=128 was beneficial"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        
        num_stages = search_space.get("num_stages", [])
        if isinstance(num_stages, list) and 5 in num_stages:
            strategy = "Aggressive num_stages=5 improved performance"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        elif isinstance(num_stages, list) and 4 in num_stages:
            strategy = "Including num_stages=4 was beneficial"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        
        num_warps = search_space.get("num_warps", [])
        if isinstance(num_warps, list) and 16 in num_warps:
            strategy = "High warp count (16) improved throughput"
            if strategy not in self.successful_strategies:
                new_strategies.append(strategy)
        
        # Add new strategies (limit total to 6)
        for strategy in new_strategies:
            if len(self.successful_strategies) < 6:
                self.successful_strategies.append(strategy)
    
    def _analyze_failed_strategy(self, policy: Dict[str, Any], reward: float) -> None:
        """
        Analyze and record what made this policy underperform.
        
        Args:
            policy: The underperforming policy
            reward: The reward achieved
        """
        weights = policy.get("objective_weights", {})
        search_space = policy.get("search_space", {})
        
        # Only analyze if significantly worse than best
        if self.best_policy_reward > 0 and reward < self.best_policy_reward * 0.8:
            new_strategies = []
            
            # Analyze weight patterns
            sm_weight = weights.get("R_sm_throughput", 0)
            if sm_weight < 0.2:
                strategy = f"Low SM throughput weight ({sm_weight:.2f}) underperformed"
                if strategy not in self.failed_strategies:
                    new_strategies.append(strategy)
            
            # Analyze search space patterns
            block_m = search_space.get("BLOCK_SIZE_M", [])
            if isinstance(block_m, list) and all(v <= 32 for v in block_m):
                strategy = "Small BLOCK_SIZE_M values limited performance"
                if strategy not in self.failed_strategies:
                    new_strategies.append(strategy)
            
            num_stages = search_space.get("num_stages", [])
            if isinstance(num_stages, list) and all(v <= 2 for v in num_stages):
                strategy = "Low num_stages (<=2) limited memory hiding"
                if strategy not in self.failed_strategies:
                    new_strategies.append(strategy)
            
            # Add new strategies (limit total to 4)
            for strategy in new_strategies:
                if len(self.failed_strategies) < 4:
                    self.failed_strategies.append(strategy)
    
    def format_feedback_for_prompt(self) -> str:
        """
        Generate formatted feedback string for injection into the LLM prompt.
        
        Returns:
            Formatted string containing feedback from previous iterations.
            Returns empty string if no policies have been evaluated yet.
        """
        if self.policies_evaluated == 0:
            return ""
        
        lines = []
        
        # Header
        lines.append("=== FEEDBACK FROM PREVIOUS ITERATIONS ===")
        lines.append(f"Policies evaluated so far: {self.policies_evaluated}")
        lines.append(f"Best overall reward achieved: {self.best_overall_reward:.2f}")
        lines.append("")
        
        # Best configurations by token count
        if self.best_configs_by_token:
            lines.append("=== BEST CONFIGURATIONS FOUND ===")
            for token_count in sorted(self.best_configs_by_token.keys()):
                config_data = self.best_configs_by_token[token_count]
                config = config_data.get("config", {})
                config_reward = config_data.get("reward", 0)
                
                config_str = ", ".join([
                    f"M={config.get('BLOCK_SIZE_M', '?')}",
                    f"N={config.get('BLOCK_SIZE_N', '?')}",
                    f"K={config.get('BLOCK_SIZE_K', '?')}",
                    f"warps={config.get('num_warps', '?')}",
                    f"stages={config.get('num_stages', '?')}"
                ])
                lines.append(f"  Token {token_count}: reward={config_reward:.2f}, {config_str}")
            lines.append("")
        
        # Best policy weights
        if self.best_policy_weights:
            lines.append("=== BEST POLICY WEIGHTS ===")
            for key, value in sorted(self.best_policy_weights.items()):
                lines.append(f"  {key}: {value:.3f}")
            lines.append("")
        
        # What worked
        if self.successful_strategies:
            lines.append("=== WHAT WORKED ===")
            for strategy in self.successful_strategies:
                lines.append(f"  ✓ {strategy}")
            lines.append("")
        
        # What didn't work
        if self.failed_strategies:
            lines.append("=== WHAT DIDN'T WORK ===")
            for strategy in self.failed_strategies:
                lines.append(f"  ✗ {strategy}")
            lines.append("")
        
        # Guidance for next iteration
        lines.append("=== YOUR TASK ===")
        lines.append("Generate an IMPROVED policy that:")
        lines.append("  1. Builds on the successful strategies above")
        lines.append("  2. Avoids the approaches that didn't work")
        lines.append(f"  3. Tries to beat the best reward of {self.best_overall_reward:.2f}")
        lines.append("")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the feedback collector state.
        
        Returns:
            Dict containing summary statistics
        """
        return {
            "policies_evaluated": self.policies_evaluated,
            "best_overall_reward": self.best_overall_reward,
            "best_configs_count": len(self.best_configs_by_token),
            "successful_strategies_count": len(self.successful_strategies),
            "failed_strategies_count": len(self.failed_strategies),
            "history_length": len(self.policy_history),
        }
    
    def _load_state(self) -> None:
        """Load persisted state from file if it exists."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            self.policies_evaluated = data.get("policies_evaluated", 0)
            self.best_overall_reward = data.get("best_overall_reward", 0.0)
            self.best_policy_weights = data.get("best_policy_weights", {})
            self.best_policy_reward = data.get("best_policy_reward", 0.0)
            self.successful_strategies = data.get("successful_strategies", [])
            self.failed_strategies = data.get("failed_strategies", [])
            self.policy_history = data.get("policy_history", [])
            
            # Convert string keys back to int for best_configs_by_token
            raw_configs = data.get("best_configs_by_token", {})
            self.best_configs_by_token = {
                int(k): v for k, v in raw_configs.items()
            }
            
            print(f"[FeedbackCollector] Loaded state from {self.state_file}")
            print(f"[FeedbackCollector] Resuming with {self.policies_evaluated} policies evaluated")
        except (json.JSONDecodeError, IOError) as e:
            print(f"[FeedbackCollector] Warning: Could not load state file: {e}")
    
    def _save_state(self) -> None:
        """Save current state to file for persistence."""
        data = {
            "policies_evaluated": self.policies_evaluated,
            "best_overall_reward": self.best_overall_reward,
            "best_policy_weights": self.best_policy_weights,
            "best_policy_reward": self.best_policy_reward,
            "successful_strategies": self.successful_strategies,
            "failed_strategies": self.failed_strategies,
            "policy_history": self.policy_history,
            # Convert int keys to strings for JSON serialization
            "best_configs_by_token": {
                str(k): v for k, v in self.best_configs_by_token.items()
            },
            "last_updated": datetime.now().isoformat(),
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file) if os.path.dirname(self.state_file) else '.', exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"[FeedbackCollector] Warning: Could not save state file: {e}")
    
    def reset(self) -> None:
        """Reset all feedback data (useful for testing or starting fresh)."""
        self.policies_evaluated = 0
        self.best_overall_reward = 0.0
        self.best_configs_by_token = {}
        self.best_policy_weights = {}
        self.best_policy_reward = 0.0
        self.successful_strategies = []
        self.failed_strategies = []
        self.policy_history = []
        
        # Remove state file if it exists
        if os.path.exists(self.state_file):
            try:
                os.remove(self.state_file)
            except IOError:
                pass
