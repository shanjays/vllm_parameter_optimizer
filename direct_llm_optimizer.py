"""
DirectLLMOptimizer - Direct LLM-based kernel optimizer without PPO.

This module implements a simplified optimization approach where the LLM
directly generates specific kernel configurations (BLOCK_SIZE_M, BLOCK_SIZE_N,
etc.) instead of generating search spaces that are explored by PPO agents.

Benefits over PPO-based approach:
- 5x-10x faster (no PPO training per token count)
- No action space mismatches
- Direct feedback loop (LLM sees previous results in prompt)
- More interpretable (LLM reasoning visible)

Expected flow:
1. LLM generates 3-5 specific configs based on baseline metrics
2. Profile each config with NCU for each token count
3. Compute rewards, update best configs
4. Run throughput validation
5. Update feedback and iterate
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

# Conditional imports for testing without full dependencies
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from config_exporter import VLLMConfigExporter, TOKEN_COUNTS_ALL
except ImportError:
    TOKEN_COUNTS_ALL = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024,
                        1536, 2048, 3072, 4096, 5120, 9216, 13312, 17408,
                        25600, 33792, 41984, 50176, 58368]
    VLLMConfigExporter = None

try:
    from feedback_collector import FeedbackCollector
except ImportError:
    FeedbackCollector = None


# Valid parameter values for kernel configuration
VALID_BLOCK_SIZE_M = [16, 32, 64, 128]
VALID_BLOCK_SIZE_N = [32, 64, 128]
VALID_BLOCK_SIZE_K = [32, 64]
VALID_NUM_WARPS = [4, 8, 16]
VALID_NUM_STAGES = [2, 3, 4, 5]

# Default configuration for fallback
DEFAULT_KERNEL_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "num_warps": 8,
    "num_stages": 4
}


class DirectLLMOptimizer:
    """
    Direct LLM-based kernel optimizer without PPO.
    
    This class replaces the PPO-based exploration approach with direct
    LLM configuration generation. The LLM analyzes NCU profiling metrics
    and generates specific kernel configurations to test.
    
    Attributes:
        profiler: Ray actor handle to ProfilingWorker
        config_exporter: VLLMConfigExporter for saving best configs
        feedback: FeedbackCollector for contextual learning
        static_args: Static benchmark arguments
        best_configs: Dict mapping token_count -> {config, reward}
        best_rewards: Dict mapping token_count -> reward
    """
    
    def __init__(
        self,
        profiling_worker: Any,
        config_exporter: VLLMConfigExporter,
        feedback_collector: FeedbackCollector,
        static_args: Dict[str, Any],
        token_counts: Optional[List[int]] = None,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        user_goal: str = "throughput"
    ):
        """
        Initialize the direct LLM optimizer.
        
        Args:
            profiling_worker: Ray actor handle to ProfilingWorker
            config_exporter: VLLMConfigExporter for saving configs
            feedback_collector: FeedbackCollector for tracking history
            static_args: Static benchmark arguments (num_experts, inter_size, etc.)
            token_counts: List of token counts to optimize
            model_name: Model name for throughput validation
            user_goal: Optimization goal ('throughput' or 'latency')
        """
        self.profiler = profiling_worker
        self.config_exporter = config_exporter
        self.feedback = feedback_collector
        self.static_args = static_args
        self.token_counts = token_counts or [1, 16, 64, 256, 1024, 4096]
        self.model_name = model_name
        self.user_goal = user_goal
        
        # Tracking
        self.best_configs: Dict[int, Dict[str, Any]] = {}
        self.best_rewards: Dict[int, float] = {}
        for tc in self.token_counts:
            self.best_rewards[tc] = float('-inf')
        
        # Default objective weights for reward calculation
        self.objective_weights = {
            "R_sm_throughput": 0.4,
            "R_dram_throughput": 0.3,
            "R_l1_hit_rate": 0.15,
            "R_l2_hit_rate": 0.15
        }
        
        print(f"[DirectLLMOptimizer] Initialized with {len(self.token_counts)} token counts")
    
    def optimize(
        self,
        llm_output: str,
        run_throughput_validation: bool = True
    ) -> Tuple[Dict[int, Dict[str, Any]], float]:
        """
        Run one optimization iteration with LLM-generated configs.
        
        Args:
            llm_output: Raw LLM output containing kernel configurations
            run_throughput_validation: Whether to run throughput validation
            
        Returns:
            Tuple of (best_configs, best_reward) where:
            - best_configs: Dict mapping token_count -> config
            - best_reward: Best throughput/reward achieved
        """
        print(f"\n--- [DirectLLMOptimizer] Starting Optimization Iteration ---")
        
        # 1. Parse configs from LLM output
        configs = self.parse_configs(llm_output)
        
        if not configs:
            print("[DirectLLMOptimizer] No valid configs generated, using defaults")
            configs = [self.get_default_config()]
        
        print(f"[DirectLLMOptimizer] Got {len(configs)} configurations to test")
        
        # 2. Profile each configuration for each token count
        all_results = []
        for config in configs:
            # Get token counts this config is suggested for
            config_token_counts = config.get('token_counts', self.token_counts)
            
            for token_count in config_token_counts:
                if token_count not in self.token_counts:
                    continue
                
                # Profile this config
                result = self._profile_config(config, token_count)
                all_results.append(result)
                
                # Update best config if improved
                if result['reward'] > self.best_rewards.get(token_count, float('-inf')):
                    self.best_rewards[token_count] = result['reward']
                    self.best_configs[token_count] = {
                        'config': result['config'],
                        'reward': result['reward']
                    }
                    
                    # Update config exporter
                    if self.config_exporter:
                        self.config_exporter.update_best_config(
                            token_count=token_count,
                            config=result['config'],
                            reward=result['reward'],
                            metrics=result.get('metrics')
                        )
                    
                    print(f"[DirectLLMOptimizer] New best for {token_count} tokens: reward={result['reward']:.2f}")
        
        # 3. Run throughput validation
        best_throughput = 0.0
        if run_throughput_validation and self.best_configs:
            print(f"[DirectLLMOptimizer] Running throughput validation...")
            best_throughput = self._run_throughput_validation()
            print(f"[DirectLLMOptimizer] Throughput: {best_throughput:.1f} tokens/sec")
        
        # 4. Update feedback collector
        if self.feedback:
            # Record as a policy result for compatibility
            policy = {
                "objective_weights": self.objective_weights,
                "search_space": {},  # Not used in direct mode
                "configs": configs
            }
            self.feedback.record_policy_result(
                policy=policy,
                reward=best_throughput,
                best_configs=self.best_configs
            )
        
        return self.best_configs, best_throughput
    
    def parse_configs(self, llm_output: str) -> List[Dict[str, Any]]:
        """
        Parse kernel configurations from LLM output.
        
        Expects LLM output in format:
        <param>
        {
          "configs": [
            {
              "token_counts": [1, 2, 4, 8, 16],
              "BLOCK_SIZE_M": 64,
              "BLOCK_SIZE_N": 64,
              ...
            },
            ...
          ]
        }
        </param>
        
        Args:
            llm_output: Raw LLM output string
            
        Returns:
            List of validated configuration dicts
        """
        configs = []
        
        # Try to extract JSON from <param> tags
        param_match = re.search(
            r'<param>\s*(\{[\s\S]*?\})\s*</param>',
            llm_output,
            re.DOTALL | re.IGNORECASE
        )
        
        if not param_match:
            # Fallback: try to find any JSON object
            json_match = re.search(r'(\{[\s\S]*\})', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                print("[DirectLLMOptimizer] No JSON found in output")
                return []
        else:
            json_str = param_match.group(1)
        
        # Clean and parse JSON
        try:
            json_str = self._clean_json(json_str)
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[DirectLLMOptimizer] JSON parse error: {e}")
            # Try to recover
            try:
                data = self._recover_json(json_str)
            except Exception:
                return []
        
        # Extract configs from parsed data
        if isinstance(data, dict):
            if 'configs' in data:
                raw_configs = data['configs']
            elif 'config' in data:
                raw_configs = [data['config']]
            else:
                # Single config at top level
                raw_configs = [data]
        elif isinstance(data, list):
            raw_configs = data
        else:
            return []
        
        # Validate each config
        for raw_config in raw_configs:
            if not isinstance(raw_config, dict):
                continue
            
            validated = self._validate_config(raw_config)
            if validated:
                configs.append(validated)
        
        return configs
    
    def _clean_json(self, json_str: str) -> str:
        """Clean JSON string for parsing."""
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # Remove trailing commas before ] or }
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Balance braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
    def _recover_json(self, json_str: str) -> Dict:
        """Try to recover valid JSON from malformed string."""
        import ast
        try:
            return ast.literal_eval(json_str)
        except Exception:
            pass
        
        # Last resort: try to extract key-value pairs
        result = {}
        for key in ['BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps', 'num_stages']:
            match = re.search(rf'"{key}"\s*:\s*(\d+)', json_str)
            if match:
                result[key] = int(match.group(1))
        
        if result:
            return {'configs': [result]}
        
        raise ValueError("Could not recover JSON")
    
    def _validate_config(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate and coerce a kernel configuration.
        
        Args:
            config: Raw configuration dict
            
        Returns:
            Validated config dict, or None if invalid
        """
        validated = {}
        
        # Extract and validate BLOCK_SIZE_M
        m = config.get('BLOCK_SIZE_M', DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_M'])
        try:
            m = int(m)
        except (TypeError, ValueError):
            m = DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_M']
        if m not in VALID_BLOCK_SIZE_M:
            # Clamp to nearest valid value
            m = min(VALID_BLOCK_SIZE_M, key=lambda x: abs(x - m))
        validated['BLOCK_SIZE_M'] = m
        
        # Extract and validate BLOCK_SIZE_N
        n = config.get('BLOCK_SIZE_N', DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_N'])
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_N']
        if n not in VALID_BLOCK_SIZE_N:
            n = min(VALID_BLOCK_SIZE_N, key=lambda x: abs(x - n))
        validated['BLOCK_SIZE_N'] = n
        
        # Extract and validate BLOCK_SIZE_K
        k = config.get('BLOCK_SIZE_K', DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_K'])
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = DEFAULT_KERNEL_CONFIG['BLOCK_SIZE_K']
        if k not in VALID_BLOCK_SIZE_K:
            k = min(VALID_BLOCK_SIZE_K, key=lambda x: abs(x - k))
        validated['BLOCK_SIZE_K'] = k
        
        # Extract and validate num_warps
        warps = config.get('num_warps', DEFAULT_KERNEL_CONFIG['num_warps'])
        try:
            warps = int(warps)
        except (TypeError, ValueError):
            warps = DEFAULT_KERNEL_CONFIG['num_warps']
        if warps not in VALID_NUM_WARPS:
            warps = min(VALID_NUM_WARPS, key=lambda x: abs(x - warps))
        validated['num_warps'] = warps
        
        # Extract and validate num_stages
        stages = config.get('num_stages', DEFAULT_KERNEL_CONFIG['num_stages'])
        try:
            stages = int(stages)
        except (TypeError, ValueError):
            stages = DEFAULT_KERNEL_CONFIG['num_stages']
        if stages not in VALID_NUM_STAGES:
            stages = min(VALID_NUM_STAGES, key=lambda x: abs(x - stages))
        validated['num_stages'] = stages
        
        # Validate against shared memory limit
        if not self._check_shared_memory_limit(validated):
            print(f"[DirectLLMOptimizer] Config exceeds shared memory limit: {validated}")
            # Try reducing stages
            for s in reversed(VALID_NUM_STAGES):
                validated['num_stages'] = s
                if self._check_shared_memory_limit(validated):
                    break
            else:
                return None
        
        # Handle token_counts
        tc = config.get('token_counts', self.token_counts)
        if isinstance(tc, list):
            validated['token_counts'] = [int(t) for t in tc if isinstance(t, (int, float))]
        else:
            validated['token_counts'] = self.token_counts
        
        return validated
    
    def _check_shared_memory_limit(self, config: Dict[str, Any]) -> bool:
        """Check if config is within H100 shared memory limit."""
        M = config.get('BLOCK_SIZE_M', 64)
        N = config.get('BLOCK_SIZE_N', 64)
        K = config.get('BLOCK_SIZE_K', 32)
        stages = config.get('num_stages', 4)
        
        # Shared memory formula: (M*K + K*N) * 2 * stages
        shared_mem = (M * K + K * N) * 2 * stages
        H100_LIMIT = 232448  # 228 KB
        
        return shared_mem <= H100_LIMIT
    
    def _profile_config(self, config: Dict[str, Any], token_count: int) -> Dict[str, Any]:
        """
        Profile a configuration for a specific token count.
        
        Args:
            config: Kernel configuration
            token_count: Number of tokens to test
            
        Returns:
            Dict with keys: config, token_count, metrics, reward
        """
        # Prepare config for profiling (remove token_counts)
        profile_config = {k: v for k, v in config.items() if k != 'token_counts'}
        
        if not RAY_AVAILABLE:
            print(f"[DirectLLMOptimizer] Ray not available, returning mock result")
            return {
                'config': profile_config,
                'token_count': token_count,
                'metrics': None,
                'reward': 0.0
            }
        
        try:
            # Call profiling worker
            result_id = self.profiler.run_kernel_profiling.remote(
                profile_config,
                self.static_args,
                self.objective_weights,
                token_count
            )
            state, reward, csv_data = ray.get(result_id)
            
            if state is not None:
                metrics = {
                    'sm_throughput': float(state[0]),
                    'dram_throughput': float(state[1]),
                    'l1_hit_rate': float(state[2]),
                    'l2_hit_rate': float(state[3])
                }
            else:
                metrics = None
                reward = reward if reward is not None else -20.0
                
        except Exception as e:
            print(f"[DirectLLMOptimizer] Profiling failed: {e}")
            metrics = None
            reward = -20.0
        
        return {
            'config': profile_config,
            'token_count': token_count,
            'metrics': metrics,
            'reward': reward
        }
    
    def _run_throughput_validation(self) -> float:
        """Run throughput validation on best configurations."""
        if not self.best_configs:
            return 0.0
        
        if not RAY_AVAILABLE:
            print(f"[DirectLLMOptimizer] Ray not available, skipping throughput validation")
            return 0.0
        
        # Get best overall config (highest reward)
        best_tc = max(self.best_configs.keys(), key=lambda tc: self.best_configs[tc].get('reward', 0))
        best_config = self.best_configs[best_tc]['config']
        
        try:
            result_id = self.profiler.run_throughput_validation.remote(
                best_config,
                self.model_name,
                self.user_goal
            )
            throughput = ray.get(result_id)
            return float(throughput)
        except Exception as e:
            print(f"[DirectLLMOptimizer] Throughput validation failed: {e}")
            return 0.0
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'token_counts': self.token_counts.copy(),
            **DEFAULT_KERNEL_CONFIG
        }
    
    def get_best_configs(self) -> Dict[int, Dict[str, Any]]:
        """Get best configurations found."""
        return self.best_configs.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            'total_token_counts': len(self.token_counts),
            'optimized_token_counts': len(self.best_configs),
            'best_rewards': {tc: r for tc, r in self.best_rewards.items() if r > float('-inf')}
        }


def build_direct_optimization_prompt(
    ncu_summary: str,
    model_name: str,
    num_experts: int,
    inter_size: int,
    hidden_size: int,
    top_k: int,
    dtype: str,
    token_counts: List[int],
    feedback_collector: Optional[FeedbackCollector] = None
) -> str:
    """
    Build optimization prompt for direct LLM configuration generation.
    
    This prompt asks the LLM to generate specific kernel configurations
    (not search spaces) based on profiling metrics and feedback.
    
    Args:
        ncu_summary: Summary of NCU profiling metrics
        model_name: Target model name
        num_experts: Number of experts (E)
        inter_size: Intermediate size (N)
        hidden_size: Hidden size
        top_k: Top-K experts
        dtype: Data type (e.g., 'bf16')
        token_counts: List of token counts to optimize
        feedback_collector: Optional feedback collector
        
    Returns:
        Formatted prompt string
    """
    token_counts_str = ", ".join(str(tc) for tc in token_counts)
    
    # Get feedback from previous iterations
    feedback_str = ""
    if feedback_collector:
        feedback_str = feedback_collector.format_feedback_for_prompt()
    
    prompt = f'''You are an expert CUDA kernel optimization engineer. Generate specific kernel configurations for the fused_moe kernel.

═══════════════════════════════════════════════════════════════════════════════
                              KERNEL DETAILS
═══════════════════════════════════════════════════════════════════════════════

KERNEL: fused_moe_kernel (Triton)
MODEL: {model_name}
- Number of Experts (E): {num_experts}
- Intermediate Size (N): {inter_size}
- Hidden Size: {hidden_size}
- Top-K Experts: {top_k}
- Data Type: {dtype}

TOKEN COUNTS TO OPTIMIZE: {token_counts_str}

═══════════════════════════════════════════════════════════════════════════════
                              HARDWARE SPECS
═══════════════════════════════════════════════════════════════════════════════

GPU: NVIDIA H100 80GB HBM3
- Shared Memory per SM: 228 KB (232,448 bytes)
- Shared memory formula: (M×K + K×N) × 2 × stages ≤ 232,448 bytes

═══════════════════════════════════════════════════════════════════════════════
                           BASELINE PROFILING METRICS
═══════════════════════════════════════════════════════════════════════════════

{ncu_summary}

═══════════════════════════════════════════════════════════════════════════════
                           VALID PARAMETER VALUES
═══════════════════════════════════════════════════════════════════════════════

- BLOCK_SIZE_M: 16, 32, 64, or 128
- BLOCK_SIZE_N: 32, 64, or 128
- BLOCK_SIZE_K: 32 or 64
- num_warps: 4, 8, or 16
- num_stages: 2, 3, 4, or 5
{feedback_str}
═══════════════════════════════════════════════════════════════════════════════
                           YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Generate 3-5 specific kernel configurations to test. For EACH token count range, suggest the best configuration based on your analysis.

Output format:
<param>
{{
  "configs": [
    {{
      "token_counts": [1, 2, 4, 8, 16],
      "BLOCK_SIZE_M": 64,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 32,
      "num_warps": 8,
      "num_stages": 4
    }},
    {{
      "token_counts": [32, 64, 128, 256],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 128,
      "BLOCK_SIZE_K": 64,
      "num_warps": 16,
      "num_stages": 5
    }},
    {{
      "token_counts": [512, 1024, 2048, 4096],
      "BLOCK_SIZE_M": 128,
      "BLOCK_SIZE_N": 64,
      "BLOCK_SIZE_K": 64,
      "num_warps": 16,
      "num_stages": 4
    }}
  ]
}}
</param>

REASONING: <Brief 2-4 sentence explanation>

NOW GENERATE YOUR CONFIGURATIONS:
<param>
'''
    return prompt
