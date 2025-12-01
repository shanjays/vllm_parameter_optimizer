import json
import ray
import numpy as np
import time
import os
import re
import ast
from kernel_tuning_env import KernelTuningEnvironment
from exploration_agent import ExplorationAgent
from profiling_worker import ProfilingWorker
from config_exporter import TOKEN_COUNTS_ALL

POWER_OF_TWO_WARPS = (2, 4, 8, 16, 32)

# Default optimization policy with multiple options for PPO exploration
DEFAULT_OPTIMIZATION_POLICY = {
    "objective_weights": {
        "R_sm_throughput": 0.4,
        "R_dram_throughput": 0.3,
        "R_l1_hit_rate": 0.15,
        "R_l2_hit_rate": 0.15
    },
    "search_space": {
        "BLOCK_SIZE_M": [32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4]
    }
}

# Backward compatibility alias
DEFAULT_MISSION_PLAN = {
    "reward_function": DEFAULT_OPTIMIZATION_POLICY["objective_weights"],
    "pruned_action_space": DEFAULT_OPTIMIZATION_POLICY["search_space"]
}

# Token counts for training (key representative values for 5-hour run)
TOKEN_COUNTS_TRAINING = [1, 16, 64, 256, 1024, 4096, 16384]

# Default token counts to test (matching vLLM's expected format)
DEFAULT_TOKEN_COUNTS = TOKEN_COUNTS_TRAINING

# Minimum training steps per token count to ensure meaningful exploration
MIN_STEPS_PER_TOKEN = 8  # Reduced from 10 for faster iteration

# Number of top results to collect per token count
RESULTS_PER_TOKEN = 3

# Validation frequency: run vLLM validation after every N token counts
VALIDATE_EVERY_N_TOKENS = 3

class MetaControllerReward:
    """
    Meta-Controller Reward Function for hierarchical kernel optimization.
    
    This class orchestrates the hierarchical optimization process by:
    1. Parsing optimization policies from LLM outputs
    2. Running exploration phases using PPO agents
    3. Validating configurations through throughput benchmarks
    4. Computing rewards to guide the meta-learning process
    
    The reward function bridges the high-level policy generation (LLM)
    with the low-level configuration exploration (PPO agent).
    """
    def __init__(self, user_goal, model_name, exploration_steps, profiling_gpu_id, static_args, 
                 config_exporter=None, token_counts=None, training_logger=None, feedback_collector=None):
        """
        Initialize the meta-controller reward function.
        
        Args:
            user_goal: Optimization target ('throughput' or 'latency')
            model_name: Target model name for benchmarking
            exploration_steps: Number of exploration steps per optimization round
            profiling_gpu_id: GPU ID for the profiling worker
            static_args: Static benchmark arguments
            config_exporter: Optional VLLMConfigExporter for saving configs
            token_counts: List of token counts to test
            training_logger: Optional HierarchicalTrainingLogger for TensorBoard logging
            feedback_collector: Optional FeedbackCollector for contextual learning feedback
        """
        self.user_goal = user_goal
        self.model_name = model_name
        self.exploration_steps = exploration_steps
        self.static_args = static_args
        self.config_exporter = config_exporter
        self.token_counts = token_counts or DEFAULT_TOKEN_COUNTS
        self.training_logger = training_logger
        self.feedback_collector = feedback_collector
        self.num_experts = static_args.get('num_experts', 128)
        self.inter_size = static_args.get('inter_size', 1536)
        
        # Set up vLLM config directory
        try:
            import vllm
            vllm_lib_path = os.path.dirname(vllm.__file__)
            self.vllm_config_dir = os.path.join(
                vllm_lib_path, "model_executor/layers/fused_moe/configs/"
            )
            if not os.path.exists(self.vllm_config_dir):
                self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        except ImportError:
            print("[MetaController] vLLM not installed, using fallback config directory")
            self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        except OSError as e:
            print(f"[MetaController] Could not create vLLM config directory: {e}")
            self.vllm_config_dir = "/tmp/vllm_configs/"
            os.makedirs(self.vllm_config_dir, exist_ok=True)
        
        print(f"[MetaController] Requesting ProfilingWorker for PHYSICAL GPU {profiling_gpu_id}")
        self.worker = ProfilingWorker.options(num_gpus=1).remote(profiling_gpu_id)
        self.initial_state = self._get_initial_state()
        print(f"[MetaController] Configured to test {len(self.token_counts)} token counts: {self.token_counts[:5]}...")

    def _clean_non_json_types(self, data):
        if isinstance(data, dict):
            return {k: self._clean_non_json_types(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._clean_non_json_types(i) for i in data]
        if isinstance(data, set):
            return list(self._clean_non_json_types(list(data)))
        if data is Ellipsis:
            return "..."
        return data

    def _get_initial_state(self):
        print("[MetaController] Getting initial state from worker...")
        try:
            job_id = self.worker.run_kernel_profiling.remote(None, self.static_args, {})
            state, reward, _ = ray.get(job_id)
            if state is None:
                raise RuntimeError("Worker failed initial profile.")
            print("[MetaController] Initial state acquired.")
            return state
        except Exception as e:
            print(f"[MetaController] ERROR: Worker failed initial state check. Using fallback. {e}")
            return np.array([32.3, 40.8, 0.05, 69.9], dtype=np.float32)

    def __call__(self, completions, **kwargs):
        rewards = []
        for i, policy_str in enumerate(completions):
            print(f"\n--- [MetaController] Processing Optimization Policy {i+1}/{len(completions)} ---")
            
            # Debug output
            print(f"[MetaController] DEBUG: Raw LLM Output:\n{policy_str}\n")

            valid = True
            policy = None
            best_configs = {}
            try:
                policy = self._extract_json(policy_str)
                policy = self._clean_non_json_types(policy)
                policy = self._validate_and_coerce_policy(policy)

                path = f"temp_optimization_policy_{int(time.time())}_{i}.json"
                with open(path, "w") as f:
                    json.dump(policy, f, indent=2)

                print(f"[MetaController] Starting exploration phase ({self.exploration_steps} steps)...")
                top_configs, best_configs = self._run_exploration_phase(path)
                print(f"[MetaController] Starting throughput validation (Top {len(top_configs)} configs)...")
                final_metric = self._run_throughput_validation(top_configs)
                rewards.append(final_metric)
                
                # Record policy result to feedback collector
                if self.feedback_collector and policy:
                    self.feedback_collector.record_policy_result(
                        policy=policy,
                        reward=final_metric,
                        best_configs=best_configs
                    )
                
                os.remove(path)
            except Exception as e:
                valid = False
                print(f"[MetaController] ERROR: Reward calculation failed. Reason: {e}")
                rewards.append(0.0)
                if not valid:
                    self._run_default_penalty_policy(i)
        
        # Print training summary
        if self.config_exporter:
            summary = self.config_exporter.get_summary()
            print(f"\n[MetaController] === Training Summary ===")
            print(f"  Total configs tested: {summary['total_experiments']}")
            print(f"  Token counts covered: {summary['total_token_counts']}")
            print(f"  Best rewards by token count:")
            for tc, reward in sorted(summary['best_rewards'].items(), key=lambda x: int(x[0])):
                print(f"    {tc} tokens: {reward:.2f}")
        
        return rewards

    def _run_default_penalty_policy(self, idx):
        print("[MetaController] Policy failed. Running exploration phase on default policy to ensure training progression.")
        default_policy = {
            "objective_weights": {
                "R_sm_throughput": 0.01,
                "R_dram_throughput": 0.0,
                "R_l1_hit_rate": 0.0,
                "R_l2_hit_rate": 0.0
            },
            "search_space": DEFAULT_OPTIMIZATION_POLICY["search_space"].copy()
        }
        path = f"temp_default_policy_{int(time.time())}_{idx}.json"
        try:
            with open(path, "w") as f:
                json.dump(default_policy, f, indent=2)
            self._run_exploration_phase(path)
        except Exception as e:
            print(f"[MetaController] WARNING: Default exploration phase also failed. {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def _normalize_unicode(self, s: str) -> str:
        replacements = {
            '\u2011': '-', '\u202f': ' ', '\u2248': '~',
            '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
            '\u00A0': ' ', '\u2013': '-', '\u2014': '-', '\u2212': '-',
            '\uFF0C': ',', '\u200B': ''  # zero-width space
        }
        for bad, good in replacements.items():
            s = s.replace(bad, good)
        return s

    def _preclean_reward_arrays(self, text: str) -> str:
        """
        Collapse patterns like "R_sm_throughput": [0.2464, "Weight for SM Utilization"]
        into "R_sm_throughput": 0.2464 before JSON parsing.
        """
        pattern = re.compile(
            r'("R_(?:sm_throughput|dram_throughput|l1_hit_rate|l2_hit_rate)"\s*:\s*)\[\s*([0-9eE+.\-]+)\s*,\s*"[^"\]]*"\s*\]'
        )
        while True:
            new_text, count = pattern.subn(r'\1\2', text)
            if count == 0:
                break
            text = new_text
        return text

    def _strip_unterminated_quotes(self, s: str) -> str:
        # If odd number of double quotes, try to remove trailing partial segment
        if s.count('"') % 2 != 0:
            # Remove everything after the last complete pair boundary
            last_brace = max(s.rfind('}'), s.rfind(']'))
            if last_brace != -1:
                s = s[:last_brace+1]
        return s

    def _clean_number_string(self, s: str) -> str:
        """
        Clean a number string by removing trailing periods, handling scientific notation,
        and other common LLM formatting issues.
        """
        s = s.strip()
        # Remove trailing periods (e.g., "42.75." -> "42.75")
        s = re.sub(r'\.+$', '', s)
        # Remove leading/trailing whitespace again after cleanup
        s = s.strip()
        # Handle case where LLM outputs just 'e' or partial scientific notation
        if s.lower() in ('e', 'e+', 'e-', '+e', '-e', ''):
            return '0.0'
        # Fix malformed scientific notation like "1.5e" -> "1.5"
        s = re.sub(r'[eE][+-]?$', '', s)
        return s

    def _extract_json(self, llm_output_str):
        llm_output_str = self._normalize_unicode(llm_output_str)

        # Pre-clean reward arrays w/ annotation before locating braces
        llm_output_str = self._preclean_reward_arrays(llm_output_str)

        # Check for verbose reasoning patterns (LLM thinking out loud)
        verbose_patterns = [
            r'^We are',
            r'^I am',
            r'^Let me',
            r'^Let\'s',
            r'^First,',
            r'^Looking at',
            r'^Based on',
            r'^Analyzing',
            r'^The data shows',
            r'^We need to',
            r'^We have',
        ]
        output_start = llm_output_str.strip()[:50].lower()
        is_verbose = any(re.match(pattern, output_start, re.IGNORECASE) for pattern in verbose_patterns)
        
        if is_verbose and '<param>' not in llm_output_str.lower():
            print("[MetaController] WARNING: LLM output contains verbose reasoning without <param> tags, using default policy")
            return self._default_safe_policy()

        # Check if output appears truncated (no <param> tags found and ends mid-sentence)
        if '<param>' not in llm_output_str.lower():
            stripped = llm_output_str.rstrip()
            if stripped and not stripped.endswith(('.', '>', '}', ']', '"')):
                print("[MetaController] WARNING: LLM output appears truncated (no <param> tags, ends mid-sentence), using default policy")
                return self._default_safe_policy()

        # Try to extract content from <param></param> XML tags first (preferred format)
        # Use non-greedy match for the JSON content and handle any garbage before <param>
        param_match = re.search(r'<param>\s*(\{[\s\S]*?\})\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
        if param_match:
            json_str = param_match.group(1).strip()
            print("[MetaController] Found JSON content within <param> tags.")
        else:
            # Fallback: Try to find any content between <param> tags 
            param_any_match = re.search(r'<param>\s*(.*?)\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
            if param_any_match:
                content = param_any_match.group(1).strip()
                # Check if it starts with { (JSON object)
                if content.startswith('{'):
                    json_str = content
                    print("[MetaController] Found content within <param> tags.")
                else:
                    # Content inside tags but not JSON-like
                    print(f"[MetaController] WARNING: Content in <param> tags is not JSON-like, using fallback")
                    json_str = None
            else:
                json_str = None
            
            if json_str is None:
                # Fallback: Try to find JSON-like content after <param> (for truncated output)
                param_start_match = re.search(r'<param>\s*(\{.*)', llm_output_str, re.DOTALL | re.IGNORECASE)
                if param_start_match:
                    json_str = param_start_match.group(1).strip()
                    print("[MetaController] Found truncated content after <param> tag, attempting recovery.")
                    # Try to fix truncated JSON
                    recovered_json = self._try_recover_json(json_str)
                    if recovered_json is not None:
                        return recovered_json
                
                # Fallback to brace matching
                match = re.search(r'(\{.*\})', llm_output_str, re.DOTALL)
                if not match:
                    # Try to recover from truncated output without closing brace
                    match_partial = re.search(r'(\{.*)', llm_output_str, re.DOTALL)
                    if match_partial:
                        json_str = match_partial.group(1).strip()
                        print("[MetaController] Found partial JSON, attempting recovery.")
                        recovered_json = self._try_recover_json(json_str)
                        if recovered_json is not None:
                            return recovered_json
                    
                    salvage = self._try_salvage_policy(llm_output_str)
                    if salvage is not None:
                        return salvage
                    print("[MetaController] No braces found; using default-safe policy for this completion.")
                    return self._default_safe_policy()
                json_str = match.group(0).strip()

        json_str = json_str.replace('```json', '').replace('```', '').strip()

        # Replace any leftover schema-like arrays (second pass)
        json_str = self._preclean_reward_arrays(json_str)

        # Remove annotation arrays like ["float", "Weight ..."] generically
        json_str = re.sub(r'\[\s*([0-9eE+.\-]+)\s*,\s*"[^"]*"\s*\]', r'\1', json_str)

        # Remove control characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)

        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # Clean number values with trailing periods or malformed scientific notation
        # Pattern explanation:
        #   [0-9]+       - one or more digits (integer part)
        #   \.?          - optional decimal point
        #   [0-9]*       - zero or more digits (fractional part)
        #   [eE]?        - optional exponent indicator
        #   [+-]?        - optional sign for exponent
        #   [0-9]*       - exponent digits
        #   \.+          - one or more trailing periods (the malformed part we're fixing)
        #   (?=\s*[,\]\}]) - lookahead for JSON delimiter (comma, bracket, or brace)
        def clean_json_numbers(match):
            return self._clean_number_string(match.group(0))
        json_str = re.sub(r'[0-9]+\.?[0-9]*[eE]?[+-]?[0-9]*\.+(?=\s*[,\]\}])', clean_json_numbers, json_str)

        # Force closure if braces unbalanced
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
            print(f"DEBUG: Added {open_braces - close_braces} closing brace(s).")

        json_str = self._strip_unterminated_quotes(json_str)

        # Attempt strict JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Attempt Python literal
        try:
            return ast.literal_eval(json_str)
        except Exception:
            # Final normalization: replace stray tokens
            normalized = re.sub(r'\bfloat\b', '0.5', json_str)
            normalized = re.sub(r'\bint\b', '64', json_str)
            try:
                return json.loads(normalized)
            except Exception:
                try:
                    return ast.literal_eval(normalized)
                except Exception:
                    return self._default_safe_policy()

    def _try_recover_json(self, json_str):
        """
        Try to recover a valid dict from a potentially truncated JSON string.
        Returns a dict if successful, None otherwise.
        """
        recovered_json = self._fix_truncated_json(json_str)
        if recovered_json is not None:
            if isinstance(recovered_json, dict):
                return recovered_json
            else:
                print(f"[MetaController] WARNING: Recovered JSON is {type(recovered_json)}, expected dict")
        return None

    def _fix_truncated_json(self, json_str):
        """Attempt to fix truncated JSON by adding missing brackets."""
        # Count brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        # Strip trailing whitespace
        json_str = json_str.rstrip()
        
        # Remove trailing comma if present
        if json_str.endswith(','):
            json_str = json_str[:-1]

        # Add missing array closures
        for _ in range(open_brackets - close_brackets):
            json_str += ']'

        # Add missing object closures
        for _ in range(open_braces - close_braces):
            json_str += '}'

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try Python literal eval
            try:
                return ast.literal_eval(json_str)
            except Exception:
                return None

    def _default_safe_policy(self):
        """Return a safe default policy with multiple options for exploration."""
        return {
            "objective_weights": DEFAULT_OPTIMIZATION_POLICY["objective_weights"].copy(),
            "search_space": {
                k: list(v) for k, v in DEFAULT_OPTIMIZATION_POLICY["search_space"].items()
            }
        }

    def _default_safe_plan(self):
        """Legacy method for backward compatibility."""
        return self._default_safe_policy()

    def _try_salvage_policy(self, s: str):
        """Try to salvage partial policy data from malformed output."""
        rf_keys = ("R_sm_throughput", "R_dram_throughput", "R_l1_hit_rate", "R_l2_hit_rate")
        pas_keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages")
        rf = {}
        pas = {}
        # Reward salvage for pattern key: [num, "desc"]
        for k in rf_keys:
            m = re.search(rf'{k}\s*[:=]\s*\[\s*([0-9eE+.\-]+)', s)
            if m:
                rf[k] = float(m.group(1))
                continue
            m2 = re.search(rf'{k}\s*[:=]\s*([0-9eE+.\-]+)', s)
            if m2:
                rf[k] = float(m2.group(1))
        # Param salvage
        for k in pas_keys:
            m_list = re.search(rf'{k}\s*[:=]\s*\[([^\]]+)\]', s)
            m_nums = re.search(rf'{k}\s*[:=]\s*([0-9][0-9,\s]*)', s)
            values = []
            src = None
            if m_list:
                src = m_list.group(1)
            elif m_nums:
                src = m_nums.group(1)
            if src:
                for tok in re.split(r'[,\s]+', src.strip()):
                    if tok:
                        try:
                            values.append(int(tok))
                        except Exception:
                            pass
                if values:
                    pas[k] = values[:3]
        if rf or pas:
            for k in rf_keys:
                rf.setdefault(k, 0.0)
            for k in pas_keys:
                if k not in pas:
                    default = 64 if "BLOCK" in k else (32 if k == "BLOCK_SIZE_K" else 4)
                    pas[k] = [default]
            return {"objective_weights": rf, "search_space": pas}
        return None

    def _try_salvage_plan(self, s: str):
        """Legacy method for backward compatibility."""
        return self._try_salvage_policy(s)

    def _validate_and_coerce_policy(self, policy):
        """Validate and coerce the optimization policy, using defaults for missing/invalid values."""
        # If policy is None or not a dict, use default
        if policy is None or not isinstance(policy, dict):
            print("[MetaController] WARNING: Invalid policy (None or not dict), using default optimization policy")
            return {
                "objective_weights": DEFAULT_OPTIMIZATION_POLICY["objective_weights"].copy(),
                "search_space": {
                    k: list(v) for k, v in DEFAULT_OPTIMIZATION_POLICY["search_space"].items()
                }
            }
        
        # Support both old and new key names for backward compatibility
        if "objective_weights" in policy:
            rf = policy.get("objective_weights", {})
        else:
            rf = policy.get("reward_function", {})
        
        if not isinstance(rf, dict):
            rf = {}
        
        # If reward values are lists, take first numeric
        def _scalar(v):
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, (int, float)):
                        return float(x)
                return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0
        
        # Ensure all required reward keys are present
        required_keys = ["R_sm_throughput", "R_dram_throughput", "R_l1_hit_rate", "R_l2_hit_rate"]
        for k in required_keys:
            rf[k] = _scalar(rf.get(k, DEFAULT_OPTIMIZATION_POLICY["objective_weights"].get(k, 0.0)))
        
        # Normalize weights to sum to 1.0 if total > 0
        total = sum(rf.values())
        if total > 0:
            rf = {k: v / total for k, v in rf.items()}

        # Support both old and new key names for search space
        if "search_space" in policy:
            ss = policy.get("search_space", {})
        else:
            ss = policy.get("pruned_action_space", {})
        
        if not isinstance(ss, dict) or not ss:
            ss = {k: list(v) for k, v in DEFAULT_OPTIMIZATION_POLICY["search_space"].items()}
            
        def _coerce_list(v, default):
            if isinstance(v, list):
                out = []
                for i in v:
                    try:
                        out.append(int(i))
                    except Exception:
                        continue
                if not out:
                    out = [default]
                return out[:3]
            try:
                return [int(v)]
            except Exception:
                return [default]
        
        # Get default values from DEFAULT_OPTIMIZATION_POLICY
        ss["BLOCK_SIZE_M"] = _coerce_list(ss.get("BLOCK_SIZE_M", DEFAULT_OPTIMIZATION_POLICY["search_space"]["BLOCK_SIZE_M"]), 64)
        ss["BLOCK_SIZE_N"] = _coerce_list(ss.get("BLOCK_SIZE_N", DEFAULT_OPTIMIZATION_POLICY["search_space"]["BLOCK_SIZE_N"]), 64)
        ss["BLOCK_SIZE_K"] = _coerce_list(ss.get("BLOCK_SIZE_K", DEFAULT_OPTIMIZATION_POLICY["search_space"]["BLOCK_SIZE_K"]), 32)
        ss["num_warps"]    = _coerce_list(ss.get("num_warps", DEFAULT_OPTIMIZATION_POLICY["search_space"]["num_warps"]), 4)
        ss["num_stages"]   = _coerce_list(ss.get("num_stages", DEFAULT_OPTIMIZATION_POLICY["search_space"]["num_stages"]), 4)
        # Enforce power-of-two warps
        ss["num_warps"] = [w for w in ss["num_warps"] if w in POWER_OF_TWO_WARPS] or [4]
        
        # H100 hardware constraint validation - clamp values to safe limits
        # Aggressive: Allow num_stages=5 for pushing boundaries!
        H100_BLOCK_SIZE_MN_LIMIT = 128
        H100_BLOCK_SIZE_K_LIMIT = 64
        H100_NUM_STAGES_LIMIT = 5  # Increased from 4 for aggressive testing
        
        ss["BLOCK_SIZE_M"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in ss["BLOCK_SIZE_M"]]
        ss["BLOCK_SIZE_N"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in ss["BLOCK_SIZE_N"]]
        ss["BLOCK_SIZE_K"] = [min(v, H100_BLOCK_SIZE_K_LIMIT) for v in ss["BLOCK_SIZE_K"]]
        ss["num_stages"] = [min(v, H100_NUM_STAGES_LIMIT) for v in ss["num_stages"]]
        
        # Remove duplicates and ensure non-empty lists
        for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages"]:
            ss[key] = list(set(ss[key])) or ([64] if "BLOCK" in key else [4])
        
        # AFTER coercing all values, ensure minimum combinations
        MIN_COMBINATIONS = 8  # Need at least 8 for meaningful PPO exploration
        MIN_VALUES_PER_DIM = 2  # Each dimension needs at least 2 values
        
        # Default values to expand narrow search spaces
        defaults = {
            "BLOCK_SIZE_M": [64, 128],
            "BLOCK_SIZE_N": [64, 128],
            "BLOCK_SIZE_K": [32, 64],
            "num_warps": [8, 16],
            "num_stages": [3, 4, 5],
        }
        
        # Ensure each dimension has at least 2 values
        for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"]:
            if len(ss[key]) < MIN_VALUES_PER_DIM:
                # Merge with defaults, remove duplicates, limit to 3 values
                # (consistent with _coerce_list which limits to 3 to keep search space manageable)
                ss[key] = list(set(ss[key] + defaults[key]))[:3]
                print(f"[MetaController] Expanded {key} to {ss[key]} (was too narrow)")
        
        # Calculate total combinations
        total_combinations = 1
        for values in ss.values():
            total_combinations *= len(values)
        
        if total_combinations < MIN_COMBINATIONS:
            print(f"[MetaController] WARNING: Only {total_combinations} combinations, expanding search space")
            # Use full default search space
            ss = {k: list(v) for k, v in DEFAULT_OPTIMIZATION_POLICY["search_space"].items()}
            total_combinations = 1
            for values in ss.values():
                total_combinations *= len(values)
        
        print(f"[MetaController] Search space has {total_combinations} combinations")
        
        return {"objective_weights": rf, "search_space": ss}

    def _validate_and_coerce_plan(self, plan):
        """Legacy method for backward compatibility."""
        return self._validate_and_coerce_policy(plan)

    def _run_exploration_phase(self, policy_config_path):
        """Run exploration phase for EACH token count.
        
        Returns:
            Tuple of (top_configs, best_configs) where:
            - top_configs: List of top configuration results
            - best_configs: Dict mapping token_count -> {config, reward}
        """
        all_top_results = []
        best_configs_for_validation = []
        best_configs = {}
        total_tokens = len(self.token_counts)
        
        # Calculate steps per token count
        steps_per_token = max(MIN_STEPS_PER_TOKEN, self.exploration_steps // len(self.token_counts))
        
        for i, token_count in enumerate(self.token_counts):
            print(f"\n[MetaController] === Token Count {i+1}/{total_tokens}: {token_count} tokens ===")
            
            env = KernelTuningEnvironment(
                policy_config_path=policy_config_path,
                profiling_worker=self.worker,
                static_args=self.static_args,
                initial_state=self.initial_state,
                config_exporter=self.config_exporter,
                current_token_count=token_count
            )
            # Use persistent log_dir for each token count (not timestamped)
            log_dir = f"./logs/exploration_agent/tokens_{token_count}/"
            prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            
            # Hide GPUs for SB3 MLP – force CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "" 
            agent = None
            try:
                # Create agent WITH loading existing model - THIS IS THE FIX!
                agent = ExplorationAgent(
                    env, 
                    log_dir=log_dir, 
                    device="cpu",
                    load_existing=True,  # Load trained model if exists!
                    training_logger=self.training_logger,
                )
                agent.train_epoch(steps=steps_per_token)
                top = env.get_top_results(n=RESULTS_PER_TOKEN)
                print(f"[MetaController] Token count {token_count}: Found {len(top)} results.")
                all_top_results.extend(top)
                best_configs_for_validation.extend(top)
                
                # Get best config from environment and save best model if improved
                if top:
                    best_result = max(top, key=lambda x: x[2])
                    best_config = best_result[0]
                    best_reward = best_result[2]
                    
                    best_configs[token_count] = {
                        'config': best_config,
                        'reward': best_reward,
                    }
                    
                    # Save best model if this is the best reward so far
                    agent.save_best_if_improved(best_reward)
                    
                    # Update config exporter
                    if self.config_exporter:
                        self.config_exporter.update_best_config(
                            token_count=token_count,
                            config=best_config,
                            reward=best_reward,
                        )
            finally:
                if agent:
                    try: agent.close()
                    except Exception: pass
                env.close()
                if prev_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
            
            # Periodic validation
            if (i + 1) % VALIDATE_EVERY_N_TOKENS == 0 and best_configs_for_validation:
                print(f"[MetaController] Running periodic throughput validation after {i+1} token counts...")
                best_config = max(best_configs_for_validation, key=lambda x: x[2])
                throughput = self._run_throughput_validation([best_config])
                print(f"[MetaController] Periodic validation throughput: {throughput} tokens/sec")
                best_configs_for_validation = []
        
        # Return combined top results from all token counts along with best_configs
        if all_top_results:
            sorted_results = sorted(all_top_results, key=lambda x: x[2], reverse=True)
            print(f"[MetaController] Exploration phase completed. Total {len(all_top_results)} results across {len(self.token_counts)} token counts.")
            return sorted_results[:5], best_configs
        return [], best_configs

    def _run_fast_loop(self, mission_plan_path):
        """Legacy method for backward compatibility."""
        top_configs, _ = self._run_exploration_phase(mission_plan_path)
        return top_configs

    def _run_throughput_validation(self, top_configs):
        """Run throughput validation on the best configurations."""
        if not top_configs:
            print("[MetaController] No valid configurations found for throughput validation.")
            return 0.0
        ids = [
            self.worker.run_throughput_validation.remote(params, self.model_name, self.user_goal)
            for params, state, reward in top_configs
        ]
        print(f"[MetaController] Awaiting validation metrics for {len(top_configs)} configs from ProfilingWorker...")
        metrics = ray.get(ids)
        if self.user_goal == "throughput":
            best = max(metrics)
        else:
            valid = [m for m in metrics if m > 0]
            best = min(valid) if valid else 0.0
        print(f"[MetaController] Throughput validation complete. Best metric: {best}")
        return best

    def _run_slow_gym_validation(self, best_configs_by_token):
        """
        Run vLLM benchmark with a COMBINED config file containing all token counts.
        
        Args:
            best_configs_by_token: Dict mapping token_count -> {"config": {...}, "reward": float}
            
        Returns:
            float: Throughput metric from vLLM benchmark
        """
        if not best_configs_by_token:
            print("[MetaController] No configs to validate.")
            return 0.0
        
        # Build combined config with ALL token counts
        combined_config = {}
        for token_count, config_data in best_configs_by_token.items():
            combined_config[str(token_count)] = config_data["config"]
        
        # Write the combined config ONCE
        config_path = os.path.join(
            self.vllm_config_dir,
            f"E={self.num_experts},N={self.inter_size},device_name=NVIDIA_H100_80GB_HBM3.json"
        )
        
        with open(config_path, "w") as f:
            json.dump(combined_config, f, indent=2)
        
        print(f"[MetaController] Wrote combined config with {len(combined_config)} token counts to: {config_path}")
        
        # Now run vLLM benchmark ONCE (it will use the right config per token count internally)
        result = self.worker.run_throughput_validation.remote(
            combined_config,
            self.model_name,
            self.user_goal
        )
        
        throughput = ray.get(result)
        print(f"[MetaController] Combined config validation throughput: {throughput} tokens/sec")
        return throughput

    def _run_slow_gym(self, top_configs):
        """Legacy method for backward compatibility."""
        return self._run_throughput_validation(top_configs)


# Backward compatibility alias
HAKT_Reward_Function = MetaControllerReward
