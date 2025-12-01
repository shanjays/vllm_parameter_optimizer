import json
import ray
import numpy as np
import time
import os
import re
import ast
from fast_gym_env import FastGymEnv
from fighter_agent import FighterPilot
from benchmark_worker import BenchmarkWorker

POWER_OF_TWO_WARPS = (2, 4, 8, 16, 32)

# Default token counts to test (matching vLLM's expected format)
DEFAULT_TOKEN_COUNTS = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 
    1536, 2048, 3072, 4096
]

# Minimum training steps per token count to ensure meaningful exploration
MIN_STEPS_PER_TOKEN = 8  # Reduced from 10 for faster iteration

# Number of top results to collect per token count
RESULTS_PER_TOKEN = 3

# Validation frequency: run vLLM validation after every N token counts
VALIDATE_EVERY_N_TOKENS = 3

class HAKT_Reward_Function:
    def __init__(self, user_goal, model_name, fast_loop_steps, worker_gpu_id, static_args, 
                 config_saver=None, token_counts=None):
        self.user_goal = user_goal
        self.model_name = model_name
        self.fast_loop_steps = fast_loop_steps
        self.static_args = static_args
        self.config_saver = config_saver
        self.token_counts = token_counts or DEFAULT_TOKEN_COUNTS
        print(f"[RewardFn] Requesting BenchmarkWorker for PHYSICAL GPU {worker_gpu_id}")
        self.worker = BenchmarkWorker.options(num_gpus=1).remote(worker_gpu_id)
        self.initial_state = self._get_initial_state()
        print(f"[RewardFn] Configured to test {len(self.token_counts)} token counts: {self.token_counts[:5]}...")

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
        print("[RewardFn] Getting initial state from worker...")
        try:
            job_id = self.worker.run_fast_gym_benchmark.remote(None, self.static_args, {})
            state, reward, _ = ray.get(job_id)
            if state is None:
                raise RuntimeError("Worker failed initial profile.")
            print("[RewardFn] Initial state acquired.")
            return state
        except Exception as e:
            print(f"[RewardFn] ERROR: Worker failed initial state check. Using fallback. {e}")
            return np.array([32.3, 40.8, 0.05, 69.9], dtype=np.float32)

    def __call__(self, completions, **kwargs):
        rewards = []
        for i, plan_str in enumerate(completions):
            print(f"\n--- [RewardFn] Processing Mission Plan {i+1}/{len(completions)} ---")
            
            # --- DEBUG PRINT (as requested) ---
            print(f"[RewardFn] DEBUG: Raw LLM Output:\n{plan_str}\n")
            # --- END DEBUG PRINT ---

            valid = True
            try:
                plan = self._extract_json(plan_str)
                plan = self._clean_non_json_types(plan)
                plan = self._validate_and_coerce_plan(plan)

                path = f"temp_mission_plan_{int(time.time())}_{i}.json"
                with open(path, "w") as f:
                    json.dump(plan, f, indent=2)

                print(f"[RewardFn] Starting 'Fast Loop' PPO training ({self.fast_loop_steps} steps)...")
                top_configs = self._run_fast_loop(path)
                print(f"[RewardFn] Starting 'Slow Gym' validation (Top {len(top_configs)} configs)...")
                final_metric = self._run_slow_gym(top_configs)
                rewards.append(final_metric)
                os.remove(path)
            except Exception as e:
                valid = False
                print(f"[RewardFn] ERROR: HAKT reward calculation failed. Reason: {e}")
                rewards.append(0.0)
                if not valid:
                    self._run_default_penalty_plan(i)
        
        # Print training summary
        if self.config_saver:
            summary = self.config_saver.get_summary()
            print(f"\n[RewardFn] === Training Summary ===")
            print(f"  Total configs tested: {summary['total_experiments']}")
            print(f"  Token counts covered: {summary['total_token_counts']}")
            print(f"  Best rewards by token count:")
            for tc, reward in sorted(summary['best_rewards'].items(), key=lambda x: int(x[0])):
                print(f"    {tc} tokens: {reward:.2f}")
        
        return rewards

    def _run_default_penalty_plan(self, idx):
        print("[RewardFn] Plan failed. Running Fast Loop on default, penalized plan to ensure training progression.")
        default_plan = {
            "reward_function": {
                "R_sm_throughput": 0.01,
                "R_dram_throughput": 0.0,
                "R_l1_hit_rate": 0.0,
                "R_l2_hit_rate": 0.0
            },
            "pruned_action_space": {
                "BLOCK_SIZE_M": [64],
                "BLOCK_SIZE_N": [64],
                "BLOCK_SIZE_K": [32],
                "num_warps": [4],
                "num_stages": [4]
            }
        }
        path = f"temp_default_plan_{int(time.time())}_{idx}.json"
        try:
            with open(path, "w") as f:
                json.dump(default_plan, f, indent=2)
            self._run_fast_loop(path)
        except Exception as e:
            print(f"[RewardFn] WARNING: Default Fast Loop also failed. {e}")
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

        # Try to extract content from <param></param> XML tags first (preferred format)
        param_match = re.search(r'<param>\s*(.*?)\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
        if param_match:
            json_str = param_match.group(1).strip()
            print("[RewardFn] Found content within <param> tags.")
        else:
            # Fallback: Try to find JSON-like content after <param> (for truncated output)
            param_start_match = re.search(r'<param>\s*(\{.*)', llm_output_str, re.DOTALL | re.IGNORECASE)
            if param_start_match:
                json_str = param_start_match.group(1).strip()
                print("[RewardFn] Found truncated content after <param> tag, attempting recovery.")
                # Try to fix truncated JSON
                recovered_json = self._fix_truncated_json(json_str)
                if recovered_json is not None:
                    return recovered_json
            
            # Fallback to brace matching
            match = re.search(r'(\{.*\})', llm_output_str, re.DOTALL)
            if not match:
                # Try to recover from truncated output without closing brace
                match_partial = re.search(r'(\{.*)', llm_output_str, re.DOTALL)
                if match_partial:
                    json_str = match_partial.group(1).strip()
                    print("[RewardFn] Found partial JSON, attempting recovery.")
                    recovered_json = self._fix_truncated_json(json_str)
                    if recovered_json is not None:
                        return recovered_json
                
                salvage = self._try_salvage_plan(llm_output_str)
                if salvage is not None:
                    return salvage
                print("[RewardFn] No braces found; using default-safe plan for this completion.")
                return self._default_safe_plan()
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
                    return self._default_safe_plan()

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

    def _default_safe_plan(self):
        return {
            "reward_function": {
                "R_sm_throughput": 0.5,
                "R_dram_throughput": 0.0,
                "R_l1_hit_rate": 0.0,
                "R_l2_hit_rate": 0.0
            },
            "pruned_action_space": {
                "BLOCK_SIZE_M": [64],
                "BLOCK_SIZE_N": [64],
                "BLOCK_SIZE_K": [32],
                "num_warps": [4],
                "num_stages": [4]
            }
        }

    def _try_salvage_plan(self, s: str):
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
            return {"reward_function": rf, "pruned_action_space": pas}
        return None

    def _validate_and_coerce_plan(self, plan):
        if not isinstance(plan, dict):
            raise ValueError("Plan is not a dict.")
        rf = plan.get("reward_function", {})
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
        for k in ("R_sm_throughput", "R_dram_throughput", "R_l1_hit_rate", "R_l2_hit_rate"):
            rf[k] = _scalar(rf.get(k, 0.0))

        pas = plan.get("pruned_action_space", {})
        if not isinstance(pas, dict):
            pas = {}
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
        pas["BLOCK_SIZE_M"] = _coerce_list(pas.get("BLOCK_SIZE_M", [64]), 64)
        pas["BLOCK_SIZE_N"] = _coerce_list(pas.get("BLOCK_SIZE_N", [64]), 64)
        pas["BLOCK_SIZE_K"] = _coerce_list(pas.get("BLOCK_SIZE_K", [32]), 32)
        pas["num_warps"]    = _coerce_list(pas.get("num_warps", [4]), 4)
        pas["num_stages"]   = _coerce_list(pas.get("num_stages", [4]), 4)
        # Enforce power-of-two warps
        pas["num_warps"] = [w for w in pas["num_warps"] if w in POWER_OF_TWO_WARPS] or [4]
        
        # H100 hardware constraint validation - clamp values to safe limits
        # H100 has 228KB shared memory per SM, we use conservative 227KB (232,448 bytes)
        # Values > 128 for M/N and > 64 for K can cause Triton shared memory overflow
        # when combined with high num_stages (e.g., 128*128 + 128*128 * 2 * 4 = 262KB)
        H100_BLOCK_SIZE_MN_LIMIT = 128
        H100_BLOCK_SIZE_K_LIMIT = 64  # Lower limit for K to avoid overflow with high M/N
        H100_NUM_STAGES_LIMIT = 4
        
        pas["BLOCK_SIZE_M"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas["BLOCK_SIZE_M"]]
        pas["BLOCK_SIZE_N"] = [min(v, H100_BLOCK_SIZE_MN_LIMIT) for v in pas["BLOCK_SIZE_N"]]
        pas["BLOCK_SIZE_K"] = [min(v, H100_BLOCK_SIZE_K_LIMIT) for v in pas["BLOCK_SIZE_K"]]
        pas["num_stages"] = [min(v, H100_NUM_STAGES_LIMIT) for v in pas["num_stages"]]
        
        # Remove duplicates and ensure non-empty lists
        for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages"]:
            pas[key] = list(set(pas[key])) or ([64] if "BLOCK" in key else [4])
        
        return {"reward_function": rf, "pruned_action_space": pas}

    def _run_fast_loop(self, mission_plan_path):
        """Run fast loop for EACH token count."""
        all_top_results = []
        best_configs_for_validation = []
        total_tokens = len(self.token_counts)
        
        # Calculate steps per token count
        steps_per_token = max(MIN_STEPS_PER_TOKEN, self.fast_loop_steps // len(self.token_counts))
        
        for i, token_count in enumerate(self.token_counts):
            print(f"\n[RewardFn] === Token Count {i+1}/{total_tokens}: {token_count} tokens ===")
            
            env = FastGymEnv(
                mission_plan_path=mission_plan_path,
                benchmark_worker=self.worker,
                static_args=self.static_args,
                initial_state=self.initial_state,
                config_saver=self.config_saver,
                current_token_count=token_count
            )
            log_dir = f"./hakt_logs/run_{int(time.time())}_tokens{token_count}/"
            prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            
            # --- THIS IS THE FIX ---
            # Hide GPUs for SB3 MLP – force CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "" 
            pilot = None
            try:
                pilot = FighterPilot(env, log_dir=log_dir, device="cpu") # Force CPU
                pilot.train_epoch(steps=steps_per_token)
                top = env.get_top_results(n=RESULTS_PER_TOKEN)
                print(f"[RewardFn] Token count {token_count}: Found {len(top)} results.")
                all_top_results.extend(top)
                best_configs_for_validation.extend(top)
            # --- END FIX ---
            finally:
                if pilot:
                    try: pilot.close()
                    except Exception: pass
                env.close()
                if prev_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
            
            # Periodic validation
            if (i + 1) % VALIDATE_EVERY_N_TOKENS == 0 and best_configs_for_validation:
                print(f"[RewardFn] Running periodic vLLM validation after {i+1} token counts...")
                # best_configs_for_validation contains tuples of (params, state, reward)
                best_config = max(best_configs_for_validation, key=lambda x: x[2])  # x[2] is reward
                throughput = self._run_slow_gym([best_config])
                print(f"[RewardFn] Periodic validation throughput: {throughput} tokens/sec")
                best_configs_for_validation = []
        
        # Return combined top results from all token counts
        if all_top_results:
            # Sort by reward and return top 5
            sorted_results = sorted(all_top_results, key=lambda x: x[2], reverse=True)
            print(f"[RewardFn] Fast Loop completed. Total {len(all_top_results)} results across {len(self.token_counts)} token counts.")
            return sorted_results[:5]
        return []

    def _run_slow_gym(self, top_configs):
        if not top_configs:
            print("[RewardFn] No valid configurations found for Slow Gym validation.")
            return 0.0
        ids = [
            self.worker.run_slow_gym_validation.remote(params, self.model_name, self.user_goal)
            for params, state, reward in top_configs
        ]
        print(f"[RewardFn] Awaiting validation metrics for {len(top_configs)} configs from BenchmarkWorker...")
        metrics = ray.get(ids)
        if self.user_goal == "throughput":
            best = max(metrics)
        else:
            valid = [m for m in metrics if m > 0]
            best = min(valid) if valid else 0.0
        print(f"[RewardFn] Slow Gym validation complete. Best metric: {best}")
        return best
