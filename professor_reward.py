import json
import ray
import numpy as np
import time
import torch
import os
import re # <-- New dependency for robust parsing
from fast_gym_env import FastGymEnv
from fighter_agent import FighterPilot
from benchmark_worker import BenchmarkWorker

# --- REVISION ---
# The "rogue" ray.init() block that was here has been REMOVED.
# hakt_meta_trainer.py is now the only script that calls ray.init().
# --- END REVISION ---

class HAKT_Reward_Function:
    
    def __init__(self, user_goal, model_name, fast_loop_steps, worker_gpu_id, static_args):
        self.user_goal = user_goal
        self.model_name = model_name
        self.fast_loop_steps = fast_loop_steps
        self.static_args = static_args
        
        print(f"[RewardFn] Requesting BenchmarkWorker for PHYSICAL GPU {worker_gpu_id}")

        self.worker = BenchmarkWorker.options(
            num_gpus=1, # Request 1 GPU from Ray's pool *for the actor process*
        ).remote(worker_gpu_id) # Pass the PHYSICAL ID (7) as an argument
        
        self.initial_state = self._get_initial_state()

    def _get_initial_state(self):
        print("[RewardFn] Getting initial state from worker...")
        try:
            job_id = self.worker.run_fast_gym_benchmark.remote(
                None, self.static_args, {}
            )
            state, reward, csv_data = ray.get(job_id)
            if state is None:
                raise Exception("Worker failed initial profile.")
            print("[RewardFn] Initial state acquired.")
            return state
        except Exception as e:
            print(f"[RewardFn] ERROR: Worker failed initial state check. Using fallback. {e}")
            # Fallback to known "bad" state from our logs
            return np.array([32.3, 40.8, 0.05, 69.9], dtype=np.float32)

    def __call__(self, completions, **kwargs):
        """
        This is the main "reward" entrypoint for the GRPOTrainer.
        'completions' is a list of generated texts (the JSON plans).
        """
        rewards = []
        for plan_str in completions:
            try:
                plan = self._extract_json(plan_str)
                
                plan_file_path = f"temp_mission_plan_{int(time.time())}.json"
                with open(plan_file_path, "w") as f:
                    # The JSON object is now guaranteed to be clean by _extract_json
                    json.dump(plan, f, indent=2)
                    
                print(f"[RewardFn] Starting 'Fast Loop' PPO training...")
                top_5_configs = self._run_fast_loop(plan_file_path)
                
                print(f"[RewardFn] Starting 'Slow Gym' validation...")
                final_metric = self._run_slow_gym(top_5_configs)
                
                rewards.append(final_metric)
                
                os.remove(plan_file_path) 
                
            except Exception as e:
                print(f"[RewardFn] ERROR: HAKT reward calculation failed. {e}")
                rewards.append(0.0) # Penalize bad/unparseable plans
        
        return rewards

    def _extract_json(self, llm_output_str):
        """
        Extracts the JSON plan from the LLM's completion using a robust regex.
        This handles surrounding text, control characters, and markdown ticks.
        """
        # --- NEW ROBUST FIX ---
        
        # 1. Regex to find the JSON block, optionally enclosed in ```json or ```
        # It handles optional newlines/whitespace around the block
        # The '(?s)' makes '.' match newlines, '.*?' is non-greedy
        match = re.search(r"```json\s*(.*?)\s*```|(\s*\{.*\}\s*)", llm_output_str, re.DOTALL)
        
        json_str = None
        if match:
            # Use the content of the first successful group capture (1 or 2)
            json_str = match.group(1) or match.group(2)
            
        if json_str is None:
            # Fallback to finding the first { and last } (original method, but less reliable)
            start_idx = llm_output_str.find('{')
            end_idx = llm_output_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = llm_output_str[start_idx : end_idx + 1]
            
        if json_str:
            # 2. Try to load the found string
            try:
                # Remove common invalid control characters (like stray backslashes) that confuse the parser
                cleaned_str = json_str.replace('\\', '').replace('\n', ' ').strip()
                return json.loads(cleaned_str)
            except json.JSONDecodeError as e:
                # If the first attempt fails, try a slightly less aggressive clean (remove only newlines)
                try:
                    cleaned_str = json_str.replace('\n', ' ').strip()
                    return json.loads(cleaned_str)
                except json.JSONDecodeError:
                    raise e # Re-raise the original error if both fail

        # --- END NEW ROBUST FIX ---
        
        # If no JSON was found or parsing failed, we hit this final except block
        raise json.JSONDecodeError("No valid JSON structure found in LLM output.", llm_output_str, 0)
            
    def _run_fast_loop(self, mission_plan_path):
        """
        Runs the *entire* PPO training loop for the "Fighter Pilot".
        """
        # The imports for FastGymEnv and FighterPilot are correct at the top of the file
        env = FastGymEnv(
            mission_plan_path=mission_plan_path,
            benchmark_worker=self.worker,
            static_args=self.static_args,
            initial_state=self.initial_state
        )
        
        epoch_log_dir = f"./hakt_logs/run_{int(time.time())}/"
        pilot = FighterPilot(env, log_dir=epoch_log_dir)
        
        print(f"[{pilot.__class__.__name__}] Training on {pilot.device} for {self.fast_loop_steps} steps...")
        pilot.train_epoch(steps=self.fast_loop_steps)
        
        top_results = env.get_top_results(n=5)
        
        env.close()
        del pilot
        del env
        
        return top_results

    def _run_slow_gym(self, top_configs_from_la):
        """
        Runs the 'vllm bench' validation on the "Slow Gym" (GPU 1).
        """
        if not top_configs_from_la:
            return 0.0 # No valid configs found

        validation_ids = []
        for config_tuple in top_configs_from_la:
            params, state, reward = config_tuple
            job_id = self.worker.run_slow_gym_validation.remote(
                params, self.model_name, self.user_goal
            )
            validation_ids.append(job_id)

        validation_metrics = ray.get(validation_ids)
        
        if self.user_goal == "throughput":
            best_metric = max(validation_metrics)
        else: # "latency"
            # Filter out 0.0 (failed runs) before finding min
            valid_latencies = [m for m in validation_metrics if m > 0]
            best_metric = min(valid_latencies) if valid_latencies else 0.0
        
        return best_metric
