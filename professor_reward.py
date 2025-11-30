import json
import ray
import numpy as np
import time
import torch
import os
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

        # The BenchmarkWorker actor will be requested from Ray's pool.
        # We pass the worker_gpu_id (e.g., 7) to its constructor.
        # The actor *itself* will use 1 GPU from Ray's pool (e.g., GPU 1),
        # but its *subprocess work* (ncu, vllm bench) will be pinned
        # to the physical GPU ID (7) that we pass in.
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
                    json.dump(plan, f)
                    
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
        """Extracts the JSON plan from the LLM's completion."""
        try:
            # Robust parsing to handle reasoning text around the JSON block
            start_marker = llm_output_str.find('```json')
            if start_marker != -1:
                start_marker += 7
                end_marker = llm_output_str.rfind('```')
                json_str = llm_output_str[start_marker:end_marker].strip()
            else:
                # Fallback to finding the first { and last }
                json_str = llm_output_str[llm_output_str.find('{'):llm_output_str.rfind('}') + 1]

            return json.loads(json_str.strip())
            
        except Exception as e:
            print(f"ERROR parsing LLM JSON: {e}")
            # Return a safe, penalized plan on failure
            return {"reward_function": {}, "pruned_action_space": {
                "BLOCK_SIZE_M": [64], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32],
                "num_warps": [4], "num_stages": [4]}
            }

    def _run_fast_loop(self, mission_plan_path):
        """
        Runs the *entire* PPO training loop for the "Fighter Pilot".
        """
        env = FastGymEnv(
            mission_plan_path=mission_plan_path,
            benchmark_worker=self.worker,
            static_args=self.static_args,
            initial_state=self.initial_state
        )
        
        epoch_log_dir = f"./hakt_logs/run_{int(time.time())}/"
        pilot = FighterPilot(env, log_dir=epoch_log_dir)
        
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