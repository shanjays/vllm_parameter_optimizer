import json
import ray
import os
import subprocess
import time
import torch
from fast_gym_env import FastGymEnv
from fighter_agent import FighterPilot
from professor_agent import ProfessorAgent
from benchmark_worker import BenchmarkWorker

# --- HAKT Configuration ---
USER_GOAL = "throughput" # or "latency"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" # Model for "Slow Gym" validation
KERNEL_TO_TUNE = "fused_moe_kernel"
RUN_SCRIPT_PATH = "run_moe_gym.py" # The script for the "Fast Gym"
PROFESSOR_MODEL = "gpt-oss-20b" # Your chosen reasoning LLM
NUM_EPOCHS = 5           # Number of times the LLM will intervene
STEPS_PER_EPOCH = 100    # Number of "Fast Loop" steps per epoch

# (This is the *full* space, before the LLM prunes it)
FULL_PARAM_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128, 256],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4, 5]
}
# -----------------------------

def get_initial_profile(benchmark_worker):
    """
    Runs ncu *once* on the benchmark worker (GPU 1) to get the 'goldmine' report.
    """
    print("[HAKT] Running initial profile on BenchmarkWorker...")
    
    # Static args for the "Fast Gym"
    static_args = {
        "run_script_path": RUN_SCRIPT_PATH,
        "kernel_name": KERNEL_TO_TUNE,
        "num_tokens": 16088, # "sweet spot"
        "num_experts": 64, "top_k": 2,
        "hidden_size": 6144, "inter_size": 11008,
        "dtype": "fp16", "num_iters": 0
    }
    
    # Run the benchmark using the default config (None)
    result_id = benchmark_worker.run_fast_gym_benchmark.remote(
        params_dict=None, 
        static_args=static_args,
        reward_weights={} # Not needed, we just want the CSV
    )
    
    # Wait for the result from GPU 1
    state, reward, csv_data = ray.get(result_id)
    
    if csv_data is None:
        raise Exception("Initial profiling failed. Check BenchmarkWorker logs.")
        
    print("[HAKT] Initial profile complete.")
    return csv_data, state, static_args # Return all info

def main():
    # Initialize Ray to use all available GPUs (assumes >= 2)
    ray.init()
    
    available_gpus = ray.available_resources().get("GPU", 0)
    if available_gpus < 2:
        print(f"ERROR: HAKT requires at least 2 GPUs, but found {available_gpus}.")
        ray.shutdown()
        return

    # --- STAGE 1: MISSION BRIEFING ---
    
    # Pin the BenchmarkWorker to 1 GPU. Ray will assign it one.
    print("[HAKT] Assigning BenchmarkWorker to its own GPU...")
    worker = BenchmarkWorker.options(num_gpus=1).remote()
    
    # Wait for the worker to start and get its GPU ID
    worker_gpu_id = ray.get(worker.get_gpu_id.remote())
    print(f"[HAKT] BenchmarkWorker is live on GPU: {worker_gpu_id}")

    # Assign the *other* GPU to the main process (agent training)
    all_gpu_ids = list(range(int(available_gpus)))
    agent_gpu_id = [gid for gid in all_gpu_ids if gid != worker_gpu_id][0]
    
    print(f"[HAKT] Main process (and FighterPilot) will run on GPU: {agent_gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(agent_gpu_id)
    # Verify torch sees the correct device
    torch.cuda.set_device(0) # Torch will now see agent_gpu_id as 'cuda:0'
    print(f"[HAKT] Torch context set to: {torch.cuda.current_device()}")

    
    initial_csv_data, initial_state, static_args = get_initial_profile(worker)
    
    professor = ProfessorAgent(model_name=PROFESSOR_MODEL, user_goal=USER_GOAL)
    
    current_plan_path = professor.create_initial_plan(
        initial_csv_data,
        FULL_PARAM_SPACE,
        initial_state
    )

    # --- STAGE 2: "FAST LOOP" TRAINING ---
    
    # Initialize the "Gym" and the "Fighter Pilot" on GPU 0
    # Pass the Ray handle for the worker to the Gym
    env = FastGymEnv(
        mission_plan_path=current_plan_path,
        benchmark_worker=worker,
        static_args=static_args,
        initial_state=initial_state
    )
    
    pilot = FighterPilot(env, log_dir="./hakt_logs/fighter_pilot/")

    all_top_results = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- HAKT Epoch {epoch}/{NUM_EPOCHS} (Training on GPU {agent_gpu_id}) ---")
        
        # 1. Train the "Fighter Pilot" (PPO.learn() runs on GPU 0)
        #    This will call env.step(), which triggers the worker on GPU 1
        pilot.train_epoch(steps=STEPS_PER_EPOCH)
        
        # 2. Get the LA's best results
        top_epoch_results = env.get_top_results(n=5)
        all_top_results.extend(top_epoch_results)
        
        if epoch < NUM_EPOCHS:
            # 3. "Professor" (on CPU) refines the plan
            current_plan_path = professor.refine_plan(
                current_plan_path,
                top_epoch_results,
                epoch_num = epoch + 1
            )
            # 4. Re-configure the gym (on GPU 0) with the *new* plan
            env.set_mission_plan(current_plan_path)
            # We also re-set the env on the pilot model
            pilot.model.set_env(env.env)

    # --- STAGE 3: FINAL VALIDATION ---
    
    # Get the unique, all-time best configs from the LA's training
    all_top_results = sorted(all_top_results, key=lambda x: x[2], reverse=True)
    unique_top_configs = []
    seen_params = set()
    for res in all_top_results:
        params_str = json.dumps(res[0])
        if params_str not in seen_params:
            unique_top_configs.append(res)
            seen_params.add(params_str)
        if len(unique_top_configs) >= 5: # Get Top 5 unique configs
            break
            
    # "Professor" (on CPU) tells the "Worker" (on GPU 1)
    # to run the final system-level benchmarks
    best_overall_config = professor.run_final_validation(
        unique_top_configs,
        MODEL_NAME,
        worker # Pass the worker handle
    )
    
    print(f"\n[HAKT] Final optimal config: {best_overall_config}")
    
    ray.shutdown()
    print("[HAKT] Tuning complete.")

if __name__ == "__main__":
    main()
