import json
import ray
import os
import subprocess
import torch
import time
from datasets import Dataset

from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

from professor_reward import HAKT_Reward_Function # The "Slow Loop" reward

# --- HAKT-R Configuration ---
AGENT_GPU_ID = 0  # Physical ID for Professor/Fighter
WORKER_GPU_ID = 7 # Physical ID for BenchmarkWorker

PROFESSOR_MODEL = "openai/gpt-oss-20b" 

USER_GOAL = "throughput" 
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
KERNEL_TO_TUNE = "fused_moe_kernel" 
RUN_SCRIPT_PATH = "run_moe_gym.py" 
FAST_LOOP_STEPS = 100 
LLM_TRAIN_STEPS = 50  

STATIC_ARGS_FOR_HAKT = {
    "run_script_path": RUN_SCRIPT_PATH,
    "kernel_name": KERNEL_TO_TUNE, 
    "num_tokens": 16088,
    "num_experts": 128,
    "top_k": 2,
    "hidden_size": 6144,
    "inter_size": 1536,
    "dtype": "fp16",
    "num_iters": 1 
}

FULL_PARAM_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128, 256],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4, 5]
}
# -----------------------------

def get_initial_profile_data():
    """
    Runs ncu *once* with default config to get the 'goldmine' report.
    This must be run *before* Ray is initialized.
    """
    print("[HAKT] Running initial profile (pre-flight check)...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(WORKER_GPU_ID) 
    
    ncu_log_file = "initial_profile.csv"
    
    command = [
        "ncu", "--csv",
        "--kernel-name", KERNEL_TO_TUNE, 
        "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
        "--target-processes", "all",
        "--force-overwrite",
        "--log-file", ncu_log_file,
        "python", RUN_SCRIPT_PATH,
        "--num-tokens", str(STATIC_ARGS_FOR_HAKT["num_tokens"]), 
        "--num-iters", str(STATIC_ARGS_FOR_HAKT["num_iters"]), 
        "--num-warmup-iters", "1",
        "--num-experts", str(STATIC_ARGS_FOR_HAKT["num_experts"]),
        "--top-k", str(STATIC_ARGS_FOR_HAKT["top_k"]),
        "--hidden-size", str(STATIC_ARGS_FOR_HAKT["hidden_size"]),
        "--inter-size", str(STATIC_ARGS_FOR_HAKT["inter_size"]),
        "--dtype", STATIC_ARGS_FOR_HAKT["dtype"]
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=env)
        
        with open(ncu_log_file, 'r') as f:
            csv_data = f.read()
        print(f"[HAKT] Initial profile '{ncu_log_file}' created successfully.")
        return csv_data

    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("[HAKT] ERROR: Initial profiling failed. The 'ncu' command returned a non-zero exit code.")
        print(f"Please ensure GPU {WORKER_GPU_ID} is available and ncu is installed.")
        print("\n>>> DID YOU RUN 'sudo sysctl -w kernel.yama.ptrace_scope=0' ? <<<")
        print(f"\nCOMMAND THAT FAILED:\n{' '.join(e.cmd)}\n")
        print("\n--- NCU STANDARD OUTPUT (if any) ---")
        print(e.stdout)
        print("\n--- NCU STANDARD ERROR (THE *REAL* ERROR) ---")
        print(e.stderr)
        print("="*80 + "\n")
        raise
    except Exception as e:
        print(f"[HAKT] ERROR: A Python-level error occurred: {e}")
        raise

def main():
    initial_ncu_report = get_initial_profile_data()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(AGENT_GPU_ID)
    torch.cuda.set_device(0) 
    
    print(f"[HAKT] Main process (Professor/Fighter) pinned to Physical GPU: {AGENT_GPU_ID}")
    print(f"[HAKT] BenchmarkWorker will be requested for Physical GPU: {WORKER_GPU_ID}")

    max_seq_length = 2048
    print(f"[HAKT] Loading Professor LLM '{PROFESSOR_MODEL}' onto GPU 0 (Logical)...")
    
    professor_llm = AutoModelForCausalLM.from_pretrained(
        PROFESSOR_MODEL,
        device_map="auto", 
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(PROFESSOR_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    professor_llm.tokenizer = tokenizer

    print("[HAKT] Adding LoRA adapters to Professor LLM...")
    
    professor_llm = prepare_model_for_kbit_training(professor_llm)
    
    peft_config = LoraConfig(
        r=64, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    professor_llm = get_peft_model(professor_llm, peft_config)

    # --- FINAL PROMPT FIX: Remove ambiguity and force API-style JSON ---
    professor_prompt = f"""
You are a world-class CUDA kernel tuning expert. Your sole function is to act as a JSON API endpoint.
You must analyze the performance report provided below and generate a JSON object containing your optimization plan and rewards.
Your response MUST contain ONLY the JSON object. Do not include any reasoning, markdown ticks (```json), or conversational text outside of the JSON structure.

USER GOAL: {USER_GOAL} (Maximize this metric.)
HARDWARE: NVIDIA H100
MODEL NAME: = {MODEL_NAME}
TARGET KERNEL: {KERNEL_TO_TUNE} (MoE Layer)
PARAMETER SPACE: {json.dumps(FULL_PARAM_SPACE)}

PERFORMANCE REPORT (CSV):
{initial_ncu_report}

Generate the JSON plan based on this structure:
{{
    "reward_function": {{
        "R_sm_throughput": [float, Weight for SM Utilization],
        "R_dram_throughput": [float, Weight for DRAM Utilization],
        "R_l1_hit_rate": [float, Weight for L1 Cache],
        "R_l2_hit_rate": [float, Weight for L2 Cache]
    }},
    "pruned_action_space": {{
        "BLOCK_SIZE_M": [list of int, Max 3 values],
        "BLOCK_SIZE_N": [list of int, Max 3 values],
        "BLOCK_SIZE_K": [list of int, Max 3 values],
        "num_warps": [list of int, Max 3 values],
        "num_stages": [list of int, Max 3 values]
    }}
}}
"""
    # --- END FINAL PROMPT FIX ---

    dataset = Dataset.from_list(
        [{"prompt": professor_prompt}] * LLM_TRAIN_STEPS
    )

    print("[HAKT] Initializing HAKT Reward Function...")
    
    reward_fn_object = HAKT_Reward_Function(
        user_goal=USER_GOAL,
        model_name=MODEL_NAME,
        fast_loop_steps=FAST_LOOP_STEPS,
        worker_gpu_id=WORKER_GPU_ID, 
        static_args=STATIC_ARGS_FOR_HAKT
    )

    def hakt_reward_function_wrapper(completions, **kwargs):
        return reward_fn_object(completions, **kwargs)

    grpo_config = GRPOConfig(
        output_dir="hakt_professor_finetune",
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=1,
        learning_rate=2e-5, 
        num_train_epochs=1, 
        max_steps=LLM_TRAIN_STEPS,
        save_steps=10, 
        logging_steps=1,
        report_to="tensorboard", 
        remove_unused_columns=False, 

        temperature=0.7, 
        max_prompt_length=1024,
        max_completion_length=1024,
        num_generations=4, 
        loss_type="grpo", 
    )

    trainer = GRPOTrainer(
        model=professor_llm,
        args=grpo_config, 
        train_dataset=dataset,
        reward_funcs=[hakt_reward_function_wrapper],
    )

    print("\n--- [HAKT] Starting 'Slow Loop' Training for Professor LLM ---")
    trainer.train()
    print("\n--- [HAKT] Professor LLM fine-tuning complete ---")

    final_model_path = "hakt_professor_final"
    trainer.model.save_pretrained(final_model_path) 
    tokenizer.save_pretrained(final_model_path)
    print(f"Final fine-tuned Professor LLM saved to: {final_model_path}")
    
    ray.shutdown()
    print("[HAKT] System shutdown complete.")

if __name__ == "__main__":
    main()
