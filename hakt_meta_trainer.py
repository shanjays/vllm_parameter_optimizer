# Key changes: Swapped transformers/peft for Unsloth's FastLanguageModel
import json
import ray
import os
import subprocess
import torch
import time
from datasets import Dataset
from importlib import metadata as importlib_metadata

# --- THIS IS THE FIX ---
# We now import Unsloth as requested
from unsloth import FastLanguageModel
# --- END FIX ---

from transformers import AutoTokenizer
# We no longer need the standard AutoModelForCausalLM, LoraConfig, or peft functions
from trl import GRPOTrainer, GRPOConfig

from professor_reward import HAKT_Reward_Function

AGENT_GPU_ID = 7
WORKER_GPU_ID = 0

# --- THIS IS THE FIX ---
# We must use an Unsloth-optimized model name.
# Your log shows your 'gpt-oss-20b' took 77.48GiB, meaning it's a 70B+ model.
# We will use an Unsloth-optimized 70B model to match your intent.
PROFESSOR_MODEL = "openai/gpt-oss-20b"
# --- END FIX ---

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

def _check_versions_and_env():
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

def get_initial_profile_data():
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
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=env)
    with open(ncu_log_file, 'r') as f:
        csv_data = f.read()
    print(f"[HAKT] Initial profile '{ncu_log_file}' created successfully.")
    return csv_data

def main():
    _check_versions_and_env()
    initial_ncu_report = get_initial_profile_data()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Set visible devices *before* loading the model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(AGENT_GPU_ID)
    try:
        torch.cuda.set_device(0) # Use logical device 0
    except Exception:
        pass

    print(f"[HAKT] Main process (Professor/Fighter) pinned to Physical GPU: {AGENT_GPU_ID}")
    print(f"[HAKT] BenchmarkWorker will be requested for Physical GPU: {WORKER_GPU_ID}")

    print(f"[HAKT] Loading Professor LLM '{PROFESSOR_MODEL}' onto GPU 0 (Logical)...")
    
    # --- THIS IS THE UNSLOTH FIX ---
    # We replace the standard loader with FastLanguageModel
    # This will load the model in 4-bit and fix the OOM error.
    # Increased max_seq_length from 2048 to 4096 for better VRAM utilization on H100
    # Note: Using 4096 instead of 8192 to balance memory usage with batch size increase
    max_seq_length = 4096
    professor_llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name = PROFESSOR_MODEL,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )
    # --- END FIX ---

    print("[HAKT] Adding LoRA adapters to Professor LLM...")
    
    # --- THIS IS THE UNSLOTH FIX ---
    # We replace prepare_model_for_kbit_training and get_peft_model
    # with Unsloth's single, optimized function.
    professor_llm = FastLanguageModel.get_peft_model(
        professor_llm,
        r = 64, # LoRA rank
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 128,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Unsloth's smart checkpointing
        random_state = 3407,
        task_type = "CAUSAL_LM",
    )
    # --- END FIX ---

    professor_prompt = f"""You are a world-class CUDA kernel tuning expert specializing in NVIDIA H100 GPU optimization.

=== HARDWARE SPECIFICATIONS (NVIDIA H100 80GB HBM3) ===
- Memory: 80GB HBM3
- Memory Bandwidth: 3.35 TB/s
- FP16 Tensor Core Performance: 989 TFLOPS
- FP32 Performance: 528 TFLOPS
- Streaming Multiprocessors (SMs): 132 SMs with 128 CUDA cores each
- L2 Cache: 50MB total, ~256KB per SM partition
- Shared Memory: 228KB per SM (configurable L1/shared split)
- Max Warps per SM: 64
- Max Threads per SM: 2048

=== TARGET KERNEL: {KERNEL_TO_TUNE} ===
The fused_moe_kernel is a Triton kernel for Mixture-of-Experts (MoE) computation.
It fuses expert routing, token dispatch, and expert computation into a single kernel.

=== KERNEL PARAMETER IMPACT ===
| Parameter | Description | Increasing Value Effect |
|-----------|-------------|------------------------|
| BLOCK_SIZE_M | Tokens processed per thread block | More tokens per block = higher throughput, but may reduce occupancy if shared memory exceeds limits |
| BLOCK_SIZE_N | Output dimension tile size | Larger tiles = better memory coalescing, but more register pressure |
| BLOCK_SIZE_K | Reduction dimension tile size | Larger K = more shared memory usage, better L1 cache utilization, fewer global memory accesses |
| GROUP_SIZE_M | Number of blocks grouped for L2 locality | Higher = better L2 reuse for weights, but may increase latency |
| num_warps | Warps per block (MUST be power of 2: 2,4,8,16,32) | More parallelism per block, but higher values may reduce occupancy |
| num_stages | Software pipelining stages (1-8) | More prefetching = hides memory latency, but uses more registers and shared memory |

=== TARGET MODEL: {MODEL_NAME} ===
- Architecture: Mixture-of-Experts (MoE)
- Number of Experts: {STATIC_ARGS_FOR_HAKT['num_experts']} experts
- Top-K Routing: {STATIC_ARGS_FOR_HAKT['top_k']}
- Hidden Size: {STATIC_ARGS_FOR_HAKT['hidden_size']}
- Intermediate Size: {STATIC_ARGS_FOR_HAKT['inter_size']}
- Data Type: {STATIC_ARGS_FOR_HAKT['dtype']}

=== USER OPTIMIZATION GOAL ===
{USER_GOAL}

=== CURRENT PARAMETER SPACE ===
{json.dumps(FULL_PARAM_SPACE, indent=2)}

=== INITIAL PERFORMANCE REPORT (NCU CSV) ===
{initial_ncu_report}

=== OUTPUT FORMAT ===
You MUST output your response within <param></param> XML tags containing ONLY a valid JSON object.
The JSON must be parseable by Python's json.loads() with no comments, trailing commas, or type annotations.

<param>
{{
  "reward_function": {{
    "R_sm_throughput": <float 0.0-1.0>,
    "R_dram_throughput": <float 0.0-1.0>,
    "R_l1_hit_rate": <float 0.0-1.0>,
    "R_l2_hit_rate": <float 0.0-1.0>
  }},
  "pruned_action_space": {{
    "BLOCK_SIZE_M": [<int>, <int>, <int>],
    "BLOCK_SIZE_N": [<int>, <int>, <int>],
    "BLOCK_SIZE_K": [<int>, <int>, <int>],
    "num_warps": [<int>, <int>, <int>],
    "num_stages": [<int>, <int>, <int>]
  }}
}}
</param>

=== EXAMPLE VALID OUTPUT (from Mistral 7B E=8 config) ===
<param>
{{
  "reward_function": {{
    "R_sm_throughput": 0.6,
    "R_dram_throughput": 0.2,
    "R_l1_hit_rate": 0.1,
    "R_l2_hit_rate": 0.1
  }},
  "pruned_action_space": {{
    "BLOCK_SIZE_M": [16, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [64, 128, 256],
    "num_warps": [4, 8, 16],
    "num_stages": [3, 4, 5]
  }}
}}
</param>

=== RULES ===
1. reward_function weights should sum to approximately 1.0
2. Each pruned_action_space list may contain 1 to 3 integers from the PARAMETER SPACE
3. num_warps values MUST be from [2, 4, 8, 16, 32] (powers of 2)
4. num_stages values should be positive integers (1-8)
5. Output ONLY the <param>...</param> block with valid JSON inside
6. NO comments, NO trailing commas, NO type names like "float" or "int"
7. All numbers must be concrete values (e.g., 0.5, 64, not "0.5" or "float")

Analyze the NCU metrics and generate an optimized mission plan for the {KERNEL_TO_TUNE} targeting {MODEL_NAME} on H100:
"""

    dataset = Dataset.from_list([{"prompt": professor_prompt}] * LLM_TRAIN_STEPS)

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

    # Increased batch sizes and completion length for better H100 VRAM utilization (60-70GB target)
    grpo_config = GRPOConfig(
        output_dir="hakt_professor_finetune",
        per_device_train_batch_size=8,  # Increased from 4 to 8 for better VRAM utilization
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=1,
        max_steps=LLM_TRAIN_STEPS,
        save_steps=10,
        logging_steps=1,
        report_to="tensorboard",
        remove_unused_columns=False,
        temperature=0.7,
        max_prompt_length=2048,  # Increased from 1024 to 2048 for longer prompts
        max_completion_length=2048,  # Increased from 1024 to 2048 for longer outputs
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

