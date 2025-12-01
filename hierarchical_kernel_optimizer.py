# Hierarchical Kernel Optimizer
# 
# This module implements a hierarchical optimization system for CUDA kernel tuning.
# It uses a meta-learning approach where an LLM generates optimization policies
# that guide an RL agent's exploration of the kernel configuration space.
#
# Key components:
# - Meta-Controller: LLM that generates optimization policies
# - Exploration Agent: PPO agent that explores configurations
# - Profiling Worker: Executes benchmarks and collects metrics

import json
import ray
import os
import subprocess
import torch
import time
from datasets import Dataset
from importlib import metadata as importlib_metadata

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from meta_controller import MetaControllerReward
from config_exporter import VLLMConfigExporter, TOKEN_COUNTS_ALL

META_CONTROLLER_GPU_ID = 0
PROFILING_GPU_ID = 7

# ============== 5-HOUR AGGRESSIVE TRAINING CONFIG ==============

# Token counts for training (key representative values)
TOKEN_COUNTS_TRAINING = [1, 16, 64, 256, 1024, 4096]

# Token counts to test (matching vLLM's expected format)
TOKEN_COUNTS_TO_TEST = TOKEN_COUNTS_TRAINING

# Meta-controller model (LLM for generating optimization policies)
META_CONTROLLER_MODEL = "openai/gpt-oss-20b"

USER_GOAL = "throughput"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
KERNEL_TO_TUNE = "fused_moe_kernel"
RUN_SCRIPT_PATH = "run_kernel_benchmark.py"

# Meta-learning (outer loop) - optimized for 5-hour run
EXPLORATION_STEPS = 10            # Increased from 8 for better exploration
META_LEARNING_STEPS = 4           # Reduced from 50 for 5-hour run
NUM_GENERATIONS = 4               # Reduced from 8
MAX_COMPLETION_LENGTH = 1536      # Increased from 1024 for aggressive prompts

# vLLM validation settings
VLLM_NUM_PROMPTS = 40             # Reduced from 100
VLLM_MAX_MODEL_LEN = 4096
VLLM_GPU_MEMORY_UTIL = 0.90       # Aggressive (was 0.85)

STATIC_BENCHMARK_ARGS = {
    "run_script_path": RUN_SCRIPT_PATH,
    "kernel_name": "fused_moe_kernel",
    "num_tokens": 4096,       # Will be varied during training
    "num_experts": 128,       # E=128 ✅
    "top_k": 8,               # 8 experts ✅ (was 2)
    "hidden_size": 6656,      # ✅ (was 6144)
    "inter_size": 768,        # N=768 ✅ (was 1536)
    "dtype": "bf16",      # bf16 for H100
    "num_iters": 3,
    "num_warmup_iters": 1,
}

# Aggressive search space for H100 - includes num_stages=5 for aggressive testing
AGGRESSIVE_SEARCH_SPACE = {
    "BLOCK_SIZE_M": [64, 128],
    "BLOCK_SIZE_N": [64, 128],
    "BLOCK_SIZE_K": [32, 64],
    "num_warps": [8, 16],
    "num_stages": [3, 4, 5],      # Include 5 for aggressive testing!
}

# Safe parameter space for H100 - validated against shared memory limits
FULL_PARAM_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [32, 64],
    "num_warps": [4, 8, 16],
    "num_stages": [2, 3, 4, 5]     # Include 5 for aggressive testing
}

def _check_versions_and_env():
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

def get_initial_profile_data():
    """Run initial NCU profiling to collect baseline performance metrics."""
    print("[HierarchicalOptimizer] Running initial profile (pre-flight check)...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(PROFILING_GPU_ID) 
    ncu_log_file = "initial_profile.csv"
    command = [
        "ncu", "--csv",
        "--kernel-name", KERNEL_TO_TUNE, 
        "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
        "--target-processes", "all",
        "--force-overwrite",
        "--log-file", ncu_log_file,
        "python", RUN_SCRIPT_PATH,
        "--num-tokens", str(STATIC_BENCHMARK_ARGS["num_tokens"]), 
        "--num-iters", str(STATIC_BENCHMARK_ARGS["num_iters"]), 
        "--num-warmup-iters", "1",
        "--num-experts", str(STATIC_BENCHMARK_ARGS["num_experts"]),
        "--top-k", str(STATIC_BENCHMARK_ARGS["top_k"]),
        "--hidden-size", str(STATIC_BENCHMARK_ARGS["hidden_size"]),
        "--inter-size", str(STATIC_BENCHMARK_ARGS["inter_size"]),
        "--dtype", STATIC_BENCHMARK_ARGS["dtype"]
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60, env=env)
    with open(ncu_log_file, 'r') as f:
        csv_data = f.read()
    print(f"[HierarchicalOptimizer] Initial profile '{ncu_log_file}' created successfully.")
    return csv_data

def main():
    """Main entry point for the hierarchical kernel optimizer."""
    _check_versions_and_env()
    initial_ncu_report = get_initial_profile_data()

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Set visible devices *before* loading the model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(META_CONTROLLER_GPU_ID)
    try:
        torch.cuda.set_device(0)  # Use logical device 0
    except Exception:
        pass

    print(f"[HierarchicalOptimizer] Main process pinned to Physical GPU: {META_CONTROLLER_GPU_ID}")
    print(f"[HierarchicalOptimizer] ProfilingWorker will be requested for Physical GPU: {PROFILING_GPU_ID}")

    print(f"[HierarchicalOptimizer] Loading Meta-Controller LLM '{META_CONTROLLER_MODEL}' onto GPU 0 (Logical)...")
    
    # Load model with Unsloth optimization
    max_seq_length = 4096
    meta_controller_llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=META_CONTROLLER_MODEL,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    print("[HierarchicalOptimizer] Adding LoRA adapters to Meta-Controller LLM...")
    
    meta_controller_llm = FastLanguageModel.get_peft_model(
        meta_controller_llm,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        task_type="CAUSAL_LM",
    )

    # Format token counts for the prompt
    token_counts_str = ", ".join(str(tc) for tc in TOKEN_COUNTS_TO_TEST)
    
    optimization_prompt = f"""You are an AGGRESSIVE CUDA kernel optimization expert. Your goal is to find the FASTEST possible fused_moe kernel configurations for maximum throughput.

=== TARGET MODEL ===
{MODEL_NAME}
Experts: {STATIC_BENCHMARK_ARGS['num_experts']}, Intermediate Size: {STATIC_BENCHMARK_ARGS['inter_size']}, Top-K: {STATIC_BENCHMARK_ARGS['top_k']}

=== HARDWARE ===
NVIDIA H100 80GB HBM3
Shared Memory: 227KB per SM (232,448 bytes)
Target: Push configs to the EDGE of hardware limits!

=== TOKEN COUNTS TO OPTIMIZE ===
You are optimizing for these specific token batch sizes:
{token_counts_str}

Each token count needs different optimal parameters:
- Small batches (1-16): May prefer smaller blocks, fewer stages
- Medium batches (64-512): Balance between parallelism and efficiency  
- Large batches (1024+): Maximize throughput with larger blocks

=== INITIAL PROFILING METRICS ===
{initial_ncu_report}

=== OPTIMIZATION POLICY FORMAT ===
Output your policy inside <param></param> tags as JSON:

<param>
{{
  "objective_weights": {{
    "R_sm_throughput": <0.0-1.0>,
    "R_dram_throughput": <0.0-1.0>,
    "R_l1_hit_rate": <0.0-1.0>,
    "R_l2_hit_rate": <0.0-1.0>
  }},
  "search_space": {{
    "BLOCK_SIZE_M": [<values from 16,32,64,128>],
    "BLOCK_SIZE_N": [<values from 32,64,128>],
    "BLOCK_SIZE_K": [<values from 32,64>],
    "num_warps": [<values from 4,8,16>],
    "num_stages": [<values from 2,3,4,5>]
  }}
}}
</param>

=== 🔥 BE AGGRESSIVE! 🔥 ===
- Configs that occasionally hit VRAM limits are GOOD - they show we're pushing boundaries!
- LARGER block sizes generally = BETTER throughput
- MORE pipeline stages = BETTER memory hiding (up to VRAM limits)
- num_stages=5 is allowed for aggressive testing!
- If configs never fail, you're being TOO CONSERVATIVE!
- Target: 90%+ GPU memory utilization

=== STRATEGY ===
1. Prioritize SM throughput (weight 0.4-0.5) - this is compute bound
2. Secondary: DRAM throughput (weight 0.3-0.4) - memory bandwidth matters
3. Cache hit rates are less critical for MoE (weight 0.1-0.15 each)

Output ONLY the <param>JSON</param> block. Keep reasoning minimal.
"""

    dataset = Dataset.from_list([{"prompt": optimization_prompt}] * META_LEARNING_STEPS)

    # Create config exporter for vLLM format
    config_exporter = VLLMConfigExporter(
        num_experts=STATIC_BENCHMARK_ARGS['num_experts'],
        inter_size=STATIC_BENCHMARK_ARGS['inter_size'],
        device_name="NVIDIA_H100_80GB_HBM3"
    )

    print("[HierarchicalOptimizer] Initializing Meta-Controller Reward Function...")
    reward_fn_object = MetaControllerReward(
        user_goal=USER_GOAL,
        model_name=MODEL_NAME,
        exploration_steps=EXPLORATION_STEPS,
        profiling_gpu_id=PROFILING_GPU_ID,
        static_args=STATIC_BENCHMARK_ARGS,
        config_exporter=config_exporter,
        token_counts=TOKEN_COUNTS_TO_TEST
    )

    def meta_controller_reward_wrapper(completions, **kwargs):
        return reward_fn_object(completions, **kwargs)

    # GRPO training configuration - optimized for 5-hour aggressive training
    grpo_config = GRPOConfig(
        output_dir="hierarchical_optimizer_finetune",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        max_steps=META_LEARNING_STEPS,
        save_steps=10,
        logging_steps=1,
        report_to="tensorboard",
        remove_unused_columns=False,
        temperature=0.7,
        max_prompt_length=2048,
        max_completion_length=MAX_COMPLETION_LENGTH,  # Use the configured value
        num_generations=NUM_GENERATIONS,               # Use the configured value
        loss_type="grpo",
    )

    trainer = GRPOTrainer(
        model=meta_controller_llm,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[meta_controller_reward_wrapper],
    )

    print("\n--- [HierarchicalOptimizer] Starting Meta-Learning Phase ---")
    trainer.train()
    print("\n--- [HierarchicalOptimizer] Meta-learning phase complete ---")

    # Save best configs in vLLM format
    print("\n--- [HierarchicalOptimizer] Saving Optimized Configs ---")
    vllm_config_path = config_exporter.save_vllm_config()
    
    # Export complete config with all token counts (interpolated)
    print("\n--- [HierarchicalOptimizer] Exporting Complete Config with ALL Token Counts ---")
    complete_config_path = config_exporter.export_complete_config()
    
    # Optionally copy to vLLM installation directory
    config_exporter.copy_to_vllm()
    
    # Print summary
    summary = config_exporter.get_summary()
    print(f"[HierarchicalOptimizer] Config Summary: {summary}")

    final_model_path = "hierarchical_optimizer_final"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final fine-tuned Meta-Controller LLM saved to: {final_model_path}")

    ray.shutdown()
    print("[HierarchicalOptimizer] System shutdown complete.")

if __name__ == "__main__":
    main()





