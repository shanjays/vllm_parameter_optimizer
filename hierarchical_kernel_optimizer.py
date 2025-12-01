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
from config_exporter import VLLMConfigExporter

META_CONTROLLER_GPU_ID = 0
PROFILING_GPU_ID = 7

# Token counts to test (matching vLLM's expected format)
# These represent different token batch sizes that the fused_moe kernel
# will encounter during inference. The optimal kernel config varies based
# on how many tokens are being processed.
TOKEN_COUNTS_TO_TEST = [
    1, 16, 64, 256, 1024, 4096
]

# Meta-controller model (LLM for generating optimization policies)
META_CONTROLLER_MODEL = "openai/gpt-oss-20b"

USER_GOAL = "throughput"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
KERNEL_TO_TUNE = "fused_moe_kernel"
RUN_SCRIPT_PATH = "run_kernel_benchmark.py"
EXPLORATION_STEPS = 20  # Each step = 1 NCU run (~30 sec), so 20 steps = ~10 minutes
META_LEARNING_STEPS = 50

STATIC_BENCHMARK_ARGS = {
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

# Safe parameter space for H100 - validated against shared memory limits
FULL_PARAM_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [32, 64],
    "num_warps": [4, 8, 16],
    "num_stages": [2, 3, 4]
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

    optimization_prompt = f"""You are a CUDA kernel optimization expert in a hierarchical optimization system.

=== HARDWARE SPECIFICATIONS (NVIDIA H100 80GB HBM3) ===
- Memory: 80GB HBM3
- Memory Bandwidth: 3.35 TB/s
- FP16 Tensor Core Performance: 989 TFLOPS
- Streaming Multiprocessors (SMs): 132 SMs with 128 CUDA cores each
- Shared Memory: 228KB per SM (configurable L1/shared split)
- Max Warps per SM: 64

=== TARGET KERNEL: {KERNEL_TO_TUNE} ===
The fused_moe_kernel is a Triton kernel for Mixture-of-Experts (MoE) computation.

=== KERNEL PARAMETER IMPACT ===
| Parameter | Description | Increasing Value Effect |
|-----------|-------------|------------------------|
| BLOCK_SIZE_M | Tokens processed per thread block | Higher throughput, may reduce occupancy |
| BLOCK_SIZE_N | Output dimension tile size | Better memory coalescing |
| BLOCK_SIZE_K | Reduction dimension tile size | More shared memory, better L1 utilization |
| num_warps | Warps per block (MUST be power of 2) | More parallelism per block |
| num_stages | Software pipelining stages | Hides memory latency |

=== TARGET MODEL: {MODEL_NAME} ===
- Number of Experts: {STATIC_BENCHMARK_ARGS['num_experts']} experts
- Top-K Routing: {STATIC_BENCHMARK_ARGS['top_k']}
- Hidden Size: {STATIC_BENCHMARK_ARGS['hidden_size']}
- Intermediate Size: {STATIC_BENCHMARK_ARGS['inter_size']}
- Data Type: {STATIC_BENCHMARK_ARGS['dtype']}

=== USER OPTIMIZATION GOAL ===
{USER_GOAL}

=== CURRENT SEARCH SPACE ===
{json.dumps(FULL_PARAM_SPACE, indent=2)}

=== INITIAL PERFORMANCE REPORT (NCU CSV) ===
{initial_ncu_report}

=== OUTPUT FORMAT ===
Output an 'optimization policy' with objective function weights and a search space.
You MUST output your response within <param></param> XML tags containing ONLY valid JSON.

<param>
{{
  "objective_weights": {{
    "R_sm_throughput": <float 0.0-1.0>,
    "R_dram_throughput": <float 0.0-1.0>,
    "R_l1_hit_rate": <float 0.0-1.0>,
    "R_l2_hit_rate": <float 0.0-1.0>
  }},
  "search_space": {{
    "BLOCK_SIZE_M": [<int>, <int>, <int>],
    "BLOCK_SIZE_N": [<int>, <int>, <int>],
    "BLOCK_SIZE_K": [<int>, <int>],
    "num_warps": [<int>, <int>, <int>],
    "num_stages": [<int>, <int>, <int>]
  }}
}}
</param>

=== HARDWARE CONSTRAINTS ===
- Shared Memory Limit: 227KB per SM
- BLOCK_SIZE_M, BLOCK_SIZE_N: ≤ 128
- BLOCK_SIZE_K: ≤ 64
- num_stages: ≤ 4
- num_warps: powers of 2 from [4, 8, 16]

=== RULES ===
1. objective_weights should sum to approximately 1.0
2. Output ONLY the <param>...</param> block with valid JSON
3. Keep reasoning BRIEF, output JSON IMMEDIATELY

Analyze the NCU metrics and generate an optimization policy:
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

    # GRPO training configuration
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
        max_completion_length=2048,
        num_generations=4,
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


