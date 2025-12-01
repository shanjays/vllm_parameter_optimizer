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
from feedback_collector import FeedbackCollector

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
MAX_COMPLETION_LENGTH = 1536      # Increased to allow JSON-first output with brief reasoning

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


def summarize_ncu_report(ncu_csv: str) -> str:
    """Extract key metrics from NCU CSV data."""
    import re
    
    metrics = {
        'sm_throughput': [],
        'dram_throughput': [],
        'l1_hit_rate': [],
        'l2_hit_rate': []
    }
    
    for match in re.finditer(r'sm__throughput\.avg\.pct_of_peak_sustained_elapsed[^0-9]*([0-9.]+)', ncu_csv):
        metrics['sm_throughput'].append(float(match.group(1)))
    for match in re.finditer(r'dram__throughput\.avg\.pct_of_peak_sustained_elapsed[^0-9]*([0-9.]+)', ncu_csv):
        metrics['dram_throughput'].append(float(match.group(1)))
    for match in re.finditer(r'l1tex__t_sector_hit_rate\.pct[^0-9]*([0-9.]+)', ncu_csv):
        metrics['l1_hit_rate'].append(float(match.group(1)))
    for match in re.finditer(r'lts__t_sector_hit_rate\.pct[^0-9]*([0-9.]+)', ncu_csv):
        metrics['l2_hit_rate'].append(float(match.group(1)))
    
    def stats(vals):
        if not vals:
            return "N/A"
        return f"min={min(vals):.1f}%, max={max(vals):.1f}%, avg={sum(vals)/len(vals):.1f}%"
    
    return f"""SM Throughput: {stats(metrics['sm_throughput'])}
DRAM Throughput: {stats(metrics['dram_throughput'])}
L1 Cache Hit Rate: {stats(metrics['l1_hit_rate'])}
L2 Cache Hit Rate: {stats(metrics['l2_hit_rate'])}"""


def build_optimization_prompt(initial_ncu_report, feedback_collector=None):
    """
    Build the optimization prompt with optional feedback from previous iterations.
    
    Uses detailed expert-level prompt to encourage JSON-first output with brief reasoning.
    
    Args:
        initial_ncu_report: Initial NCU profiling data (raw CSV)
        feedback_collector: Optional FeedbackCollector for contextual feedback
        
    Returns:
        str: The formatted optimization prompt
    """
    # Format token counts for the prompt
    token_counts_str = ", ".join(str(tc) for tc in TOKEN_COUNTS_TO_TEST)
    
    # Summarize NCU data instead of including raw CSV
    ncu_summary = summarize_ncu_report(initial_ncu_report)
    
    # Get feedback from previous iterations if available
    feedback_section = ""
    if feedback_collector:
        feedback_section = feedback_collector.format_feedback_for_prompt()
    
    optimization_prompt = f'''You are an expert CUDA kernel optimization engineer specializing in NVIDIA GPU architectures and Triton kernel tuning. Your task is to generate an optimal configuration policy for the fused_moe (Mixture of Experts) kernel.

═══════════════════════════════════════════════════════════════════════════════
                              KERNEL DETAILS
═══════════════════════════════════════════════════════════════════════════════

KERNEL: fused_moe_kernel (Triton)
PURPOSE: Fused Mixture-of-Experts computation combining expert routing, matrix multiplication, and reduction in a single kernel to minimize memory bandwidth.

MODEL CONFIGURATION:
- Model: {MODEL_NAME}
- Number of Experts (E): {STATIC_BENCHMARK_ARGS['num_experts']}
- Intermediate Size (N): {STATIC_BENCHMARK_ARGS['inter_size']}
- Hidden Size: {STATIC_BENCHMARK_ARGS['hidden_size']}
- Top-K Experts: {STATIC_BENCHMARK_ARGS['top_k']}
- Data Type: {STATIC_BENCHMARK_ARGS['dtype']}

TOKEN BATCH SIZES TO OPTIMIZE: {token_counts_str}

═══════════════════════════════════════════════════════════════════════════════
                              HARDWARE SPECS
═══════════════════════════════════════════════════════════════════════════════

GPU: NVIDIA H100 80GB HBM3
- SM Count: 132 Streaming Multiprocessors
- Shared Memory per SM: 228 KB (232,448 bytes)
- L2 Cache: 50 MB
- HBM3 Bandwidth: 3.35 TB/s
- FP16/BF16 Tensor Core Peak: 1,979 TFLOPS

SHARED MEMORY USAGE FORMULA:
  shared_mem = (BLOCK_SIZE_M × BLOCK_SIZE_K + BLOCK_SIZE_K × BLOCK_SIZE_N) × 2 × num_stages
  Must be ≤ 232,448 bytes

═══════════════════════════════════════════════════════════════════════════════
                           BASELINE PROFILING METRICS
═══════════════════════════════════════════════════════════════════════════════

Current NCU profiling results:
{ncu_summary}

Analyze these metrics to determine if the kernel is compute-bound or memory-bound.

═══════════════════════════════════════════════════════════════════════════════
                           TUNING PARAMETERS
═══════════════════════════════════════════════════════════════════════════════

You must choose 2-3 values for each parameter:

1. BLOCK_SIZE_M - Tile size in M dimension
   Valid: [16, 32, 64, 128]
   Larger = better arithmetic intensity, Smaller = better for small batches

2. BLOCK_SIZE_N - Tile size in N dimension
   Valid: [32, 64, 128]
   Larger = better memory coalescing

3. BLOCK_SIZE_K - Tile size in K dimension (reduction)
   Valid: [32, 64]
   K=64 better for compute, K=32 for memory-bound

4. num_warps - Warps per thread block
   Valid: [4, 8, 16]
   More warps = higher occupancy

5. num_stages - Software pipelining stages
   Valid: [2, 3, 4, 5]
   More stages = better latency hiding (limited by shared memory)

═══════════════════════════════════════════════════════════════════════════════
                           OBJECTIVE WEIGHTS
═══════════════════════════════════════════════════════════════════════════════

Specify weights for the reward function (must sum to 1.0):

- R_sm_throughput: SM compute efficiency (typical: 0.3-0.5)
- R_dram_throughput: Memory bandwidth utilization (typical: 0.25-0.4)
- R_l1_hit_rate: L1 cache efficiency (typical: 0.05-0.15)
- R_l2_hit_rate: L2 cache efficiency (typical: 0.1-0.2)
{feedback_section}
═══════════════════════════════════════════════════════════════════════════════
                           YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

1. Analyze the baseline metrics above
2. Determine if kernel is compute-bound or memory-bound
3. Choose appropriate objective weights based on your analysis
4. Select 2-3 promising values for each search_space parameter
5. Output your policy JSON FIRST, then provide brief reasoning

═══════════════════════════════════════════════════════════════════════════════
                           OUTPUT FORMAT (IMPORTANT!)
═══════════════════════════════════════════════════════════════════════════════

You MUST output in this EXACT format:
1. FIRST: Output your policy JSON inside <param></param> tags
2. THEN: Provide brief reasoning (2-4 sentences) explaining your choices

<param>
{{
  "objective_weights": {{
    "R_sm_throughput": <your_value>,
    "R_dram_throughput": <your_value>,
    "R_l1_hit_rate": <your_value>,
    "R_l2_hit_rate": <your_value>
  }},
  "search_space": {{
    "BLOCK_SIZE_M": [<2-3 values>],
    "BLOCK_SIZE_N": [<2-3 values>],
    "BLOCK_SIZE_K": [<2 values>],
    "num_warps": [<2-3 values>],
    "num_stages": [<2-3 values>]
  }}
}}
</param>

REASONING: <Your brief 2-4 sentence explanation of why you chose these values>

═══════════════════════════════════════════════════════════════════════════════

NOW GENERATE YOUR OPTIMIZED POLICY (JSON first, then reasoning):

<param>
'''
    return optimization_prompt


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

    # Initialize feedback collector for contextual learning
    print("[HierarchicalOptimizer] Initializing Feedback Collector for contextual learning...")
    feedback_collector = FeedbackCollector(state_file="./feedback_state.json")
    
    # Build initial prompt (will be regenerated with feedback each step)
    optimization_prompt = build_optimization_prompt(initial_ncu_report, feedback_collector)

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
        token_counts=TOKEN_COUNTS_TO_TEST,
        feedback_collector=feedback_collector
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
        max_prompt_length=4096,
        max_completion_length=MAX_COMPLETION_LENGTH,  # Use the configured value
        num_generations=NUM_GENERATIONS,               # Use the configured value
        loss_type="grpo",
    )

    # Create dataset with prompts that include contextual feedback
    # Note: HuggingFace Dataset uses static snapshots, but feedback will be recorded
    # after each policy evaluation and persisted for use in subsequent training runs
    dataset = Dataset.from_list([{"prompt": build_optimization_prompt(initial_ncu_report, feedback_collector)}] * META_LEARNING_STEPS)

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
    
    # Print feedback collector summary
    feedback_summary = feedback_collector.get_summary()
    print(f"[HierarchicalOptimizer] Feedback Summary: {feedback_summary}")

    final_model_path = "hierarchical_optimizer_final"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final fine-tuned Meta-Controller LLM saved to: {final_model_path}")

    ray.shutdown()
    print("[HierarchicalOptimizer] System shutdown complete.")

if __name__ == "__main__":
    main()






