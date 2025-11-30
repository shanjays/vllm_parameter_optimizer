import argparse
import torch
import json
import os
from contextlib import nullcontext

#
# 1. THE CRITICAL FIX
#
# This prevents the "Duplicate registration" error when this script
# is repeatedly called as a subprocess.
#
import sys
if "vllm.model_executor.layers.fused_moe.fused_moe" in sys.modules:
    del sys.modules["vllm.model_executor.layers.fused_moe.fused_moe"]
#
# End of fix
#

#
# 2. NOW, WE CAN SAFELY IMPORT THE KERNEL FUNCTIONS
#
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, fused_experts, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe import override_config
from vllm.platforms import current_platform

# This is the "DEFAULT_CONFIG" your agent will be tuning
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
}


def benchmark_moe(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    inter_size: int,
    dtype: torch.dtype,
    num_warmup_iters: int,
    num_iters: int,
    config_path: str
):
    """
    This is our simple "Fast Gym" benchmark executable.
    It runs on the GPU specified by CUDA_VISIBLE_DEVICES.
    """
    
    # This script will run on the GPU specified by CUDA_VISIBLE_DEVICES
    # set by the BenchmarkWorker (e.g., GPU 1)
    torch.set_default_device("cuda")
    current_platform.seed_everything(42)

    # 1. Create Tensors
    N = inter_size // 2
    K = hidden_size
    E = num_experts

    x = torch.randn(num_tokens, K, dtype=dtype)
    w1 = torch.randn(E, N, K, dtype=dtype)
    w2 = torch.randn(E, K, N, dtype=dtype)
    gating_output = torch.randn(num_tokens, E, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(quant_dtype=None)

    # --- MODIFICATION ---
    # Load the config from the path provided by the worker
    config_to_use = DEFAULT_CONFIG
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_to_use = json.load(f)
        except json.JSONDecodeError:
            print(f"[run_moe_gym] Warning: {config_path} is corrupted, using default.")
            pass # Use default
    # --- END MODIFICATION ---

    # 2. Define the function to run
    def run():
        with override_config(config_to_use):
            topk_weights, topk_ids, _ = fused_topk(x, gating_output, top_k,
                                                   renormalize=True)
            fused_experts(
                x,
                w1,
                w2,
                topk_weights,
                topk_ids,
                inplace=True,
                quant_config=quant_config,
            )

    # 3. JIT compilation & warmup
    
    for _ in range(num_warmup_iters):
        run()
    torch.cuda.synchronize()

    # 4. Run benchmark
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run()
    end_event.record()

    torch.cuda.synchronize()
    
    # Only print latency if num_iters > 0
    if num_iters > 0:
        latency_ms = start_event.elapsed_time(end_event) / num_iters
        print(f"\n--- Benchmark Complete ---")
        print(f"Avg. Latency: {latency_ms:.4f} ms")
    else:
        # This is the normal path for ncu
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple benchmark 'gym' for fused_moe_kernel.")

    # Workload parameters
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--inter-size", type=int, default=11008)
    parser.add_argument("--num-tokens", type=int, default=16088) # Safe value
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    
    # Iteration parameters
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=1,
        help="Number of warmup runs (for JIT, etc.)"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=0, # Set default to 0 for clean ncu runs
        help="Number of measured benchmark runs."
    )
    
    # Config path from worker
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to temp_config.json written by worker"
    )

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    benchmark_moe(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        inter_size=args.inter_size,
        dtype=dtype,
        num_warmup_iters=args.num_warmup_iters,
        num_iters=args.num_iters,
        config_path=args.config_path
    )