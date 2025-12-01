"""
Kernel Benchmark Script

This script executes the fused_moe kernel with specified configurations
for profiling and performance measurement. It is invoked by the ProfilingWorker
to collect NCU metrics for different kernel parameter combinations.
"""

import argparse
import torch
import json
import os
from contextlib import nullcontext

# Prevent duplicate registration error when called repeatedly as subprocess
import sys
if "vllm.model_executor.layers.fused_moe.fused_moe" in sys.modules:
    del sys.modules["vllm.model_executor.layers.fused_moe.fused_moe"]

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, fused_experts, FusedMoEQuantConfig)
from vllm.model_executor.layers.fused_moe import override_config
from vllm.platforms import current_platform

# Default kernel configuration
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
    Execute fused_moe kernel benchmark with the specified configuration.
    
    This function runs on the GPU specified by CUDA_VISIBLE_DEVICES,
    which is set by the ProfilingWorker.
    
    Args:
        num_tokens: Number of tokens to process
        num_experts: Number of MoE experts
        top_k: Top-K routing value
        hidden_size: Model hidden dimension
        inter_size: Intermediate size
        dtype: Data type for computation
        num_warmup_iters: Number of warmup iterations
        num_iters: Number of measurement iterations
        config_path: Path to configuration JSON file
    """
    torch.set_default_device("cuda")
    current_platform.seed_everything(42)

    # Create tensors for MoE computation
    N = inter_size // 2
    K = hidden_size
    E = num_experts

    x = torch.randn(num_tokens, K, dtype=dtype)
    w1 = torch.randn(E, N, K, dtype=dtype)
    w2 = torch.randn(E, K, N, dtype=dtype)
    gating_output = torch.randn(num_tokens, E, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(quant_dtype=None)

    # Load configuration from file if provided
    config_to_use = DEFAULT_CONFIG
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_to_use = json.load(f)
        except json.JSONDecodeError:
            print(f"[run_kernel_benchmark] Warning: {config_path} is corrupted, using default.")
            pass

    # Define the kernel execution function
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

    # Warmup iterations
    for _ in range(num_warmup_iters):
        run()
    torch.cuda.synchronize()

    # Measurement iterations
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        run()
    end_event.record()

    torch.cuda.synchronize()
    
    if num_iters > 0:
        latency_ms = start_event.elapsed_time(end_event) / num_iters
        print(f"\n--- Benchmark Complete ---")
        print(f"Avg. Latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kernel benchmark for fused_moe optimization.")

    # Workload parameters
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--inter-size", type=int, default=11008)
    parser.add_argument("--num-tokens", type=int, default=16088)
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
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=0,
        help="Number of measurement iterations"
    )
    
    # Config path from profiling worker
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to kernel configuration JSON file"
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