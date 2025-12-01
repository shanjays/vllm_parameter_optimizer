"""
DeepSeek-MoE-16B-Base Inference Script

This script uses trained models from hierarchical_kernel_optimizer.py to optimize
DeepSeek-MoE-16B-Base. It loads the fine-tuned LLM and optionally uses trained PPO
agents for transfer learning.

Key Features:
1. Load fine-tuned LLM from ./hierarchical_optimizer_final/
2. Load trained PPO agents for transfer learning
3. Use framework: ProfilingWorker, KernelTuningEnvironment, ExplorationAgent
4. Export vLLM config: E=66,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json

Usage:
    python inference_deepseek.py --llm-path ./hierarchical_optimizer_final --use-ppo
    python inference_deepseek.py --llm-path ./hierarchical_optimizer_final
    python inference_deepseek.py --use-ppo --gpu 7
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import ray
import torch

from profiling_worker import ProfilingWorker
from kernel_tuning_env import KernelTuningEnvironment
from exploration_agent import ExplorationAgent
from config_exporter import VLLMConfigExporter, TOKEN_COUNTS_ALL


# DeepSeek-MoE-16B-Base model configuration
DEEPSEEK_CONFIG = {
    "model_name": "deepseek-ai/deepseek-moe-16b-base",
    "hidden_size": 2048,
    "moe_intermediate_size": 1408,  # N value for vLLM config
    "num_routed_experts": 64,
    "num_shared_experts": 2,
    "top_k": 6,
    "dtype": "fp16",
}

# Total experts for vLLM config filename: E = routed + shared = 66
NUM_EXPERTS = DEEPSEEK_CONFIG["num_routed_experts"] + DEEPSEEK_CONFIG["num_shared_experts"]
INTER_SIZE = DEEPSEEK_CONFIG["moe_intermediate_size"]

# Default paths
DEFAULT_LLM_PATH = "./hierarchical_optimizer_final"
DEFAULT_PPO_LOG_DIR = "./logs/exploration_agent"
DEFAULT_OUTPUT_DIR = "./optimized_configs"

# GPU configuration
DEFAULT_META_CONTROLLER_GPU = 0
DEFAULT_PROFILING_GPU = 7

# Token counts to optimize
TOKEN_COUNTS_QUICK = [1, 16, 64, 256, 1024, 4096]
TOKEN_COUNTS_FULL = TOKEN_COUNTS_ALL

# Exploration parameters
DEFAULT_EXPLORATION_STEPS = 10

# Fallback initial state values (typical NCU metrics from H100)
# These are representative values when profiling cannot complete:
# SM throughput: 32.3%, DRAM throughput: 40.8%, L1 hit rate: 0.05%, L2 hit rate: 69.9%
FALLBACK_INITIAL_STATE = [32.3, 40.8, 0.05, 69.9]

# Default optimization policy
DEFAULT_OPTIMIZATION_POLICY = {
    "objective_weights": {
        "R_sm_throughput": 0.4,
        "R_dram_throughput": 0.3,
        "R_l1_hit_rate": 0.15,
        "R_l2_hit_rate": 0.15
    },
    "search_space": {
        "BLOCK_SIZE_M": [32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128],
        "BLOCK_SIZE_K": [32, 64],
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4]
    }
}


def load_fine_tuned_llm(llm_path, gpu_id=0):
    """
    Load the fine-tuned LLM for generating optimization policies.
    
    Args:
        llm_path: Path to the fine-tuned model directory
        gpu_id: GPU ID to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"[DeepSeekInference] Loading fine-tuned LLM from: {llm_path}")
    
    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM path not found: {llm_path}")
    
    # Save and set GPU for model loading
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        print(f"[DeepSeekInference] LLM loaded successfully on GPU {gpu_id}")
        return model, tokenizer
        
    except ImportError:
        print("[DeepSeekInference] WARNING: unsloth not available, skipping LLM loading")
        return None, None
    except Exception as e:
        print(f"[DeepSeekInference] WARNING: Failed to load LLM: {e}")
        return None, None


def generate_optimization_policy(model, tokenizer, ncu_report=""):
    """
    Generate an optimization policy using the fine-tuned LLM.
    
    Args:
        model: Fine-tuned LLM model
        tokenizer: Tokenizer for the model
        ncu_report: Optional NCU profiling report for context
        
    Returns:
        Dict containing the optimization policy
    """
    if model is None or tokenizer is None:
        print("[DeepSeekInference] LLM not available, using default policy")
        return DEFAULT_OPTIMIZATION_POLICY.copy()
    
    # Build prompt for DeepSeek optimization
    token_counts_str = ", ".join(str(tc) for tc in TOKEN_COUNTS_QUICK)
    
    prompt = f"""You are an expert CUDA kernel optimizer. Generate optimal fused_moe kernel parameters for DeepSeek-MoE-16B-Base.

=== TARGET MODEL ===
deepseek-ai/deepseek-moe-16b-base
Experts: {NUM_EXPERTS} (64 routed + 2 shared)
Intermediate Size: {INTER_SIZE}
Top-K: {DEEPSEEK_CONFIG['top_k']}
Hidden Size: {DEEPSEEK_CONFIG['hidden_size']}

=== HARDWARE ===
NVIDIA H100 80GB HBM3

=== TOKEN COUNTS ===
{token_counts_str}

{f"=== NCU PROFILING DATA ===\n{ncu_report}" if ncu_report else ""}

=== OUTPUT FORMAT ===
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
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from <param></param> tags
        policy = _extract_policy_from_response(response)
        print(f"[DeepSeekInference] Generated optimization policy from LLM")
        return policy
        
    except Exception as e:
        print(f"[DeepSeekInference] LLM generation failed: {e}, using default policy")
        return DEFAULT_OPTIMIZATION_POLICY.copy()


def _extract_policy_from_response(response):
    """Extract optimization policy JSON from LLM response."""
    # Try to extract content from <param></param> tags
    param_match = re.search(r'<param>\s*(\{[\s\S]*?\})\s*</param>', response, re.DOTALL)
    if param_match:
        json_str = param_match.group(1).strip()
    else:
        # Fallback: find JSON-like content
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
        else:
            return DEFAULT_OPTIMIZATION_POLICY.copy()
    
    # Clean up JSON string
    json_str = json_str.replace('```json', '').replace('```', '').strip()
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # Remove trailing commas
    
    try:
        policy = json.loads(json_str)
        return _validate_policy(policy)
    except json.JSONDecodeError:
        return DEFAULT_OPTIMIZATION_POLICY.copy()


def _validate_policy(policy):
    """Validate and coerce the optimization policy."""
    if not isinstance(policy, dict):
        return DEFAULT_OPTIMIZATION_POLICY.copy()
    
    # Ensure required keys exist
    if "objective_weights" not in policy:
        policy["objective_weights"] = DEFAULT_OPTIMIZATION_POLICY["objective_weights"].copy()
    if "search_space" not in policy:
        policy["search_space"] = DEFAULT_OPTIMIZATION_POLICY["search_space"].copy()
    
    # Validate objective weights
    weights = policy["objective_weights"]
    for key in ["R_sm_throughput", "R_dram_throughput", "R_l1_hit_rate", "R_l2_hit_rate"]:
        if key not in weights or not isinstance(weights[key], (int, float)):
            weights[key] = DEFAULT_OPTIMIZATION_POLICY["objective_weights"][key]
    
    # Validate search space
    ss = policy["search_space"]
    for key in ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"]:
        if key not in ss or not isinstance(ss[key], list) or not ss[key]:
            ss[key] = DEFAULT_OPTIMIZATION_POLICY["search_space"][key]
    
    # Ensure num_warps values are powers of 2
    valid_warps = [w for w in ss["num_warps"] if w in (2, 4, 8, 16, 32)]
    ss["num_warps"] = valid_warps if valid_warps else [4, 8]
    
    return policy


def get_initial_state(profiling_worker, static_args):
    """
    Get initial state by running a profiling pass.
    
    Args:
        profiling_worker: Ray handle to ProfilingWorker
        static_args: Static benchmark arguments
        
    Returns:
        Initial state as numpy array
    """
    print("[DeepSeekInference] Getting initial state from profiling worker...")
    try:
        job_id = profiling_worker.run_kernel_profiling.remote(None, static_args, {})
        state, reward, _ = ray.get(job_id)
        if state is None:
            raise RuntimeError("Worker failed initial profile.")
        print(f"[DeepSeekInference] Initial state: {state}")
        return state
    except Exception as e:
        print(f"[DeepSeekInference] Initial state failed: {e}, using fallback")
        return np.array(FALLBACK_INITIAL_STATE, dtype=np.float32)


def run_exploration_phase(
    policy_config_path,
    profiling_worker,
    static_args,
    initial_state,
    config_exporter,
    token_counts,
    exploration_steps,
    use_ppo=False,
    ppo_log_dir=None
):
    """
    Run exploration phase for all token counts.
    
    Args:
        policy_config_path: Path to optimization policy JSON
        profiling_worker: Ray handle to ProfilingWorker
        static_args: Static benchmark arguments
        initial_state: Initial observation state
        config_exporter: VLLMConfigExporter instance
        token_counts: List of token counts to optimize
        exploration_steps: Number of exploration steps per token count
        use_ppo: Whether to use trained PPO agents
        ppo_log_dir: Directory containing trained PPO models
        
    Returns:
        Dict mapping token count to best result
    """
    results = {}
    steps_per_token = max(4, exploration_steps // len(token_counts))
    
    for i, token_count in enumerate(token_counts):
        print(f"\n[DeepSeekInference] === Token Count {i+1}/{len(token_counts)}: {token_count} ===")
        
        # Create environment for this token count
        env = KernelTuningEnvironment(
            policy_config_path=policy_config_path,
            profiling_worker=profiling_worker,
            static_args=static_args,
            initial_state=initial_state,
            config_exporter=config_exporter,
            current_token_count=token_count
        )
        
        # Set up log directory for this token count
        log_dir = os.path.join(
            ppo_log_dir or DEFAULT_PPO_LOG_DIR,
            f"deepseek_tokens_{token_count}"
        )
        
        # Hide GPUs for SB3 MLP - force CPU
        prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        agent = None
        try:
            # Create exploration agent
            agent = ExplorationAgent(
                env,
                log_dir=log_dir,
                device="cpu",
                load_existing=use_ppo,  # Load existing model if use_ppo is True
            )
            
            # Run exploration
            agent.train_epoch(steps=steps_per_token)
            
            # Get results from environment
            top_results = env.get_top_results(n=3)
            
            if top_results:
                best_result = max(top_results, key=lambda x: x[2])
                results[token_count] = {
                    'config': best_result[0],
                    'state': best_result[1],
                    'reward': best_result[2]
                }
                
                # Update config exporter
                config_exporter.update_best_config(
                    token_count=token_count,
                    config=best_result[0],
                    reward=best_result[2]
                )
                
                # Save best model if improved
                agent.save_best_if_improved(best_result[2])
                
                print(f"[DeepSeekInference] Token {token_count}: Best reward = {best_result[2]:.2f}")
            
            # Also check environment's best_config tracking
            if env.best_config is not None:
                current_best_reward = results.get(token_count, {}).get('reward', float('-inf'))
                if env.best_reward > current_best_reward:
                    results[token_count] = {
                        'config': env.best_config,
                        'reward': env.best_reward
                    }
                    print(f"[DeepSeekInference] Token {token_count}: Updated from env tracking, reward = {env.best_reward:.2f}")
                
        finally:
            if agent:
                try:
                    agent.close()
                except Exception:
                    pass
            env.close()
            
            # Restore CUDA_VISIBLE_DEVICES
            if prev_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
    
    return results


def main():
    """Main entry point for DeepSeek inference optimization."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-MoE-16B-Base kernel parameter optimization using trained models"
    )
    
    parser.add_argument(
        "--llm-path",
        type=str,
        default=DEFAULT_LLM_PATH,
        help=f"Path to fine-tuned LLM directory (default: {DEFAULT_LLM_PATH})"
    )
    parser.add_argument(
        "--use-ppo",
        action="store_true",
        help="Use trained PPO agents for transfer learning"
    )
    parser.add_argument(
        "--ppo-log-dir",
        type=str,
        default=DEFAULT_PPO_LOG_DIR,
        help=f"Directory containing trained PPO models (default: {DEFAULT_PPO_LOG_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output configs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--meta-gpu",
        type=int,
        default=DEFAULT_META_CONTROLLER_GPU,
        help=f"GPU ID for meta-controller LLM (default: {DEFAULT_META_CONTROLLER_GPU})"
    )
    parser.add_argument(
        "--profiling-gpu",
        type=int,
        default=DEFAULT_PROFILING_GPU,
        help=f"GPU ID for profiling worker (default: {DEFAULT_PROFILING_GPU})"
    )
    parser.add_argument(
        "--exploration-steps",
        type=int,
        default=DEFAULT_EXPLORATION_STEPS,
        help=f"Number of exploration steps (default: {DEFAULT_EXPLORATION_STEPS})"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick token count list (6 values instead of all)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM loading and use default optimization policy"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("[DeepSeekInference] DeepSeek-MoE-16B-Base Optimization")
    print("=" * 60)
    print(f"  Model: {DEEPSEEK_CONFIG['model_name']}")
    print(f"  Experts: {NUM_EXPERTS} (E value)")
    print(f"  Intermediate Size: {INTER_SIZE} (N value)")
    print(f"  Top-K: {DEEPSEEK_CONFIG['top_k']}")
    print(f"  LLM Path: {args.llm_path}")
    print(f"  Use PPO: {args.use_ppo}")
    print(f"  Meta GPU: {args.meta_gpu}")
    print(f"  Profiling GPU: {args.profiling_gpu}")
    print(f"  Quick Mode: {args.quick}")
    print("=" * 60)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    # Load fine-tuned LLM (optional)
    model, tokenizer = None, None
    if not args.skip_llm:
        try:
            model, tokenizer = load_fine_tuned_llm(args.llm_path, args.meta_gpu)
        except Exception as e:
            print(f"[DeepSeekInference] LLM loading failed: {e}")
    
    # Create config exporter for DeepSeek
    config_exporter = VLLMConfigExporter(
        num_experts=NUM_EXPERTS,
        inter_size=INTER_SIZE,
        device_name="NVIDIA_H100_80GB_HBM3"
    )
    
    # Static benchmark arguments for DeepSeek
    static_args = {
        "run_script_path": "run_kernel_benchmark.py",
        "kernel_name": "fused_moe_kernel",
        "num_tokens": 4096,
        "num_experts": NUM_EXPERTS,
        "top_k": DEEPSEEK_CONFIG["top_k"],
        "hidden_size": DEEPSEEK_CONFIG["hidden_size"],
        "inter_size": INTER_SIZE,
        "dtype": DEEPSEEK_CONFIG["dtype"],
        "num_iters": 3,
        "num_warmup_iters": 1,
    }
    
    # Create profiling worker
    print(f"[DeepSeekInference] Creating ProfilingWorker on GPU {args.profiling_gpu}...")
    profiling_worker = ProfilingWorker.options(num_gpus=1).remote(args.profiling_gpu)
    
    # Get initial state
    initial_state = get_initial_state(profiling_worker, static_args)
    
    # Generate optimization policy
    policy = generate_optimization_policy(model, tokenizer)
    
    # Save policy to temp file using tempfile module for cross-platform compatibility
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.json', 
        prefix='deepseek_policy_', 
        delete=False
    ) as f:
        json.dump(policy, f, indent=2)
        policy_path = f.name
    print(f"[DeepSeekInference] Optimization policy saved to: {policy_path}")
    
    # Select token counts
    token_counts = TOKEN_COUNTS_QUICK if args.quick else TOKEN_COUNTS_FULL
    print(f"[DeepSeekInference] Optimizing for {len(token_counts)} token counts...")
    
    try:
        # Run exploration phase
        results = run_exploration_phase(
            policy_config_path=policy_path,
            profiling_worker=profiling_worker,
            static_args=static_args,
            initial_state=initial_state,
            config_exporter=config_exporter,
            token_counts=token_counts,
            exploration_steps=args.exploration_steps,
            use_ppo=args.use_ppo,
            ppo_log_dir=args.ppo_log_dir
        )
        
        # Export vLLM config
        print("\n[DeepSeekInference] Exporting vLLM configuration...")
        vllm_config_path = config_exporter.save_vllm_config(args.output_dir)
        
        # Export complete config with all token counts
        complete_config_path = config_exporter.export_complete_config()
        
        # Copy to vLLM installation if available
        config_exporter.copy_to_vllm()
        
        # Print summary
        print("\n" + "=" * 60)
        print("[DeepSeekInference] OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"  Config filename: {config_exporter.get_config_filename()}")
        print(f"  Token counts optimized: {len(results)}")
        print(f"  Output directory: {args.output_dir}")
        
        if results:
            print("\n  Best configs by token count:")
            for tc in sorted(results.keys()):
                reward = results[tc].get('reward', 0)
                print(f"    {tc:>5} tokens: reward = {reward:.2f}")
        
        summary = config_exporter.get_summary()
        print(f"\n  Total experiments: {summary['total_experiments']}")
        print("=" * 60)
        
    finally:
        # Cleanup
        if os.path.exists(policy_path):
            os.remove(policy_path)
        ray.shutdown()
    
    print("[DeepSeekInference] Done.")


if __name__ == "__main__":
    main()
