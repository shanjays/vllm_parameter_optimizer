import ray
import subprocess
import json
import csv
import os
import numpy as np
from io import StringIO
import time
# We use lazy imports for vllm inside methods to save RAM

# This is the "DEFAULT_CONFIG" your agent will be tuning
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
}

@ray.remote(num_gpus=1) # This actor requires 1 GPU from Ray's pool
class BenchmarkWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        pid = os.getpid()
        self.ncu_log_path = f"/tmp/hakt_ncu_log_gpu{self.gpu_id}_{pid}.csv"
        self.temp_config_path = f"/tmp/hakt_temp_config_gpu{self.gpu_id}_{pid}.json"
        
        try:
            import vllm
            vllm_lib_path = os.path.dirname(vllm.__file__)
            # vllm 0.10.2 might have a different config path
            self.vllm_config_dir = os.path.join(
                vllm_lib_path, "model_executor/layers/fused_moe/configs/"
            )
            if not os.path.exists(self.vllm_config_dir):
                 self.vllm_config_dir = "/tmp/vllm_configs/"
                 
            os.makedirs(self.vllm_config_dir, exist_ok=True)
            print(f"[BenchmarkWorker] vLLM config path set to: {self.vllm_config_dir}")
            del vllm 
        except Exception as e:
            print(f"[BenchmarkWorker] WARNING: Could not find vLLM path. {e}")
            self.vllm_config_dir = "/tmp/vllm_configs/" 
            os.makedirs(self.vllm_config_dir, exist_ok=True)
            
        print(f"[BenchmarkWorker] Initialized on PHYSICAL GPU {self.gpu_id} (PID: {pid})")

    def get_gpu_id(self):
        return self.gpu_id

    def _validate_triton_config(self, config):
        """
        Check if config fits H100 shared memory limit.
        
        H100 has 227KB (232,448 bytes) of shared memory per SM.
        Shared memory usage ≈ (M*K + K*N) * 2 * stages bytes for FP16.
        
        Returns True if config is valid, False otherwise.
        """
        M = config.get('BLOCK_SIZE_M', 64)
        N = config.get('BLOCK_SIZE_N', 64)
        K = config.get('BLOCK_SIZE_K', 32)
        stages = config.get('num_stages', 4)
        
        # FP16 = 2 bytes per element
        # Shared memory for A tile (M x K) and B tile (K x N)
        shared_mem = (M * K + K * N) * 2 * stages
        
        # H100 has 228KB shared memory per SM, we use conservative 227KB (232,448 bytes)
        H100_LIMIT = 232448  # bytes (~227KB, conservative limit)
        if shared_mem > H100_LIMIT:
            print(f"[BenchmarkWorker] Config exceeds shared memory: {shared_mem} bytes > {H100_LIMIT} bytes")
            print(f"[BenchmarkWorker] Offending config: BLOCK_SIZE_M={M}, BLOCK_SIZE_N={N}, BLOCK_SIZE_K={K}, num_stages={stages}")
            return False
        return True

    def run_fast_gym_benchmark(self, params_dict, static_args, reward_weights, num_tokens=None):
        """
        Runs the 'Fast Gym' (ncu) on this worker's GPU (GPU 1).
        Returns (state, reward, csv_data)
        
        Args:
            params_dict: Kernel configuration parameters
            static_args: Static arguments for the benchmark
            reward_weights: Weights for reward calculation
            num_tokens: Override token count (for multi-token testing)
        """
        
        # Use override token count if provided
        effective_static_args = static_args
        if num_tokens is not None:
            effective_static_args = static_args.copy()
            effective_static_args['num_tokens'] = num_tokens
        
        # --- FIX for GROUP_SIZE_M error ---
        config_to_use = DEFAULT_CONFIG.copy()
        if params_dict:
            config_to_use.update(params_dict)
        # --- END FIX ---
        
        # Validate num_warps is a power of 2
        num_warps = config_to_use.get('num_warps', 4)
        if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            print(f"[BenchmarkWorker] WARNING: Invalid num_warps={num_warps}, using 4")
            config_to_use['num_warps'] = 4
        
        # Validate config fits H100 shared memory before running NCU
        if not self._validate_triton_config(config_to_use):
            print(f"[BenchmarkWorker] Returning penalty for invalid config")
            return None, -100.0, None
            
        with open(self.temp_config_path, "w") as f:
            json.dump(config_to_use, f)

        ncu_command = [
            "ncu", "--csv",
            "--kernel-name", effective_static_args['kernel_name'], 
            "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", self.ncu_log_path,
        ]

        python_command = [
            "python", effective_static_args['run_script_path'],
            "--config-path", self.temp_config_path,
            "--num-experts", str(effective_static_args['num_experts']),
            "--top-k", str(effective_static_args['top_k']),
            "--hidden-size", str(effective_static_args['hidden_size']),
            "--inter-size", str(effective_static_args['inter_size']),
            "--num-tokens", str(effective_static_args['num_tokens']),
            "--dtype", effective_static_args['dtype'],
            "--num-iters", str(effective_static_args['num_iters']),
            "--num-warmup-iters", "1", 
        ]
        
        full_command = ncu_command + python_command
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id) 

        try:
            subprocess.run(full_command, check=True, capture_output=True, text=True, timeout=30, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[BenchmarkWorker] ERROR: NCU run failed on GPU {self.gpu_id}. STDERR:\n{e.stderr}")
            return None, -100.0, None 
        except subprocess.TimeoutExpired:
            print(f"[BenchmarkWorker] ERROR: NCU run timed out on GPU {self.gpu_id}.")
            return None, -100.0, None

        # --- THIS IS THE FIX: Robust "Long Format" CSV Parsing ---
        try:
            csv_data = ""
            # We will group metrics by Kernel Name and Invocation ID
            # Ex: { "fused_moe_kernel_1": {'sm': 10.0, 'dram': 20.0, ...} }
            kernel_invocations = {}
            
            clean_csv_lines = []
            header_found = False

            with open(self.ncu_log_path, 'r') as f:
                for line in f:
                    csv_data += line # Store all data for the reward fn
                    
                    # 1. Find the real header row, skipping garbage lines
                    if not header_found and line.strip().startswith('"ID"'):
                        header_found = True
                    
                    # 2. Once header is found, append all subsequent lines
                    if header_found:
                        clean_csv_lines.append(line)
                
            if not header_found:
                raise Exception("NCU ran but did not produce a valid CSV header (no 'ID' row found).")

            # 3. Process the *clean* CSV data
            csv_reader = csv.DictReader(clean_csv_lines)
            
            for row in csv_reader:
                try:
                    # Create a unique ID for each kernel invocation
                    invocation_key = f"{row['Kernel Name']}_{row['ID']}"
                    
                    if invocation_key not in kernel_invocations:
                        kernel_invocations[invocation_key] = {}
                    
                    # Store the metric value
                    metric_name = row['Metric Name']
                    # This is where the 'NoneType' error happened.
                    # We add a check for None before calling .replace()
                    metric_value_str = row.get('Metric Value')
                    if metric_value_str is None:
                        continue # Skip this malformed row
                        
                    metric_value = float(metric_value_str.replace('%', '').strip())
                    
                    if metric_name == 'sm__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['sm'] = metric_value
                    elif metric_name == 'dram__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['dram'] = metric_value
                    elif metric_name == 'l1tex__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l1'] = metric_value
                    elif metric_name == 'lts__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l2'] = metric_value
                        
                except (KeyError, ValueError, TypeError) as e:
                    # This warning is normal for garbage lines that slip through
                    # print(f"[BenchmarkWorker] WARNING: Skipping malformed/unexpected NCU row. Error: {e}. Row: {row}")
                    continue
            
            # 4. Now, aggregate the metrics from all valid invocations
            metrics = {'sm': [], 'dram': [], 'l1': [], 'l2': []}
            for invocation in kernel_invocations.values():
                # Only count *complete* data points
                if all(k in invocation for k in ['sm', 'dram', 'l1', 'l2']):
                    metrics['sm'].append(invocation['sm'])
                    metrics['dram'].append(invocation['dram'])
                    metrics['l1'].append(invocation['l1'])
                    metrics['l2'].append(invocation['l2'])
            
            if not metrics['sm']: # Check if we found any valid, complete invocations
                raise Exception(f"NCU ran but found 0 *complete* kernel metric sets matching '{static_args['kernel_name']}'.")
            
            # --- END FIX ---

            state = np.array([
                np.mean(metrics['sm']),
                np.mean(metrics['dram']),
                np.mean(metrics['l1']),
                np.mean(metrics['l2'])
            ], dtype=np.float32)
            
            reward = self._calculate_reward(state, reward_weights)
            return state, reward, csv_data

        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: NCU CSV parsing failed on GPU {self.gpu_id}. {e}")
            return None, -100.0, None

    def _calculate_reward(self, state, reward_weights):
        sm, dram, l1, l2 = state
        reward = (
            sm * reward_weights.get('R_sm_throughput', 0.0) +
            dram * reward_weights.get('R_dram_throughput', 0.0) +
            l1 * reward_weights.get('R_l1_hit_rate', 0.0) +
            l2 * reward_weights.get('R_l2_hit_rate', 0.0)
        )
        return float(reward)

    def run_slow_gym_validation(self, params_dict, model_name, user_goal):
        """
        Runs the 'Slow Gym' (vllm bench) on this worker's GPU (GPU 1).
        """
        
        try:
            import vllm
        except ImportError:
            print("[BenchmarkWorker] ERROR: vLLM not found in worker process.")
            return 0.0
        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: vLLM import failed in worker. {e}")
            return 0.0

        # Qwen 30B MoE configuration for vLLM config filename
        # E = number of experts (128 for Qwen 30B)
        # N = intermediate size for MoE FFN (inter_size value from model config)
        static_args = {
            "num_experts": 128,  # Qwen 30B has 128 experts
            "inter_size": 1536,  # Intermediate size from model config
        }
        E = static_args['num_experts']    
        N = static_args['inter_size']  # Use full inter_size, not halved
        
        config_filename = f"E={E},N={N},device_name=NVIDIA_H100_80GB_HBM3.json" 
        config_path = os.path.join(self.vllm_config_dir, config_filename)
        
        vllm_config_data = { 
            "16088": params_dict, 
            "default": params_dict  
        }
        
        try:
            with open(config_path, "w") as f:
                json.dump(vllm_config_data, f, indent=2)
            print(f"[BenchmarkWorker] Wrote config to vLLM default path: {config_path}")
        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: Failed to write vLLM config file. {e}")
            return 0.0
            
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        command = [
            "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", model_name,
            "--dataset-path", "ShareGPT_Vicuna_unfiltered.json", 
            "--num-prompts", "200", 
            "--trust-remote-code",
            "--enforce-eager", 
            "--tensor-parallel-size", "1"
        ]
        
        try:
            print(f"[BenchmarkWorker] Running Slow Gym: {' '.join(command)}")
            output = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True, timeout=300
            ).stdout
            
            metric = self._parse_vllm_bench_output(output, user_goal)
            print(f"[BenchmarkWorker] Slow Gym Result: {metric} tokens/sec")
            
            os.remove(config_path)
            return metric

        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: vllm bench failed on GPU {self.gpu_id}. {e}")
            if os.path.exists(config_path):
                os.remove(config_path)
            return 0.0

    def _parse_vllm_bench_output(self, output, goal):
        for line in output.strip().split('\n'):
            if goal == 'throughput' and line.startswith("Throughput:"):
                try:
                    parts = line.split(',')
                    output_tok_s = parts[2].strip().split(' ')[0]
                    return float(output_tok_s)
                except Exception:
                    continue
            
        print(f"[BenchmarkWorker] ERROR: Could not parse vllM bench output.")
        return 0.0
