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

# Error penalty constants for different failure types
PENALTY_SHARED_MEMORY = -50.0
PENALTY_REGISTER_OVERFLOW = -75.0
PENALTY_TIMEOUT = -75.0
PENALTY_PARSE_ERROR = -25.0
PENALTY_TOTAL_FAILURE = -100.0

@ray.remote(num_gpus=1) # This actor requires 1 GPU from Ray's pool
class BenchmarkWorker:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        pid = os.getpid()
        self.ncu_log_path = f"/tmp/hakt_ncu_log_gpu{self.gpu_id}_{pid}.csv"
        self.temp_config_path = f"/tmp/hakt_temp_config_gpu{self.gpu_id}_{pid}.json"
        self.failed_configs = []  # Track failed configs for analysis
        self.max_retries = 2  # Number of retries before giving up
        
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

    def _reduce_config(self, config):
        """
        Try to reduce config to fit within hardware limits.
        Returns reduced config or None if can't reduce further.
        """
        reduced = config.copy()
        
        # Try reducing in order of impact
        reductions = [
            ('num_stages', lambda x: max(2, x - 1)),
            ('BLOCK_SIZE_M', lambda x: max(16, x // 2)),
            ('BLOCK_SIZE_K', lambda x: max(32, x // 2)),
            ('BLOCK_SIZE_N', lambda x: max(32, x // 2)),
        ]
        
        for key, reduce_fn in reductions:
            if key in reduced and reduced[key] > reduce_fn(reduced[key]):
                old_val = reduced[key]
                reduced[key] = reduce_fn(reduced[key])
                print(f"[BenchmarkWorker] Reduced {key}: {old_val} -> {reduced[key]}")
                
                if self._validate_triton_config(reduced):
                    return reduced
        
        return None  # Can't reduce further
    
    def _categorize_error(self, stderr):
        """Categorize error type from stderr output."""
        if stderr is None:
            return "unknown", PENALTY_TOTAL_FAILURE
        
        stderr_lower = stderr.lower()
        
        if "out of resource: shared memory" in stderr_lower:
            return "shared_memory", PENALTY_SHARED_MEMORY
        elif "insufficient registers" in stderr_lower:
            return "register_overflow", PENALTY_REGISTER_OVERFLOW
        elif "timeout" in stderr_lower:
            return "timeout", PENALTY_TIMEOUT
        elif "cuda out of memory" in stderr_lower or "oom" in stderr_lower:
            return "cuda_oom", PENALTY_TOTAL_FAILURE
        else:
            return "unknown", PENALTY_TOTAL_FAILURE
    
    def _log_failed_config(self, config, error_type, error_msg):
        """Log failed config for analysis."""
        self.failed_configs.append({
            'config': config.copy(),
            'error_type': error_type,
            'error_msg': str(error_msg)[:200],  # Truncate long messages
            'timestamp': time.time()
        })
        
        # Keep only last 100 failures to avoid memory bloat
        if len(self.failed_configs) > 100:
            self.failed_configs = self.failed_configs[-100:]
    
    def get_failure_stats(self):
        """Return statistics about failed configs."""
        if not self.failed_configs:
            return {"total_failures": 0}
        
        error_counts = {}
        for failure in self.failed_configs:
            error_type = failure['error_type']
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_failures": len(self.failed_configs),
            "by_type": error_counts,
            "recent_failures": self.failed_configs[-5:]
        }

    def run_fast_gym_benchmark(self, params_dict, static_args, reward_weights):
        """
        Runs the 'Fast Gym' (ncu) on this worker's GPU.
        Returns (state, reward, csv_data)
        
        Includes retry logic and automatic config reduction.
        """
        
        config_to_use = DEFAULT_CONFIG.copy()
        if params_dict:
            config_to_use.update(params_dict)
        
        # Validate num_warps is a power of 2
        num_warps = config_to_use.get('num_warps', 4)
        if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            print(f"[BenchmarkWorker] WARNING: Invalid num_warps={num_warps}, using 4")
            config_to_use['num_warps'] = 4
        
        # Pre-flight validation with automatic reduction
        if not self._validate_triton_config(config_to_use):
            print(f"[BenchmarkWorker] Config exceeds limits, attempting reduction...")
            reduced_config = self._reduce_config(config_to_use)
            
            if reduced_config is None:
                print(f"[BenchmarkWorker] Cannot reduce config further, returning penalty")
                self._log_failed_config(config_to_use, "shared_memory", "Config exceeds limits and cannot be reduced")
                return None, PENALTY_SHARED_MEMORY, None
            
            config_to_use = reduced_config
            print(f"[BenchmarkWorker] Using reduced config: {config_to_use}")
        
        # Retry loop
        last_error = None
        last_stderr = None
        
        for attempt in range(self.max_retries):
            if attempt > 0:
                print(f"[BenchmarkWorker] Retry attempt {attempt + 1}/{self.max_retries}")
                time.sleep(1)  # Brief pause between retries
            
            with open(self.temp_config_path, "w") as f:
                json.dump(config_to_use, f)

            ncu_command = [
                "ncu", "--csv",
                "--kernel-name", static_args['kernel_name'], 
                "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
                "--target-processes", "all",
                "--force-overwrite",
                "--log-file", self.ncu_log_path,
            ]

            python_command = [
                "python", static_args['run_script_path'],
                "--config-path", self.temp_config_path,
                "--num-experts", str(static_args['num_experts']),
                "--top-k", str(static_args['top_k']),
                "--hidden-size", str(static_args['hidden_size']),
                "--inter-size", str(static_args['inter_size']),
                "--num-tokens", str(static_args['num_tokens']),
                "--dtype", static_args['dtype'],
                "--num-iters", str(static_args['num_iters']),
                "--num-warmup-iters", "1", 
            ]
            
            full_command = ncu_command + python_command
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id) 

            try:
                subprocess.run(
                    full_command, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=60,  # Increased timeout
                    env=env
                )
                # Success! Break out of retry loop
                break
                
            except subprocess.CalledProcessError as e:
                last_error = e
                last_stderr = e.stderr
                error_type, penalty = self._categorize_error(e.stderr)
                print(f"[BenchmarkWorker] NCU failed (attempt {attempt + 1}): {error_type}")
                
                # Don't retry for config-related errors
                if error_type in ["shared_memory", "register_overflow"]:
                    self._log_failed_config(config_to_use, error_type, e.stderr)
                    return None, penalty, None
                    
            except subprocess.TimeoutExpired as e:
                last_error = e
                last_stderr = "Timeout expired"
                print(f"[BenchmarkWorker] NCU timed out (attempt {attempt + 1})")
                
        else:
            # All retries exhausted
            error_type, penalty = self._categorize_error(last_stderr)
            print(f"[BenchmarkWorker] All retries exhausted. Error type: {error_type}")
            self._log_failed_config(config_to_use, error_type, str(last_error))
            return None, penalty, None

        # Parse NCU output
        try:
            csv_data = ""
            kernel_invocations = {}
            clean_csv_lines = []
            header_found = False

            with open(self.ncu_log_path, 'r') as f:
                for line in f:
                    csv_data += line
                    if not header_found and line.strip().startswith('"ID"'):
                        header_found = True
                    if header_found:
                        clean_csv_lines.append(line)
                
            if not header_found:
                raise Exception("NCU ran but did not produce a valid CSV header")

            csv_reader = csv.DictReader(clean_csv_lines)
            
            for row in csv_reader:
                try:
                    invocation_key = f"{row['Kernel Name']}_{row['ID']}"
                    if invocation_key not in kernel_invocations:
                        kernel_invocations[invocation_key] = {}
                    
                    metric_name = row['Metric Name']
                    metric_value_str = row.get('Metric Value')
                    if metric_value_str is None:
                        continue
                        
                    metric_value = float(metric_value_str.replace('%', '').strip())
                    
                    if metric_name == 'sm__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['sm'] = metric_value
                    elif metric_name == 'dram__throughput.avg.pct_of_peak_sustained_elapsed':
                        kernel_invocations[invocation_key]['dram'] = metric_value
                    elif metric_name == 'l1tex__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l1'] = metric_value
                    elif metric_name == 'lts__t_sector_hit_rate.pct':
                        kernel_invocations[invocation_key]['l2'] = metric_value
                        
                except (KeyError, ValueError, TypeError):
                    continue
            
            metrics = {'sm': [], 'dram': [], 'l1': [], 'l2': []}
            for invocation in kernel_invocations.values():
                if all(k in invocation for k in ['sm', 'dram', 'l1', 'l2']):
                    metrics['sm'].append(invocation['sm'])
                    metrics['dram'].append(invocation['dram'])
                    metrics['l1'].append(invocation['l1'])
                    metrics['l2'].append(invocation['l2'])
            
            if not metrics['sm']:
                raise Exception(f"NCU found 0 complete kernel metric sets")

            state = np.array([
                np.mean(metrics['sm']),
                np.mean(metrics['dram']),
                np.mean(metrics['l1']),
                np.mean(metrics['l2'])
            ], dtype=np.float32)
            
            reward = self._calculate_reward(state, reward_weights)
            return state, reward, csv_data

        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: NCU CSV parsing failed. {e}")
            self._log_failed_config(config_to_use, "parse_error", str(e))
            return None, PENALTY_PARSE_ERROR, None

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
