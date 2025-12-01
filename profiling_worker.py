import ray
import subprocess
import json
import csv
import os
import numpy as np
from io import StringIO
import time

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4,
}

# Aggressive penalty constants - crashes near limits are informative!
PENALTY_CUDA_OOM = -20.0          # Was -100. OOM tells us we found the limit!
PENALTY_SHARED_MEMORY = -25.0     # Was -50. Shared mem limit is useful info
PENALTY_REGISTER_OVERFLOW = -30.0 # Was -75. Register limit is useful info
PENALTY_TIMEOUT = -50.0           # Was -75. Timeouts waste time
PENALTY_PARSE_ERROR = -25.0
PENALTY_TOTAL_FAILURE = -100.0    # Keep high for unknown failures
PENALTY_TRITON_ERROR = -25.0      # Triton compilation failures

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1
DEFAULT_NCU_TIMEOUT = 60

@ray.remote(num_gpus=1)
class ProfilingWorker:
    """
    Profiling worker for kernel configuration benchmarking.
    
    This worker runs on a dedicated GPU and executes NCU profiling
    to collect performance metrics for different kernel configurations.
    It communicates with the meta-controller via Ray.
    """
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        pid = os.getpid()
        self.ncu_log_path = f"/tmp/profiling_ncu_log_gpu{self.gpu_id}_{pid}.csv"
        self.temp_config_path = f"/tmp/profiling_temp_config_gpu{self.gpu_id}_{pid}.json"
        self.failed_configs = []
        self.max_retries = DEFAULT_MAX_RETRIES
        self.retry_delay = DEFAULT_RETRY_DELAY
        self.ncu_timeout = DEFAULT_NCU_TIMEOUT
        
        try:
            import vllm
            vllm_lib_path = os.path.dirname(vllm.__file__)
            self.vllm_config_dir = os.path.join(
                vllm_lib_path, "model_executor/layers/fused_moe/configs/"
            )
            if not os.path.exists(self.vllm_config_dir):
                 self.vllm_config_dir = "/tmp/vllm_configs/"
                 
            os.makedirs(self.vllm_config_dir, exist_ok=True)
            print(f"[ProfilingWorker] vLLM config path set to: {self.vllm_config_dir}")
            del vllm 
        except Exception as e:
            print(f"[ProfilingWorker] WARNING: Could not find vLLM path. {e}")
            self.vllm_config_dir = "/tmp/vllm_configs/" 
            os.makedirs(self.vllm_config_dir, exist_ok=True)
            
        print(f"[ProfilingWorker] Initialized on PHYSICAL GPU {self.gpu_id} (PID: {pid})")

    def get_gpu_id(self):
        return self.gpu_id

    def _validate_triton_config(self, config):
        M = config.get('BLOCK_SIZE_M', 64)
        N = config.get('BLOCK_SIZE_N', 64)
        K = config.get('BLOCK_SIZE_K', 32)
        stages = config.get('num_stages', 4)
        
        shared_mem = (M * K + K * N) * 2 * stages
        H100_LIMIT = 232448
        if shared_mem > H100_LIMIT:
            print(f"[ProfilingWorker] Config exceeds shared memory: {shared_mem} bytes > {H100_LIMIT} bytes")
            print(f"[ProfilingWorker] Offending config: BLOCK_SIZE_M={M}, BLOCK_SIZE_N={N}, BLOCK_SIZE_K={K}, num_stages={stages}")
            return False
        return True

    def _reduce_config(self, config):
        reduced = config.copy()
        reductions = [
            ('num_stages', lambda x: max(2, x - 1)),
            ('BLOCK_SIZE_M', lambda x: max(16, x // 2)),
            ('BLOCK_SIZE_K', lambda x: max(32, x // 2)),
            ('BLOCK_SIZE_N', lambda x: max(32, x // 2)),
        ]
        for key, reduce_fn in reductions:
            if key in reduced:
                old_val = reduced[key]
                new_val = reduce_fn(reduced[key])
                if new_val < old_val:
                    reduced[key] = new_val
                    print(f"[ProfilingWorker] Reduced {key}: {old_val} -> {new_val}")
                    if self._validate_triton_config(reduced):
                        return reduced
        return None

    def _categorize_error(self, stderr):
        """Categorize error and return appropriate penalty."""
        if stderr is None:
            return "unknown", PENALTY_TOTAL_FAILURE
        
        stderr_lower = stderr.lower() if isinstance(stderr, str) else str(stderr).lower()
        
        # CUDA OOM - we found the memory limit! This is valuable info.
        if "cuda out of memory" in stderr_lower or "out of memory" in stderr_lower or "oom" in stderr_lower:
            print(f"[ProfilingWorker] CUDA OOM detected - boundary found!")
            return "cuda_oom", PENALTY_CUDA_OOM
        
        # Shared memory exceeded
        if "shared memory" in stderr_lower or "smem" in stderr_lower:
            print(f"[ProfilingWorker] Shared memory limit hit - boundary found!")
            return "shared_memory", PENALTY_SHARED_MEMORY
        
        # Register overflow
        if "register" in stderr_lower or "spill" in stderr_lower:
            print(f"[ProfilingWorker] Register overflow - boundary found!")
            return "register_overflow", PENALTY_REGISTER_OVERFLOW
        
        # Timeout
        if "timeout" in stderr_lower:
            return "timeout", PENALTY_TIMEOUT
        
        # Triton compilation errors (often due to aggressive configs)
        if "triton" in stderr_lower and ("error" in stderr_lower or "failed" in stderr_lower):
            print(f"[ProfilingWorker] Triton compilation failed - config too aggressive")
            return "triton_error", PENALTY_TRITON_ERROR
        
        return "unknown", PENALTY_TOTAL_FAILURE

    def _track_boundary_configs(self, config, error_type):
        """Track configs that hit hardware limits for learning."""
        if not hasattr(self, 'boundary_configs'):
            self.boundary_configs = []
        
        self.boundary_configs.append({
            'config': config.copy(),
            'error_type': error_type,
            'timestamp': time.time()
        })
        
        # Keep last 50 boundary configs
        if len(self.boundary_configs) > 50:
            self.boundary_configs = self.boundary_configs[-50:]

    def _log_failed_config(self, config, error_type, error_msg):
        self.failed_configs.append({
            'config': config.copy(),
            'error_type': error_type,
            'error_msg': str(error_msg)[:200],
            'timestamp': time.time()
        })
        if len(self.failed_configs) > 100:
            self.failed_configs = self.failed_configs[-100:]

    def run_kernel_profiling(self, params_dict, static_args, objective_weights, num_tokens=None):
        """
        Execute kernel profiling with the given configuration.
        
        Args:
            params_dict: Kernel configuration parameters
            static_args: Static arguments for the benchmark
            objective_weights: Weights for computing the objective function
            num_tokens: Optional number of tokens for the benchmark
            
        Returns:
            Tuple of (state, reward, csv_data)
        """
        effective_static_args = static_args
        if num_tokens is not None:
            effective_static_args = static_args.copy()
            effective_static_args['num_tokens'] = num_tokens
        
        config_to_use = DEFAULT_CONFIG.copy()
        if params_dict:
            config_to_use.update(params_dict)
        
        num_warps = config_to_use.get('num_warps', 4)
        if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
            print(f"[ProfilingWorker] WARNING: Invalid num_warps={num_warps}, using 4")
            config_to_use['num_warps'] = 4
        
        if not self._validate_triton_config(config_to_use):
            print(f"[ProfilingWorker] Config exceeds limits, attempting reduction...")
            reduced_config = self._reduce_config(config_to_use)
            if reduced_config is None:
                print(f"[ProfilingWorker] Cannot reduce config further, returning penalty")
                self._log_failed_config(config_to_use, "shared_memory", "Config exceeds limits")
                return None, PENALTY_SHARED_MEMORY, None
            config_to_use = reduced_config
            print(f"[ProfilingWorker] Using reduced config: {config_to_use}")
        
        last_error = None
        last_stderr = None
        
        for attempt in range(self.max_retries):
            if attempt > 0:
                print(f"[ProfilingWorker] Retry attempt {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)
            
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
                subprocess.run(
                    full_command, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.ncu_timeout,
                    env=env
                )
                break
                
            except subprocess.CalledProcessError as e:
                last_error = e
                last_stderr = e.stderr
                error_type, penalty = self._categorize_error(e.stderr)
                print(f"[ProfilingWorker] NCU failed (attempt {attempt + 1}): {error_type}")
                if error_type in ["shared_memory", "register_overflow"]:
                    self._log_failed_config(config_to_use, error_type, e.stderr)
                    return None, penalty, None
                    
            except subprocess.TimeoutExpired as e:
                last_error = e
                last_stderr = "Timeout expired"
                print(f"[ProfilingWorker] NCU timed out (attempt {attempt + 1})")
                
        else:
            error_type, penalty = self._categorize_error(last_stderr)
            print(f"[ProfilingWorker] All retries exhausted. Error type: {error_type}")
            self._log_failed_config(config_to_use, error_type, str(last_error))
            return None, penalty, None

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
            
            reward = self._calculate_reward(state, objective_weights)
            return state, reward, csv_data

        except Exception as e:
            print(f"[ProfilingWorker] ERROR: NCU CSV parsing failed. {e}")
            self._log_failed_config(config_to_use, "parse_error", str(e))
            return None, PENALTY_PARSE_ERROR, None

    def _calculate_objective(self, state, objective_weights):
        """Calculate the objective function value from state and weights."""
        sm, dram, l1, l2 = state
        objective = (
            sm * objective_weights.get('R_sm_throughput', 0.0) +
            dram * objective_weights.get('R_dram_throughput', 0.0) +
            l1 * objective_weights.get('R_l1_hit_rate', 0.0) +
            l2 * objective_weights.get('R_l2_hit_rate', 0.0)
        )
        return float(objective)

    def _calculate_reward(self, state, reward_weights):
        """Legacy method for backward compatibility."""
        return self._calculate_objective(state, reward_weights)

    def run_throughput_validation(self, params_dict, model_name, user_goal):
        """
        Run throughput validation using vLLM benchmarking.
        
        This method validates kernel configurations by running actual
        inference throughput tests with vLLM.
        
        Args:
            params_dict: Kernel configuration parameters
            model_name: Name of the model to benchmark
            user_goal: Optimization goal ('throughput' or 'latency')
            
        Returns:
            float: Throughput metric in tokens/sec
        """
        try:
            import vllm
        except ImportError:
            print("[ProfilingWorker] ERROR: vLLM not found in worker process.")
            return 0.0
        except Exception as e:
            print(f"[ProfilingWorker] ERROR: vLLM import failed in worker. {e}")
            return 0.0

        static_args = {
            "num_experts": 128,
            "inter_size": 768,
        }
        E = static_args['num_experts']    
        N = static_args['inter_size']
        
        config_filename = f"E={E},N={N},device_name=NVIDIA_H100_80GB_HBM3.json" 
        config_path = os.path.join(self.vllm_config_dir, config_filename)
        
        vllm_config_data = { 
            "16088": params_dict, 
            "default": params_dict  
        }
        
        try:
            with open(config_path, "w") as f:
                json.dump(vllm_config_data, f, indent=2)
            print(f"[ProfilingWorker] Wrote config to vLLM default path: {config_path}")
        except Exception as e:
            print(f"[ProfilingWorker] ERROR: Failed to write vLLM config file. {e}")
            return 0.0
            
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Clear CUDA cache before running to avoid memory conflicts after NCU profiling
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Aggressive settings for maximum throughput testing
        command = [
            "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", model_name,
            "--dataset-path", "ShareGPT_Vicuna_unfiltered.json", 
            "--num-prompts", "40",              # Reduced from 100 for faster iterations
            "--trust-remote-code",
            "--enforce-eager", 
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.90",  # Aggressive (was 0.85)
        ]
        
        try:
            print(f"[ProfilingWorker] Running throughput validation: {' '.join(command)}")
            result = subprocess.run(
                command, 
                env=env, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=600  # Increased timeout for model loading
            )
            output = result.stdout
            
            metric = self._parse_vllm_bench_output(output, user_goal)
            print(f"[ProfilingWorker] Throughput validation result: {metric} tokens/sec")
            
            # Clean up config file
            if os.path.exists(config_path):
                os.remove(config_path)
            return metric

        except subprocess.CalledProcessError as e:
            print(f"[ProfilingWorker] ERROR: vllm bench failed. Return code: {e.returncode}")
            print(f"[ProfilingWorker] STDERR: {e.stderr[:500] if e.stderr else 'None'}")
            if os.path.exists(config_path):
                os.remove(config_path)
            return 0.0
            
        except subprocess.TimeoutExpired:
            print(f"[ProfilingWorker] ERROR: vllm bench timed out after 600s")
            if os.path.exists(config_path):
                os.remove(config_path)
            return 0.0
            
        except Exception as e:
            print(f"[ProfilingWorker] ERROR: vllm bench failed with exception: {e}")
            if os.path.exists(config_path):
                os.remove(config_path)
            return 0.0

    def run_slow_gym_validation(self, params_dict, model_name, user_goal):
        """Legacy method for backward compatibility."""
        return self.run_throughput_validation(params_dict, model_name, user_goal)

    def _parse_vllm_bench_output(self, output, goal):
        for line in output.strip().split('\n'):
            if goal == 'throughput' and line.startswith("Throughput:"):
                try:
                    parts = line.split(',')
                    output_tok_s = parts[2].strip().split(' ')[0]
                    return float(output_tok_s)
                except Exception:
                    continue
            
        print(f"[ProfilingWorker] ERROR: Could not parse vllM bench output.")
        return 0.0

    def run_fast_gym_benchmark(self, params_dict, static_args, reward_weights, num_tokens=None):
        """Legacy method for backward compatibility."""
        return self.run_kernel_profiling(params_dict, static_args, reward_weights, num_tokens)


# Backward compatibility alias
BenchmarkWorker = ProfilingWorker


