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
            # vllm 0.10.2 has a different path
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

    def run_fast_gym_benchmark(self, params_dict, static_args, reward_weights):
        """
        Runs the 'Fast Gym' (ncu) on this worker's GPU (GPU 1).
        Returns (state, reward, csv_data)
        """
        
        config_to_use = params_dict if params_dict else DEFAULT_CONFIG
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
            "--num-warmup-iters", "1", # This argument was missing
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

        # --- THIS IS THE FIX: Robust CSV Parsing ---
        try:
            csv_data = ""
            metrics = {'sm': [], 'dram': [], 'l1': [], 'l2': []}
            row_count = 0

            with open(self.ncu_log_path, 'r') as f:
                # 1. Find the real header row, skipping garbage lines
                header_line = None
                for line in f:
                    csv_data += line # Store all data for the reward fn
                    # A real ncu header will start with "ID" or "Kernel Name"
                    if line.strip().startswith('"ID"') or line.strip().startswith('"Kernel Name"'):
                        header_line = line
                        break
                
                # If we never found a header, ncu failed
                if header_line is None:
                    raise Exception("NCU ran but did not produce a valid CSV header. Check ptrace_scope or permissions.")

                # 2. We found the header. Now, process the *rest* of the file.
                # We combine the header_line we found with the rest of the lines
                csv_reader = csv.DictReader([header_line] + f.readlines())
                
                for row in csv_reader:
                    try:
                        metrics['sm'].append(float(row['sm__throughput.avg.pct_of_peak_sustained_elapsed']))
                        metrics['dram'].append(float(row['dram__throughput.avg.pct_of_peak_sustained_elapsed']))
                        metrics['l1'].append(float(row['l1tex__t_sector_hit_rate.pct']))
                        metrics['l2'].append(float(row['lts__t_sector_hit_rate.pct']))
                        row_count += 1
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"[BenchmarkWorker] WARNING: Skipping malformed/unexpected NCU row. Error: {e}. Row: {row}")
                        continue
            
            # This check is still valid
            if row_count == 0:
                raise Exception(f"NCU ran but found 0 valid kernels matching '{static_args['kernel_name']}'.")
            
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

        # This is a bit of a hack. We should pass static_args to the worker.
        # For now, we'll hardcode the values we know.
        static_args = {
            "num_experts": 128,
            "inter_size": 1536,
        }
        E = static_args['num_experts']    
        N = static_args['inter_size'] // 2 
        
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
