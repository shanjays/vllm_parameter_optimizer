import ray
import subprocess
import json
import csv
import os
import numpy as np
from io import StringIO
import time
import vllm # Import vllm to find its path

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
        
        # --- REVISION ---
        # Find the vLLM default config path dynamically
        try:
            vllm_lib_path = os.path.dirname(vllm.__file__)
            self.vllm_config_dir = os.path.join(
                vllm_lib_path, "model_executor/layers/fused_moe/configs/"
            )
            os.makedirs(self.vllm_config_dir, exist_ok=True)
            print(f"[BenchmarkWorker] vLLM config path set to: {self.vllm_config_dir}")
        except Exception as e:
            print(f"[BenchmarkWorker] WARNING: Could not find vLLM path. {e}")
            # Fallback to user's hard-coded path from log
            self.vllm_config_dir = "~/vllm/vllm/model_executor/layers/fused_moe/configs/"
            
        print(f"[BenchmarkWorker] Initialized on GPU {self.gpu_id} (PID: {pid})")

    def get_gpu_id(self):
        return self.gpu_id

    def run_fast_gym_benchmark(self, params_dict, static_args, reward_weights):
        """
        Runs the 'Fast Gym' (ncu) on this worker's GPU (GPU 1).
        Returns (state, reward, csv_data)
        """
        
        # 1. Set the kernel config
        config_to_use = params_dict if params_dict else DEFAULT_CONFIG
        with open(self.temp_config_path, "w") as f:
            json.dump(config_to_use, f)

        # 2. Build NCU Command
        ncu_command = [
            "ncu", "--csv",
            "--kernel-name", static_args['kernel_name'],
            "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", self.ncu_log_path,
        ]

        # 3. Build Python Command
        python_command = [
            "python", static_args['run_script_path'],
            "--config-path", self.temp_config_path, # Tell script where to find config
            "--num-experts", str(static_args['num_experts']),
            "--top-k", str(static_args['top_k']),
            "--hidden-size", str(static_args['hidden_size']),
            "--inter-size", str(static_args['inter_size']),
            "--num-tokens", str(static_args['num_tokens']),
            "--dtype", static_args['dtype'],
            "--num-iters", str(static_args['num_iters'])
        ]
        
        full_command = ncu_command + python_command
        
        # 4. Set CUDA_VISIBLE_DEVICES for this subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        try:
            subprocess.run(full_command, check=True, capture_output=True, text=True, timeout=30, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[BenchmarkWorker] ERROR: NCU run failed on GPU {self.gpu_id}. {e.stderr}")
            return None, -100.0, None # Return failure
        except subprocess.TimeoutExpired:
            print(f"[BenchmarkWorker] ERROR: NCU run timed out on GPU {self.gpu_id}.")
            return None, -100.0, None # Return failure

        # 5. Parse the ncu_log.csv
        try:
            with open(self.ncu_log_path, 'r') as f:
                csv_data = f.read()
                csv_reader = csv.DictReader(StringIO(csv_data))
                
                metrics = {'sm': [], 'dram': [], 'l1': [], 'l2': []}
                row_count = 0
                for row in csv_reader:
                    if row['sm__throughput.avg.pct_of_peak_sustained_elapsed'] == 'nan':
                        continue
                    metrics['sm'].append(float(row['sm__throughput.avg.pct_of_peak_sustained_elapsed']))
                    metrics['dram'].append(float(row['dram__throughput.avg.pct_of_peak_sustained_elapsed']))
                    metrics['l1'].append(float(row['l1tex__t_sector_hit_rate.pct']))
                    metrics['l2'].append(float(row['lts__t_sector_hit_rate.pct']))
                    row_count += 1
                
                if row_count == 0:
                    raise Exception("NCU output was empty or contained only 'nan' rows.")

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
            return None, -100.0, None # Return failure

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
        This is now revised based on your logs.
        """
        
        # --- REVISION ---
        # 1. Create the *exact* filename vLLM is looking for.
        #    We get E and N from the *correct* static args.
        E = self.static_args['num_experts']    # 128
        N = self.static_args['inter_size'] // 2 # 1536 // 2 = 768
        
        # We assume H100 based on your logs.
        config_filename = f"E={E},N={N},device_name=NVIDIA_H100_80GB_HBM3.json" 
        config_path = os.path.join(self.vllm_config_dir, config_filename)
        
        # 2. Write the config file to the *default* vLLM path.
        #    vLLM expects { "batch_size_str": { "config": ... } }
        vllm_config_data = { 
            "16088": params_dict, # Use our fast-gym batch size as the key
            "default": params_dict  # Add a default
        }
        
        try:
            with open(config_path, "w") as f:
                json.dump(vllm_config_data, f, indent=2)
            print(f"[BenchmarkWorker] Wrote config to vLLM default path: {config_path}")
        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: Failed to write vLLM config file. {e}")
            return 0.0 # Return failure
            
        # 3. Set the correct GPU for the subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # We NO LONGER need VLLM_TUNED_CONFIG_FOLDER

        # 4. Run the "Slow Gym" (vllm bench)
        #    This command is based on *your* successful log.
        command = [
            "python", "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", model_name,
            "--dataset-path", "ShareGPT_Vicuna_unfiltered.json", # From your log
            "--num-prompts", "200", # Shorter than your test, but still valid
            "--trust-remote-code",
            "--enforce-eager", # Crucial: ensures our kernel is used
            "--tensor-parallel-size", "1"
        ]
        
        try:
            print(f"[BenchmarkWorker] Running Slow Gym: {' '.join(command)}")
            output = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True, timeout=300
            ).stdout
            
            # 5. Parse the system-level metric
            metric = self._parse_vllm_bench_output(output, user_goal)
            print(f"[BenchmarkWorker] Slow Gym Result: {metric} tokens/sec")
            
            # 6. Clean up the config file
            os.remove(config_path)
            
            return metric

        except Exception as e:
            print(f"[BenchmarkWorker] ERROR: vllm bench failed on GPU {self.gpu_id}. {e}")
            # Clean up the config file even on failure
            if os.path.exists(config_path):
                os.remove(config_path)
            return 0.0 # Return failure

    def _parse_vllm_bench_output(self, output, goal):
        # --- REVISION ---
        # This parses the "throughput" benchmark output, not "serve"
        for line in output.strip().split('\n'):
            if goal == 'throughput' and line.startswith("Throughput:"):
                # Line is "Throughput: 25.78 requests/s, 11076.96 total tokens/s, 5319.96 output tokens/s"
                try:
                    parts = line.split(',')
                    output_tok_s = parts[2].strip().split(' ')[0]
                    return float(output_tok_s)
                except Exception:
                    continue
            # Add a latency parser if needed
            
        print(f"[BenchmarkWorker] ERROR: Could not parse vllm bench output.")
        return 0.0