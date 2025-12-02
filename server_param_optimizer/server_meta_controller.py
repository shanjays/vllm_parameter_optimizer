"""
Server Meta-Controller for LLM-Guided Parameter Optimization

Uses gpt-oss-20b LLM with LoRA adapters to generate server parameter configurations.
Generates --max-num-seqs and --max-num-batched-tokens configurations
based on feedback from previous optimization iterations.

Target: NVIDIA H100 80GB with meta-llama/Llama-3.1-8B-Instruct
"""

import json
import re
import ast
from typing import Dict, List, Optional, Any

# Try to import LLM dependencies (optional for testing)
try:
    import torch
    from unsloth import FastLanguageModel
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    torch = None
    FastLanguageModel = None


# Parameter space for server optimization
PARAM_SPACE = {
    'max_num_seqs': [4, 8, 16, 32, 64, 128, 256],
    'max_num_batched_tokens': [2048, 4096, 8192, 16384, 32768],
}

# LLM configuration
META_CONTROLLER_MODEL = "openai/gpt-oss-20b"
MAX_SEQ_LENGTH = 4096
MAX_COMPLETION_LENGTH = 1024

# H100 80GB thermal configuration
GPU_CONFIG = {
    'name': 'NVIDIA H100 80GB',
    'memory_gb': 80,
    'tdp_watts': 350,
    'max_safe_temp': 85,
    'target_sustained_temp': 75,
}

# Minimum tokens per sequence for constraint validation
# This represents the minimum average token count per sequence
# Constraint: max_num_batched_tokens >= max_num_seqs * MIN_TOKENS_PER_SEQUENCE
MIN_TOKENS_PER_SEQUENCE = 128


class ServerMetaController:
    """LLM-based meta-controller for server parameter optimization.
    
    Uses gpt-oss-20b with LoRA adapters to generate server configurations
    based on optimization feedback from previous iterations.
    
    Example:
        controller = ServerMetaController()
        configs = controller.generate_configs(feedback_collector)
        
        # Each config has:
        # - name: descriptive name
        # - max_num_seqs: maximum concurrent sequences
        # - max_num_batched_tokens: maximum tokens per batch
        # - rationale: explanation for the configuration
    """
    
    def __init__(
        self,
        model_name: str = META_CONTROLLER_MODEL,
        gpu_id: int = 0,
        max_seq_length: int = MAX_SEQ_LENGTH,
        load_in_4bit: bool = True
    ):
        """Initialize the meta-controller.
        
        Args:
            model_name: LLM model to use for config generation
            gpu_id: GPU device index for LLM
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load in 4-bit quantization
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        
        self.llm = None
        self.tokenizer = None
        self._initialized = False
        
        # Set device
        if LLM_AVAILABLE and torch is not None and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cpu"
        
        print(f"[ServerMetaController] Configured with model: {model_name}")
        print(f"[ServerMetaController] Device: {self.device}")
    
    def _load_model(self) -> None:
        """Load the LLM with LoRA adapters on the specified GPU."""
        if self._initialized:
            return
            
        if not LLM_AVAILABLE:
            print("[ServerMetaController] Warning: LLM dependencies not available, using fallback mode")
            self._initialized = True
            return
        
        print(f"[ServerMetaController] Loading LLM '{self.model_name}' on GPU {self.gpu_id}...")
        
        try:
            # Load model with Unsloth optimization
            # Use device_map for explicit GPU targeting without modifying environment variables
            self.llm, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                device_map={"": self.gpu_id},  # Explicit GPU mapping
            )
            
            print("[ServerMetaController] Adding LoRA adapters...")
            
            self.llm = FastLanguageModel.get_peft_model(
                self.llm,
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
            
            print(f"[ServerMetaController] LLM loaded successfully on GPU {self.gpu_id}")
            self._initialized = True
            
        except Exception as e:
            print(f"[ServerMetaController] Error loading LLM: {e}")
            print("[ServerMetaController] Falling back to default configs")
            self._initialized = True
    
    def generate_configs(
        self,
        feedback_collector: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate server configurations using the LLM.
        
        Args:
            feedback_collector: ServerFeedbackCollector with previous results
            
        Returns:
            List of configuration dictionaries, each containing:
            - name: descriptive name
            - max_num_seqs: maximum concurrent sequences
            - max_num_batched_tokens: maximum tokens per batch
            - rationale: explanation for the configuration
        """
        self._load_model()
        
        # Get feedback string
        feedback_str = ""
        if feedback_collector:
            feedback_str = feedback_collector.get_feedback_for_prompt()
        
        if not feedback_str:
            feedback_str = "No previous results. This is the first iteration."
        
        # Build the prompt
        prompt = self._build_prompt(feedback_str)
        
        # Generate configs using LLM or fallback
        if self.llm is not None and self.tokenizer is not None:
            configs = self._generate_with_llm(prompt)
        else:
            configs = self._generate_default_configs()
        
        # Validate and return configs
        validated_configs = []
        for config in configs:
            if self._validate_config(config):
                validated_configs.append(config)
            else:
                print(f"[ServerMetaController] Invalid config skipped: {config}")
        
        if not validated_configs:
            print("[ServerMetaController] No valid configs generated, using defaults")
            validated_configs = self._generate_default_configs()
        
        print(f"[ServerMetaController] Generated {len(validated_configs)} valid configurations")
        return validated_configs
    
    def _build_prompt(self, feedback_str: str) -> str:
        """Build the optimization prompt for the LLM.
        
        Args:
            feedback_str: Formatted feedback from previous iterations
            
        Returns:
            Complete prompt string for the LLM
        """
        prompt = f'''You are an expert in optimizing vLLM server parameters for maximum throughput.

═══════════════════════════════════════════════════════════════════════════════
                              SERVER CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

MODEL: meta-llama/Llama-3.1-8B-Instruct
GPU: NVIDIA H100 80GB
  - Memory: 80 GB HBM3
  - TDP: 350W
  - Max Safe Temp: 85°C (throttling)
  - Target Sustained Temp: 75°C

═══════════════════════════════════════════════════════════════════════════════
                           PARAMETERS TO OPTIMIZE
═══════════════════════════════════════════════════════════════════════════════

--max-num-seqs: Maximum concurrent sequences
  Valid values: 4, 8, 16, 32, 64, 128, 256

--max-num-batched-tokens: Maximum tokens per batch
  Valid values: 2048, 4096, 8192, 16384, 32768

Constraint: max_num_batched_tokens >= max_num_seqs * 128

═══════════════════════════════════════════════════════════════════════════════
                           PREVIOUS RESULTS
═══════════════════════════════════════════════════════════════════════════════

{feedback_str}

═══════════════════════════════════════════════════════════════════════════════
                           YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Generate 2-4 configurations to test. Consider:
1. One aggressive config (maximize throughput)
2. One conservative config (stay below 75°C)
3. Configurations that explore untested regions

Output format:
<param>
{{
  "configs": [
    {{
      "name": "aggressive_high",
      "max_num_seqs": 128,
      "max_num_batched_tokens": 32768,
      "rationale": "Push for maximum throughput"
    }},
    {{
      "name": "balanced",
      "max_num_seqs": 64,
      "max_num_batched_tokens": 16384,
      "rationale": "Balance throughput and thermals"
    }}
  ]
}}
</param>

NOW GENERATE YOUR CONFIGURATIONS:
<param>
'''
        return prompt
    
    def _generate_with_llm(self, prompt: str) -> List[Dict[str, Any]]:
        """Generate configs using the LLM.
        
        Args:
            prompt: The optimization prompt
            
        Returns:
            List of generated configurations
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LENGTH
            )
            
            # Move inputs to the same device as the model
            if self.llm is not None and hasattr(self.llm, 'device'):
                model_device = self.llm.device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            elif torch.cuda.is_available() and self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the part after the prompt
            if prompt in generated_text:
                llm_output = generated_text[len(prompt):]
            else:
                llm_output = generated_text
            
            print(f"[ServerMetaController] LLM output:\n{llm_output[:500]}...")
            
            # Parse configs from output
            configs = self._parse_configs(llm_output)
            return configs
            
        except Exception as e:
            print(f"[ServerMetaController] Error generating with LLM: {e}")
            return self._generate_default_configs()
    
    def _parse_configs(self, llm_output: str) -> List[Dict[str, Any]]:
        """Parse configuration JSON from LLM output.
        
        Args:
            llm_output: Raw LLM output text
            
        Returns:
            List of parsed configurations
        """
        # Try to extract content from <param></param> tags
        param_match = re.search(
            r'<param>\s*(\{[\s\S]*?\})\s*</param>',
            llm_output,
            re.DOTALL | re.IGNORECASE
        )
        
        if param_match:
            json_str = param_match.group(1).strip()
        else:
            # Fallback: Try to find JSON content after <param>
            param_start_match = re.search(
                r'<param>\s*(\{.*)',
                llm_output,
                re.DOTALL | re.IGNORECASE
            )
            if param_start_match:
                json_str = param_start_match.group(1).strip()
            else:
                # Fallback: Try to find any JSON-like content
                match = re.search(r'(\{.*\})', llm_output, re.DOTALL)
                if not match:
                    print("[ServerMetaController] No JSON found in LLM output")
                    return []
                json_str = match.group(0).strip()
        
        # Clean up JSON string
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        
        # Fix unbalanced braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        # Remove trailing commas
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Try to parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_str)
            except Exception:
                print("[ServerMetaController] Failed to parse JSON from LLM output")
                return []
        
        # Extract configs list
        if isinstance(data, dict) and 'configs' in data:
            return data['configs']
        elif isinstance(data, list):
            return data
        else:
            print("[ServerMetaController] Unexpected JSON structure")
            return []
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate a configuration against constraints.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        if not isinstance(config, dict):
            return False
        
        # Check required fields
        max_num_seqs = config.get('max_num_seqs')
        max_num_batched_tokens = config.get('max_num_batched_tokens')
        
        if max_num_seqs is None or max_num_batched_tokens is None:
            return False
        
        # Validate values are in allowed range
        if max_num_seqs not in PARAM_SPACE['max_num_seqs']:
            # Try to find closest valid value
            valid_seqs = PARAM_SPACE['max_num_seqs']
            max_num_seqs = min(valid_seqs, key=lambda x: abs(x - max_num_seqs))
            config['max_num_seqs'] = max_num_seqs
        
        if max_num_batched_tokens not in PARAM_SPACE['max_num_batched_tokens']:
            # Try to find closest valid value
            valid_tokens = PARAM_SPACE['max_num_batched_tokens']
            max_num_batched_tokens = min(valid_tokens, key=lambda x: abs(x - max_num_batched_tokens))
            config['max_num_batched_tokens'] = max_num_batched_tokens
        
        # Check constraint: max_num_batched_tokens >= max_num_seqs * MIN_TOKENS_PER_SEQUENCE
        if max_num_batched_tokens < max_num_seqs * MIN_TOKENS_PER_SEQUENCE:
            print(f"[ServerMetaController] Config violates constraint: "
                  f"tokens={max_num_batched_tokens} < seqs={max_num_seqs} * {MIN_TOKENS_PER_SEQUENCE}")
            return False
        
        return True
    
    def _generate_default_configs(self) -> List[Dict[str, Any]]:
        """Generate default configurations when LLM is unavailable.
        
        Returns:
            List of default configurations to test
        """
        return [
            {
                "name": "aggressive_high",
                "max_num_seqs": 128,
                "max_num_batched_tokens": 32768,
                "rationale": "Push for maximum throughput with large batch"
            },
            {
                "name": "balanced",
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "rationale": "Balance throughput and thermals"
            },
            {
                "name": "conservative",
                "max_num_seqs": 32,
                "max_num_batched_tokens": 8192,
                "rationale": "Conservative config for thermal safety"
            },
            {
                "name": "low_concurrency",
                "max_num_seqs": 16,
                "max_num_batched_tokens": 4096,
                "rationale": "Low concurrency for minimal thermal impact"
            }
        ]
    
    def get_param_space(self) -> Dict[str, List[int]]:
        """Get the parameter space for server optimization.
        
        Returns:
            Dictionary with parameter names and valid values
        """
        return PARAM_SPACE.copy()
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available for config generation.
        
        Returns:
            True if LLM dependencies are installed
        """
        return LLM_AVAILABLE
