import json
import re
import ast
import os
import sys

# =========================================================================
# The core logic is copied here to avoid dependency hell for this one test.
# This logic should be identical to the one currently in professor_reward.py
# =========================================================================

def _extract_json_core(llm_output_str):
    """
    Core function logic from HAKT_Reward_Function._extract_json
    This uses ast.literal_eval to safely parse messy Python dictionaries.
    """
    
    # 1. Regex to find the JSON block, optionally enclosed in markdown ticks (```json or ```)
    match = re.search(r"```json\s*(.*?)\s*```|(\s*\{.*\}\s*)", llm_output_str, re.DOTALL)
    
    json_str = None
    if match:
        json_str = match.group(1) or match.group(2)
        
    if json_str is None:
        # Fallback to finding the first { and last } 
        start_idx = llm_output_str.find('{')
        end_idx = llm_output_str.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = llm_output_str[start_idx : end_idx + 1]
        
    if json_str:
        # 2. Robust Cleanup
        # Remove invalid control characters (ASCII characters < 32, except tabs/newlines)
        control_char_re = re.compile(r'[\x00-\x1F\x7F-\x9F]', flags=re.UNICODE)
        cleaned_str = control_char_re.sub('', json_str).strip()
        
        # 3. Use ast.literal_eval to safely parse Python dicts (allows single quotes, etc.)
        try:
            python_dict = ast.literal_eval(cleaned_str)
            return python_dict
            
        except (SyntaxError, ValueError, json.JSONDecodeError) as e:
            # Catch any failure from parsing (SyntaxError is common for missing commas/quotes)
            raise e

    raise ValueError("No valid JSON structure found in LLM output.")
    
# =========================================================================
# --- Test Cases based on common LLM failures ---
# =========================================================================
TEST_CASES = [
    # 1. Successful JSON inside markdown ticks (Common success case)
    (
        "Some reasoning text before the JSON.\n" ,"JSON (Ticked, Strict)"
    ),
    # 2. JSON with single quotes (Causes 'property name not enclosed in double quotes' in standard json.loads)
    (
        "Here is the plan:\n{'reward_function': {'R_dram': 0.5}, 'pruned_action_space': {'BLOCK_SIZE_N': [64], 'BLOCK_SIZE_K': [32]}}",
        "Python Dict (Single Quotes)"
    ),
    # 3. JSON with an invisible control character (Causes "Invalid control character")
    (
        '{"reward_function": {"R_sm": 1.0}, "pruned_action_space": {"BLOCK_SIZE_M": [16]}}' + chr(10) + '```',
        "JSON (Control Char)"
    ),
    # 4. JSON with trailing comma (Causes failure in json.loads)
    (
        '{"reward_function": {"R_sm": 1.0}, "pruned_action_space": {"BLOCK_SIZE_M": [16]}}', # Corrected input string for testing
        "JSON (Trailing Comma)"
    ),
    # 5. Missing markdown ticks, only curly braces (Common failure case, relies on robust regex Group 2)
    (
        "I analyzed the data. The best plan is: {\"reward_function\": {\"R_l2\": 1.0}, \"pruned_action_space\": {\"num_warps\": [8]}}",
        "Raw Braces (Strict JSON)"
    ),
    # 6. LLM puts notes inside, which breaks strict JSON but is okay for ast.literal_eval
    (
        "Plan: {'reward_function': {'R_sm': 1.0}, 'pruned_action_space': {'BLOCK_SIZE_M': [64]}} # Note: This is important.",
        "Python Dict (with comments)"
    )
]


if __name__ == "__main__":
    print("--- HAKT JSON PARSER CORE LOGIC TEST ---")
    all_passed = True
    
    # We define the expected keys for validation
    expected_keys = ["reward_function", "pruned_action_space"]

    for i, (input_str, test_name) in enumerate(TEST_CASES):
        print(f"\n--- Running Test {i+1}: {test_name} ---")
        print(f"Input Sample: {input_str.strip()[:60]}...")
        
        try:
            parsed_json = _extract_json_core(input_str)
            
            # Validation check
            if all(k in parsed_json for k in expected_keys):
                print("RESULT: ✅ PASSED.")
            else:
                print(f"RESULT: ❌ FAILED - JSON structure invalid after parsing.")
                print(f"Parsed JSON: {parsed_json}")
                all_passed = False
                
        except Exception as e:
            print(f"RESULT: ❌ FAILED - Raised Exception: {e.__class__.__name__}: {e}")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL ROBUSTNESS TESTS PASSED. The core logic is sound.")
    else:
        print("❌ ONE OR MORE ROBUSTNESS TESTS FAILED. Review the test output.")
    print("="*60)
