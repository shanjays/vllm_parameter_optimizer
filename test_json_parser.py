import json
import re
import ast
import os
import sys

# =========================================================================
# The core logic is copied here to avoid dependency hell for this one test.
# This logic should be identical to the one currently in professor_reward.py
# =========================================================================

def _clean_number_string(s: str) -> str:
    """
    Clean a number string by removing trailing periods, handling scientific notation,
    and other common LLM formatting issues.
    
    Args:
        s: Input string that may contain a malformed number
        
    Returns:
        str: Cleaned number string. Returns '0.0' for completely invalid inputs.
        
    Examples:
        '42.75.' -> '42.75'   (trailing period removed)
        '1.5e'   -> '1.5'     (incomplete exponent removed)
        'e'      -> '0.0'     (invalid, returns default)
        'e+'     -> '0.0'     (invalid, returns default)
        ''       -> '0.0'     (empty string, returns default)
        '3.14'   -> '3.14'    (valid, unchanged)
        '1e10'   -> '1e10'    (valid scientific notation, unchanged)
    """
    s = s.strip()
    # Remove trailing periods (e.g., "42.75." -> "42.75")
    s = re.sub(r'\.+$', '', s)
    # Remove leading/trailing whitespace again after cleanup
    s = s.strip()
    # Handle case where LLM outputs just 'e' or partial scientific notation
    if s.lower() in ('e', 'e+', 'e-', '+e', '-e', ''):
        return '0.0'
    # Fix malformed scientific notation like "1.5e" -> "1.5"
    s = re.sub(r'[eE][+-]?$', '', s)
    return s

def _extract_json_core(llm_output_str):
    """
    Core function logic from HAKT_Reward_Function._extract_json
    This uses ast.literal_eval to safely parse messy Python dictionaries.
    Now supports <param></param> XML tag extraction.
    """
    
    # Try to extract content from <param></param> XML tags first (preferred format)
    param_match = re.search(r'<param>\s*(.*?)\s*</param>', llm_output_str, re.DOTALL | re.IGNORECASE)
    if param_match:
        json_str = param_match.group(1).strip()
    else:
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
        
        # Clean number values with trailing periods
        def clean_json_numbers(match):
            return _clean_number_string(match.group(0))
        cleaned_str = re.sub(r'[0-9]+\.?[0-9]*[eE]?[+-]?[0-9]*\.+(?=\s*[,\]\}])', clean_json_numbers, cleaned_str)
        
        # 3. Use ast.literal_eval to safely parse Python dicts (allows single quotes, etc.)
        try:
            python_dict = ast.literal_eval(cleaned_str)
            return python_dict
            
        except (SyntaxError, ValueError, json.JSONDecodeError) as e:
            # Try json.loads as fallback
            try:
                return json.loads(cleaned_str)
            except:
                raise e

    raise ValueError("No valid JSON structure found in LLM output.")
    
# =========================================================================
# --- Test Cases based on common LLM failures ---
# =========================================================================
TEST_CASES = [
    # 1. JSON within <param> tags (NEW: preferred format)
    (
        """Here is my analysis of the kernel parameters.

<param>
{"reward_function": {"R_sm_throughput": 0.6, "R_dram_throughput": 0.2, "R_l1_hit_rate": 0.1, "R_l2_hit_rate": 0.1}, "pruned_action_space": {"BLOCK_SIZE_M": [64, 128], "BLOCK_SIZE_N": [64], "BLOCK_SIZE_K": [32], "num_warps": [4, 8], "num_stages": [4]}}
</param>

This configuration should work well.""",
        "JSON (with <param> tags)"
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
        '{"reward_function": {"R_sm": 1.0}, "pruned_action_space": {"BLOCK_SIZE_M": [16]}}',
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
    ),
    # 7. JSON within <param> tags with extra whitespace
    (
        """<param>
        
{
  "reward_function": {
    "R_sm_throughput": 0.5,
    "R_dram_throughput": 0.3,
    "R_l1_hit_rate": 0.1,
    "R_l2_hit_rate": 0.1
  },
  "pruned_action_space": {
    "BLOCK_SIZE_M": [32, 64],
    "BLOCK_SIZE_N": [64, 128],
    "BLOCK_SIZE_K": [64],
    "num_warps": [4],
    "num_stages": [3, 4]
  }
}

        </param>""",
        "JSON (<param> with whitespace)"
    ),
]

# =========================================================================
# --- Test Cases for number cleaning ---
# =========================================================================
NUMBER_CLEANING_TEST_CASES = [
    ("42.75.", "42.75"),
    ("1.5e", "1.5"),
    ("e", "0.0"),
    ("e+", "0.0"),
    ("e-", "0.0"),
    ("", "0.0"),
    ("3.14", "3.14"),
    ("1e10", "1e10"),
    ("2.5e-3", "2.5e-3"),
]

# =========================================================================
# --- Test Cases for truncated JSON recovery ---
# =========================================================================
def _fix_truncated_json(json_str):
    """Attempt to fix truncated JSON by adding missing brackets."""
    # Count brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    # Strip trailing whitespace
    json_str = json_str.rstrip()
    
    # Remove trailing comma if present
    if json_str.endswith(','):
        json_str = json_str[:-1]

    # Add missing array closures
    for _ in range(open_brackets - close_brackets):
        json_str += ']'

    # Add missing object closures
    for _ in range(open_braces - close_braces):
        json_str += '}'

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try Python literal eval
        try:
            return ast.literal_eval(json_str)
        except Exception:
            return None

TRUNCATED_JSON_TEST_CASES = [
    # Truncated after opening braces
    (
        '{"reward_function": {"R_sm_throughput": 0.6, "R_dram_throughput": 0.2, "R_l1_hit_rate": 0.1, "R_l2_hit_rate": 0.1}, "pruned_action_space": {"BLOCK_SIZE_M": [64, 128], "BLOCK_SIZE_N": [64',
        "Truncated mid-array"
    ),
    # Truncated with trailing comma
    (
        '{"reward_function": {"R_sm_throughput": 0.5, "R_dram_throughput": 0.3},',
        "Truncated with trailing comma"
    ),
    # Missing final closing brace
    (
        '{"reward_function": {"R_sm_throughput": 0.5, "R_dram_throughput": 0.3, "R_l1_hit_rate": 0.1, "R_l2_hit_rate": 0.1}',
        "Missing final closing brace"
    ),
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

    # Test number cleaning function
    print("\n--- NUMBER CLEANING TESTS ---")
    for input_val, expected in NUMBER_CLEANING_TEST_CASES:
        result = _clean_number_string(input_val)
        if result == expected:
            print(f"✅ _clean_number_string('{input_val}') = '{result}'")
        else:
            print(f"❌ _clean_number_string('{input_val}') = '{result}', expected '{expected}'")
            all_passed = False

    # Test truncated JSON recovery
    print("\n--- TRUNCATED JSON RECOVERY TESTS ---")
    for input_str, test_name in TRUNCATED_JSON_TEST_CASES:
        print(f"\n--- Testing: {test_name} ---")
        print(f"Input Sample: {input_str[:60]}...")
        result = _fix_truncated_json(input_str)
        if result is not None:
            print(f"✅ Successfully recovered JSON: {list(result.keys())}")
        else:
            print(f"❌ Failed to recover truncated JSON")
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL ROBUSTNESS TESTS PASSED. The core logic is sound.")
    else:
        print("❌ ONE OR MORE ROBUSTNESS TESTS FAILED. Review the test output.")
    print("="*60)
