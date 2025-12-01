"""
Tests for FeedbackCollector - Contextual feedback collection for LLM prompts.

These tests verify:
1. Policy result recording and tracking
2. Feedback formatting for prompt injection
3. Successful/failed strategy analysis
4. State persistence across runs
"""

import json
import os
import tempfile
import shutil
from feedback_collector import FeedbackCollector


def test_initial_state():
    """Test that FeedbackCollector initializes with correct default state."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        assert collector.policies_evaluated == 0
        assert collector.best_overall_reward == 0.0
        assert collector.best_configs_by_token == {}
        assert collector.best_policy_weights == {}
        assert collector.successful_strategies == []
        assert collector.failed_strategies == []
        
        print("✅ test_initial_state PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_record_policy_result():
    """Test recording a policy result."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        policy = {
            "objective_weights": {
                "R_sm_throughput": 0.45,
                "R_dram_throughput": 0.35,
                "R_l1_hit_rate": 0.10,
                "R_l2_hit_rate": 0.10
            },
            "search_space": {
                "BLOCK_SIZE_M": [64, 128],
                "BLOCK_SIZE_N": [64, 128],
                "BLOCK_SIZE_K": [32, 64],
                "num_warps": [8, 16],
                "num_stages": [3, 4, 5]
            }
        }
        
        best_configs = {
            1: {"config": {"BLOCK_SIZE_M": 64, "num_warps": 8}, "reward": 45.20},
            64: {"config": {"BLOCK_SIZE_M": 128, "num_warps": 16}, "reward": 51.20}
        }
        
        collector.record_policy_result(policy, reward=51.20, best_configs=best_configs)
        
        assert collector.policies_evaluated == 1
        assert collector.best_overall_reward == 51.20
        assert 1 in collector.best_configs_by_token
        assert 64 in collector.best_configs_by_token
        assert len(collector.policy_history) == 1
        
        print("✅ test_record_policy_result PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_record_multiple_policies():
    """Test recording multiple policy results."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # First policy
        policy1 = {
            "objective_weights": {"R_sm_throughput": 0.4},
            "search_space": {"BLOCK_SIZE_M": [64]}
        }
        collector.record_policy_result(policy1, reward=40.0)
        
        # Second policy (better)
        policy2 = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128]}
        }
        collector.record_policy_result(policy2, reward=50.0)
        
        assert collector.policies_evaluated == 2
        assert collector.best_overall_reward == 50.0
        assert collector.best_policy_weights.get("R_sm_throughput") == 0.5
        
        print("✅ test_record_multiple_policies PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_format_feedback_empty():
    """Test that format_feedback_for_prompt returns empty string when no policies evaluated."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        feedback = collector.format_feedback_for_prompt()
        assert feedback == "", "Should return empty string when no policies evaluated"
        
        print("✅ test_format_feedback_empty PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_format_feedback_with_data():
    """Test that format_feedback_for_prompt returns formatted feedback."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        policy = {
            "objective_weights": {
                "R_sm_throughput": 0.45,
                "R_dram_throughput": 0.35,
                "R_l1_hit_rate": 0.10,
                "R_l2_hit_rate": 0.10
            },
            "search_space": {
                "BLOCK_SIZE_M": [64, 128],
                "num_stages": [4, 5]
            }
        }
        
        best_configs = {
            1: {"config": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 8, "num_stages": 3}, "reward": 45.20},
            64: {"config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_warps": 16, "num_stages": 4}, "reward": 51.20}
        }
        
        collector.record_policy_result(policy, reward=51.20, best_configs=best_configs)
        
        feedback = collector.format_feedback_for_prompt()
        
        # Check that key sections are present (structured format)
        assert "FEEDBACK FROM PREVIOUS ITERATIONS" in feedback
        assert "Policies Evaluated: 1" in feedback
        assert "Best Reward Achieved: 51.20" in feedback
        assert "BEST CONFIGURATIONS FOUND:" in feedback
        assert "Token 1:" in feedback
        assert "Token 64:" in feedback
        
        print("✅ test_format_feedback_with_data PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_successful_strategy_analysis():
    """Test that successful strategies are analyzed and recorded."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # Policy with high SM throughput weight and BLOCK_SIZE_M=128
        policy = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128], "num_stages": [5]}
        }
        
        collector.record_policy_result(policy, reward=55.0)
        
        # Check that strategies were recorded
        assert len(collector.successful_strategies) > 0
        
        # Check for specific strategies
        strategies_text = " ".join(collector.successful_strategies)
        assert "SM throughput" in strategies_text or "BLOCK_SIZE_M=128" in strategies_text or "num_stages=5" in strategies_text
        
        print("✅ test_successful_strategy_analysis PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_failed_strategy_analysis():
    """Test that failed strategies are analyzed and recorded."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # First, record a good policy to set the baseline
        good_policy = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128]}
        }
        collector.record_policy_result(good_policy, reward=100.0)
        
        # Then record a bad policy (significantly worse)
        bad_policy = {
            "objective_weights": {"R_sm_throughput": 0.1},
            "search_space": {"BLOCK_SIZE_M": [32], "num_stages": [2]}
        }
        collector.record_policy_result(bad_policy, reward=50.0)  # Less than 80% of best
        
        # Check that failed strategies were recorded
        assert len(collector.failed_strategies) > 0
        
        print("✅ test_failed_strategy_analysis PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_state_persistence():
    """Test that state is persisted and loaded correctly."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        
        # Create collector and record data
        collector1 = FeedbackCollector(state_file=state_file)
        policy = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128]}
        }
        best_configs = {
            64: {"config": {"BLOCK_SIZE_M": 128}, "reward": 55.0}
        }
        collector1.record_policy_result(policy, reward=55.0, best_configs=best_configs)
        
        # Verify state file was created
        assert os.path.exists(state_file), "State file should be created"
        
        # Create new collector and verify state was loaded
        collector2 = FeedbackCollector(state_file=state_file)
        
        assert collector2.policies_evaluated == 1
        assert collector2.best_overall_reward == 55.0
        assert 64 in collector2.best_configs_by_token
        
        print("✅ test_state_persistence PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_best_configs_update():
    """Test that best configs are updated correctly when better reward found."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # First policy with lower reward
        policy1 = {"objective_weights": {"R_sm_throughput": 0.4}, "search_space": {}}
        best_configs1 = {64: {"config": {"BLOCK_SIZE_M": 64}, "reward": 40.0}}
        collector.record_policy_result(policy1, reward=40.0, best_configs=best_configs1)
        
        # Second policy with better reward for same token count
        policy2 = {"objective_weights": {"R_sm_throughput": 0.5}, "search_space": {}}
        best_configs2 = {64: {"config": {"BLOCK_SIZE_M": 128}, "reward": 55.0}}
        collector.record_policy_result(policy2, reward=55.0, best_configs=best_configs2)
        
        # Verify the better config was kept
        assert collector.best_configs_by_token[64]["reward"] == 55.0
        assert collector.best_configs_by_token[64]["config"]["BLOCK_SIZE_M"] == 128
        
        print("✅ test_best_configs_update PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_get_summary():
    """Test that get_summary returns correct summary."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        policy = {
            "objective_weights": {"R_sm_throughput": 0.5},
            "search_space": {"BLOCK_SIZE_M": [128]}
        }
        best_configs = {
            1: {"config": {}, "reward": 45.0},
            64: {"config": {}, "reward": 55.0}
        }
        collector.record_policy_result(policy, reward=55.0, best_configs=best_configs)
        
        summary = collector.get_summary()
        
        assert summary["policies_evaluated"] == 1
        assert summary["best_overall_reward"] == 55.0
        assert summary["best_configs_count"] == 2
        assert summary["history_length"] == 1
        
        print("✅ test_get_summary PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_reset():
    """Test that reset clears all data."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # Record some data
        policy = {"objective_weights": {"R_sm_throughput": 0.5}, "search_space": {}}
        collector.record_policy_result(policy, reward=55.0)
        
        # Reset
        collector.reset()
        
        assert collector.policies_evaluated == 0
        assert collector.best_overall_reward == 0.0
        assert collector.best_configs_by_token == {}
        assert not os.path.exists(state_file)
        
        print("✅ test_reset PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_feedback_includes_policy_weights():
    """Test that formatted feedback includes what worked strategies and objective weights."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        policy = {
            "objective_weights": {
                "R_sm_throughput": 0.45,
                "R_dram_throughput": 0.35,
                "R_l1_hit_rate": 0.10,
                "R_l2_hit_rate": 0.10
            },
            "search_space": {"BLOCK_SIZE_M": [128]}
        }
        collector.record_policy_result(policy, reward=50.0)
        
        feedback = collector.format_feedback_for_prompt()
        
        # Check that feedback contains structured format and objective weights
        assert "FEEDBACK FROM PREVIOUS ITERATIONS" in feedback
        assert "Policies Evaluated: 1" in feedback
        assert "Best Reward Achieved: 50.00" in feedback
        assert "BEST OBJECTIVE WEIGHTS:" in feedback
        # Since we have successful strategies, we should see "WHAT WORKED"
        if collector.successful_strategies:
            assert "WHAT WORKED:" in feedback
        
        print("✅ test_feedback_includes_policy_weights PASSED")
    finally:
        shutil.rmtree(temp_dir)


def test_strategy_limit():
    """Test that successful and failed strategies are limited."""
    temp_dir = tempfile.mkdtemp()
    try:
        state_file = os.path.join(temp_dir, "feedback_state.json")
        collector = FeedbackCollector(state_file=state_file)
        
        # Record many policies to potentially generate many strategies
        for i in range(20):
            policy = {
                "objective_weights": {"R_sm_throughput": 0.5 + i * 0.01},
                "search_space": {"BLOCK_SIZE_M": [128], "num_stages": [5], "num_warps": [16]}
            }
            collector.record_policy_result(policy, reward=50.0 + i)
        
        # Verify strategy lists are bounded
        assert len(collector.successful_strategies) <= 6
        assert len(collector.failed_strategies) <= 4
        
        print("✅ test_strategy_limit PASSED")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("--- FEEDBACK COLLECTOR TESTS ---\n")
    
    print("=== Initialization Tests ===")
    test_initial_state()
    
    print("\n=== Recording Tests ===")
    test_record_policy_result()
    test_record_multiple_policies()
    test_best_configs_update()
    
    print("\n=== Feedback Formatting Tests ===")
    test_format_feedback_empty()
    test_format_feedback_with_data()
    test_feedback_includes_policy_weights()
    
    print("\n=== Strategy Analysis Tests ===")
    test_successful_strategy_analysis()
    test_failed_strategy_analysis()
    test_strategy_limit()
    
    print("\n=== Persistence Tests ===")
    test_state_persistence()
    
    print("\n=== Utility Tests ===")
    test_get_summary()
    test_reset()
    
    print("\n" + "="*60)
    print("✅ ALL FEEDBACK COLLECTOR TESTS PASSED")
    print("="*60)
