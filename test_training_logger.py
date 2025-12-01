"""
Tests for TrainingLogger and visualization functionality.

These tests verify:
1. TrainingLogger correctly logs per-step data
2. TrainingLogger exports to CSV and JSON formats
3. TrainingLogger can be loaded from saved files
4. TrainingLogger summary statistics are correct
5. VLLMConfigExporter.export_visualization_data works correctly
6. Visualization functions generate valid output
"""

import json
import os
import tempfile
import shutil

from training_logger import TrainingLogger
from config_exporter import VLLMConfigExporter


class TestTrainingLogger:
    """Tests for TrainingLogger class."""
    
    def test_log_step_basic(self):
        """Test basic step logging."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        entry = logger.log_step(
            step=1,
            reward=50.0,
            config={'BLOCK_SIZE_M': 64, 'num_warps': 4},
            metrics={'sm_throughput': 45.5},
            token_count=16
        )
        
        assert entry['step'] == 1
        assert entry['reward'] == 50.0
        assert entry['token_count'] == 16
        assert entry['BLOCK_SIZE_M'] == 64
        assert entry['sm_throughput'] == 45.5
        assert len(logger.entries) == 1
        print("✅ test_log_step_basic PASSED")
    
    def test_log_step_auto_increment(self):
        """Test that step auto-increments when not provided."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        entry1 = logger.log_step(reward=10.0)
        entry2 = logger.log_step(reward=20.0)
        entry3 = logger.log_step(reward=30.0)
        
        assert entry1['step'] == 0
        assert entry2['step'] == 1
        assert entry3['step'] == 2
        print("✅ test_log_step_auto_increment PASSED")
    
    def test_cumulative_stats(self):
        """Test cumulative statistics tracking."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        logger.log_step(reward=10.0)
        logger.log_step(reward=20.0)
        logger.log_step(reward=30.0)
        
        assert logger._cumulative_reward == 60.0
        assert logger._best_reward == 30.0
        
        # Check last entry has correct cumulative values
        assert logger.entries[-1]['cumulative_reward'] == 60.0
        assert logger.entries[-1]['best_reward'] == 30.0
        print("✅ test_cumulative_stats PASSED")
    
    def test_save_csv(self):
        """Test saving to CSV file."""
        temp_dir = tempfile.mkdtemp()
        try:
            logger = TrainingLogger(output_dir=temp_dir)
            
            logger.log_step(step=1, reward=50.0, token_count=16)
            logger.log_step(step=2, reward=75.0, token_count=16)
            
            csv_path = logger.save_csv()
            
            assert os.path.exists(csv_path)
            
            # Verify content
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 3  # Header + 2 entries
            assert 'step' in lines[0]
            assert 'reward' in lines[0]
            print("✅ test_save_csv PASSED")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_json(self):
        """Test saving to JSON file."""
        temp_dir = tempfile.mkdtemp()
        try:
            logger = TrainingLogger(output_dir=temp_dir)
            
            logger.log_step(step=1, reward=50.0, token_count=16)
            logger.log_step(step=2, reward=75.0, token_count=16)
            
            json_path = logger.save_json()
            
            assert os.path.exists(json_path)
            
            # Verify content
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert data['total_steps'] == 2
            assert len(data['entries']) == 2
            assert data['cumulative_reward'] == 125.0
            assert data['best_reward'] == 75.0
            print("✅ test_save_json PASSED")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_from_csv(self):
        """Test loading from CSV file."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create and save logger
            logger1 = TrainingLogger(output_dir=temp_dir)
            logger1.log_step(step=1, reward=50.0, token_count=16)
            logger1.log_step(step=2, reward=75.0, token_count=16)
            csv_path = logger1.save_csv()
            
            # Load from CSV
            logger2 = TrainingLogger.load_from_csv(csv_path)
            
            assert len(logger2.entries) == 2
            assert logger2.entries[0]['reward'] == 50.0
            assert logger2.entries[1]['reward'] == 75.0
            print("✅ test_load_from_csv PASSED")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_load_from_json(self):
        """Test loading from JSON file."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create and save logger
            logger1 = TrainingLogger(output_dir=temp_dir)
            logger1.log_step(step=1, reward=50.0, token_count=16)
            logger1.log_step(step=2, reward=75.0, token_count=16)
            json_path = logger1.save_json()
            
            # Load from JSON
            logger2 = TrainingLogger.load_from_json(json_path)
            
            assert len(logger2.entries) == 2
            assert logger2._cumulative_reward == 125.0
            assert logger2._best_reward == 75.0
            print("✅ test_load_from_json PASSED")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_get_summary(self):
        """Test summary statistics."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        logger.log_step(step=1, reward=50.0, token_count=16)
        logger.log_step(step=2, reward=75.0, token_count=16)
        logger.log_step(step=3, reward=60.0, token_count=64)
        
        summary = logger.get_summary()
        
        assert summary['total_steps'] == 3
        assert summary['cumulative_reward'] == 185.0
        assert summary['best_reward'] == 75.0
        assert summary['mean_reward'] == 185.0 / 3
        assert set(summary['token_counts_covered']) == {16, 64}
        print("✅ test_get_summary PASSED")
    
    def test_get_entries_by_token_count(self):
        """Test filtering entries by token count."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        logger.log_step(step=1, reward=50.0, token_count=16)
        logger.log_step(step=2, reward=75.0, token_count=16)
        logger.log_step(step=3, reward=60.0, token_count=64)
        
        entries_16 = logger.get_entries_by_token_count(16)
        entries_64 = logger.get_entries_by_token_count(64)
        
        assert len(entries_16) == 2
        assert len(entries_64) == 1
        print("✅ test_get_entries_by_token_count PASSED")
    
    def test_get_best_config_by_token_count(self):
        """Test getting best config for a token count."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        logger.log_step(step=1, reward=50.0, token_count=16, 
                       config={'BLOCK_SIZE_M': 64})
        logger.log_step(step=2, reward=75.0, token_count=16,
                       config={'BLOCK_SIZE_M': 128})
        logger.log_step(step=3, reward=60.0, token_count=64,
                       config={'BLOCK_SIZE_M': 64})
        
        best_16 = logger.get_best_config_by_token_count(16)
        
        assert best_16 is not None
        assert best_16['reward'] == 75.0
        assert best_16['BLOCK_SIZE_M'] == 128
        print("✅ test_get_best_config_by_token_count PASSED")
    
    def test_export_visualization_data(self):
        """Test exporting data for visualization."""
        logger = TrainingLogger(output_dir=tempfile.mkdtemp())
        
        logger.log_step(step=1, reward=50.0, token_count=16,
                       config={'BLOCK_SIZE_M': 64, 'num_warps': 4},
                       metrics={'sm_throughput': 45.5})
        logger.log_step(step=2, reward=75.0, token_count=16,
                       config={'BLOCK_SIZE_M': 128, 'num_warps': 8},
                       metrics={'sm_throughput': 55.0})
        
        viz_data = logger.export_visualization_data()
        
        assert viz_data['steps'] == [1, 2]
        assert viz_data['rewards'] == [50.0, 75.0]
        assert viz_data['token_counts'] == [16, 16]
        assert len(viz_data['configs']) == 2
        assert viz_data['sm_throughputs'] == [45.5, 55.0]
        print("✅ test_export_visualization_data PASSED")


class TestConfigExporterVisualization:
    """Tests for VLLMConfigExporter visualization export."""
    
    def test_export_visualization_data_basic(self):
        """Test basic visualization data export."""
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
        
        exporter.update_best_config(
            token_count=16,
            config={'BLOCK_SIZE_M': 64},
            reward=50.0,
            metrics={'sm_throughput': 45.5}
        )
        exporter.update_best_config(
            token_count=64,
            config={'BLOCK_SIZE_M': 128},
            reward=70.0,
            metrics={'sm_throughput': 55.0}
        )
        
        viz_data = exporter.export_visualization_data()
        
        assert len(viz_data['steps']) == 2
        assert viz_data['rewards'] == [50.0, 70.0]
        assert viz_data['token_counts'] == [16, 64]
        print("✅ test_export_visualization_data_basic PASSED")
    
    def test_export_visualization_data_cumulative_stats(self):
        """Test cumulative statistics in visualization export."""
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
        
        exporter.update_best_config(16, {'BLOCK_SIZE_M': 64}, 50.0)
        exporter.update_best_config(64, {'BLOCK_SIZE_M': 128}, 70.0)
        exporter.update_best_config(128, {'BLOCK_SIZE_M': 128}, 60.0)
        
        viz_data = exporter.export_visualization_data()
        stats = viz_data['cumulative_stats']
        
        assert stats['total_experiments'] == 3
        assert stats['mean_reward'] == 60.0
        assert stats['best_reward'] == 70.0
        assert stats['min_reward'] == 50.0
        print("✅ test_export_visualization_data_cumulative_stats PASSED")
    
    def test_export_visualization_data_empty(self):
        """Test visualization export with no data."""
        exporter = VLLMConfigExporter(num_experts=128, inter_size=1536)
        
        viz_data = exporter.export_visualization_data()
        
        assert viz_data['steps'] == []
        assert viz_data['rewards'] == []
        assert viz_data['cumulative_stats']['total_experiments'] == 0
        print("✅ test_export_visualization_data_empty PASSED")


def run_training_logger_tests():
    """Run all TrainingLogger tests."""
    tests = TestTrainingLogger()
    tests.test_log_step_basic()
    tests.test_log_step_auto_increment()
    tests.test_cumulative_stats()
    tests.test_save_csv()
    tests.test_save_json()
    tests.test_load_from_csv()
    tests.test_load_from_json()
    tests.test_get_summary()
    tests.test_get_entries_by_token_count()
    tests.test_get_best_config_by_token_count()
    tests.test_export_visualization_data()


def run_config_exporter_viz_tests():
    """Run all ConfigExporter visualization tests."""
    tests = TestConfigExporterVisualization()
    tests.test_export_visualization_data_basic()
    tests.test_export_visualization_data_cumulative_stats()
    tests.test_export_visualization_data_empty()


if __name__ == "__main__":
    print("--- TRAINING LOGGER TESTS ---\n")
    
    print("=== TrainingLogger Tests ===")
    run_training_logger_tests()
    
    print("\n=== ConfigExporter Visualization Tests ===")
    run_config_exporter_viz_tests()
    
    print("\n" + "="*60)
    print("✅ ALL TRAINING LOGGER TESTS PASSED")
    print("="*60)
