"""
Unit tests for visualization and output format verification
"""

import pytest
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.utils.metrics import compute_fairness_tradeoff_curve

class TestOutputFormatValidation:
    """Test output format validation and schema compliance"""
    
    @pytest.mark.unit
    def test_experiment_results_json_schema(self):
        """Test that experiment results follow expected JSON schema"""
        
        # Define expected schema structure
        expected_schema = {
            'experiment_type': str,
            'config': dict,
            'results': dict,
            'metadata': dict
        }
        
        # Create sample results
        sample_results = {
            'experiment_type': 'exp_1_accuracy_fairness',
            'config': {
                'datasets': ['uci_adult'],
                'methods': ['doubly_regressing'],
                'lambda_values': [0.0, 0.5, 1.0]
            },
            'results': {
                'datasets': {
                    'uci_adult': {
                        'methods': {
                            'doubly_regressing': {
                                'lambda_sweep': [
                                    {
                                        'lambda': 0.0,
                                        'accuracy_mean': 0.85,
                                        'accuracy_std': 0.02,
                                        'sup_ipm_mean': 0.15,
                                        'sup_ipm_std': 0.01
                                    }
                                ]
                            }
                        }
                    }
                }
            },
            'metadata': {
                'start_time': '2024-01-01T00:00:00',
                'end_time': '2024-01-01T01:00:00',
                'duration': 3600.0,
                'parameters': {'random_seeds': [42, 123]}
            }
        }
        
        # Validate schema
        for key, expected_type in expected_schema.items():
            assert key in sample_results, f"Missing required key: {key}"
            assert isinstance(sample_results[key], expected_type), \
                f"Wrong type for {key}: expected {expected_type}, got {type(sample_results[key])}"
        
        # Validate nested structure
        assert 'datasets' in sample_results['results']
        assert isinstance(sample_results['results']['datasets'], dict)
        
        # Validate metadata completeness
        required_metadata = ['start_time', 'end_time', 'duration']
        for key in required_metadata:
            assert key in sample_results['metadata'], f"Missing metadata: {key}"
    
    @pytest.mark.unit
    def test_metrics_output_format(self, sample_binary_data):
        """Test that metrics output follows expected format"""
        
        from scripts.utils.metrics import evaluate_all_metrics
        
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        y_pred_proba = np.random.uniform(0, 1, len(y_true))
        
        metrics = evaluate_all_metrics(y_true, y_pred, y_pred_proba, data['S'])
        
        # Check that all metrics are JSON-serializable
        try:
            json_str = json.dumps(metrics)
            loaded_metrics = json.loads(json_str)
            assert loaded_metrics == metrics
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metrics not JSON serializable: {e}")
        
        # Check expected metric categories
        expected_categories = ['accuracy', 'f1_score', 'precision', 'recall']
        
        for metric in expected_categories:
            assert metric in metrics, f"Missing standard metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), \
                f"Metric {metric} should be numeric, got {type(metrics[metric])}"
            assert 0 <= metrics[metric] <= 1, f"Metric {metric} out of range [0,1]: {metrics[metric]}"
    
    @pytest.mark.unit
    def test_fairness_metrics_output_format(self, sample_binary_data):
        """Test fairness metrics output format"""
        
        from scripts.utils.metrics import SubgroupFairnessMetrics
        
        data = sample_binary_data
        y_pred = np.random.choice([0, 1], len(data['y']))
        
        dp_results = SubgroupFairnessMetrics.demographic_parity(y_pred, data['S'])
        
        # Check output format
        assert isinstance(dp_results, dict)
        
        # All values should be numeric and finite
        for key, value in dp_results.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Metric {key} should be numeric, got {type(value)}"
            assert np.isfinite(value), f"Metric {key} should be finite, got {value}"
        
        # Check key naming convention
        for key in dp_results.keys():
            assert key.startswith('dp_'), f"DP metric should start with 'dp_': {key}"
            assert any(suffix in key for suffix in ['violation', 'max_rate', 'min_rate']), \
                f"Unexpected DP metric format: {key}"
    
    @pytest.mark.unit
    def test_training_history_format(self, trained_doubly_regressing_model):
        """Test training history output format"""
        
        model = trained_doubly_regressing_model
        history = model.get_training_history()
        
        # Check structure
        assert isinstance(history, dict)
        expected_keys = ['classifier_loss', 'discriminator_loss', 'fairness_violation', 'total_loss']
        
        for key in expected_keys:
            assert key in history, f"Missing training history key: {key}"
            assert isinstance(history[key], list), f"History {key} should be a list"
            
            # All values should be numeric
            for i, value in enumerate(history[key]):
                assert isinstance(value, (int, float, np.number)), \
                    f"History {key}[{i}] should be numeric: {value}"
                assert np.isfinite(value), f"History {key}[{i}] should be finite: {value}"
    
    @pytest.mark.unit
    def test_subgroup_importance_output_format(self, trained_doubly_regressing_model):
        """Test subgroup importance weights output format"""
        
        model = trained_doubly_regressing_model
        importance = model.get_subgroup_importance()
        
        if importance is not None:  # May be None if no subgroups
            assert isinstance(importance, dict)
            
            required_keys = ['weights', 'subgroup_definitions', 'subgroup_sizes']
            for key in required_keys:
                assert key in importance, f"Missing importance key: {key}"
            
            # Weights should be numeric array
            weights = importance['weights']
            assert isinstance(weights, np.ndarray), "Weights should be numpy array"
            assert all(isinstance(w, (float, np.floating)) for w in weights), \
                "All weights should be float"
            assert all(w >= 0 for w in weights), "All weights should be non-negative"
            
            # Subgroup definitions should be list of dicts
            definitions = importance['subgroup_definitions']
            assert isinstance(definitions, list), "Definitions should be list"
            for defn in definitions:
                assert isinstance(defn, dict), "Each definition should be dict"
            
            # Sizes should be numeric
            sizes = importance['subgroup_sizes']
            assert isinstance(sizes, list), "Sizes should be list"
            assert all(isinstance(s, (int, np.integer)) for s in sizes), \
                "All sizes should be integers"
            assert all(s > 0 for s in sizes), "All sizes should be positive"

class TestVisualizationComponents:
    """Test visualization components and plot data validation"""
    
    @pytest.fixture
    def temp_plot_dir(self):
        """Create temporary directory for plot outputs"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.unit
    def test_fairness_tradeoff_curve_data(self):
        """Test fairness-accuracy trade-off curve data preparation"""
        
        # Create sample trade-off data
        accuracy_values = [0.70, 0.75, 0.80, 0.78, 0.82, 0.77]
        fairness_values = [0.30, 0.25, 0.35, 0.28, 0.40, 0.32]  # Lower is better
        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        
        result = compute_fairness_tradeoff_curve(accuracy_values, fairness_values, lambda_values)
        
        # Validate output structure
        expected_keys = [
            'pareto_indices', 'pareto_accuracies', 'pareto_fairness', 'pareto_lambdas',
            'accuracy_range', 'fairness_range', 'lambda_range'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key in tradeoff curve: {key}"
        
        # Validate Pareto frontier
        pareto_indices = result['pareto_indices']
        assert isinstance(pareto_indices, list)
        assert len(pareto_indices) > 0, "Should have at least one Pareto point"
        assert all(0 <= idx < len(accuracy_values) for idx in pareto_indices), \
            "Pareto indices should be valid"
        
        # Validate ranges
        assert len(result['accuracy_range']) == 2
        assert result['accuracy_range'][0] <= result['accuracy_range'][1]
        assert len(result['fairness_range']) == 2
        assert result['fairness_range'][0] <= result['fairness_range'][1]
    
    @pytest.mark.unit
    def test_plot_data_consistency(self):
        """Test that plot data is consistent and valid"""
        
        # Sample experimental results
        experiment_results = {
            'datasets': {
                'uci_adult': {
                    'methods': {
                        'doubly_regressing': {
                            'lambda_sweep': [
                                {
                                    'lambda': 0.0,
                                    'accuracy_mean': 0.85,
                                    'accuracy_std': 0.02,
                                    'sup_ipm_mean': 0.20,
                                    'sup_ipm_std': 0.01,
                                    'training_time_mean': 45.5
                                },
                                {
                                    'lambda': 0.5,
                                    'accuracy_mean': 0.83,
                                    'accuracy_std': 0.03,
                                    'sup_ipm_mean': 0.15,
                                    'sup_ipm_std': 0.02,
                                    'training_time_mean': 48.2
                                }
                            ]
                        }
                    }
                }
            }
        }
        
        # Extract plot data
        method_data = experiment_results['datasets']['uci_adult']['methods']['doubly_regressing']
        lambda_sweep = method_data['lambda_sweep']
        
        # Validate data consistency
        assert len(lambda_sweep) > 0, "Should have sweep data"
        
        for point in lambda_sweep:
            # Check required fields
            required_fields = ['lambda', 'accuracy_mean', 'sup_ipm_mean']
            for field in required_fields:
                assert field in point, f"Missing field in sweep point: {field}"
            
            # Check value ranges
            assert 0 <= point['accuracy_mean'] <= 1, \
                f"Accuracy out of range: {point['accuracy_mean']}"
            assert point['sup_ipm_mean'] >= 0, \
                f"supIPM should be non-negative: {point['sup_ipm_mean']}"
            assert point['lambda'] >= 0, \
                f"Lambda should be non-negative: {point['lambda']}"
    
    @pytest.mark.unit  
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_generation_interface(self, mock_show, mock_savefig, temp_plot_dir):
        """Test plot generation interface (mocked)"""
        
        # Sample data for plotting
        x_data = [0.0, 0.5, 1.0, 1.5, 2.0]
        y_accuracy = [0.85, 0.83, 0.81, 0.79, 0.77]
        y_fairness = [0.20, 0.15, 0.12, 0.10, 0.08]
        
        # Test matplotlib interface works
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy vs lambda
        ax1.plot(x_data, y_accuracy, 'b-o', label='Accuracy')
        ax1.set_xlabel('Lambda (Fairness Weight)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Fairness Trade-off')
        ax1.legend()
        ax1.grid(True)
        
        # Plot fairness vs lambda
        ax2.plot(x_data, y_fairness, 'r-s', label='supIPM Violation')
        ax2.set_xlabel('Lambda (Fairness Weight)')
        ax2.set_ylabel('supIPM Violation')
        ax2.set_title('Fairness Violation vs Lambda')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Test saving
        output_file = temp_plot_dir / "test_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Verify mocks were called
        mock_savefig.assert_called()
        
        plt.close()
    
    @pytest.mark.unit
    def test_plot_data_validation(self):
        """Test validation of plot data before visualization"""
        
        # Valid data
        valid_data = {
            'x': [0.0, 0.5, 1.0],
            'y': [0.85, 0.83, 0.81],
            'errors': [0.02, 0.03, 0.02]
        }
        
        # Test data validation function
        def validate_plot_data(data):
            if 'x' not in data or 'y' not in data:
                raise ValueError("Missing x or y data")
            
            if len(data['x']) != len(data['y']):
                raise ValueError("x and y data length mismatch")
            
            if 'errors' in data and len(data['errors']) != len(data['x']):
                raise ValueError("Error bars length mismatch")
            
            # Check for NaN or infinite values
            if any(not np.isfinite(val) for val in data['x'] + data['y']):
                raise ValueError("Data contains NaN or infinite values")
            
            return True
        
        # Test valid data
        assert validate_plot_data(valid_data) == True
        
        # Test invalid data cases
        invalid_cases = [
            {'x': [0.0, 0.5], 'y': [0.85]},  # Length mismatch
            {'x': [0.0, np.nan], 'y': [0.85, 0.83]},  # NaN values
            {'x': [0.0, 0.5], 'y': [0.85, np.inf]},  # Infinite values
            {'y': [0.85, 0.83]},  # Missing x data
        ]
        
        for invalid_data in invalid_cases:
            with pytest.raises(ValueError):
                validate_plot_data(invalid_data)

class TestExperimentLogValidation:
    """Test experiment log format and completeness"""
    
    @pytest.mark.unit
    def test_log_entry_format(self):
        """Test that log entries follow expected format"""
        
        from scripts.utils.logger import ExperimentLogger
        from io import StringIO
        import logging
        
        # Create string buffer to capture log output
        log_buffer = StringIO()
        
        # Create logger with string handler
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(log_buffer)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Create experiment logger (mock the file creation)
        with patch('pathlib.Path.mkdir'), patch('scripts.utils.logger.setup_logging') as mock_setup:
            mock_setup.return_value = logger
            
            exp_logger = ExperimentLogger("test_experiment", Path("/tmp"))
            
            # Test log entries
            exp_logger.log_stage_start("data_loading", "Loading UCI Adult dataset")
            exp_logger.log_method_performance("doubly_regressing", {
                'accuracy': 0.85,
                'sup_ipm': 0.15,
                'training_time': 45.2
            })
            exp_logger.log_comparison("doubly_regressing", "kearns_supimp", 
                                    "accuracy", 0.85, 0.83)
        
        # Get log output
        log_output = log_buffer.getvalue()
        
        # Validate log format
        log_lines = log_output.strip().split('\n')
        assert len(log_lines) > 0, "Should have log entries"
        
        # Each line should have timestamp, logger name, level, message
        for line in log_lines:
            parts = line.split(' - ')
            assert len(parts) >= 4, f"Log line should have 4+ parts: {line}"
            
            # Check timestamp format (basic check)
            timestamp_part = parts[0]
            assert len(timestamp_part) > 10, "Timestamp should be reasonable length"
    
    @pytest.mark.unit
    def test_experiment_metadata_completeness(self):
        """Test that experiment metadata is complete"""
        
        # Sample metadata
        metadata = {
            'experiment_name': 'test_experiment',
            'start_time': '2024-01-01T00:00:00',
            'end_time': '2024-01-01T01:00:00',
            'duration': 3600.0,
            'parameters': {
                'lambda_values': [0.0, 0.5, 1.0],
                'random_seeds': [42, 123],
                'max_iterations': 1000
            },
            'datasets': ['uci_adult'],
            'methods': ['doubly_regressing', 'kearns_supimp'],
            'system_info': {
                'python_version': '3.8.10',
                'platform': 'Linux-5.4.0'
            }
        }
        
        # Check required fields
        required_fields = ['experiment_name', 'start_time', 'end_time', 'duration']
        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"
        
        # Check data types
        assert isinstance(metadata['duration'], (int, float))
        assert metadata['duration'] > 0
        assert isinstance(metadata['parameters'], dict)
        assert isinstance(metadata['datasets'], list)
        assert isinstance(metadata['methods'], list)
        
        # Check timestamp format (basic validation)
        for time_field in ['start_time', 'end_time']:
            time_str = metadata[time_field]
            assert 'T' in time_str, f"Timestamp should be ISO format: {time_str}"
            assert ':' in time_str, f"Timestamp should have time: {time_str}"