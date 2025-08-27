"""
Integration tests for the main experimental pipeline
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.experiments.main_experiment_pipeline import ExperimentRunner
from experimental_config import ExperimentType, ExperimentParams, EXPERIMENT_CONFIGS

class TestExperimentRunner:
    """Test the main ExperimentRunner class"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    def test_experiment_runner_initialization(self, temp_output_dir):
        """Test ExperimentRunner initialization"""
        runner = ExperimentRunner(
            output_dir=str(temp_output_dir),
            log_level="DEBUG"
        )
        
        assert runner.output_dir == temp_output_dir
        assert runner.logger is not None
        assert isinstance(runner.results, dict)
        
        # Check output directory created
        assert temp_output_dir.exists()
        
        # Check log file created
        log_files = list(temp_output_dir.glob("*.log"))
        assert len(log_files) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_single_experiment_accuracy_fairness(self, temp_output_dir, mock_experiment_config):
        """Test running accuracy-fairness trade-off experiment"""
        runner = ExperimentRunner(
            output_dir=str(temp_output_dir),
            log_level="INFO"
        )
        
        # Modify config for faster testing
        test_config = mock_experiment_config.copy()
        test_config['lambda_values'] = [0.0, 0.5, 1.0]  # Fewer lambda values
        
        # Mock the dataset loading to use smaller synthetic data
        with patch('scripts.experiments.main_experiment_pipeline.load_datasets') as mock_load:
            mock_load.return_value = {
                'test_dataset': {
                    'train': pd.DataFrame({
                        'feature_1': np.random.randn(200),
                        'feature_2': np.random.randn(200),
                        'target': np.random.choice([0, 1], 200)
                    }),
                    'test': pd.DataFrame({
                        'feature_1': np.random.randn(50),
                        'feature_2': np.random.randn(50),
                        'target': np.random.choice([0, 1], 50)
                    }),
                    'sensitive_attrs': ['attr1', 'attr2'],
                    'feature_names': ['feature_1', 'feature_2']
                }
            }
            
            # Mock EXPERIMENT_CONFIGS to use our test config
            with patch.dict('scripts.experiments.main_experiment_pipeline.EXPERIMENT_CONFIGS', 
                          {ExperimentType.ACCURACY_FAIRNESS_TRADEOFF: test_config}):
                
                # Create simplified experiment parameters
                params = ExperimentParams()
                params.random_seeds = [42, 123]  # Fewer seeds
                
                # Run experiment
                results = runner.run_experiment(
                    ExperimentType.ACCURACY_FAIRNESS_TRADEOFF, 
                    params
                )
        
        # Check results structure
        assert 'experiment_type' in results
        assert 'config' in results
        assert 'results' in results
        assert 'metadata' in results
        
        # Check metadata
        metadata = results['metadata']
        assert 'start_time' in metadata
        assert 'end_time' in metadata
        assert 'duration' in metadata
        assert metadata['duration'] > 0
        
        # Check experiment directory created
        exp_dir = temp_output_dir / ExperimentType.ACCURACY_FAIRNESS_TRADEOFF.value
        assert exp_dir.exists()
        
        # Check results file saved
        results_file = exp_dir / "results.json"
        assert results_file.exists()
    
    @pytest.mark.integration
    def test_run_experiment_with_invalid_type(self, temp_output_dir):
        """Test running experiment with invalid experiment type"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        
        # This should raise an error or handle gracefully
        with pytest.raises((KeyError, ValueError, AttributeError)):
            # Try to access non-existent experiment type
            fake_experiment_type = "fake_experiment"
            # This will fail when trying to access EXPERIMENT_CONFIGS
            runner.run_experiment(fake_experiment_type)
    
    @pytest.mark.integration
    def test_load_experiment_datasets(self, temp_output_dir):
        """Test dataset loading functionality"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        
        # Mock dataset configs
        dataset_configs = [
            {
                'name': 'synthetic',
                'sample_size': 100,
                'n_sensitive_attrs': 2
            }
        ]
        
        with patch('scripts.experiments.main_experiment_pipeline.load_datasets') as mock_load:
            mock_load.return_value = {
                'train': pd.DataFrame({'feature_1': [1, 2, 3], 'target': [0, 1, 0]}),
                'test': pd.DataFrame({'feature_1': [4, 5], 'target': [1, 0]}),
                'sensitive_attrs': ['attr1', 'attr2'],
                'feature_names': ['feature_1']
            }
            
            datasets = runner._load_experiment_datasets(dataset_configs)
            
            assert len(datasets) == 1
            assert 'synthetic' in datasets
            assert 'train' in datasets['synthetic']
            assert 'test' in datasets['synthetic']
    
    @pytest.mark.integration
    def test_save_results_functionality(self, temp_output_dir):
        """Test results saving functionality"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        
        # Create test results with various data types
        test_results = {
            'experiment_type': 'test',
            'metrics': {
                'accuracy': 0.85,
                'fairness_violation': 0.12,
                'training_time': 45.5
            },
            'arrays': {
                'predictions': np.array([0, 1, 0, 1]),
                'probabilities': np.array([0.2, 0.8, 0.3, 0.9])
            },
            'metadata': {
                'timestamp': '2024-01-01T00:00:00',
                'parameters': {'lambda': 0.5}
            }
        }
        
        # Save results
        results_file = temp_output_dir / "test_results.json"
        runner._save_results(test_results, results_file)
        
        # Check file was created
        assert results_file.exists()
        
        # Load and verify content
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['experiment_type'] == 'test'
        assert loaded_results['metrics']['accuracy'] == 0.85
        
        # Check numpy arrays were converted to lists
        assert isinstance(loaded_results['arrays']['predictions'], list)
        assert loaded_results['arrays']['predictions'] == [0, 1, 0, 1]
    
    @pytest.mark.integration
    def test_train_and_evaluate_functionality(self, temp_output_dir, sample_binary_data):
        """Test train and evaluate functionality"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        data = sample_binary_data
        
        # Create train/test split
        n_train = 800
        train_data = pd.DataFrame({
            'feature_1': data['X']['feature_1'][:n_train],
            'feature_2': data['X']['feature_2'][:n_train],
            'target': data['y'][:n_train]
        })
        
        test_data = pd.DataFrame({
            'feature_1': data['X']['feature_1'][n_train:],
            'feature_2': data['X']['feature_2'][n_train:],
            'target': data['y'][n_train:]
        })
        
        sensitive_attrs = ['gender', 'race']
        
        # Mock a simple model
        class MockModel:
            def fit(self, X, y, S):
                self.is_trained = True
                return self
                
            def predict(self, X):
                return np.random.choice([0, 1], len(X))
                
            def predict_proba(self, X):
                probs = np.random.uniform(0, 1, len(X))
                return np.column_stack([1-probs, probs])
        
        model = MockModel()
        
        # Test training and evaluation
        metrics = runner._train_and_evaluate(
            model, train_data, test_data, sensitive_attrs
        )
        
        # Check metrics structure
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'training_time' in metrics
        assert metrics['training_time'] > 0
        
        # Check values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['training_time'] < 1000  # Reasonable training time
    
    @pytest.mark.integration
    def test_aggregate_seed_results(self, temp_output_dir):
        """Test seed results aggregation"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        
        # Create mock results from multiple seeds
        seed_results = [
            {'accuracy': 0.85, 'fairness': 0.12, 'training_time': 45.2},
            {'accuracy': 0.87, 'fairness': 0.15, 'training_time': 43.8},
            {'accuracy': 0.83, 'fairness': 0.11, 'training_time': 46.1},
            {'accuracy': 0.86, 'fairness': 0.13, 'training_time': 44.5}
        ]
        
        aggregated = runner._aggregate_seed_results(seed_results)
        
        # Check aggregated structure
        assert 'accuracy_mean' in aggregated
        assert 'accuracy_std' in aggregated
        assert 'accuracy_values' in aggregated
        
        # Check calculations
        expected_acc_mean = np.mean([0.85, 0.87, 0.83, 0.86])
        assert abs(aggregated['accuracy_mean'] - expected_acc_mean) < 1e-10
        
        expected_acc_std = np.std([0.85, 0.87, 0.83, 0.86])
        assert abs(aggregated['accuracy_std'] - expected_acc_std) < 1e-10
        
        assert aggregated['accuracy_values'] == [0.85, 0.87, 0.83, 0.86]
    
    @pytest.mark.integration
    def test_empty_seed_results_aggregation(self, temp_output_dir):
        """Test aggregation with empty seed results"""
        runner = ExperimentRunner(output_dir=str(temp_output_dir))
        
        aggregated = runner._aggregate_seed_results([])
        
        assert aggregated == {}  # Should return empty dict

class TestExperimentWorkflow:
    """Test end-to-end experiment workflows"""
    
    @pytest.fixture
    def minimal_runner(self, temp_dir):
        """Create minimal experiment runner for testing"""
        return ExperimentRunner(output_dir=str(temp_dir), log_level="ERROR")  # Minimal logging
    
    @pytest.mark.integration 
    @pytest.mark.slow
    def test_full_experiment_workflow_synthetic_data(self, minimal_runner):
        """Test complete experiment workflow with synthetic data"""
        
        # Mock all the slow/complex parts for integration test
        with patch('scripts.experiments.main_experiment_pipeline.load_datasets') as mock_load_datasets, \
             patch('scripts.baselines.baseline_factory.create_baseline_method') as mock_create_method:
            
            # Mock dataset loading
            mock_dataset = {
                'synthetic': {
                    'train': pd.DataFrame({
                        'f1': np.random.randn(100),
                        'f2': np.random.randn(100), 
                        'target': np.random.choice([0, 1], 100)
                    }),
                    'test': pd.DataFrame({
                        'f1': np.random.randn(30),
                        'f2': np.random.randn(30),
                        'target': np.random.choice([0, 1], 30)  
                    }),
                    'sensitive_attrs': ['attr1'],
                    'feature_names': ['f1', 'f2']
                }
            }
            mock_load_datasets.return_value = mock_dataset
            
            # Mock model creation
            class FastMockModel:
                def fit(self, X, y, S):
                    return self
                def predict(self, X):
                    return np.random.choice([0, 1], len(X))
                def predict_proba(self, X):
                    p = np.random.uniform(0, 1, len(X))
                    return np.column_stack([1-p, p])
                    
            mock_create_method.return_value = FastMockModel()
            
            # Test with reduced parameters
            params = ExperimentParams()
            params.random_seeds = [42]  # Single seed
            params.lambda_values = [0.0, 1.0]  # Two values only
            
            # Mock config with minimal settings
            test_config = {
                'datasets': [{'name': 'synthetic', 'sample_size': 100}],
                'methods': ['doubly_regressing'],
                'lambda_values': [0.0, 1.0],
                'primary_metrics': ['accuracy'],
                'output_plots': []
            }
            
            with patch.dict('scripts.experiments.main_experiment_pipeline.EXPERIMENT_CONFIGS',
                          {ExperimentType.ACCURACY_FAIRNESS_TRADEOFF: test_config}):
                
                # Run the experiment
                results = minimal_runner.run_experiment(
                    ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
                    params
                )
        
        # Verify results structure
        assert 'experiment_type' in results
        assert 'results' in results
        assert 'metadata' in results
        
        # Check timing information
        assert results['metadata']['duration'] > 0
        assert 'start_time' in results['metadata']
        assert 'end_time' in results['metadata']
    
    @pytest.mark.integration
    def test_experiment_error_handling(self, minimal_runner):
        """Test experiment error handling and recovery"""
        
        # Test with mock that raises an exception
        with patch('scripts.experiments.main_experiment_pipeline.load_datasets') as mock_load:
            mock_load.side_effect = Exception("Dataset loading failed")
            
            with pytest.raises(Exception):
                minimal_runner.run_experiment(ExperimentType.ACCURACY_FAIRNESS_TRADEOFF)
    
    @pytest.mark.integration
    def test_experiment_results_persistence(self, minimal_runner):
        """Test that experiment results are properly saved"""
        
        # Mock successful experiment
        with patch('scripts.experiments.main_experiment_pipeline.load_datasets') as mock_load, \
             patch('scripts.baselines.baseline_factory.create_baseline_method') as mock_create:
            
            # Simple mock data
            mock_load.return_value = {
                'test': {
                    'train': pd.DataFrame({'f1': [1, 2], 'target': [0, 1]}),
                    'test': pd.DataFrame({'f1': [3], 'target': [1]}),
                    'sensitive_attrs': ['attr'],
                    'feature_names': ['f1']
                }
            }
            
            class SimpleMockModel:
                def fit(self, X, y, S): return self
                def predict(self, X): return np.array([0])
                def predict_proba(self, X): return np.array([[0.4, 0.6]])
                    
            mock_create.return_value = SimpleMockModel()
            
            # Run with minimal config
            minimal_config = {
                'datasets': [{'name': 'test'}],
                'methods': ['test_method'],
                'lambda_values': [0.5],
                'primary_metrics': ['accuracy']
            }
            
            with patch.dict('scripts.experiments.main_experiment_pipeline.EXPERIMENT_CONFIGS',
                          {ExperimentType.ACCURACY_FAIRNESS_TRADEOFF: minimal_config}):
                
                params = ExperimentParams()
                params.random_seeds = [42]
                
                results = minimal_runner.run_experiment(
                    ExperimentType.ACCURACY_FAIRNESS_TRADEOFF,
                    params
                )
        
        # Check that results directory exists
        exp_dir = minimal_runner.output_dir / ExperimentType.ACCURACY_FAIRNESS_TRADEOFF.value
        assert exp_dir.exists()
        
        # Check results file exists
        results_file = exp_dir / "results.json"
        assert results_file.exists()
        
        # Verify file content
        with open(results_file, 'r') as f:
            saved_results = json.load(f)
            
        assert saved_results['experiment_type'] == ExperimentType.ACCURACY_FAIRNESS_TRADEOFF.value

class TestExperimentConfiguration:
    """Test experiment configuration handling"""
    
    @pytest.mark.integration
    def test_experiment_params_defaults(self):
        """Test ExperimentParams default values"""
        params = ExperimentParams()
        
        # Check defaults are set
        assert params.lambda_values is not None
        assert params.n_low_values is not None
        assert params.learning_rates is not None
        assert params.random_seeds is not None
        
        # Check reasonable defaults
        assert len(params.lambda_values) > 0
        assert len(params.random_seeds) > 1  # Multiple seeds for robustness
        assert params.max_iterations > 0
        assert params.cv_folds > 0
    
    @pytest.mark.integration
    def test_experiment_config_completeness(self):
        """Test that experiment configurations are complete"""
        
        for exp_type, config in EXPERIMENT_CONFIGS.items():
            assert isinstance(exp_type, ExperimentType)
            assert isinstance(config, dict)
            
            # Check required fields
            required_fields = ['datasets', 'methods', 'primary_metrics']
            for field in required_fields:
                assert field in config, f"Missing {field} in {exp_type.value}"
            
            # Check datasets is a list
            assert isinstance(config['datasets'], list)
            assert len(config['datasets']) > 0
            
            # Check methods is a list  
            assert isinstance(config['methods'], list)
            assert len(config['methods']) > 0
            
            # Check metrics is a list
            assert isinstance(config['primary_metrics'], list)
            assert len(config['primary_metrics']) > 0
    
    @pytest.mark.integration
    def test_all_experiment_types_have_configs(self):
        """Test that all experiment types have configurations"""
        
        for exp_type in ExperimentType:
            assert exp_type in EXPERIMENT_CONFIGS, f"Missing config for {exp_type.value}"
            
            config = EXPERIMENT_CONFIGS[exp_type]
            assert len(config) > 0, f"Empty config for {exp_type.value}"