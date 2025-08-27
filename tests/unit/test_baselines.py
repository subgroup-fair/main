"""
Unit tests for baseline fairness methods
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.baselines.doubly_regressing import DoublyRegressingFairness
from scripts.baselines.kearns_subipm import KearnsSubgroupFairness
from scripts.baselines.fams_fairness import FAMSFairness
from scripts.baselines.baseline_factory import (
    create_baseline_method, 
    get_method_info, 
    list_available_methods
)
from tests.conftest import assert_model_trained, assert_valid_probabilities

class TestDoublyRegressingFairness:
    """Test DoublyRegressingFairness implementation"""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test proper initialization of DoublyRegressingFairness"""
        model = DoublyRegressingFairness(
            lambda_fair=0.8,
            n_low=30,
            max_iterations=5000,
            random_state=42
        )
        
        assert model.lambda_fair == 0.8
        assert model.n_low == 30
        assert model.max_iterations == 5000
        assert model.random_state == 42
        assert model.classifier is None  # Not trained yet
        assert model.discriminator is None
        
        # Check training history initialized
        assert 'classifier_loss' in model.training_history
        assert len(model.training_history['classifier_loss']) == 0
    
    @pytest.mark.unit
    def test_fit_basic_functionality(self, sample_binary_data):
        """Test basic fitting functionality"""
        data = sample_binary_data
        
        # Use small iterations for faster testing
        model = DoublyRegressingFairness(
            lambda_fair=0.5,
            n_low=20,
            max_iterations=10,  # Very small for unit test
            random_state=42,
            verbose=False
        )
        
        # Fit model
        model.fit(data['X'], data['y'], data['S'])
        
        # Check model is trained
        assert_model_trained(model)
        assert model.classifier is not None
        assert model.discriminator is not None
        assert model.scaler is not None
        
        # Check training history recorded
        assert len(model.training_history['classifier_loss']) > 0
    
    @pytest.mark.unit
    def test_predict_after_training(self, trained_doubly_regressing_model, sample_binary_data):
        """Test prediction functionality after training"""
        model = trained_doubly_regressing_model
        data = sample_binary_data
        
        # Test predictions
        y_pred = model.predict(data['X'])
        
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(data['X'])
        assert set(np.unique(y_pred)).issubset({0, 1})  # Binary predictions
    
    @pytest.mark.unit
    def test_predict_proba_after_training(self, trained_doubly_regressing_model, sample_binary_data):
        """Test probability prediction functionality"""
        model = trained_doubly_regressing_model
        data = sample_binary_data
        
        # Test probability predictions
        y_pred_proba = model.predict_proba(data['X'])
        
        assert isinstance(y_pred_proba, np.ndarray)
        assert y_pred_proba.shape == (len(data['X']), 2)  # [P(class=0), P(class=1)]
        assert_valid_probabilities(y_pred_proba)
    
    @pytest.mark.unit
    def test_predict_without_training(self, sample_binary_data):
        """Test prediction without training raises appropriate error"""
        model = DoublyRegressingFairness()
        data = sample_binary_data
        
        with pytest.raises((AttributeError, RuntimeError)):
            model.predict(data['X'])
    
    @pytest.mark.unit
    def test_subgroup_enumeration(self, sample_binary_data):
        """Test subgroup enumeration functionality"""
        data = sample_binary_data
        
        model = DoublyRegressingFairness(n_low=10, random_state=42)
        
        # Test subgroup enumeration
        covered_subgroups = model._enumerate_covered_subgroups(data['S'])
        
        assert isinstance(covered_subgroups, list)
        assert len(covered_subgroups) > 0
        
        # Each subgroup should have required fields
        for subgroup in covered_subgroups:
            assert 'definition' in subgroup
            assert 'mask' in subgroup
            assert 'size' in subgroup
            assert subgroup['size'] >= model.n_low
    
    @pytest.mark.unit
    def test_different_lambda_values(self, sample_binary_data):
        """Test model behavior with different lambda values"""
        data = sample_binary_data
        
        lambda_values = [0.0, 0.5, 1.0, 2.0]
        models = {}
        
        for lambda_val in lambda_values:
            model = DoublyRegressingFairness(
                lambda_fair=lambda_val,
                max_iterations=5,  # Very small for testing
                random_state=42,
                verbose=False
            )
            
            model.fit(data['X'], data['y'], data['S'])
            models[lambda_val] = model
        
        # All models should be trained successfully
        for lambda_val, model in models.items():
            assert_model_trained(model)
    
    @pytest.mark.unit
    def test_get_training_history(self, trained_doubly_regressing_model):
        """Test training history retrieval"""
        model = trained_doubly_regressing_model
        
        history = model.get_training_history()
        
        assert isinstance(history, dict)
        expected_keys = ['classifier_loss', 'discriminator_loss', 'fairness_violation', 'total_loss']
        
        for key in expected_keys:
            assert key in history
            assert isinstance(history[key], list)
            assert len(history[key]) > 0  # Should have recorded some iterations
    
    @pytest.mark.unit
    def test_get_subgroup_importance(self, trained_doubly_regressing_model):
        """Test subgroup importance weights retrieval"""
        model = trained_doubly_regressing_model
        
        importance = model.get_subgroup_importance()
        
        if importance is not None:  # May be None if no subgroups found
            assert isinstance(importance, dict)
            assert 'weights' in importance
            assert 'subgroup_definitions' in importance
            assert 'subgroup_sizes' in importance
            
            # Weights should sum to approximately 1 (softmax output)
            weights = importance['weights']
            assert abs(weights.sum() - 1.0) < 0.1
    
    @pytest.mark.unit
    def test_model_with_small_dataset(self):
        """Test model behavior with very small dataset"""
        # Create tiny dataset
        n_samples = 50
        X = pd.DataFrame(np.random.randn(n_samples, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.choice([0, 1], n_samples))
        S = pd.DataFrame({'attr': np.random.choice(['A', 'B'], n_samples)})
        
        model = DoublyRegressingFairness(
            n_low=5,  # Very low threshold
            max_iterations=5,
            random_state=42,
            verbose=False
        )
        
        # Should handle small dataset gracefully
        model.fit(X, y, S)
        y_pred = model.predict(X)
        
        assert len(y_pred) == n_samples

class TestKearnsSubgroupFairness:
    """Test KearnsSubgroupFairness implementation"""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test proper initialization of KearnsSubgroupFairness"""
        model = KearnsSubgroupFairness(
            gamma=0.05,
            fairness_penalty=2.0,
            max_iterations=500,
            random_state=42
        )
        
        assert model.gamma == 0.05
        assert model.fairness_penalty == 2.0
        assert model.max_iterations == 500
        assert model.random_state == 42
        assert model.base_classifier is None  # Not trained yet
    
    @pytest.mark.unit
    def test_fit_basic_functionality(self, sample_binary_data):
        """Test basic fitting functionality"""
        data = sample_binary_data
        
        model = KearnsSubgroupFairness(
            gamma=0.1,
            max_iterations=10,  # Small for testing
            random_state=42
        )
        
        # Fit model
        model.fit(data['X'], data['y'], data['S'])
        
        # Check model is trained
        assert model.base_classifier is not None
        assert hasattr(model, 'subgroups')
        assert len(model.subgroups) > 0
        assert hasattr(model, 'lagrange_multipliers')
    
    @pytest.mark.unit
    def test_predict_after_training(self, trained_kearns_model, sample_binary_data):
        """Test prediction functionality after training"""
        model = trained_kearns_model
        data = sample_binary_data
        
        # Test predictions
        y_pred = model.predict(data['X'])
        
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(data['X'])
        assert set(np.unique(y_pred)).issubset({0, 1})  # Binary predictions
    
    @pytest.mark.unit
    def test_predict_proba_after_training(self, trained_kearns_model, sample_binary_data):
        """Test probability prediction functionality"""
        model = trained_kearns_model
        data = sample_binary_data
        
        # Test probability predictions
        y_pred_proba = model.predict_proba(data['X'])
        
        assert isinstance(y_pred_proba, np.ndarray)
        assert y_pred_proba.shape == (len(data['X']), 2)  # [P(class=0), P(class=1)]
        assert_valid_probabilities(y_pred_proba)
    
    @pytest.mark.unit
    def test_subgroup_enumeration(self, sample_binary_data):
        """Test subgroup enumeration in Kearns method"""
        data = sample_binary_data
        
        model = KearnsSubgroupFairness(max_subgroup_size=2, random_state=42)
        subgroups = model._enumerate_subgroups(data['S'], max_combinations=20)
        
        assert isinstance(subgroups, list)
        assert len(subgroups) > 0
        
        # Each subgroup should have required structure
        for subgroup in subgroups:
            assert 'definition' in subgroup
            assert 'mask' in subgroup
            assert 'size' in subgroup
            assert subgroup['size'] >= 10  # Minimum size threshold
    
    @pytest.mark.unit
    def test_get_subgroup_violations(self, trained_kearns_model, sample_binary_data):
        """Test subgroup violation reporting"""
        model = trained_kearns_model
        data = sample_binary_data
        
        violations = model.get_subgroup_violations(data['X'], data['y'], data['S'])
        
        assert isinstance(violations, dict)
        
        # Each violation entry should have required fields
        for key, violation_info in violations.items():
            assert 'definition' in violation_info
            assert 'size' in violation_info
            assert 'rate' in violation_info
            assert 'violation' in violation_info
            assert 'multiplier' in violation_info
            
            # Values should be valid
            assert violation_info['size'] > 0
            assert 0 <= violation_info['rate'] <= 1
            assert violation_info['violation'] >= 0
            assert violation_info['multiplier'] >= 0
    
    @pytest.mark.unit
    def test_training_history(self, trained_kearns_model):
        """Test training history recording"""
        model = trained_kearns_model
        
        history = model.get_training_history()
        
        assert isinstance(history, dict)
        expected_keys = ['objective', 'fairness_violations', 'accuracy']
        
        for key in expected_keys:
            assert key in history
            assert isinstance(history[key], list)
            assert len(history[key]) > 0

class TestBaselineFactory:
    """Test baseline method factory functionality"""
    
    @pytest.mark.unit
    def test_create_doubly_regressing_method(self):
        """Test creation of doubly regressing method"""
        model = create_baseline_method('doubly_regressing', lambda_val=0.7, random_state=42)
        
        assert isinstance(model, DoublyRegressingFairness)
        assert model.lambda_fair == 0.7
        assert model.random_state == 42
    
    @pytest.mark.unit
    def test_create_kearns_method(self):
        """Test creation of Kearns method"""
        model = create_baseline_method('kearns_supipm', lambda_val=1.2, random_state=123)
        
        assert isinstance(model, KearnsSubgroupFairness)
        assert model.fairness_penalty == 1.2
        assert model.random_state == 123
    
    @pytest.mark.unit
    def test_create_fams_method(self):
        """Test creation of FAMS method"""
        model = create_baseline_method('fams_fairness', lambda_val=0.6, random_state=456)
        
        assert isinstance(model, FAMSFairness)
        assert model.weight_kld == 0.6
        assert model.random_state == 456
    
    @pytest.mark.unit
    def test_create_placeholder_methods(self):
        """Test creation of placeholder methods"""
        placeholder_methods = ['molina_bounds', 'agarwal_marginal', 'multicalibration', 'foulds_intersectional']
        
        for method_name in placeholder_methods:
            model = create_baseline_method(method_name, random_state=42)
            
            # Should create some object (placeholder implementation)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
    
    @pytest.mark.unit
    def test_invalid_method_name(self):
        """Test error handling for invalid method names"""
        with pytest.raises(ValueError, match="Unknown method"):
            create_baseline_method('nonexistent_method')
    
    @pytest.mark.unit
    def test_get_method_info(self):
        """Test method information retrieval"""
        info = get_method_info('doubly_regressing')
        
        assert isinstance(info, dict)
        expected_keys = ['name', 'description', 'paper', 'supports_lambda', 'partial_fairness']
        
        for key in expected_keys:
            assert key in info
        
        assert info['supports_lambda'] is True
        assert info['partial_fairness'] is True
    
    @pytest.mark.unit
    def test_list_available_methods(self):
        """Test listing available methods"""
        methods = list_available_methods()
        
        assert isinstance(methods, list)
        assert len(methods) > 0
        
        # Should include main methods
        expected_methods = ['doubly_regressing', 'kearns_supipm', 'fams_fairness']
        for method in expected_methods:
            assert method in methods
    
    @pytest.mark.unit
    def test_method_creation_with_additional_params(self):
        """Test method creation with additional parameters"""
        model = create_baseline_method(
            'doubly_regressing', 
            lambda_val=0.5,
            random_state=42,
            n_low=100,
            max_iterations=1000
        )
        
        assert model.lambda_fair == 0.5
        assert model.random_state == 42
        assert model.n_low == 100
        assert model.max_iterations == 1000

class TestModelCompatibility:
    """Test sklearn-style compatibility of models"""
    
    @pytest.mark.unit
    def test_sklearn_interface_doubly_regressing(self, sample_binary_data):
        """Test sklearn-compatible interface for DoublyRegressingFairness"""
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        model = DoublyRegressingFairness(max_iterations=5, random_state=42, verbose=False)
        
        # Should inherit from sklearn base classes
        assert isinstance(model, BaseEstimator)
        assert isinstance(model, ClassifierMixin)
        
        data = sample_binary_data
        
        # Should support sklearn-style fit/predict
        model.fit(data['X'], data['y'], data['S'])
        y_pred = model.predict(data['X'])
        y_pred_proba = model.predict_proba(data['X'])
        
        assert len(y_pred) == len(data['X'])
        assert len(y_pred_proba) == len(data['X'])
    
    @pytest.mark.unit
    def test_sklearn_interface_kearns(self, sample_binary_data):
        """Test sklearn-compatible interface for KearnsSubgroupFairness"""
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        model = KearnsSubgroupFairness(max_iterations=5, random_state=42)
        
        # Should inherit from sklearn base classes  
        assert isinstance(model, BaseEstimator)
        assert isinstance(model, ClassifierMixin)
        
        data = sample_binary_data
        
        # Should support sklearn-style fit/predict
        model.fit(data['X'], data['y'], data['S'])
        y_pred = model.predict(data['X'])
        y_pred_proba = model.predict_proba(data['X'])
        
        assert len(y_pred) == len(data['X'])
        assert len(y_pred_proba) == len(data['X'])

class TestFAMSFairness:
    """Test FAMS (Fairness and Accuracy on Multiple Subgroups) implementation"""
    
    @pytest.mark.unit
    def test_fams_initialization(self):
        """Test proper initialization of FAMSFairness"""
        model = FAMSFairness(
            hidden_dims=[32, 16],
            n_subtasks=50,
            training_epochs=10,
            weight_kld=0.5,
            random_state=42
        )
        
        assert model.hidden_dims == [32, 16]
        assert model.n_subtasks == 50
        assert model.training_epochs == 10
        assert model.weight_kld == 0.5
        assert model.random_state == 42
        assert model.prior_model is None  # Not trained yet
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_fams_fit_basic_functionality(self, sample_binary_data):
        """Test basic fitting functionality for FAMS"""
        data = sample_binary_data
        
        # Use very small parameters for unit testing
        model = FAMSFairness(
            hidden_dims=[8, 4],
            n_subtasks=5,  # Very few subtasks
            training_epochs=2,  # Very few epochs
            batch_size=32,
            max_inner_steps=1,  # Single inner step
            n_mc_samples=2,  # Few MC samples
            random_state=42,
            verbose=False
        )
        
        # Fit model
        model.fit(data['X'], data['y'], data['S'])
        
        # Check model is trained
        assert model.prior_model is not None
        assert len(model.posterior_models) > 0
        assert model.scaler is not None
        assert len(model.subgroup_tasks) > 0
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_fams_predict_after_training(self, sample_binary_data):
        """Test prediction functionality after training"""
        data = sample_binary_data
        
        # Quick training setup
        model = FAMSFairness(
            hidden_dims=[4],
            n_subtasks=3,
            training_epochs=1,
            max_inner_steps=1,
            random_state=42,
            verbose=False
        )
        
        model.fit(data['X'], data['y'], data['S'])
        
        # Test predictions
        y_pred = model.predict(data['X'])
        
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(data['X'])
        assert set(np.unique(y_pred)).issubset({0, 1})  # Binary predictions
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_fams_predict_proba_after_training(self, sample_binary_data):
        """Test probability prediction functionality"""
        data = sample_binary_data
        
        model = FAMSFairness(
            hidden_dims=[4],
            n_subtasks=3,
            training_epochs=1,
            max_inner_steps=1,
            random_state=42,
            verbose=False
        )
        
        model.fit(data['X'], data['y'], data['S'])
        
        # Test probability predictions
        y_pred_proba = model.predict_proba(data['X'])
        
        assert isinstance(y_pred_proba, np.ndarray)
        assert y_pred_proba.shape == (len(data['X']), 2)  # [P(class=0), P(class=1)]
        assert_valid_probabilities(y_pred_proba)
    
    @pytest.mark.unit
    def test_fams_subgroup_task_creation(self, sample_binary_data):
        """Test subgroup task creation"""
        data = sample_binary_data
        
        model = FAMSFairness(n_subtasks=10, random_state=42)
        
        # Mock tensor conversion for testing
        X_tensor = torch.FloatTensor(data['X'].values)
        y_tensor = torch.FloatTensor(data['y'].values)
        
        tasks = model._create_subgroup_tasks(X_tensor, y_tensor, data['S'])
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        
        # Each task should have required fields
        for task in tasks:
            assert 'X' in task
            assert 'y' in task
            assert 'definition' in task
            assert 'size' in task
            assert task['size'] >= 10  # Minimum task size
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_fams_training_history(self, sample_binary_data):
        """Test training history recording"""
        data = sample_binary_data
        
        model = FAMSFairness(
            training_epochs=2,
            n_subtasks=3,
            random_state=42,
            verbose=False
        )
        
        model.fit(data['X'], data['y'], data['S'])
        history = model.get_training_history()
        
        assert isinstance(history, dict)
        expected_keys = ['prior_loss', 'posterior_losses', 'kld_losses', 'total_loss']
        
        for key in expected_keys:
            assert key in history
            assert isinstance(history[key], list)
            assert len(history[key]) > 0  # Should have recorded some epochs
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_fams_subgroup_info(self, sample_binary_data):
        """Test subgroup information retrieval"""
        data = sample_binary_data
        
        model = FAMSFairness(
            n_subtasks=5,
            training_epochs=1,
            random_state=42,
            verbose=False
        )
        
        model.fit(data['X'], data['y'], data['S'])
        info = model.get_subgroup_info()
        
        assert isinstance(info, dict)
        expected_keys = ['n_subgroup_tasks', 'n_posterior_models', 'task_sizes', 'task_definitions']
        
        for key in expected_keys:
            assert key in info
        
        assert info['n_subgroup_tasks'] > 0
        assert info['n_posterior_models'] > 0
        assert len(info['task_sizes']) == info['n_subgroup_tasks']

class TestModelRobustness:
    """Test robustness of models to edge cases"""
    
    @pytest.mark.unit
    def test_models_with_single_sensitive_attribute(self):
        """Test models with single sensitive attribute"""
        n_samples = 200
        X = pd.DataFrame(np.random.randn(n_samples, 3), columns=['f1', 'f2', 'f3'])
        y = pd.Series(np.random.choice([0, 1], n_samples))
        S = pd.DataFrame({'single_attr': np.random.choice(['A', 'B'], n_samples)})
        
        # Test DoublyRegressing
        model_dr = DoublyRegressingFairness(max_iterations=5, n_low=10, random_state=42, verbose=False)
        model_dr.fit(X, y, S)
        y_pred_dr = model_dr.predict(X)
        assert len(y_pred_dr) == n_samples
        
        # Test Kearns
        model_k = KearnsSubgroupFairness(max_iterations=5, random_state=42)
        model_k.fit(X, y, S)
        y_pred_k = model_k.predict(X)
        assert len(y_pred_k) == n_samples
    
    @pytest.mark.unit
    def test_models_with_many_sensitive_attributes(self):
        """Test models with many sensitive attributes"""
        n_samples = 500
        X = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], n_samples))
        
        # Create many sensitive attributes
        S_data = {}
        for i in range(6):  # 6 sensitive attributes
            S_data[f'attr_{i}'] = np.random.choice([f'val_{j}' for j in range(3)], n_samples)
        S = pd.DataFrame(S_data)
        
        # Test with reduced complexity for speed
        model = DoublyRegressingFairness(
            max_iterations=3, 
            n_low=20,  # Higher threshold to reduce subgroups
            random_state=42, 
            verbose=False
        )
        
        # Should handle many attributes (may take longer)
        model.fit(X, y, S)
        y_pred = model.predict(X)
        assert len(y_pred) == n_samples
    
    @pytest.mark.unit
    def test_models_with_imbalanced_classes(self):
        """Test models with highly imbalanced classes"""
        n_samples = 1000
        X = pd.DataFrame(np.random.randn(n_samples, 4), columns=[f'f{i}' for i in range(4)])
        
        # Create highly imbalanced target (5% positive class)
        n_positive = 50
        y = pd.Series([1] * n_positive + [0] * (n_samples - n_positive))
        S = pd.DataFrame({'attr': np.random.choice(['A', 'B'], n_samples)})
        
        model = DoublyRegressingFairness(max_iterations=5, n_low=10, random_state=42, verbose=False)
        
        # Should handle imbalanced classes
        model.fit(X, y, S)
        y_pred = model.predict(X)
        
        assert len(y_pred) == n_samples
        assert set(np.unique(y_pred)).issubset({0, 1})
    
    @pytest.mark.unit
    def test_model_reproducibility(self, sample_binary_data):
        """Test that models produce reproducible results"""
        data = sample_binary_data
        
        # Train two models with same random state
        model1 = DoublyRegressingFairness(max_iterations=5, random_state=42, verbose=False)
        model2 = DoublyRegressingFairness(max_iterations=5, random_state=42, verbose=False)
        
        model1.fit(data['X'], data['y'], data['S'])
        model2.fit(data['X'], data['y'], data['S'])
        
        y_pred1 = model1.predict(data['X'])
        y_pred2 = model2.predict(data['X'])
        
        # Should be identical (or very close due to floating point)
        assert np.allclose(y_pred1, y_pred2, atol=1e-10)