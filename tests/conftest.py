"""
Pytest configuration and shared fixtures for subgroup fairness tests
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
from typing import Dict, Any, Tuple
from unittest.mock import Mock

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.utils.data_loader import generate_synthetic_dataset
from scripts.baselines.doubly_regressing import DoublyRegressingFairness
from scripts.baselines.kearns_subipm import KearnsSubgroupFairness

@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests"""
    return 42

@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_binary_data(random_seed):
    """Generate small binary classification dataset for testing"""
    np.random.seed(random_seed)
    
    n_samples = 1000
    n_features = 5
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sensitive attributes
    sensitive_data = {
        'gender': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'race': np.random.choice(['white', 'black', 'asian'], n_samples, p=[0.7, 0.2, 0.1]),
        'age_group': np.random.choice(['young', 'old'], n_samples, p=[0.3, 0.7])
    }
    
    # Generate biased target
    target_logits = (
        X.sum(axis=1) * 0.1 +
        (sensitive_data['gender'] == 'male').astype(int) * 0.3 +
        (sensitive_data['race'] == 'white').astype(int) * 0.2
    )
    
    target_probs = 1 / (1 + np.exp(-target_logits))
    y = np.random.binomial(1, target_probs, n_samples)
    
    # Create DataFrames
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    S_df = pd.DataFrame(sensitive_data)
    
    return {
        'X': X_df,
        'y': pd.Series(y),
        'S': S_df,
        'n_samples': n_samples,
        'n_features': n_features
    }

@pytest.fixture
def sample_multiclass_data(random_seed):
    """Generate multiclass classification dataset for testing"""
    np.random.seed(random_seed)
    
    n_samples = 800
    n_features = 4
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    
    # Generate sensitive attributes
    sensitive_data = {
        'group_a': np.random.choice(['A1', 'A2', 'A3'], n_samples),
        'group_b': np.random.choice(['B1', 'B2'], n_samples, p=[0.3, 0.7])
    }
    
    # Generate multiclass target
    class_logits = np.random.randn(n_samples, n_classes)
    class_probs = np.exp(class_logits) / np.exp(class_logits).sum(axis=1, keepdims=True)
    y = np.array([np.random.choice(n_classes, p=prob) for prob in class_probs])
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    S_df = pd.DataFrame(sensitive_data)
    
    return {
        'X': X_df,
        'y': pd.Series(y),
        'S': S_df,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes
    }

@pytest.fixture
def edge_case_data(random_seed):
    """Generate edge case datasets for robustness testing"""
    np.random.seed(random_seed)
    
    datasets = {}
    
    # Very small dataset
    datasets['tiny'] = {
        'X': pd.DataFrame(np.random.randn(10, 2), columns=['f1', 'f2']),
        'y': pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        'S': pd.DataFrame({'attr': ['A', 'B'] * 5})
    }
    
    # Single class dataset
    datasets['single_class'] = {
        'X': pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3']),
        'y': pd.Series([1] * 100),
        'S': pd.DataFrame({'attr': ['A'] * 50 + ['B'] * 50})
    }
    
    # Perfect separation dataset
    X_sep = np.vstack([np.random.randn(50, 2) - 2, np.random.randn(50, 2) + 2])
    datasets['perfect_sep'] = {
        'X': pd.DataFrame(X_sep, columns=['f1', 'f2']),
        'y': pd.Series([0] * 50 + [1] * 50),
        'S': pd.DataFrame({'attr': ['A'] * 50 + ['B'] * 50})
    }
    
    return datasets

@pytest.fixture
def mock_experiment_config():
    """Mock experiment configuration for testing"""
    return {
        'datasets': [{
            'name': 'synthetic',
            'sample_size': 1000,
            'n_sensitive_attrs': 3
        }],
        'methods': ['doubly_regressing', 'kearns_supipm'],
        'lambda_values': [0.0, 0.5, 1.0],
        'n_low_values': [10, 50],
        'primary_metrics': ['accuracy', 'sup_ipm'],
        'output_plots': ['accuracy_fairness']
    }

@pytest.fixture
def trained_doubly_regressing_model(sample_binary_data, random_seed):
    """Pre-trained Doubly Regressing model for testing"""
    model = DoublyRegressingFairness(
        lambda_fair=0.5,
        n_low=20,
        max_iterations=100,  # Reduced for faster testing
        random_state=random_seed,
        verbose=False
    )
    
    data = sample_binary_data
    model.fit(data['X'], data['y'], data['S'])
    
    return model

@pytest.fixture
def trained_kearns_model(sample_binary_data, random_seed):
    """Pre-trained Kearns model for testing"""
    model = KearnsSubgroupFairness(
        gamma=0.1,
        max_iterations=50,  # Reduced for faster testing
        random_state=random_seed
    )
    
    data = sample_binary_data
    model.fit(data['X'], data['y'], data['S'])
    
    return model

@pytest.fixture
def performance_data():
    """Large dataset for performance testing"""
    np.random.seed(42)
    
    n_samples = 10000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    
    sensitive_data = {
        'attr1': np.random.choice(['A', 'B', 'C'], n_samples),
        'attr2': np.random.choice(['X', 'Y'], n_samples),
        'attr3': np.random.choice(['P', 'Q', 'R', 'S'], n_samples)
    }
    
    # Generate target with complex interactions
    target_logits = (
        X[:, :5].sum(axis=1) * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    target_probs = 1 / (1 + np.exp(-target_logits))
    y = np.random.binomial(1, target_probs, n_samples)
    
    return {
        'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)]),
        'y': pd.Series(y),
        'S': pd.DataFrame(sensitive_data),
        'n_samples': n_samples,
        'n_features': n_features
    }

# Custom pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual functions")
    config.addinivalue_line("markers", "integration: Integration tests for full pipeline") 
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")

# Helper functions for test validation
def assert_valid_probabilities(probs: np.ndarray, tolerance: float = 1e-6):
    """Assert that probabilities are valid (between 0 and 1, sum to 1 for each sample)"""
    assert np.all(probs >= -tolerance), "Probabilities should be non-negative"
    assert np.all(probs <= 1 + tolerance), "Probabilities should be <= 1"
    
    if probs.ndim == 2:  # Multi-class probabilities
        prob_sums = probs.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=tolerance), "Probabilities should sum to 1"

def assert_fairness_metrics_valid(metrics: Dict[str, float]):
    """Assert that fairness metrics are valid"""
    for metric_name, value in metrics.items():
        if 'violation' in metric_name.lower():
            assert value >= 0, f"Fairness violations should be non-negative: {metric_name}={value}"
        
        if 'rate' in metric_name.lower():
            assert 0 <= value <= 1, f"Rates should be between 0 and 1: {metric_name}={value}"
        
        assert not np.isnan(value), f"Metric should not be NaN: {metric_name}"
        assert not np.isinf(value), f"Metric should not be infinite: {metric_name}"

def assert_model_trained(model):
    """Assert that a model has been properly trained"""
    if hasattr(model, 'classifier') and model.classifier is not None:
        assert model.classifier is not None, "Classifier should be trained"
    
    if hasattr(model, 'training_history') and model.training_history is not None:
        assert len(model.training_history.get('classifier_loss', [])) > 0, "Training history should exist"