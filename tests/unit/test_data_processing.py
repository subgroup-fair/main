"""
Unit tests for data processing components
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.utils.data_loader import (
    load_datasets, 
    preprocess_sensitive_attributes,
    create_train_test_split,
    generate_synthetic_dataset,
    load_uci_adult_dataset,
    load_communities_crime_dataset
)
from tests.mock_data.generators import generate_biased_dataset, generate_fair_dataset

class TestDataLoading:
    """Test data loading functionalities"""
    
    @pytest.mark.unit
    def test_load_synthetic_dataset_basic(self):
        """Test basic synthetic dataset loading"""
        config = {
            'name': 'synthetic',
            'sample_size': 1000,
            'n_sensitive_attrs': 3,
            'noise_level': 0.0
        }
        
        result = load_datasets(config)
        
        assert 'train' in result
        assert 'test' in result
        assert 'sensitive_attrs' in result
        assert 'feature_names' in result
        
        # Check data shapes
        train_data = result['train']
        test_data = result['test']
        
        assert len(train_data) + len(test_data) == 1000
        assert len(train_data) > len(test_data)  # 80-20 split
        
        # Check target column exists
        assert 'target' in train_data.columns
        assert 'target' in test_data.columns
        
        # Check sensitive attributes exist
        assert len(result['sensitive_attrs']) == 3
    
    @pytest.mark.unit
    def test_load_synthetic_dataset_with_noise(self):
        """Test synthetic dataset loading with noise"""
        config = {
            'name': 'synthetic',
            'sample_size': 500,
            'n_sensitive_attrs': 2,
            'noise_level': 0.2
        }
        
        result = load_datasets(config)
        
        # Should still have valid structure with noise
        assert 'train' in result
        assert result['metadata']['noise_level'] == 0.2
        
        # Check data integrity with noise
        train_data = result['train']
        target_values = train_data['target'].unique()
        assert set(target_values).issubset({0, 1})  # Binary target should still be valid
    
    @pytest.mark.unit
    def test_load_uci_adult_dataset(self):
        """Test UCI Adult dataset loading (synthetic version)"""
        config = {
            'name': 'uci_adult',
            'sample_size': 2000,
            'sensitive_attrs': ['age_group', 'race', 'sex']
        }
        
        result = load_uci_adult_dataset(config)
        
        assert result['name'] == 'uci_adult'
        assert len(result['sensitive_attrs']) == 3
        
        # Check specific sensitive attributes exist
        expected_attrs = ['age_group', 'race', 'sex']
        assert all(attr in result['sensitive_attrs'] for attr in expected_attrs)
        
        # Check data types and ranges
        train_data = result['train']
        assert train_data['target'].dtype in [int, np.int32, np.int64]
        assert set(train_data['target'].unique()).issubset({0, 1})
    
    @pytest.mark.unit
    def test_load_communities_crime_dataset(self):
        """Test Communities Crime dataset loading (synthetic version)"""
        config = {
            'name': 'communities_crime',
            'sample_size': 1500,
            'sensitive_attrs': ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
        }
        
        result = load_communities_crime_dataset(config)
        
        assert result['name'] == 'communities_crime'
        assert len(result['sensitive_attrs']) == 4
        
        # Check racial proportion features are in [0, 1] range
        train_data = result['train']
        for attr in ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']:
            if attr in train_data.columns:
                values = train_data[attr]
                assert values.min() >= 0, f"{attr} should be non-negative"
                assert values.max() <= 1, f"{attr} should be <= 1"
    
    @pytest.mark.unit
    def test_invalid_dataset_name(self):
        """Test error handling for invalid dataset names"""
        config = {
            'name': 'nonexistent_dataset',
            'sample_size': 100
        }
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_datasets(config)

class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    @pytest.mark.unit
    def test_preprocess_continuous_sensitive_attributes(self):
        """Test preprocessing of continuous sensitive attributes"""
        # Create test data with continuous sensitive attributes
        df = pd.DataFrame({
            'age': np.random.uniform(18, 80, 1000),
            'income': np.random.lognormal(10, 1, 1000),
            'categorical_attr': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        sensitive_attrs = ['age', 'income', 'categorical_attr']
        result = preprocess_sensitive_attributes(df, sensitive_attrs)
        
        # Continuous attributes should be binned
        assert result['age'].dtype == 'category'
        assert result['income'].dtype == 'category'
        
        # Categorical should remain unchanged
        assert all(val in ['A', 'B', 'C'] for val in result['categorical_attr'])
        
        # Check binning creates 4 categories
        assert len(result['age'].cat.categories) == 4
        assert len(result['income'].cat.categories) == 4
    
    @pytest.mark.unit
    def test_preprocess_mixed_sensitive_attributes(self):
        """Test preprocessing with mixed data types"""
        df = pd.DataFrame({
            'continuous_1': np.random.normal(0, 1, 500),
            'continuous_2': np.random.exponential(2, 500),
            'categorical_1': np.random.choice(['X', 'Y'], 500),
            'categorical_2': np.random.choice(['P', 'Q', 'R'], 500),
            'target': np.random.choice([0, 1], 500)
        })
        
        sensitive_attrs = ['continuous_1', 'continuous_2', 'categorical_1', 'categorical_2']
        result = preprocess_sensitive_attributes(df, sensitive_attrs)
        
        # Check output shape
        assert result.shape[0] == 500
        assert result.shape[1] == 4
        
        # Verify all columns are present
        assert all(attr in result.columns for attr in sensitive_attrs)
    
    @pytest.mark.unit
    def test_preprocess_empty_sensitive_attributes(self):
        """Test preprocessing with empty sensitive attributes list"""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        result = preprocess_sensitive_attributes(df, [])
        
        assert result.shape[1] == 0
        assert len(result) == 100

class TestDataSplitting:
    """Test train-test splitting functionality"""
    
    @pytest.mark.unit
    def test_basic_train_test_split(self):
        """Test basic train-test splitting"""
        df = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        train_df, test_df = create_train_test_split(df, target_col='target', test_size=0.3, random_state=42)
        
        # Check sizes
        assert len(train_df) == 700
        assert len(test_df) == 300
        assert len(train_df) + len(test_df) == 1000
        
        # Check column consistency
        assert list(train_df.columns) == list(test_df.columns)
        assert list(train_df.columns) == list(df.columns)
    
    @pytest.mark.unit
    def test_stratified_split_preserves_distribution(self):
        """Test that stratified split preserves class distribution"""
        # Create imbalanced dataset
        n_positive = 200
        n_negative = 800
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'target': [1] * n_positive + [0] * n_negative
        })
        
        train_df, test_df = create_train_test_split(df, target_col='target', test_size=0.2, random_state=42)
        
        # Check class distribution preservation
        original_ratio = n_positive / 1000
        train_ratio = train_df['target'].mean()
        test_ratio = test_df['target'].mean()
        
        # Should be approximately equal (within 5% tolerance)
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05
    
    @pytest.mark.unit
    def test_split_with_different_test_sizes(self):
        """Test splitting with different test sizes"""
        df = pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'target': np.random.choice([0, 1], 1000)
        })
        
        test_sizes = [0.1, 0.2, 0.3, 0.4]
        
        for test_size in test_sizes:
            train_df, test_df = create_train_test_split(df, test_size=test_size, random_state=42)
            
            expected_test_size = int(1000 * test_size)
            expected_train_size = 1000 - expected_test_size
            
            assert len(test_df) == expected_test_size
            assert len(train_df) == expected_train_size

class TestDataValidation:
    """Test data validation and edge cases"""
    
    @pytest.mark.unit
    def test_data_integrity_after_loading(self, sample_binary_data):
        """Test data integrity after loading operations"""
        data = sample_binary_data
        
        # Check no NaN values
        assert not data['X'].isnull().any().any()
        assert not data['y'].isnull().any()
        assert not data['S'].isnull().any().any()
        
        # Check data types
        assert data['X'].dtypes.apply(lambda x: x in [np.float64, np.int64]).all()
        assert data['y'].dtype in [np.int64, int]
        
        # Check value ranges
        assert data['y'].min() >= 0
        assert data['y'].max() <= 1
        assert set(data['y'].unique()).issubset({0, 1})
    
    @pytest.mark.unit
    def test_handle_missing_columns(self):
        """Test handling of missing columns in dataset"""
        # Dataset missing target column
        df_no_target = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        
        with pytest.raises((KeyError, ValueError)):
            create_train_test_split(df_no_target, target_col='target')
    
    @pytest.mark.unit
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            create_train_test_split(empty_df, target_col='target')
    
    @pytest.mark.unit
    def test_single_class_dataset(self):
        """Test handling of single-class datasets"""
        # Dataset with only one class
        df_single_class = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'target': [1] * 100  # All positive class
        })
        
        with pytest.raises((ValueError)):
            # Should fail due to stratification with single class
            create_train_test_split(df_single_class, target_col='target')
    
    @pytest.mark.unit 
    def test_very_small_dataset(self):
        """Test handling of very small datasets"""
        df_tiny = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        # Should handle gracefully but may have issues with small size
        train_df, test_df = create_train_test_split(df_tiny, test_size=0.33, random_state=42)
        assert len(train_df) + len(test_df) == 3
        assert len(test_df) >= 1  # At least one sample in test set

class TestMockDataGenerators:
    """Test mock data generators for consistency"""
    
    @pytest.mark.unit
    def test_biased_dataset_generation(self):
        """Test biased dataset generation"""
        dataset = generate_biased_dataset(n_samples=1000, bias_strength=0.7, random_state=42)
        
        assert dataset['X'].shape[0] == 1000
        assert dataset['y'].shape[0] == 1000
        assert dataset['S'].shape[0] == 1000
        
        # Check bias is actually present
        assert dataset['metadata']['bias_strength'] == 0.7
        assert dataset['metadata']['type'] == 'biased'
        
        # Verify target is binary
        assert set(dataset['y'].unique()).issubset({0, 1})
    
    @pytest.mark.unit
    def test_fair_dataset_generation(self):
        """Test fair dataset generation"""
        dataset = generate_fair_dataset(n_samples=800, random_state=42)
        
        assert dataset['X'].shape[0] == 800
        assert dataset['metadata']['type'] == 'fair'
        assert dataset['metadata']['bias_strength'] <= 0.1  # Should be minimal bias
    
    @pytest.mark.unit
    def test_reproducibility_of_mock_data(self):
        """Test that mock data generation is reproducible"""
        dataset1 = generate_biased_dataset(n_samples=500, random_state=123)
        dataset2 = generate_biased_dataset(n_samples=500, random_state=123)
        
        # Should be identical with same random state
        pd.testing.assert_frame_equal(dataset1['X'], dataset2['X'])
        pd.testing.assert_series_equal(dataset1['y'], dataset2['y'])
        pd.testing.assert_frame_equal(dataset1['S'], dataset2['S'])
    
    @pytest.mark.unit
    def test_different_random_states_produce_different_data(self):
        """Test that different random states produce different data"""
        dataset1 = generate_biased_dataset(n_samples=500, random_state=111)
        dataset2 = generate_biased_dataset(n_samples=500, random_state=222)
        
        # Should be different with different random states
        assert not dataset1['X'].equals(dataset2['X'])
        assert not dataset1['y'].equals(dataset2['y'])

class TestDataEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.mark.unit
    def test_negative_sample_size(self):
        """Test handling of negative sample sizes"""
        with pytest.raises((ValueError, AssertionError)):
            generate_biased_dataset(n_samples=-100)
    
    @pytest.mark.unit
    def test_zero_sample_size(self):
        """Test handling of zero sample size"""
        with pytest.raises((ValueError, AssertionError)):
            generate_biased_dataset(n_samples=0)
    
    @pytest.mark.unit
    def test_extreme_bias_strength(self):
        """Test handling of extreme bias strengths"""
        # Very high bias
        dataset = generate_biased_dataset(n_samples=1000, bias_strength=2.0, random_state=42)
        assert dataset['X'].shape[0] == 1000  # Should still work
        
        # Negative bias
        dataset_neg = generate_biased_dataset(n_samples=1000, bias_strength=-0.5, random_state=42)
        assert dataset_neg['X'].shape[0] == 1000  # Should handle gracefully