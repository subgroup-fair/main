"""
Unit tests for statistical analysis and fairness metrics
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.utils.metrics import (
    SubgroupFairnessMetrics,
    evaluate_all_metrics,
    compute_fairness_tradeoff_curve
)
from tests.conftest import assert_fairness_metrics_valid, assert_valid_probabilities

class TestSubgroupFairnessMetrics:
    """Test SubgroupFairnessMetrics class methods"""
    
    @pytest.mark.unit
    def test_demographic_parity_basic(self, sample_binary_data):
        """Test basic demographic parity calculation"""
        data = sample_binary_data
        y_pred = np.random.choice([0, 1], len(data['y']))
        
        results = SubgroupFairnessMetrics.demographic_parity(y_pred, data['S'])
        
        # Check output structure
        assert isinstance(results, dict)
        
        # Should have results for each sensitive attribute
        for attr in data['S'].columns:
            assert f"dp_violation_{attr}" in results
            assert f"dp_max_rate_{attr}" in results
            assert f"dp_min_rate_{attr}" in results
        
        # Validate metric values
        assert_fairness_metrics_valid(results)
    
    @pytest.mark.unit
    def test_demographic_parity_perfect_fairness(self):
        """Test demographic parity with perfectly fair predictions"""
        n_samples = 1000
        
        # Create balanced sensitive attributes
        S = pd.DataFrame({
            'gender': ['male', 'female'] * (n_samples // 2),
            'race': ['white', 'black', 'asian', 'hispanic'] * (n_samples // 4)
        })
        
        # Create perfectly fair predictions (same rate across groups)
        y_pred = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        results = SubgroupFairnessMetrics.demographic_parity(y_pred, S)
        
        # Violations should be small (due to random sampling noise)
        for key, value in results.items():
            if 'violation' in key:
                assert value < 0.1, f"Perfect fairness should have low violation: {key}={value}"
    
    @pytest.mark.unit
    def test_demographic_parity_extreme_bias(self):
        """Test demographic parity with extreme bias"""
        n_samples = 1000
        
        S = pd.DataFrame({
            'group': ['A'] * 500 + ['B'] * 500
        })
        
        # Extreme bias: Group A always gets 1, Group B always gets 0
        y_pred = np.array([1] * 500 + [0] * 500)
        
        results = SubgroupFairnessMetrics.demographic_parity(y_pred, S)
        
        # Should detect maximum violation
        assert results['dp_violation_group'] == 1.0
        assert results['dp_max_rate_group'] == 1.0
        assert results['dp_min_rate_group'] == 0.0
    
    @pytest.mark.unit 
    def test_equal_opportunity_basic(self, sample_binary_data):
        """Test basic equal opportunity calculation"""
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        
        results = SubgroupFairnessMetrics.equal_opportunity(y_true, y_pred, data['S'])
        
        # Check output structure
        assert isinstance(results, dict)
        
        # Should have results for attributes with positive samples
        for attr in data['S'].columns:
            # May not have results if no positive samples in some groups
            violation_key = f"eo_violation_{attr}"
            if violation_key in results:
                assert results[violation_key] >= 0
    
    @pytest.mark.unit
    def test_equal_opportunity_edge_cases(self):
        """Test equal opportunity with edge cases"""
        # Case 1: No positive samples
        S = pd.DataFrame({'attr': ['A', 'B', 'A', 'B']})
        y_true = np.array([0, 0, 0, 0])  # No positive cases
        y_pred = np.array([0, 1, 0, 1])
        
        results = SubgroupFairnessMetrics.equal_opportunity(y_true, y_pred, S)
        
        # Should handle gracefully (may return empty dict)
        assert isinstance(results, dict)
        
        # Case 2: All positive samples
        y_true_all_pos = np.array([1, 1, 1, 1])
        y_pred_perfect = np.array([1, 1, 1, 1])  # Perfect TPR
        
        results_perfect = SubgroupFairnessMetrics.equal_opportunity(y_true_all_pos, y_pred_perfect, S)
        
        # Should have zero violation for perfect TPR
        if 'eo_violation_attr' in results_perfect:
            assert results_perfect['eo_violation_attr'] == 0.0
    
    @pytest.mark.unit
    def test_equalized_odds_basic(self, sample_binary_data):
        """Test basic equalized odds calculation"""
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        
        results = SubgroupFairnessMetrics.equalized_odds(y_true, y_pred, data['S'])
        
        # Check output structure
        assert isinstance(results, dict)
        
        # Should have TPR and FPR violations for each attribute
        for attr in data['S'].columns:
            tpr_key = f"eqodds_tpr_violation_{attr}"
            fpr_key = f"eqodds_fpr_violation_{attr}"
            combined_key = f"eqodds_violation_{attr}"
            
            # At least some of these should exist
            keys_exist = any(key in results for key in [tpr_key, fpr_key, combined_key])
            if not keys_exist:
                # May happen if groups are too small
                continue
                
            # If they exist, they should be valid
            for key in [tpr_key, fpr_key, combined_key]:
                if key in results:
                    assert results[key] >= 0, f"Violation should be non-negative: {key}={results[key]}"
    
    @pytest.mark.unit
    def test_sup_ipm_metric_basic(self, sample_binary_data):
        """Test basic supIPM metric calculation"""
        data = sample_binary_data
        n_samples = len(data['y'])
        
        # Create prediction probabilities
        y_pred_proba = np.random.uniform(0, 1, n_samples)
        
        results = SubgroupFairnessMetrics.sup_ipm_metric(y_pred_proba, data['S'])
        
        # Check output structure
        assert isinstance(results, dict)
        assert 'sup_ipm' in results
        assert 'mean_ipm' in results
        assert 'num_subgroups_evaluated' in results
        
        # Check value validity
        assert results['sup_ipm'] >= 0, "supIPM should be non-negative"
        assert results['mean_ipm'] >= 0, "mean IPM should be non-negative"
        assert results['num_subgroups_evaluated'] >= 0, "Number of subgroups should be non-negative"
    
    @pytest.mark.unit
    def test_sup_ipm_metric_with_2d_probabilities(self, sample_binary_data):
        """Test supIPM with 2D probability array (sklearn format)"""
        data = sample_binary_data
        n_samples = len(data['y'])
        
        # Create 2D probabilities [P(class=0), P(class=1)]
        y_pred_proba_2d = np.random.dirichlet([1, 1], n_samples)
        
        results = SubgroupFairnessMetrics.sup_ipm_metric(y_pred_proba_2d, data['S'])
        
        assert 'sup_ipm' in results
        assert results['sup_ipm'] >= 0
        assert not np.isnan(results['sup_ipm'])
    
    @pytest.mark.unit
    def test_sup_ipm_perfect_fairness(self):
        """Test supIPM with perfectly fair predictions"""
        n_samples = 1000
        
        # Create balanced groups
        S = pd.DataFrame({
            'group': ['A', 'B'] * (n_samples // 2)
        })
        
        # Perfectly fair predictions (same rate for all)
        fair_rate = 0.4
        y_pred_proba = np.full(n_samples, fair_rate)
        
        results = SubgroupFairnessMetrics.sup_ipm_metric(y_pred_proba, S)
        
        # Should have zero violation for perfect fairness
        assert results['sup_ipm'] == 0.0, "Perfect fairness should have zero supIPM"
    
    @pytest.mark.unit
    def test_subgroup_coverage_metrics(self, sample_binary_data):
        """Test subgroup coverage metrics"""
        data = sample_binary_data
        
        results = SubgroupFairnessMetrics.subgroup_coverage_metrics(data['S'], min_size=10)
        
        # Check output structure
        expected_keys = [
            'total_subgroups', 'covered_subgroups', 'coverage_rate',
            'mean_subgroup_size', 'min_subgroup_size', 'max_subgroup_size'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        
        # Check value validity
        assert results['total_subgroups'] >= results['covered_subgroups']
        assert 0 <= results['coverage_rate'] <= 1
        assert results['mean_subgroup_size'] > 0
        assert results['min_subgroup_size'] >= 0
        assert results['max_subgroup_size'] >= results['min_subgroup_size']
    
    @pytest.mark.unit
    def test_enumerate_subgroups_functionality(self, sample_binary_data):
        """Test subgroup enumeration"""
        data = sample_binary_data
        
        subgroups = SubgroupFairnessMetrics._enumerate_subgroups(data['S'], max_combinations=50)
        
        # Should return list of subgroup definitions
        assert isinstance(subgroups, list)
        assert len(subgroups) > 0
        
        # Each subgroup should be a dictionary
        for subgroup in subgroups:
            assert isinstance(subgroup, dict)
            
            # Should have definition with attribute-value pairs
            for attr, value in subgroup.items():
                assert attr in data['S'].columns
                assert value in data['S'][attr].unique()
    
    @pytest.mark.unit
    def test_get_subgroup_mask_functionality(self, sample_binary_data):
        """Test subgroup mask generation"""
        data = sample_binary_data
        
        # Create a specific subgroup definition
        subgroup_def = {
            'gender': 'male',
            'race': 'white'
        }
        
        mask = SubgroupFairnessMetrics._get_subgroup_mask(data['S'], subgroup_def)
        
        # Check mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(data['S'])
        
        # Verify mask correctness
        masked_data = data['S'][mask]
        if len(masked_data) > 0:
            assert all(masked_data['gender'] == 'male')
            assert all(masked_data['race'] == 'white')

class TestEvaluateAllMetrics:
    """Test the comprehensive metrics evaluation function"""
    
    @pytest.mark.unit
    def test_evaluate_all_metrics_basic(self, sample_binary_data):
        """Test basic comprehensive metrics evaluation"""
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        y_pred_proba = np.random.uniform(0, 1, len(y_true))
        
        results = evaluate_all_metrics(y_true, y_pred, y_pred_proba, data['S'])
        
        # Check accuracy metrics
        accuracy_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        for metric in accuracy_metrics:
            assert metric in results, f"Missing accuracy metric: {metric}"
            assert 0 <= results[metric] <= 1, f"Accuracy metric out of range: {metric}={results[metric]}"
        
        # Check AUC if available
        if 'auc_roc' in results:
            assert 0 <= results['auc_roc'] <= 1, f"AUC out of range: {results['auc_roc']}"
        
        # Check fairness metrics exist
        fairness_metrics_present = any('violation' in key for key in results.keys())
        assert fairness_metrics_present, "Should have some fairness metrics"
    
    @pytest.mark.unit
    def test_evaluate_all_metrics_without_probabilities(self, sample_binary_data):
        """Test metrics evaluation without prediction probabilities"""
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        
        results = evaluate_all_metrics(y_true, y_pred, None, data['S'])
        
        # Should still have basic metrics
        assert 'accuracy' in results
        assert 'f1_score' in results
        
        # Should not have AUC or supIPM
        assert results.get('auc_roc', 0.0) == 0.0  # Default value when no probabilities
        assert 'sup_ipm' not in results
    
    @pytest.mark.unit
    def test_evaluate_all_metrics_with_coverage(self, sample_binary_data):
        """Test metrics evaluation with coverage analysis"""
        data = sample_binary_data
        y_true = data['y'].values
        y_pred = np.random.choice([0, 1], len(y_true))
        y_pred_proba = np.random.uniform(0, 1, len(y_true))
        
        results = evaluate_all_metrics(y_true, y_pred, y_pred_proba, data['S'], include_coverage=True)
        
        # Should include coverage metrics
        coverage_metrics = ['total_subgroups', 'covered_subgroups', 'coverage_rate']
        for metric in coverage_metrics:
            assert metric in results, f"Missing coverage metric: {metric}"
    
    @pytest.mark.unit
    def test_evaluate_metrics_edge_cases(self):
        """Test metrics evaluation with edge cases"""
        # Perfect predictions
        y_true = np.array([0, 1, 0, 1] * 50)
        y_pred = y_true.copy()  # Perfect predictions
        y_pred_proba = y_true.astype(float)  # Perfect probabilities
        
        S = pd.DataFrame({
            'group': ['A', 'B'] * 100
        })
        
        results = evaluate_all_metrics(y_true, y_pred, y_pred_proba, S)
        
        # Perfect accuracy
        assert results['accuracy'] == 1.0
        assert results['f1_score'] == 1.0
        
        # AUC should be perfect
        if 'auc_roc' in results:
            assert results['auc_roc'] == 1.0

class TestFairnessTradeoffCurve:
    """Test fairness-accuracy tradeoff analysis"""
    
    @pytest.mark.unit
    def test_pareto_frontier_basic(self):
        """Test basic Pareto frontier computation"""
        # Create sample tradeoff data
        accuracy_values = [0.7, 0.75, 0.8, 0.77, 0.85, 0.82]
        fairness_values = [0.3, 0.25, 0.4, 0.27, 0.5, 0.35]  # Lower is better (unfairness)
        lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        
        result = compute_fairness_tradeoff_curve(accuracy_values, fairness_values, lambda_values)
        
        # Check output structure
        expected_keys = [
            'pareto_indices', 'pareto_accuracies', 'pareto_fairness', 'pareto_lambdas',
            'accuracy_range', 'fairness_range', 'lambda_range'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check Pareto optimality
        pareto_indices = result['pareto_indices']
        assert len(pareto_indices) > 0, "Should have at least one Pareto optimal point"
        
        # Check ranges
        assert len(result['accuracy_range']) == 2
        assert len(result['fairness_range']) == 2
        assert result['accuracy_range'][0] <= result['accuracy_range'][1]
        assert result['fairness_range'][0] <= result['fairness_range'][1]
    
    @pytest.mark.unit
    def test_pareto_frontier_dominated_points(self):
        """Test Pareto frontier with clearly dominated points"""
        # Point 2 dominates point 1 (higher accuracy, lower unfairness)
        accuracy_values = [0.6, 0.8, 0.7]  
        fairness_values = [0.4, 0.2, 0.3]   # Lower is better
        lambda_values = [0.0, 0.5, 1.0]
        
        result = compute_fairness_tradeoff_curve(accuracy_values, fairness_values, lambda_values)
        
        # Point 1 should not be Pareto optimal (dominated by point 2)
        pareto_indices = result['pareto_indices']
        assert 0 not in pareto_indices, "Dominated point should not be Pareto optimal"
        assert 1 in pareto_indices, "Dominant point should be Pareto optimal"
    
    @pytest.mark.unit
    def test_single_point_pareto(self):
        """Test Pareto frontier with single point"""
        accuracy_values = [0.8]
        fairness_values = [0.3]
        lambda_values = [0.5]
        
        result = compute_fairness_tradeoff_curve(accuracy_values, fairness_values, lambda_values)
        
        assert result['pareto_indices'] == [0]
        assert result['pareto_accuracies'] == [0.8]
        assert result['pareto_fairness'] == [0.3]

class TestMetricsRobustness:
    """Test robustness of metrics calculations"""
    
    @pytest.mark.unit
    def test_metrics_with_nan_values(self):
        """Test metrics handling with NaN values"""
        S = pd.DataFrame({
            'attr': ['A', 'B', 'A', 'B']
        })
        
        y_pred = np.array([0.5, np.nan, 0.3, 0.7])
        
        # Should handle NaN gracefully
        try:
            results = SubgroupFairnessMetrics.demographic_parity(y_pred, S)
            # If it doesn't raise an exception, check results are valid
            for key, value in results.items():
                assert not np.isnan(value), f"Result should not be NaN: {key}={value}"
        except (ValueError, RuntimeError):
            # Acceptable to raise an error for invalid input
            pass
    
    @pytest.mark.unit
    def test_metrics_with_infinite_values(self):
        """Test metrics handling with infinite values"""
        S = pd.DataFrame({
            'attr': ['A', 'B', 'A', 'B']
        })
        
        y_pred = np.array([0.5, np.inf, 0.3, 0.7])
        
        # Should handle inf gracefully
        try:
            results = SubgroupFairnessMetrics.demographic_parity(y_pred, S)
            for key, value in results.items():
                assert np.isfinite(value), f"Result should be finite: {key}={value}"
        except (ValueError, RuntimeError):
            # Acceptable to raise an error for invalid input
            pass
    
    @pytest.mark.unit
    def test_metrics_with_extreme_probabilities(self):
        """Test metrics with extreme probability values"""
        S = pd.DataFrame({
            'attr': ['A', 'B'] * 50
        })
        
        # Extreme probabilities (0 and 1)
        y_pred_proba = np.array([0.0] * 50 + [1.0] * 50)
        
        results = SubgroupFairnessMetrics.sup_ipm_metric(y_pred_proba, S)
        
        # Should handle extreme values
        assert 'sup_ipm' in results
        assert np.isfinite(results['sup_ipm']) or results['sup_ipm'] == float('inf')
    
    @pytest.mark.unit
    def test_metrics_reproducibility(self, sample_binary_data):
        """Test that metrics calculations are reproducible"""
        data = sample_binary_data
        y_true = data['y'].values
        
        # Fixed predictions for reproducibility
        np.random.seed(42)
        y_pred = np.random.choice([0, 1], len(y_true))
        y_pred_proba = np.random.uniform(0, 1, len(y_true))
        
        # Calculate metrics twice
        results1 = evaluate_all_metrics(y_true, y_pred, y_pred_proba, data['S'])
        results2 = evaluate_all_metrics(y_true, y_pred, y_pred_proba, data['S'])
        
        # Should be identical
        for key in results1.keys():
            if key in results2:
                assert results1[key] == results2[key], f"Results should be reproducible: {key}"