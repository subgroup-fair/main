"""
Performance benchmark tests for subgroup fairness methods
"""

import pytest
import numpy as np
import pandas as pd
import time
import psutil
import gc
from pathlib import Path
import sys
from unittest.mock import patch

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.baselines.doubly_regressing import DoublyRegressingFairness
from scripts.baselines.kearns_subipm import KearnsSubgroupFairness
from scripts.baselines.baseline_factory import create_baseline_method
from tests.mock_data.generators import generate_biased_dataset, generate_high_dimensional_dataset

class TestPerformanceBenchmarks:
    """Benchmark tests for computational performance"""
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_doubly_regressing_training_time(self, benchmark, performance_data):
        """Benchmark training time for DoublyRegressingFairness"""
        
        data = performance_data
        
        def train_model():
            model = DoublyRegressingFairness(
                lambda_fair=0.5,
                n_low=50,
                max_iterations=100,  # Reasonable for benchmarking
                random_state=42,
                verbose=False
            )
            model.fit(data['X'], data['y'], data['S'])
            return model
        
        # Benchmark the training
        result = benchmark(train_model)
        
        # Basic assertion that model was trained
        assert result.classifier is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_kearns_training_time(self, benchmark, performance_data):
        """Benchmark training time for KearnsSubgroupFairness"""
        
        data = performance_data
        
        def train_model():
            model = KearnsSubgroupFairness(
                gamma=0.1,
                max_iterations=100,
                random_state=42
            )
            model.fit(data['X'], data['y'], data['S'])
            return model
        
        result = benchmark(train_model)
        assert result.base_classifier is not None
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_prediction_time(self, benchmark, trained_doubly_regressing_model, performance_data):
        """Benchmark prediction time"""
        
        model = trained_doubly_regressing_model
        data = performance_data
        
        def predict():
            return model.predict(data['X'])
        
        predictions = benchmark(predict)
        assert len(predictions) == len(data['X'])
    
    @pytest.mark.performance
    @pytest.mark.benchmark  
    def test_probability_prediction_time(self, benchmark, trained_doubly_regressing_model, performance_data):
        """Benchmark probability prediction time"""
        
        model = trained_doubly_regressing_model
        data = performance_data
        
        def predict_proba():
            return model.predict_proba(data['X'])
        
        probabilities = benchmark(predict_proba)
        assert probabilities.shape == (len(data['X']), 2)

class TestScalabilityBenchmarks:
    """Test scalability with different dataset sizes"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scaling_with_sample_size(self):
        """Test how performance scales with dataset size"""
        
        sample_sizes = [1000, 2000, 5000, 10000]
        results = {}
        
        for size in sample_sizes:
            # Generate data of specific size
            dataset = generate_biased_dataset(
                n_samples=size,
                n_features=10,
                bias_strength=0.5,
                random_state=42
            )
            
            # Time the training
            start_time = time.time()
            
            model = DoublyRegressingFairness(
                lambda_fair=0.5,
                n_low=min(50, size // 20),  # Adaptive threshold
                max_iterations=50,  # Fixed for comparison
                random_state=42,
                verbose=False
            )
            
            model.fit(dataset['X'], dataset['y'], dataset['S'])
            
            end_time = time.time()
            training_time = end_time - start_time
            
            results[size] = {
                'training_time': training_time,
                'samples_per_second': size / training_time if training_time > 0 else 0
            }
        
        # Check that scaling is reasonable (not exponential)
        # Training time should roughly scale linearly or sub-quadratically
        times = [results[size]['training_time'] for size in sample_sizes]
        
        # Basic sanity checks
        assert all(t > 0 for t in times), "All training times should be positive"
        
        # Check that larger datasets take longer (monotonic increasing is expected)
        # But allow some variance due to noise
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # Each step should not be more than 10x slower (reasonable scaling)
        assert all(ratio < 10 for ratio in time_ratios), f"Training time scaling too steep: {time_ratios}"
    
    @pytest.mark.performance
    @pytest.mark.slow  
    def test_scaling_with_feature_dimensionality(self):
        """Test how performance scales with number of features"""
        
        feature_counts = [5, 10, 20, 50]
        results = {}
        
        for n_features in feature_counts:
            # Generate high-dimensional data
            dataset = generate_high_dimensional_dataset(
                n_samples=2000,
                n_features=n_features,
                random_state=42
            )
            
            # Limit to relevant features to keep test reasonable
            feature_cols = [col for col in dataset['X'].columns if 'feature' in col][:n_features]
            X_subset = dataset['X'][feature_cols]
            
            start_time = time.time()
            
            model = DoublyRegressingFairness(
                lambda_fair=0.5,
                n_low=50,
                max_iterations=20,  # Reduced for high-dim testing
                random_state=42,
                verbose=False
            )
            
            model.fit(X_subset, dataset['y'], dataset['S'])
            
            end_time = time.time()
            training_time = end_time - start_time
            
            results[n_features] = {
                'training_time': training_time,
                'features_per_second': n_features / training_time if training_time > 0 else 0
            }
        
        # Check scaling behavior
        times = [results[n_feat]['training_time'] for n_feat in feature_counts]
        
        assert all(t > 0 for t in times), "All training times should be positive"
        
        # Features scaling should be reasonable (not exponential)
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        assert all(ratio < 20 for ratio in time_ratios), f"Feature scaling too steep: {time_ratios}"
    
    @pytest.mark.performance
    def test_scaling_with_sensitive_attributes(self):
        """Test how performance scales with number of sensitive attributes"""
        
        sensitive_attr_counts = [2, 3, 4, 5]
        results = {}
        
        for n_attrs in sensitive_attr_counts:
            # Generate synthetic data with varying sensitive attributes
            np.random.seed(42)
            n_samples = 1000
            
            X = pd.DataFrame(np.random.randn(n_samples, 5), 
                           columns=[f'feature_{i}' for i in range(5)])
            y = pd.Series(np.random.choice([0, 1], n_samples))
            
            # Create sensitive attributes
            S_data = {}
            for i in range(n_attrs):
                S_data[f'attr_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
            S = pd.DataFrame(S_data)
            
            start_time = time.time()
            
            model = DoublyRegressingFairness(
                lambda_fair=0.5,
                n_low=30,  # Lower threshold for more subgroups
                max_iterations=20,
                random_state=42,
                verbose=False
            )
            
            model.fit(X, y, S)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            results[n_attrs] = {
                'training_time': training_time,
                'num_covered_subgroups': len(model.covered_subgroups) if hasattr(model, 'covered_subgroups') else 0
            }
        
        # Check that training time increases with more sensitive attributes
        times = [results[n_attrs]['training_time'] for n_attrs in sensitive_attr_counts]
        
        assert all(t > 0 for t in times), "All training times should be positive"
        
        # More sensitive attributes should generally take longer (combinatorial complexity)
        # But allow for some noise in timing
        for i in range(len(times)-1):
            if times[i+1] > times[i] * 15:  # Very generous limit
                pytest.fail(f"Training time increased too much with sensitive attributes: {times}")

class TestMemoryUsageBenchmarks:
    """Test memory usage of algorithms"""
    
    @pytest.mark.performance
    def test_memory_usage_during_training(self, performance_data):
        """Test memory usage during model training"""
        
        data = performance_data
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()  # Force garbage collection
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train model and monitor memory
        model = DoublyRegressingFairness(
            lambda_fair=0.5,
            n_low=50,
            max_iterations=50,
            random_state=42,
            verbose=False
        )
        
        model.fit(data['X'], data['y'], data['S'])
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory usage should be reasonable (less than 1GB increase for 10K samples)
        assert memory_increase < 1024, f"Memory usage too high: {memory_increase:.1f} MB"
        
        # Clean up
        del model
        gc.collect()
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_scaling_with_data_size(self):
        """Test how memory usage scales with dataset size"""
        
        sizes = [1000, 5000, 10000]
        memory_usage = {}
        
        for size in sizes:
            # Generate data
            dataset = generate_biased_dataset(
                n_samples=size,
                n_features=10,
                random_state=42
            )
            
            process = psutil.Process()
            gc.collect()
            
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Train model
            model = DoublyRegressingFairness(
                lambda_fair=0.5,
                n_low=min(50, size // 20),
                max_iterations=20,  # Reduced for memory testing
                random_state=42,
                verbose=False
            )
            
            model.fit(dataset['X'], dataset['y'], dataset['S'])
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - baseline_memory
            
            memory_usage[size] = memory_increase
            
            # Clean up for next iteration
            del model, dataset
            gc.collect()
        
        # Memory should scale reasonably (roughly linear with data size)
        memory_per_sample = {size: mem / size for size, mem in memory_usage.items()}
        
        # All should use reasonable memory per sample
        for size, mem_per_sample in memory_per_sample.items():
            assert mem_per_sample < 1.0, f"Memory per sample too high: {mem_per_sample:.3f} MB/sample for size {size}"

class TestComparativeBenchmarks:
    """Compare performance between different methods"""
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_method_comparison_training_time(self, benchmark, sample_binary_data):
        """Compare training time between different methods"""
        
        data = sample_binary_data
        methods = ['doubly_regressing', 'kearns_supipm', 'fams_fairness']
        results = {}
        
        for method_name in methods:
            def train_method():
                model = create_baseline_method(
                    method_name,
                    lambda_val=0.5,
                    random_state=42
                )
                
                # Adjust parameters for fair comparison
                if hasattr(model, 'max_iterations'):
                    model.max_iterations = 20
                if hasattr(model, 'training_epochs'):
                    model.training_epochs = 5  # For FAMS
                if hasattr(model, 'n_subtasks'):
                    model.n_subtasks = 10  # For FAMS
                if hasattr(model, 'verbose'):
                    model.verbose = False
                
                model.fit(data['X'], data['y'], data['S'])
                return model
            
            # Use pytest-benchmark to time it
            result = benchmark.pedantic(train_method, rounds=3, iterations=1)
            results[method_name] = result
        
        # Both methods should complete successfully
        assert all(hasattr(model, 'fit') for model in results.values())
    
    @pytest.mark.performance
    def test_method_comparison_prediction_speed(self, sample_binary_data):
        """Compare prediction speed between methods"""
        
        data = sample_binary_data
        methods = ['doubly_regressing', 'kearns_supipm', 'fams_fairness']
        
        # Train models first
        trained_models = {}
        for method_name in methods:
            model = create_baseline_method(method_name, random_state=42)
            
            if hasattr(model, 'max_iterations'):
                model.max_iterations = 10  # Quick training
            if hasattr(model, 'training_epochs'):
                model.training_epochs = 3  # For FAMS
            if hasattr(model, 'n_subtasks'):
                model.n_subtasks = 5  # For FAMS
            if hasattr(model, 'verbose'):
                model.verbose = False
            
            model.fit(data['X'], data['y'], data['S'])
            trained_models[method_name] = model
        
        # Compare prediction times
        prediction_times = {}
        
        for method_name, model in trained_models.items():
            start_time = time.time()
            
            # Make multiple predictions to get stable timing
            for _ in range(10):
                predictions = model.predict(data['X'])
            
            end_time = time.time()
            avg_prediction_time = (end_time - start_time) / 10
            
            prediction_times[method_name] = avg_prediction_time
        
        # All prediction times should be reasonable
        for method, pred_time in prediction_times.items():
            assert pred_time < 1.0, f"Prediction time too slow for {method}: {pred_time:.3f}s"
    
    @pytest.mark.performance
    def test_convergence_speed_comparison(self, sample_binary_data):
        """Compare convergence speed between methods"""
        
        data = sample_binary_data
        
        # Test DoublyRegressing convergence
        model_dr = DoublyRegressingFairness(
            lambda_fair=0.5,
            max_iterations=1000,
            tolerance=1e-4,
            random_state=42,
            verbose=False
        )
        
        model_dr.fit(data['X'], data['y'], data['S'])
        dr_history = model_dr.get_training_history()
        dr_iterations = len(dr_history['total_loss'])
        
        # Test Kearns convergence
        model_k = KearnsSubgroupFairness(
            gamma=0.1,
            max_iterations=1000,
            tolerance=1e-4,
            random_state=42
        )
        
        model_k.fit(data['X'], data['y'], data['S'])
        k_history = model_k.get_training_history()
        k_iterations = len(k_history['objective'])
        
        # Both should converge in reasonable time
        assert dr_iterations > 0, "DoublyRegressing should have training history"
        assert k_iterations > 0, "Kearns should have training history"
        assert dr_iterations < 1000, f"DoublyRegressing took too many iterations: {dr_iterations}"
        assert k_iterations < 1000, f"Kearns took too many iterations: {k_iterations}"

class TestRegressionBenchmarks:
    """Test for performance regression detection"""
    
    @pytest.mark.performance
    def test_baseline_performance_regression(self, performance_data):
        """Test that performance hasn't regressed from baseline"""
        
        data = performance_data
        
        # Define performance baseline (these should be updated based on actual measurements)
        PERFORMANCE_BASELINES = {
            'doubly_regressing_training_time': 30.0,  # seconds for 10k samples
            'kearns_training_time': 20.0,  # seconds for 10k samples
            'prediction_time_per_1k_samples': 0.1,  # seconds per 1k predictions
        }
        
        # Test DoublyRegressing training time
        start_time = time.time()
        
        model_dr = DoublyRegressingFairness(
            lambda_fair=0.5,
            n_low=50,
            max_iterations=50,
            random_state=42,
            verbose=False
        )
        model_dr.fit(data['X'], data['y'], data['S'])
        
        dr_training_time = time.time() - start_time
        
        # Test Kearns training time
        start_time = time.time()
        
        model_k = KearnsSubgroupFairness(
            gamma=0.1,
            max_iterations=50,
            random_state=42
        )
        model_k.fit(data['X'], data['y'], data['S'])
        
        k_training_time = time.time() - start_time
        
        # Test prediction time
        start_time = time.time()
        predictions = model_dr.predict(data['X'])
        prediction_time = time.time() - start_time
        prediction_time_per_1k = (prediction_time * 1000) / len(data['X'])
        
        # Check against baselines (allow 50% margin for system variation)
        margin = 1.5
        
        assert dr_training_time < PERFORMANCE_BASELINES['doubly_regressing_training_time'] * margin, \
            f"DoublyRegressing training regression: {dr_training_time:.2f}s > {PERFORMANCE_BASELINES['doubly_regressing_training_time']*margin:.2f}s"
        
        assert k_training_time < PERFORMANCE_BASELINES['kearns_training_time'] * margin, \
            f"Kearns training regression: {k_training_time:.2f}s > {PERFORMANCE_BASELINES['kearns_training_time']*margin:.2f}s"
        
        assert prediction_time_per_1k < PERFORMANCE_BASELINES['prediction_time_per_1k_samples'] * margin, \
            f"Prediction time regression: {prediction_time_per_1k:.3f}s > {PERFORMANCE_BASELINES['prediction_time_per_1k_samples']*margin:.3f}s"