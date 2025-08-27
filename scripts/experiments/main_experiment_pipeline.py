#!/usr/bin/env python3
"""
Main Experimental Pipeline for Subgroup Fairness Research
Author: Kyungseon Lee
Date: August 26, 2025
"""

import argparse
import logging
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from experimental_config import (
    ExperimentType, ExperimentParams, BaselineConfig, 
    MetricsConfig, EXPERIMENT_CONFIGS
)
from scripts.utils.logger import setup_logging
from scripts.utils.data_loader import load_datasets
from scripts.utils.metrics import evaluate_all_metrics
from scripts.baselines.doubly_regressing import DoublyRegressingFairness
from scripts.baselines.kearns_subipm import KearnsSubgroupFairness
from scripts.baselines.baseline_factory import create_baseline_method

class ExperimentRunner:
    """Main class for running subgroup fairness experiments"""
    
    def __init__(self, output_dir: str = "results", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            log_file=self.output_dir / "experiment.log",
            level=log_level
        )
        
        # Initialize results storage
        self.results = {}
        
        self.logger.info("ExperimentRunner initialized")
    
    def run_experiment(self, experiment_type: ExperimentType, 
                      params: ExperimentParams = None) -> Dict[str, Any]:
        """Run a single experiment"""
        
        if params is None:
            params = ExperimentParams()
            
        config = EXPERIMENT_CONFIGS[experiment_type]
        self.logger.info(f"Starting experiment: {experiment_type.value}")
        
        # Create experiment output directory
        exp_dir = self.output_dir / experiment_type.value
        exp_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        results = {
            'experiment_type': experiment_type.value,
            'config': config,
            'results': {},
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'parameters': params.__dict__
            }
        }
        
        # Load datasets
        datasets = self._load_experiment_datasets(config['datasets'])
        
        # Run experiment based on type
        if experiment_type == ExperimentType.ACCURACY_FAIRNESS_TRADEOFF:
            results['results'] = self._run_accuracy_fairness_experiment(
                datasets, config, params, exp_dir
            )
        elif experiment_type == ExperimentType.COMPUTATIONAL_EFFICIENCY:
            results['results'] = self._run_computational_experiment(
                datasets, config, params, exp_dir
            )
        elif experiment_type == ExperimentType.PARTIAL_VS_COMPLETE:
            results['results'] = self._run_partial_complete_experiment(
                datasets, config, params, exp_dir
            )
        elif experiment_type == ExperimentType.MARGINAL_VS_INTERSECTIONAL:
            results['results'] = self._run_marginal_intersectional_experiment(
                datasets, config, params, exp_dir
            )
        elif experiment_type == ExperimentType.SCALABILITY_ROBUSTNESS:
            results['results'] = self._run_scalability_experiment(
                datasets, config, params, exp_dir
            )
        
        # Record completion
        results['metadata']['end_time'] = datetime.now().isoformat()
        results['metadata']['duration'] = time.time() - start_time
        
        # Save results
        self._save_results(results, exp_dir / "results.json")
        
        self.logger.info(f"Experiment {experiment_type.value} completed in {results['metadata']['duration']:.2f}s")
        return results
    
    def _run_accuracy_fairness_experiment(self, datasets: Dict, config: Dict, 
                                        params: ExperimentParams, output_dir: Path) -> Dict:
        """Experiment 1: Accuracy vs Fairness Trade-off Analysis"""
        
        results = {'datasets': {}}
        
        for dataset_name, dataset_data in datasets.items():
            self.logger.info(f"Running accuracy-fairness experiment on {dataset_name}")
            dataset_results = {'methods': {}}
            
            # For each baseline method
            for method_name in config['methods']:
                method_results = {'lambda_sweep': []}
                
                # For each lambda value
                for lambda_val in config['lambda_values']:
                    self.logger.info(f"Testing {method_name} with λ={lambda_val}")
                    
                    # Run multiple seeds for statistical significance
                    seed_results = []
                    for seed in params.random_seeds:
                        # Create and train model
                        model = create_baseline_method(method_name, 
                                                     lambda_val=lambda_val, 
                                                     random_state=seed)
                        
                        # Train model
                        train_metrics = self._train_and_evaluate(
                            model, dataset_data['train'], dataset_data['test'],
                            dataset_data['sensitive_attrs']
                        )
                        
                        seed_results.append(train_metrics)
                    
                    # Aggregate results across seeds
                    aggregated = self._aggregate_seed_results(seed_results)
                    aggregated['lambda'] = lambda_val
                    method_results['lambda_sweep'].append(aggregated)
                
                dataset_results['methods'][method_name] = method_results
            
            results['datasets'][dataset_name] = dataset_results
            
            # Generate plots for this dataset
            self._generate_accuracy_fairness_plots(dataset_results, 
                                                 output_dir / f"{dataset_name}_plots")
        
        return results
    
    def _run_computational_experiment(self, datasets: Dict, config: Dict,
                                    params: ExperimentParams, output_dir: Path) -> Dict:
        """Experiment 2: Computational Efficiency Comparison"""
        
        results = {'sensitivity_analysis': {}}
        
        # Test scalability with varying number of sensitive attributes
        for n_attrs in config['sensitive_attr_counts']:
            self.logger.info(f"Testing computational efficiency with {n_attrs} sensitive attributes")
            
            # Generate synthetic data with n_attrs sensitive attributes
            synthetic_data = self._generate_synthetic_data(
                n_samples=10000, n_sensitive_attrs=n_attrs
            )
            
            attr_results = {'methods': {}}
            
            for method_name in config['methods']:
                self.logger.info(f"Benchmarking {method_name}")
                
                # Measure computational metrics
                comp_metrics = self._benchmark_computational_efficiency(
                    method_name, synthetic_data, n_attrs
                )
                
                attr_results['methods'][method_name] = comp_metrics
            
            results['sensitivity_analysis'][n_attrs] = attr_results
        
        # Generate computational efficiency plots
        self._generate_computational_plots(results, output_dir / "computational_plots")
        
        return results
    
    def _run_partial_complete_experiment(self, datasets: Dict, config: Dict,
                                       params: ExperimentParams, output_dir: Path) -> Dict:
        """Experiment 3: Partial vs Complete Subgroup Fairness"""
        
        results = {'threshold_analysis': {}}
        
        dataset_name = list(datasets.keys())[0]  # Use first dataset
        dataset_data = datasets[dataset_name]
        
        for n_low in config['n_low_values']:
            self.logger.info(f"Testing partial fairness with n_low={n_low}")
            
            threshold_results = {'coverage_analysis': []}
            
            # Run multiple seeds
            for seed in params.random_seeds:
                # Create model with threshold
                model = DoublyRegressingFairness(n_low=n_low, random_state=seed)
                
                # Train and evaluate
                metrics = self._train_and_evaluate(
                    model, dataset_data['train'], dataset_data['test'],
                    dataset_data['sensitive_attrs'], 
                    include_coverage=True
                )
                
                threshold_results['coverage_analysis'].append(metrics)
            
            # Aggregate results
            results['threshold_analysis'][n_low] = self._aggregate_seed_results(
                threshold_results['coverage_analysis']
            )
        
        # Generate coverage analysis plots
        self._generate_coverage_plots(results, output_dir / "coverage_plots")
        
        return results
    
    def _run_marginal_intersectional_experiment(self, datasets: Dict, config: Dict,
                                              params: ExperimentParams, output_dir: Path) -> Dict:
        """Experiment 4: Marginal vs Intersectional Fairness"""
        
        results = {'fairness_comparison': {}}
        
        for dataset_name, dataset_data in datasets.items():
            self.logger.info(f"Comparing marginal vs intersectional fairness on {dataset_name}")
            
            dataset_results = {'methods': {}}
            
            for method_name in config['methods']:
                # Train model
                model = create_baseline_method(method_name)
                
                # Evaluate both marginal and intersectional fairness
                fairness_metrics = self._evaluate_fairness_types(
                    model, dataset_data['train'], dataset_data['test'],
                    dataset_data['sensitive_attrs']
                )
                
                dataset_results['methods'][method_name] = fairness_metrics
            
            results['fairness_comparison'][dataset_name] = dataset_results
        
        # Generate fairness comparison plots
        self._generate_fairness_comparison_plots(results, 
                                               output_dir / "fairness_plots")
        
        return results
    
    def _run_scalability_experiment(self, datasets: Dict, config: Dict,
                                  params: ExperimentParams, output_dir: Path) -> Dict:
        """Experiment 5: Scalability and Robustness Analysis"""
        
        results = {'scalability': {}, 'robustness': {}}
        
        # Scalability analysis
        for sample_size in config['sample_sizes']:
            self.logger.info(f"Testing scalability with {sample_size} samples")
            
            synthetic_data = self._generate_synthetic_data(
                n_samples=sample_size, n_sensitive_attrs=4
            )
            
            size_results = {'methods': {}}
            
            for method_name in config['methods']:
                scalability_metrics = self._benchmark_scalability(
                    method_name, synthetic_data, sample_size
                )
                
                size_results['methods'][method_name] = scalability_metrics
            
            results['scalability'][sample_size] = size_results
        
        # Robustness analysis
        for noise_level in config['noise_levels']:
            self.logger.info(f"Testing robustness with noise level {noise_level}")
            
            noisy_data = self._generate_synthetic_data(
                n_samples=10000, n_sensitive_attrs=4, noise_level=noise_level
            )
            
            noise_results = {'methods': {}}
            
            for method_name in config['methods']:
                robustness_metrics = self._benchmark_robustness(
                    method_name, noisy_data, noise_level
                )
                
                noise_results['methods'][method_name] = robustness_metrics
            
            results['robustness'][noise_level] = noise_results
        
        # Generate scalability plots
        self._generate_scalability_plots(results, output_dir / "scalability_plots")
        
        return results
    
    def _train_and_evaluate(self, model, train_data: pd.DataFrame, 
                           test_data: pd.DataFrame, sensitive_attrs: List[str],
                           include_coverage: bool = False) -> Dict[str, float]:
        """Train model and compute all evaluation metrics"""
        
        # Extract features and labels
        X_train = train_data.drop(['target'], axis=1)
        y_train = train_data['target']
        X_test = test_data.drop(['target'], axis=1)
        y_test = test_data['target']
        
        # Extract sensitive attributes
        S_train = train_data[sensitive_attrs]
        S_test = test_data[sensitive_attrs]
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train, S_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Compute all metrics
        metrics = evaluate_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            sensitive_attrs=S_test,
            include_coverage=include_coverage
        )
        
        metrics['training_time'] = training_time
        
        return metrics
    
    def _aggregate_seed_results(self, seed_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across multiple random seeds"""
        
        if not seed_results:
            return {}
        
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for result in seed_results:
            all_keys.update(result.keys())
        
        # Compute mean and std for each metric
        for key in all_keys:
            values = [result.get(key, np.nan) for result in seed_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_values"] = values
        
        return aggregated
    
    def _load_experiment_datasets(self, dataset_configs: List[Dict]) -> Dict[str, Any]:
        """Load datasets specified in experiment config"""
        datasets = {}
        
        for config in dataset_configs:
            dataset_name = config['name']
            self.logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                dataset_data = load_datasets(config)
                datasets[dataset_name] = dataset_data
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        return datasets
    
    def _save_results(self, results: Dict, filepath: Path):
        """Save results to JSON file"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    # Placeholder methods for specific implementations
    def _generate_synthetic_data(self, n_samples: int, n_sensitive_attrs: int, 
                               noise_level: float = 0.0) -> pd.DataFrame:
        """Generate synthetic dataset - to be implemented"""
        pass
    
    def _benchmark_computational_efficiency(self, method_name: str, 
                                          data: pd.DataFrame, n_attrs: int) -> Dict:
        """Benchmark computational efficiency - to be implemented"""
        pass
    
    def _generate_accuracy_fairness_plots(self, results: Dict, output_dir: Path):
        """Generate accuracy vs fairness plots - to be implemented"""
        pass
    
    def _generate_computational_plots(self, results: Dict, output_dir: Path):
        """Generate computational efficiency plots - to be implemented"""
        pass
    
    def _generate_coverage_plots(self, results: Dict, output_dir: Path):
        """Generate coverage analysis plots - to be implemented"""
        pass
    
    def _generate_fairness_comparison_plots(self, results: Dict, output_dir: Path):
        """Generate fairness comparison plots - to be implemented"""
        pass
    
    def _generate_scalability_plots(self, results: Dict, output_dir: Path):
        """Generate scalability plots - to be implemented"""
        pass

def main():
    """Main entry point for experiment pipeline"""
    
    parser = argparse.ArgumentParser(description="Run Subgroup Fairness Experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=[e.value for e in ExperimentType],
                       help="Experiment type to run")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--config-file", type=str, 
                       help="Custom configuration file (optional)")
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir, log_level=args.log_level)
    
    # Parse experiment type
    experiment_type = ExperimentType(args.experiment)
    
    # Load custom config if provided
    params = ExperimentParams()
    if args.config_file:
        # Load custom parameters - to be implemented
        pass
    
    # Run experiment
    try:
        results = runner.run_experiment(experiment_type, params)
        print(f"Experiment {experiment_type.value} completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())