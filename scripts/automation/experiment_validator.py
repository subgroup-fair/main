"""
Experiment Validation System
Validates inputs, configurations, and results against expected outcomes
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import jsonschema
from dataclasses import asdict, is_dataclass

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from experimental_config import ExperimentType, ExperimentParams, EXPERIMENT_CONFIGS

class ExperimentValidator:
    """
    Comprehensive validation system for experiments
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger("experiment_validator")
        
        # Load validation schemas
        self.schemas = self._load_validation_schemas()
        
        # Load expected outcome patterns
        self.expected_outcomes = self._load_expected_outcomes()
        
        # Load custom validation rules
        if config_path:
            self.custom_rules = self._load_custom_rules(config_path)
        else:
            self.custom_rules = {}
        
        self.logger.info("ExperimentValidator initialized")
    
    def _load_validation_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for validation"""
        
        # Schema for experiment job
        job_schema = {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "minLength": 1},
                "experiment_type": {"type": "string"},
                "params": {"type": "object"},
                "priority": {"type": "integer", "minimum": 0, "maximum": 10},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 10},
                "timeout_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                "resource_limits": {
                    "type": "object",
                    "properties": {
                        "max_memory_gb": {"type": "number", "minimum": 0.1},
                        "max_cpu_percent": {"type": "number", "minimum": 1, "maximum": 100},
                        "max_disk_gb": {"type": "number", "minimum": 0.1}
                    }
                }
            },
            "required": ["job_id", "experiment_type", "params"],
            "additionalProperties": True
        }
        
        # Schema for experiment results
        results_schema = {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "status": {"type": "string", "enum": ["completed", "failed", "timeout", "interrupted"]},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "results": {"type": "object"},
                "error_message": {"type": ["string", "null"]},
                "resource_usage": {"type": ["object", "null"]},
                "validation_results": {"type": ["object", "null"]}
            },
            "required": ["job_id", "status", "start_time"],
            "additionalProperties": True
        }
        
        # Schema for experiment parameters
        params_schema = {
            "type": "object",
            "properties": {
                "lambda_values": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 10}
                },
                "n_low_values": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1}
                },
                "max_iterations": {"type": "integer", "minimum": 1},
                "cv_folds": {"type": "integer", "minimum": 2, "maximum": 20},
                "random_seeds": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1
                }
            },
            "additionalProperties": True
        }
        
        return {
            "job": job_schema,
            "results": results_schema,
            "params": params_schema
        }
    
    def _load_expected_outcomes(self) -> Dict[str, Dict]:
        """Load expected outcome patterns for different experiments"""
        
        return {
            "accuracy_bounds": {
                "min_accuracy": 0.45,  # Minimum reasonable accuracy
                "max_accuracy": 1.0,
                "typical_range": [0.65, 0.95]
            },
            "fairness_bounds": {
                "min_sup_ipm": 0.0,
                "max_sup_ipm": 2.0,  # Reasonable upper bound
                "good_fairness_threshold": 0.1
            },
            "training_time_bounds": {
                "min_seconds": 1.0,
                "max_seconds": 7200,  # 2 hours max for single experiment
                "typical_range": [10, 600]  # 10s to 10 minutes
            },
            "convergence_patterns": {
                "max_loss_increase_ratio": 0.1,  # Loss shouldn't increase by more than 10%
                "min_improvement_threshold": 1e-6
            }
        }
    
    def _load_custom_rules(self, config_path: Path) -> Dict[str, Any]:
        """Load custom validation rules from configuration file"""
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load custom rules from {config_path}: {e}")
            return {}
    
    def validate_experiment_job(self, job) -> Dict[str, Any]:
        """Validate experiment job configuration"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Convert job to dict if it's a dataclass
            if is_dataclass(job):
                job_dict = asdict(job)
            else:
                job_dict = job.__dict__ if hasattr(job, '__dict__') else dict(job)
            
            # Schema validation
            try:
                jsonschema.validate(job_dict, self.schemas['job'])
                validation_result['checks_performed'].append('schema_validation')
            except jsonschema.ValidationError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Schema validation failed: {e.message}")
            
            # Experiment type validation
            if hasattr(job, 'experiment_type'):
                if job.experiment_type not in ExperimentType:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Invalid experiment type: {job.experiment_type}")
                else:
                    validation_result['checks_performed'].append('experiment_type_validation')
            
            # Parameter validation
            if hasattr(job, 'params'):
                param_validation = self._validate_experiment_params(job.params)
                validation_result['errors'].extend(param_validation['errors'])
                validation_result['warnings'].extend(param_validation['warnings'])
                validation_result['checks_performed'].extend(param_validation['checks_performed'])
                
                if not param_validation['valid']:
                    validation_result['valid'] = False
            
            # Resource limits validation
            if hasattr(job, 'resource_limits') and job.resource_limits:
                resource_validation = self._validate_resource_limits(job.resource_limits)
                validation_result['errors'].extend(resource_validation['errors'])
                validation_result['warnings'].extend(resource_validation['warnings'])
                validation_result['checks_performed'].append('resource_limits_validation')
                
                if not resource_validation['valid']:
                    validation_result['valid'] = False
            
            # Dependency validation
            if hasattr(job, 'dependencies') and job.dependencies:
                dep_validation = self._validate_job_dependencies(job.dependencies)
                validation_result['errors'].extend(dep_validation['errors'])
                validation_result['warnings'].extend(dep_validation['warnings'])
                validation_result['checks_performed'].append('dependency_validation')
                
                if not dep_validation['valid']:
                    validation_result['valid'] = False
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_experiment_params(self, params) -> Dict[str, Any]:
        """Validate experiment parameters"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Convert params to dict
            if is_dataclass(params):
                params_dict = asdict(params)
            else:
                params_dict = params.__dict__ if hasattr(params, '__dict__') else dict(params)
            
            # Schema validation for params
            try:
                jsonschema.validate(params_dict, self.schemas['params'])
                validation_result['checks_performed'].append('params_schema_validation')
            except jsonschema.ValidationError as e:
                validation_result['warnings'].append(f"Parameter schema warning: {e.message}")
            
            # Lambda values validation
            if 'lambda_values' in params_dict and params_dict['lambda_values']:
                lambda_vals = params_dict['lambda_values']
                if not isinstance(lambda_vals, list) or not lambda_vals:
                    validation_result['errors'].append("lambda_values must be a non-empty list")
                    validation_result['valid'] = False
                elif any(val < 0 for val in lambda_vals):
                    validation_result['errors'].append("lambda_values must be non-negative")
                    validation_result['valid'] = False
                elif len(lambda_vals) > 20:
                    validation_result['warnings'].append("Large number of lambda values may slow down experiments")
                
                validation_result['checks_performed'].append('lambda_values_validation')
            
            # Random seeds validation
            if 'random_seeds' in params_dict and params_dict['random_seeds']:
                seeds = params_dict['random_seeds']
                if not isinstance(seeds, list) or not seeds:
                    validation_result['errors'].append("random_seeds must be a non-empty list")
                    validation_result['valid'] = False
                elif len(seeds) < 3:
                    validation_result['warnings'].append("Fewer than 3 random seeds may affect statistical reliability")
                elif len(seeds) > 10:
                    validation_result['warnings'].append("Many random seeds will increase computation time")
                
                validation_result['checks_performed'].append('random_seeds_validation')
            
            # Iterations validation
            if 'max_iterations' in params_dict:
                max_iter = params_dict['max_iterations']
                if max_iter < 10:
                    validation_result['warnings'].append("Very low max_iterations may prevent convergence")
                elif max_iter > 50000:
                    validation_result['warnings'].append("Very high max_iterations may cause long training times")
                
                validation_result['checks_performed'].append('max_iterations_validation')
            
            # CV folds validation
            if 'cv_folds' in params_dict:
                cv_folds = params_dict['cv_folds']
                if cv_folds < 2:
                    validation_result['errors'].append("cv_folds must be at least 2")
                    validation_result['valid'] = False
                elif cv_folds > 10:
                    validation_result['warnings'].append("High cv_folds value will increase computation time")
                
                validation_result['checks_performed'].append('cv_folds_validation')
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Parameter validation error: {str(e)}")
        
        return validation_result
    
    def _validate_resource_limits(self, resource_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource limit configuration"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Memory limit validation
        if 'max_memory_gb' in resource_limits:
            memory_limit = resource_limits['max_memory_gb']
            if memory_limit < 0.5:
                validation_result['errors'].append("max_memory_gb too low (< 0.5 GB)")
                validation_result['valid'] = False
            elif memory_limit > 64:
                validation_result['warnings'].append("Very high memory limit (> 64 GB)")
        
        # CPU limit validation
        if 'max_cpu_percent' in resource_limits:
            cpu_limit = resource_limits['max_cpu_percent']
            if cpu_limit < 10:
                validation_result['warnings'].append("Very low CPU limit may slow experiments")
            elif cpu_limit > 100:
                validation_result['errors'].append("CPU limit cannot exceed 100%")
                validation_result['valid'] = False
        
        # Disk limit validation
        if 'max_disk_gb' in resource_limits:
            disk_limit = resource_limits['max_disk_gb']
            if disk_limit < 1:
                validation_result['errors'].append("max_disk_gb too low (< 1 GB)")
                validation_result['valid'] = False
        
        return validation_result
    
    def _validate_job_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Validate job dependencies"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if len(dependencies) > 10:
            validation_result['warnings'].append("Many dependencies may create complex execution order")
        
        # Check for potential circular dependencies (simplified)
        if len(dependencies) != len(set(dependencies)):
            validation_result['errors'].append("Duplicate dependencies detected")
            validation_result['valid'] = False
        
        return validation_result
    
    def validate_experiment_results(self, result) -> Dict[str, Any]:
        """Validate experiment results against expected outcomes"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': [],
            'quality_scores': {}
        }
        
        try:
            # Convert result to dict if needed
            if is_dataclass(result):
                result_dict = asdict(result)
            else:
                result_dict = result.__dict__ if hasattr(result, '__dict__') else dict(result)
            
            # Schema validation
            try:
                jsonschema.validate(result_dict, self.schemas['results'])
                validation_result['checks_performed'].append('results_schema_validation')
            except jsonschema.ValidationError as e:
                validation_result['warnings'].append(f"Result schema warning: {e.message}")
            
            # Status validation
            if result_dict.get('status') == 'completed':
                if 'results' not in result_dict or not result_dict['results']:
                    validation_result['errors'].append("Completed job missing results")
                    validation_result['valid'] = False
                else:
                    # Validate experiment metrics
                    metrics_validation = self._validate_experiment_metrics(result_dict['results'])
                    validation_result['errors'].extend(metrics_validation['errors'])
                    validation_result['warnings'].extend(metrics_validation['warnings'])
                    validation_result['checks_performed'].extend(metrics_validation['checks_performed'])
                    validation_result['quality_scores'].update(metrics_validation['quality_scores'])
                    
                    if not metrics_validation['valid']:
                        validation_result['valid'] = False
            
            # Resource usage validation
            if 'resource_usage' in result_dict and result_dict['resource_usage']:
                resource_validation = self._validate_resource_usage(result_dict['resource_usage'])
                validation_result['warnings'].extend(resource_validation['warnings'])
                validation_result['checks_performed'].append('resource_usage_validation')
            
            # Timing validation
            timing_validation = self._validate_experiment_timing(result_dict)
            validation_result['errors'].extend(timing_validation['errors'])
            validation_result['warnings'].extend(timing_validation['warnings'])
            validation_result['checks_performed'].append('timing_validation')
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Result validation error: {str(e)}")
        
        return validation_result
    
    def _validate_experiment_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment metrics against expected bounds"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': [],
            'quality_scores': {}
        }
        
        try:
            # Extract metrics from nested results structure
            all_metrics = self._extract_metrics_from_results(results)
            
            if not all_metrics:
                validation_result['errors'].append("No metrics found in results")
                validation_result['valid'] = False
                return validation_result
            
            # Accuracy validation
            accuracy_scores = [m.get('accuracy_mean', m.get('accuracy', None)) 
                             for m in all_metrics if 'accuracy_mean' in m or 'accuracy' in m]
            
            if accuracy_scores:
                accuracy_validation = self._validate_accuracy_scores(accuracy_scores)
                validation_result['errors'].extend(accuracy_validation['errors'])
                validation_result['warnings'].extend(accuracy_validation['warnings'])
                validation_result['quality_scores']['accuracy'] = accuracy_validation['quality_score']
                validation_result['checks_performed'].append('accuracy_validation')
                
                if not accuracy_validation['valid']:
                    validation_result['valid'] = False
            
            # Fairness validation
            fairness_scores = [m.get('sup_ipm_mean', m.get('sup_ipm', None))
                             for m in all_metrics if 'sup_ipm_mean' in m or 'sup_ipm' in m]
            
            if fairness_scores:
                fairness_validation = self._validate_fairness_scores(fairness_scores)
                validation_result['errors'].extend(fairness_validation['errors'])
                validation_result['warnings'].extend(fairness_validation['warnings'])
                validation_result['quality_scores']['fairness'] = fairness_validation['quality_score']
                validation_result['checks_performed'].append('fairness_validation')
            
            # Training time validation
            training_times = [m.get('training_time_mean', m.get('training_time', None))
                            for m in all_metrics if 'training_time_mean' in m or 'training_time' in m]
            
            if training_times:
                time_validation = self._validate_training_times(training_times)
                validation_result['warnings'].extend(time_validation['warnings'])
                validation_result['quality_scores']['efficiency'] = time_validation['quality_score']
                validation_result['checks_performed'].append('training_time_validation')
            
            # Convergence validation
            if 'metadata' in results and 'training_history' in results.get('metadata', {}):
                conv_validation = self._validate_convergence_patterns(results['metadata']['training_history'])
                validation_result['warnings'].extend(conv_validation['warnings'])
                validation_result['quality_scores']['convergence'] = conv_validation['quality_score']
                validation_result['checks_performed'].append('convergence_validation')
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Metrics validation error: {str(e)}")
        
        return validation_result
    
    def _extract_metrics_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all metrics from nested results structure"""
        
        metrics = []
        
        try:
            if 'datasets' in results:
                for dataset_name, dataset_results in results['datasets'].items():
                    if 'methods' in dataset_results:
                        for method_name, method_results in dataset_results['methods'].items():
                            if 'lambda_sweep' in method_results:
                                for sweep_point in method_results['lambda_sweep']:
                                    metrics.append(sweep_point)
                            elif isinstance(method_results, dict):
                                metrics.append(method_results)
            
            # Also check top-level metrics
            if any(key in results for key in ['accuracy', 'sup_ipm', 'training_time']):
                metrics.append(results)
        
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
        
        return metrics
    
    def _validate_accuracy_scores(self, scores: List[float]) -> Dict[str, Any]:
        """Validate accuracy scores against expected bounds"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        bounds = self.expected_outcomes['accuracy_bounds']
        
        # Remove None values
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        
        if not valid_scores:
            validation_result['errors'].append("No valid accuracy scores found")
            validation_result['valid'] = False
            return validation_result
        
        min_acc = min(valid_scores)
        max_acc = max(valid_scores)
        mean_acc = np.mean(valid_scores)
        
        # Bound checks
        if min_acc < bounds['min_accuracy']:
            validation_result['errors'].append(f"Accuracy too low: {min_acc:.3f} < {bounds['min_accuracy']}")
            validation_result['valid'] = False
        
        if max_acc > bounds['max_accuracy']:
            validation_result['errors'].append(f"Accuracy too high: {max_acc:.3f} > {bounds['max_accuracy']}")
            validation_result['valid'] = False
        
        if mean_acc < bounds['typical_range'][0]:
            validation_result['warnings'].append(f"Mean accuracy below typical range: {mean_acc:.3f}")
        
        # Quality score (0-1, higher is better)
        if bounds['typical_range'][0] <= mean_acc <= bounds['typical_range'][1]:
            validation_result['quality_score'] = 1.0
        else:
            # Penalize for being outside typical range
            distance_from_range = min(abs(mean_acc - bounds['typical_range'][0]),
                                    abs(mean_acc - bounds['typical_range'][1]))
            validation_result['quality_score'] = max(0, 1 - distance_from_range * 2)
        
        return validation_result
    
    def _validate_fairness_scores(self, scores: List[float]) -> Dict[str, Any]:
        """Validate fairness scores (supIPM) against expected bounds"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        bounds = self.expected_outcomes['fairness_bounds']
        
        # Remove None values
        valid_scores = [s for s in scores if s is not None and not np.isnan(s)]
        
        if not valid_scores:
            validation_result['warnings'].append("No fairness scores found")
            return validation_result
        
        min_fair = min(valid_scores)
        max_fair = max(valid_scores)
        mean_fair = np.mean(valid_scores)
        
        # Bound checks
        if min_fair < bounds['min_sup_ipm']:
            validation_result['errors'].append(f"Negative fairness score: {min_fair:.3f}")
            validation_result['valid'] = False
        
        if max_fair > bounds['max_sup_ipm']:
            validation_result['warnings'].append(f"Very high unfairness: {max_fair:.3f}")
        
        # Quality score (lower unfairness is better)
        if mean_fair <= bounds['good_fairness_threshold']:
            validation_result['quality_score'] = 1.0
        else:
            # Penalize high unfairness
            validation_result['quality_score'] = max(0, 1 - (mean_fair - bounds['good_fairness_threshold']) / bounds['max_sup_ipm'])
        
        return validation_result
    
    def _validate_training_times(self, times: List[float]) -> Dict[str, Any]:
        """Validate training times against expected bounds"""
        
        validation_result = {
            'warnings': [],
            'quality_score': 0.0
        }
        
        bounds = self.expected_outcomes['training_time_bounds']
        
        # Remove None values
        valid_times = [t for t in times if t is not None and not np.isnan(t)]
        
        if not valid_times:
            return validation_result
        
        min_time = min(valid_times)
        max_time = max(valid_times)
        mean_time = np.mean(valid_times)
        
        if min_time < bounds['min_seconds']:
            validation_result['warnings'].append(f"Very fast training time: {min_time:.1f}s")
        
        if max_time > bounds['max_seconds']:
            validation_result['warnings'].append(f"Very slow training time: {max_time:.1f}s")
        
        # Quality score (prefer typical range)
        if bounds['typical_range'][0] <= mean_time <= bounds['typical_range'][1]:
            validation_result['quality_score'] = 1.0
        else:
            # Penalize for being outside typical range
            if mean_time < bounds['typical_range'][0]:
                # Too fast might indicate incomplete training
                validation_result['quality_score'] = 0.7
            else:
                # Too slow is inefficient
                excess_ratio = (mean_time - bounds['typical_range'][1]) / bounds['typical_range'][1]
                validation_result['quality_score'] = max(0.1, 1 - excess_ratio)
        
        return validation_result
    
    def _validate_convergence_patterns(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate convergence patterns from training history"""
        
        validation_result = {
            'warnings': [],
            'quality_score': 0.0
        }
        
        try:
            # Look for loss values in history
            loss_keys = ['total_loss', 'classifier_loss', 'objective']
            loss_values = None
            
            for key in loss_keys:
                if key in history and history[key]:
                    loss_values = history[key]
                    break
            
            if not loss_values or len(loss_values) < 5:
                validation_result['quality_score'] = 0.5
                return validation_result
            
            # Check for convergence (decreasing trend)
            initial_loss = np.mean(loss_values[:3])
            final_loss = np.mean(loss_values[-3:])
            
            if final_loss > initial_loss:
                validation_result['warnings'].append("Loss increased during training")
                validation_result['quality_score'] = 0.3
            else:
                # Good convergence
                improvement_ratio = (initial_loss - final_loss) / initial_loss
                validation_result['quality_score'] = min(1.0, improvement_ratio * 5)  # Scale to 0-1
            
            # Check for oscillations
            if len(loss_values) > 10:
                recent_losses = loss_values[-10:]
                oscillation_score = np.std(recent_losses) / np.mean(recent_losses)
                
                if oscillation_score > 0.1:
                    validation_result['warnings'].append("High oscillation in final training phase")
                    validation_result['quality_score'] *= 0.8
        
        except Exception as e:
            self.logger.debug(f"Convergence validation error: {e}")
            validation_result['quality_score'] = 0.5
        
        return validation_result
    
    def _validate_resource_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource usage patterns"""
        
        validation_result = {
            'warnings': []
        }
        
        # Memory usage warnings
        if 'memory_gb' in usage and usage['memory_gb'] > 16:
            validation_result['warnings'].append(f"High memory usage: {usage['memory_gb']:.1f} GB")
        
        # CPU usage warnings
        if 'cpu_percent' in usage and usage['cpu_percent'] > 95:
            validation_result['warnings'].append(f"Very high CPU usage: {usage['cpu_percent']:.1f}%")
        elif 'cpu_percent' in usage and usage['cpu_percent'] < 10:
            validation_result['warnings'].append(f"Very low CPU usage: {usage['cpu_percent']:.1f}%")
        
        return validation_result
    
    def _validate_experiment_timing(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment timing"""
        
        validation_result = {
            'errors': [],
            'warnings': []
        }
        
        try:
            if 'start_time' in result and 'end_time' in result:
                start_time = datetime.fromisoformat(result['start_time'].replace('Z', '+00:00'))
                
                if result['end_time']:
                    end_time = datetime.fromisoformat(result['end_time'].replace('Z', '+00:00'))
                    duration = (end_time - start_time).total_seconds()
                    
                    if duration < 0:
                        validation_result['errors'].append("End time before start time")
                    elif duration < 1:
                        validation_result['warnings'].append(f"Very short experiment duration: {duration:.1f}s")
                    elif duration > 14400:  # 4 hours
                        validation_result['warnings'].append(f"Very long experiment duration: {duration/3600:.1f}h")
        
        except Exception as e:
            validation_result['warnings'].append(f"Could not validate timing: {str(e)}")
        
        return validation_result
    
    def generate_validation_report(self, validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_validations = len(validations)
        valid_count = sum(1 for v in validations if v.get('valid', False))
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        all_checks = set()
        quality_scores = {}
        
        for validation in validations:
            all_errors.extend(validation.get('errors', []))
            all_warnings.extend(validation.get('warnings', []))
            all_checks.update(validation.get('checks_performed', []))
            
            if 'quality_scores' in validation:
                for metric, score in validation['quality_scores'].items():
                    if metric not in quality_scores:
                        quality_scores[metric] = []
                    quality_scores[metric].append(score)
        
        # Calculate average quality scores
        avg_quality_scores = {
            metric: np.mean(scores) for metric, scores in quality_scores.items()
        }
        
        report = {
            'summary': {
                'total_validations': total_validations,
                'valid_count': valid_count,
                'invalid_count': total_validations - valid_count,
                'success_rate': valid_count / total_validations if total_validations > 0 else 0,
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings)
            },
            'quality_metrics': avg_quality_scores,
            'checks_performed': sorted(list(all_checks)),
            'common_issues': self._analyze_common_issues(all_errors, all_warnings),
            'detailed_validations': validations
        }
        
        return report
    
    def _analyze_common_issues(self, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
        """Analyze common issues across validations"""
        
        # Count frequency of different issue types
        error_patterns = {}
        warning_patterns = {}
        
        for error in errors:
            # Simple pattern matching
            if 'accuracy' in error.lower():
                error_patterns['accuracy_issues'] = error_patterns.get('accuracy_issues', 0) + 1
            elif 'schema' in error.lower():
                error_patterns['schema_issues'] = error_patterns.get('schema_issues', 0) + 1
            elif 'resource' in error.lower():
                error_patterns['resource_issues'] = error_patterns.get('resource_issues', 0) + 1
            else:
                error_patterns['other_errors'] = error_patterns.get('other_errors', 0) + 1
        
        for warning in warnings:
            if 'memory' in warning.lower():
                warning_patterns['memory_warnings'] = warning_patterns.get('memory_warnings', 0) + 1
            elif 'time' in warning.lower():
                warning_patterns['timing_warnings'] = warning_patterns.get('timing_warnings', 0) + 1
            elif 'convergence' in warning.lower():
                warning_patterns['convergence_warnings'] = warning_patterns.get('convergence_warnings', 0) + 1
            else:
                warning_patterns['other_warnings'] = warning_patterns.get('other_warnings', 0) + 1
        
        return {
            'error_patterns': error_patterns,
            'warning_patterns': warning_patterns,
            'most_common_error': max(error_patterns.items(), key=lambda x: x[1])[0] if error_patterns else None,
            'most_common_warning': max(warning_patterns.items(), key=lambda x: x[1])[0] if warning_patterns else None
        }

# Convenience functions
def validate_single_experiment(job, result=None) -> Dict[str, Any]:
    """Quick validation of a single experiment"""
    
    validator = ExperimentValidator()
    
    validations = []
    
    # Validate job
    job_validation = validator.validate_experiment_job(job)
    validations.append(job_validation)
    
    # Validate result if provided
    if result:
        result_validation = validator.validate_experiment_results(result)
        validations.append(result_validation)
    
    return validator.generate_validation_report(validations)

def run_validation_suite(jobs: List, results: List = None) -> Dict[str, Any]:
    """Run comprehensive validation on multiple experiments"""
    
    validator = ExperimentValidator()
    all_validations = []
    
    # Validate all jobs
    for job in jobs:
        validation = validator.validate_experiment_job(job)
        all_validations.append(validation)
    
    # Validate results if provided
    if results:
        for result in results:
            validation = validator.validate_experiment_results(result)
            all_validations.append(validation)
    
    return validator.generate_validation_report(all_validations)

if __name__ == "__main__":
    # Example usage
    print("ExperimentValidator - validate experiment configurations and results")
    print("Use validate_single_experiment() or run_validation_suite() for validation")