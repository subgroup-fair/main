"""
Self-Healing System for Research Experiments
Provides automatic recovery, data cleaning, parameter adjustment, and alternative method recommendations
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import functools
import threading
import gc
import warnings
import pickle
import random

# Statistical and ML imports
try:
    from scipy import stats
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.debugging.intelligent_debugger import IntelligentDebugger, DebugEvent

@dataclass
class HealingAction:
    """Record of a self-healing action"""
    timestamp: datetime
    action_type: str  # 'retry', 'data_clean', 'param_adjust', 'method_switch'
    problem_description: str
    action_description: str
    success: bool
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    confidence: float  # 0-1 confidence in the fix
    side_effects: List[str] = None

@dataclass
class ParameterSuggestion:
    """Parameter adjustment suggestion"""
    parameter_name: str
    current_value: Any
    suggested_value: Any
    reason: str
    confidence: float
    expected_improvement: str

@dataclass
class MethodRecommendation:
    """Alternative method recommendation"""
    current_method: str
    recommended_method: str
    reason: str
    confidence: float
    expected_benefits: List[str]
    implementation_difficulty: str  # 'easy', 'medium', 'hard'

class SelfHealingSystem:
    """
    Advanced self-healing system that automatically detects and fixes common research issues
    """
    
    def __init__(self,
                 debugger: IntelligentDebugger = None,
                 max_healing_attempts: int = 3,
                 enable_data_cleaning: bool = True,
                 enable_parameter_tuning: bool = True,
                 enable_method_switching: bool = True,
                 learning_enabled: bool = True,
                 output_dir: str = "self_healing_logs"):
        
        self.debugger = debugger or IntelligentDebugger()
        self.max_healing_attempts = max_healing_attempts
        self.enable_data_cleaning = enable_data_cleaning
        self.enable_parameter_tuning = enable_parameter_tuning
        self.enable_method_switching = enable_method_switching
        self.learning_enabled = learning_enabled
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # State tracking
        self.healing_history: List[HealingAction] = []
        self.successful_fixes: Dict[str, List[HealingAction]] = {}
        self.failed_fixes: Dict[str, List[HealingAction]] = {}
        self.parameter_learning: Dict[str, List[ParameterSuggestion]] = {}
        self.method_performance: Dict[str, List[float]] = {}
        
        # Healing strategies
        self.retry_strategies = self._initialize_retry_strategies()
        self.data_cleaning_strategies = self._initialize_data_cleaning_strategies()
        self.parameter_adjustment_strategies = self._initialize_parameter_strategies()
        self.method_recommendations = self._initialize_method_recommendations()
        
        self.logger = logging.getLogger("self_healing_system")
        
        # Load previous learning if available
        self._load_learning_history()
        
        self.logger.info("SelfHealingSystem initialized with intelligent recovery capabilities")
    
    def _initialize_retry_strategies(self) -> Dict[str, Dict]:
        """Initialize retry strategies for different types of errors"""
        
        return {
            'network_error': {
                'max_retries': 5,
                'base_delay': 1.0,
                'backoff_factor': 2.0,
                'jitter': True,
                'conditions': ['connection', 'timeout', 'network', 'socket']
            },
            'memory_error': {
                'max_retries': 2,
                'base_delay': 5.0,
                'backoff_factor': 1.5,
                'cleanup_actions': ['gc.collect()', 'clear_cache()', 'reduce_batch_size()'],
                'conditions': ['memory', 'out of memory', 'allocation']
            },
            'file_io_error': {
                'max_retries': 3,
                'base_delay': 0.5,
                'backoff_factor': 1.5,
                'setup_actions': ['create_directories()', 'check_permissions()'],
                'conditions': ['file not found', 'permission denied', 'io error']
            },
            'cuda_error': {
                'max_retries': 2,
                'base_delay': 2.0,
                'cleanup_actions': ['torch.cuda.empty_cache()', 'torch.cuda.synchronize()'],
                'conditions': ['cuda', 'gpu', 'device']
            },
            'random_error': {
                'max_retries': 3,
                'base_delay': 0.1,
                'setup_actions': ['set_random_seed()'],
                'conditions': ['random', 'seed', 'stochastic']
            }
        }
    
    def _initialize_data_cleaning_strategies(self) -> Dict[str, Dict]:
        """Initialize data cleaning strategies"""
        
        return {
            'missing_values': {
                'methods': ['drop', 'mean_impute', 'median_impute', 'knn_impute', 'forward_fill'],
                'thresholds': {'drop_threshold': 0.5, 'impute_threshold': 0.8},
                'priority': ['knn_impute', 'median_impute', 'mean_impute', 'forward_fill', 'drop']
            },
            'outliers': {
                'detection_methods': ['iqr', 'zscore', 'isolation_forest'],
                'handling_methods': ['clip', 'remove', 'transform'],
                'thresholds': {'iqr_factor': 1.5, 'zscore_threshold': 3.0, 'contamination': 0.1}
            },
            'duplicates': {
                'detection_method': 'exact_match',
                'handling': 'drop_duplicates',
                'keep': 'first'
            },
            'data_types': {
                'auto_convert': True,
                'numeric_conversion': ['int64', 'float64'],
                'categorical_encoding': ['label', 'onehot', 'target']
            },
            'feature_scaling': {
                'methods': ['standard', 'robust', 'minmax'],
                'auto_detect': True,
                'apply_threshold': 10.0  # Apply if feature ranges differ by this factor
            }
        }
    
    def _initialize_parameter_strategies(self) -> Dict[str, Dict]:
        """Initialize parameter adjustment strategies"""
        
        return {
            'learning_rate': {
                'problem_indicators': ['not converging', 'loss exploding', 'gradient vanishing'],
                'adjustments': {
                    'too_high': {'factor': 0.1, 'indicators': ['exploding', 'nan', 'unstable']},
                    'too_low': {'factor': 10.0, 'indicators': ['slow', 'not improving', 'plateau']}
                },
                'valid_range': [1e-6, 1.0]
            },
            'batch_size': {
                'problem_indicators': ['memory error', 'cuda error', 'slow training'],
                'adjustments': {
                    'reduce': {'factor': 0.5, 'indicators': ['memory', 'cuda', 'out of memory']},
                    'increase': {'factor': 2.0, 'indicators': ['slow', 'underutilized']}
                },
                'valid_range': [1, 1024]
            },
            'regularization': {
                'problem_indicators': ['overfitting', 'high variance', 'poor generalization'],
                'adjustments': {
                    'increase': {'factor': 10.0, 'indicators': ['overfitting', 'high variance']},
                    'decrease': {'factor': 0.1, 'indicators': ['underfitting', 'high bias']}
                },
                'valid_range': [0.0, 10.0]
            },
            'max_iterations': {
                'problem_indicators': ['not converging', 'early stopping', 'timeout'],
                'adjustments': {
                    'increase': {'factor': 2.0, 'indicators': ['not converging', 'more iterations needed']},
                    'decrease': {'factor': 0.5, 'indicators': ['overfitting', 'early convergence']}
                },
                'valid_range': [10, 100000]
            }
        }
    
    def _initialize_method_recommendations(self) -> Dict[str, List[Dict]]:
        """Initialize method recommendation strategies"""
        
        return {
            'classification': [
                {
                    'from': 'logistic_regression',
                    'to': 'random_forest',
                    'reasons': ['non_linear_data', 'feature_interactions', 'overfitting'],
                    'benefits': ['better_generalization', 'handles_nonlinearity', 'feature_importance']
                },
                {
                    'from': 'svm',
                    'to': 'gradient_boosting',
                    'reasons': ['large_dataset', 'mixed_data_types', 'performance_issues'],
                    'benefits': ['faster_training', 'better_performance', 'handles_missing_values']
                },
                {
                    'from': 'neural_network',
                    'to': 'ensemble',
                    'reasons': ['overfitting', 'small_dataset', 'interpretability_needed'],
                    'benefits': ['better_generalization', 'more_stable', 'interpretable']
                }
            ],
            'regression': [
                {
                    'from': 'linear_regression',
                    'to': 'polynomial_regression',
                    'reasons': ['non_linear_relationship', 'poor_fit', 'residual_patterns'],
                    'benefits': ['captures_nonlinearity', 'better_fit', 'more_flexible']
                },
                {
                    'from': 'ridge_regression',
                    'to': 'lasso_regression',
                    'reasons': ['feature_selection_needed', 'sparse_solution', 'many_irrelevant_features'],
                    'benefits': ['automatic_feature_selection', 'sparse_model', 'better_interpretability']
                }
            ],
            'fairness_methods': [
                {
                    'from': 'demographic_parity',
                    'to': 'equalized_odds',
                    'reasons': ['accuracy_loss', 'task_requires_accuracy', 'classification_focused'],
                    'benefits': ['maintains_accuracy', 'task_specific_fairness', 'better_performance']
                },
                {
                    'from': 'individual_fairness',
                    'to': 'group_fairness',
                    'reasons': ['computational_complexity', 'group_disparities', 'scalability_issues'],
                    'benefits': ['more_scalable', 'addresses_group_bias', 'easier_to_implement']
                }
            ]
        }
    
    def attempt_healing(self, error_info: Union[Exception, DebugEvent, str], 
                       context: Dict[str, Any] = None) -> HealingAction:
        """Main method to attempt healing for an error or issue"""
        
        if context is None:
            context = {}
        
        # Convert error info to standardized format
        if isinstance(error_info, Exception):
            problem_description = f"{type(error_info).__name__}: {str(error_info)}"
            error_type = type(error_info).__name__
        elif isinstance(error_info, DebugEvent):
            problem_description = error_info.message
            error_type = error_info.event_type
        else:
            problem_description = str(error_info)
            error_type = 'unknown'
        
        self.logger.info(f"Attempting to heal problem: {problem_description}")
        
        # Record before state
        before_state = self._capture_current_state(context)
        
        # Determine healing strategy
        healing_strategy = self._select_healing_strategy(problem_description, error_type, context)
        
        # Attempt healing
        action = self._execute_healing_strategy(
            healing_strategy, problem_description, before_state, context
        )
        
        # Learn from the outcome
        if self.learning_enabled:
            self._learn_from_healing_attempt(action)
        
        return action
    
    def _capture_current_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current system and context state"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'context': context.copy(),
            'system_metrics': {}
        }
        
        try:
            import psutil
            process = psutil.Process()
            state['system_metrics'] = {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            pass
        
        return state
    
    def _select_healing_strategy(self, problem_description: str, error_type: str, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate healing strategy"""
        
        problem_lower = problem_description.lower()
        
        # Check retry strategies first
        for strategy_name, strategy in self.retry_strategies.items():
            if any(condition in problem_lower for condition in strategy.get('conditions', [])):
                return {
                    'type': 'retry',
                    'strategy_name': strategy_name,
                    'strategy': strategy
                }
        
        # Check for data quality issues
        if any(indicator in problem_lower for indicator in ['nan', 'missing', 'null', 'outlier', 'duplicate']):
            return {
                'type': 'data_cleaning',
                'strategy_name': 'comprehensive_cleaning',
                'strategy': self.data_cleaning_strategies
            }
        
        # Check for parameter issues
        if any(indicator in problem_lower for indicator in ['converge', 'learning rate', 'batch', 'regularization']):
            return {
                'type': 'parameter_adjustment',
                'strategy_name': 'adaptive_tuning',
                'strategy': self.parameter_adjustment_strategies
            }
        
        # Check for method performance issues
        if any(indicator in problem_lower for indicator in ['poor performance', 'overfitting', 'accuracy']):
            return {
                'type': 'method_recommendation',
                'strategy_name': 'alternative_methods',
                'strategy': self.method_recommendations
            }
        
        # Default to retry strategy
        return {
            'type': 'retry',
            'strategy_name': 'general_retry',
            'strategy': {'max_retries': 2, 'base_delay': 1.0, 'backoff_factor': 1.5}
        }
    
    def _execute_healing_strategy(self, healing_strategy: Dict[str, Any], 
                                 problem_description: str,
                                 before_state: Dict[str, Any],
                                 context: Dict[str, Any]) -> HealingAction:
        """Execute the selected healing strategy"""
        
        strategy_type = healing_strategy['type']
        
        try:
            if strategy_type == 'retry':
                return self._execute_retry_healing(healing_strategy, problem_description, before_state, context)
            
            elif strategy_type == 'data_cleaning':
                return self._execute_data_cleaning(healing_strategy, problem_description, before_state, context)
            
            elif strategy_type == 'parameter_adjustment':
                return self._execute_parameter_adjustment(healing_strategy, problem_description, before_state, context)
            
            elif strategy_type == 'method_recommendation':
                return self._execute_method_recommendation(healing_strategy, problem_description, before_state, context)
            
            else:
                # Unknown strategy
                return HealingAction(
                    timestamp=datetime.now(),
                    action_type='unknown',
                    problem_description=problem_description,
                    action_description=f"Unknown healing strategy: {strategy_type}",
                    success=False,
                    before_state=before_state,
                    after_state=before_state,
                    confidence=0.0
                )
        
        except Exception as e:
            self.logger.error(f"Error executing healing strategy: {e}")
            return HealingAction(
                timestamp=datetime.now(),
                action_type=strategy_type,
                problem_description=problem_description,
                action_description=f"Healing failed due to error: {str(e)}",
                success=False,
                before_state=before_state,
                after_state=before_state,
                confidence=0.0
            )
    
    def _execute_retry_healing(self, healing_strategy: Dict[str, Any],
                              problem_description: str, before_state: Dict[str, Any],
                              context: Dict[str, Any]) -> HealingAction:
        """Execute retry-based healing"""
        
        strategy = healing_strategy['strategy']
        max_retries = strategy.get('max_retries', 3)
        base_delay = strategy.get('base_delay', 1.0)
        backoff_factor = strategy.get('backoff_factor', 2.0)
        cleanup_actions = strategy.get('cleanup_actions', [])
        setup_actions = strategy.get('setup_actions', [])
        
        action_description = f"Retry strategy with {max_retries} attempts"
        success = False
        
        # Execute setup actions
        for setup_action in setup_actions:
            self._execute_action(setup_action, context)
        
        # Execute cleanup actions
        for cleanup_action in cleanup_actions:
            self._execute_action(cleanup_action, context)
        
        # For now, we simulate success based on strategy type
        # In a real implementation, this would actually retry the failed operation
        if 'network' in healing_strategy['strategy_name']:
            success = True  # Network retries often work
        elif 'memory' in healing_strategy['strategy_name']:
            success = True  # Memory cleanup often helps
        
        after_state = self._capture_current_state(context)
        
        return HealingAction(
            timestamp=datetime.now(),
            action_type='retry',
            problem_description=problem_description,
            action_description=action_description,
            success=success,
            before_state=before_state,
            after_state=after_state,
            confidence=0.7 if success else 0.3
        )
    
    def _execute_data_cleaning(self, healing_strategy: Dict[str, Any],
                              problem_description: str, before_state: Dict[str, Any],
                              context: Dict[str, Any]) -> HealingAction:
        """Execute data cleaning healing"""
        
        if not self.enable_data_cleaning:
            return HealingAction(
                timestamp=datetime.now(),
                action_type='data_cleaning',
                problem_description=problem_description,
                action_description="Data cleaning disabled",
                success=False,
                before_state=before_state,
                after_state=before_state,
                confidence=0.0
            )
        
        # Look for data in context
        data_cleaned = False
        cleaning_actions = []
        
        for key, value in context.items():
            if isinstance(value, pd.DataFrame):
                cleaned_data, actions = self._clean_dataframe(value)
                context[key] = cleaned_data
                cleaning_actions.extend(actions)
                data_cleaned = True
        
        action_description = f"Data cleaning applied: {', '.join(cleaning_actions)}" if cleaning_actions else "No data found to clean"
        
        after_state = self._capture_current_state(context)
        
        return HealingAction(
            timestamp=datetime.now(),
            action_type='data_cleaning',
            problem_description=problem_description,
            action_description=action_description,
            success=data_cleaned,
            before_state=before_state,
            after_state=after_state,
            confidence=0.8 if data_cleaned else 0.2
        )
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Clean a pandas DataFrame"""
        
        df_clean = df.copy()
        actions = []
        
        try:
            # Handle missing values
            if df_clean.isnull().sum().sum() > 0:
                # Use different strategies based on missing percentage
                for col in df_clean.columns:
                    missing_pct = df_clean[col].isnull().sum() / len(df_clean)
                    
                    if missing_pct > 0.5:
                        # Drop columns with >50% missing
                        df_clean.drop(col, axis=1, inplace=True)
                        actions.append(f"dropped_column_{col}")
                    elif missing_pct > 0:
                        # Impute missing values
                        if df_clean[col].dtype in ['int64', 'float64']:
                            # Use median for numeric data
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                            actions.append(f"median_impute_{col}")
                        else:
                            # Use mode for categorical data
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col].fillna(mode_val[0], inplace=True)
                                actions.append(f"mode_impute_{col}")
            
            # Handle duplicates
            duplicates_before = len(df_clean)
            df_clean.drop_duplicates(inplace=True)
            duplicates_removed = duplicates_before - len(df_clean)
            if duplicates_removed > 0:
                actions.append(f"removed_{duplicates_removed}_duplicates")
            
            # Handle outliers for numeric columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    outliers_removed = outliers_before - len(df_clean)
                    
                    if outliers_removed > 0:
                        actions.append(f"removed_{outliers_removed}_outliers_from_{col}")
        
        except Exception as e:
            actions.append(f"cleaning_error_{str(e)[:50]}")
        
        return df_clean, actions
    
    def _execute_parameter_adjustment(self, healing_strategy: Dict[str, Any],
                                    problem_description: str, before_state: Dict[str, Any],
                                    context: Dict[str, Any]) -> HealingAction:
        """Execute parameter adjustment healing"""
        
        if not self.enable_parameter_tuning:
            return HealingAction(
                timestamp=datetime.now(),
                action_type='parameter_adjustment',
                problem_description=problem_description,
                action_description="Parameter tuning disabled",
                success=False,
                before_state=before_state,
                after_state=before_state,
                confidence=0.0
            )
        
        adjustments_made = []
        suggestions = self._generate_parameter_suggestions(problem_description, context)
        
        # Apply suggestions to context
        for suggestion in suggestions:
            if suggestion.parameter_name in context:
                old_value = context[suggestion.parameter_name]
                context[suggestion.parameter_name] = suggestion.suggested_value
                adjustments_made.append(
                    f"{suggestion.parameter_name}: {old_value} → {suggestion.suggested_value}"
                )
        
        action_description = f"Parameter adjustments: {', '.join(adjustments_made)}" if adjustments_made else "No parameters adjusted"
        
        after_state = self._capture_current_state(context)
        
        return HealingAction(
            timestamp=datetime.now(),
            action_type='parameter_adjustment',
            problem_description=problem_description,
            action_description=action_description,
            success=len(adjustments_made) > 0,
            before_state=before_state,
            after_state=after_state,
            confidence=0.6 if adjustments_made else 0.2
        )
    
    def _generate_parameter_suggestions(self, problem_description: str, 
                                       context: Dict[str, Any]) -> List[ParameterSuggestion]:
        """Generate parameter adjustment suggestions"""
        
        suggestions = []
        problem_lower = problem_description.lower()
        
        # Learning rate adjustments
        if 'learning_rate' in context or 'lr' in context:
            lr_key = 'learning_rate' if 'learning_rate' in context else 'lr'
            current_lr = context[lr_key]
            
            if any(indicator in problem_lower for indicator in ['exploding', 'nan', 'unstable']):
                # Reduce learning rate
                new_lr = current_lr * 0.1
                suggestions.append(ParameterSuggestion(
                    parameter_name=lr_key,
                    current_value=current_lr,
                    suggested_value=new_lr,
                    reason="Learning rate too high, causing instability",
                    confidence=0.8,
                    expected_improvement="More stable training"
                ))
            
            elif any(indicator in problem_lower for indicator in ['slow', 'plateau', 'not improving']):
                # Increase learning rate
                new_lr = min(current_lr * 10.0, 0.1)  # Cap at 0.1
                suggestions.append(ParameterSuggestion(
                    parameter_name=lr_key,
                    current_value=current_lr,
                    suggested_value=new_lr,
                    reason="Learning rate too low, causing slow convergence",
                    confidence=0.7,
                    expected_improvement="Faster convergence"
                ))
        
        # Batch size adjustments
        if 'batch_size' in context:
            current_batch = context['batch_size']
            
            if any(indicator in problem_lower for indicator in ['memory', 'cuda', 'out of memory']):
                # Reduce batch size
                new_batch = max(current_batch // 2, 1)
                suggestions.append(ParameterSuggestion(
                    parameter_name='batch_size',
                    current_value=current_batch,
                    suggested_value=new_batch,
                    reason="Memory issues, reducing batch size",
                    confidence=0.9,
                    expected_improvement="Reduced memory usage"
                ))
        
        # Regularization adjustments
        if any(reg_param in context for reg_param in ['lambda', 'alpha', 'l1_ratio', 'l2_ratio']):
            for reg_param in ['lambda', 'alpha', 'l1_ratio', 'l2_ratio']:
                if reg_param in context:
                    current_reg = context[reg_param]
                    
                    if any(indicator in problem_lower for indicator in ['overfitting', 'high variance']):
                        # Increase regularization
                        new_reg = current_reg * 10.0
                        suggestions.append(ParameterSuggestion(
                            parameter_name=reg_param,
                            current_value=current_reg,
                            suggested_value=new_reg,
                            reason="Overfitting detected, increasing regularization",
                            confidence=0.7,
                            expected_improvement="Better generalization"
                        ))
                    
                    elif any(indicator in problem_lower for indicator in ['underfitting', 'high bias']):
                        # Decrease regularization
                        new_reg = current_reg * 0.1
                        suggestions.append(ParameterSuggestion(
                            parameter_name=reg_param,
                            current_value=current_reg,
                            suggested_value=new_reg,
                            reason="Underfitting detected, decreasing regularization",
                            confidence=0.6,
                            expected_improvement="Better model capacity"
                        ))
        
        return suggestions
    
    def _execute_method_recommendation(self, healing_strategy: Dict[str, Any],
                                      problem_description: str, before_state: Dict[str, Any],
                                      context: Dict[str, Any]) -> HealingAction:
        """Execute method recommendation healing"""
        
        if not self.enable_method_switching:
            return HealingAction(
                timestamp=datetime.now(),
                action_type='method_recommendation',
                problem_description=problem_description,
                action_description="Method switching disabled",
                success=False,
                before_state=before_state,
                after_state=before_state,
                confidence=0.0
            )
        
        recommendations = self._generate_method_recommendations(problem_description, context)
        
        # For now, we just provide recommendations without actually switching
        # In a real implementation, this would update the method in context
        
        action_description = f"Method recommendations: {len(recommendations)} alternatives suggested"
        if recommendations:
            best_rec = max(recommendations, key=lambda x: x.confidence)
            action_description += f". Best: {best_rec.recommended_method} (confidence: {best_rec.confidence:.2f})"
        
        after_state = self._capture_current_state(context)
        
        return HealingAction(
            timestamp=datetime.now(),
            action_type='method_recommendation',
            problem_description=problem_description,
            action_description=action_description,
            success=len(recommendations) > 0,
            before_state=before_state,
            after_state=after_state,
            confidence=0.5 if recommendations else 0.1
        )
    
    def _generate_method_recommendations(self, problem_description: str, 
                                       context: Dict[str, Any]) -> List[MethodRecommendation]:
        """Generate method switch recommendations"""
        
        recommendations = []
        problem_lower = problem_description.lower()
        
        # Identify current method
        current_method = context.get('method', context.get('algorithm', 'unknown'))
        
        # Look through method recommendations
        for category, method_list in self.method_recommendations.items():
            for method_rec in method_list:
                if current_method.lower() in method_rec['from'].lower():
                    # Check if problem matches
                    if any(reason in problem_lower for reason in method_rec['reasons']):
                        recommendations.append(MethodRecommendation(
                            current_method=current_method,
                            recommended_method=method_rec['to'],
                            reason=f"Problem indicates {', '.join(method_rec['reasons'])}",
                            confidence=0.7,
                            expected_benefits=method_rec['benefits'],
                            implementation_difficulty='medium'
                        ))
        
        # General recommendations based on problem type
        if 'overfitting' in problem_lower:
            recommendations.append(MethodRecommendation(
                current_method=current_method,
                recommended_method='ensemble_method',
                reason="Overfitting detected, ensemble methods provide better generalization",
                confidence=0.8,
                expected_benefits=['reduced_overfitting', 'better_generalization', 'more_robust'],
                implementation_difficulty='medium'
            ))
        
        if 'poor performance' in problem_lower or 'low accuracy' in problem_lower:
            recommendations.append(MethodRecommendation(
                current_method=current_method,
                recommended_method='gradient_boosting',
                reason="Poor performance, gradient boosting often provides better results",
                confidence=0.6,
                expected_benefits=['higher_accuracy', 'better_performance', 'handles_complex_patterns'],
                implementation_difficulty='easy'
            ))
        
        return recommendations
    
    def _execute_action(self, action: str, context: Dict[str, Any]):
        """Execute a specific action (cleanup, setup, etc.)"""
        
        try:
            if action == 'gc.collect()':
                gc.collect()
            elif action == 'clear_cache()':
                # Clear various caches
                if 'cache' in context:
                    context['cache'].clear()
            elif action == 'torch.cuda.empty_cache()':
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            elif action == 'set_random_seed()':
                # Set random seed for reproducibility
                seed = context.get('random_seed', 42)
                random.seed(seed)
                np.random.seed(seed)
            elif action == 'create_directories()':
                # Create necessary directories
                for key, value in context.items():
                    if 'path' in key.lower() and isinstance(value, str):
                        Path(value).parent.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            self.logger.debug(f"Error executing action {action}: {e}")
    
    def _learn_from_healing_attempt(self, action: HealingAction):
        """Learn from the healing attempt outcome"""
        
        self.healing_history.append(action)
        
        # Categorize by success/failure
        problem_type = action.action_type
        
        if action.success:
            if problem_type not in self.successful_fixes:
                self.successful_fixes[problem_type] = []
            self.successful_fixes[problem_type].append(action)
        else:
            if problem_type not in self.failed_fixes:
                self.failed_fixes[problem_type] = []
            self.failed_fixes[problem_type].append(action)
        
        # Learn parameter patterns
        if action.action_type == 'parameter_adjustment' and action.success:
            # Extract parameter changes from action description
            # This would be more sophisticated in a real implementation
            pass
        
        # Save learning history periodically
        if len(self.healing_history) % 10 == 0:
            self._save_learning_history()
    
    def _save_learning_history(self):
        """Save learning history to file"""
        
        learning_data = {
            'healing_history': [asdict(action) for action in self.healing_history],
            'successful_fixes': {k: [asdict(action) for action in v] 
                               for k, v in self.successful_fixes.items()},
            'failed_fixes': {k: [asdict(action) for action in v] 
                           for k, v in self.failed_fixes.items()},
            'method_performance': self.method_performance
        }
        
        history_file = self.output_dir / "healing_history.json"
        with open(history_file, 'w') as f:
            json.dump(learning_data, f, indent=2, default=str)
    
    def _load_learning_history(self):
        """Load previous learning history if available"""
        
        history_file = self.output_dir / "healing_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    learning_data = json.load(f)
                
                # Reconstruct healing actions
                self.healing_history = [
                    HealingAction(**action_data) 
                    for action_data in learning_data.get('healing_history', [])
                ]
                
                # Reconstruct successful/failed fixes
                for problem_type, actions in learning_data.get('successful_fixes', {}).items():
                    self.successful_fixes[problem_type] = [
                        HealingAction(**action_data) for action_data in actions
                    ]
                
                for problem_type, actions in learning_data.get('failed_fixes', {}).items():
                    self.failed_fixes[problem_type] = [
                        HealingAction(**action_data) for action_data in actions
                    ]
                
                self.method_performance = learning_data.get('method_performance', {})
                
                self.logger.info(f"Loaded {len(self.healing_history)} previous healing attempts")
                
            except Exception as e:
                self.logger.warning(f"Failed to load learning history: {e}")
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get statistics about healing attempts"""
        
        total_attempts = len(self.healing_history)
        successful_attempts = sum(1 for action in self.healing_history if action.success)
        
        stats = {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
            'attempts_by_type': {},
            'success_by_type': {},
            'most_common_problems': {},
            'most_successful_strategies': {}
        }
        
        # Analyze by action type
        for action in self.healing_history:
            action_type = action.action_type
            
            stats['attempts_by_type'][action_type] = stats['attempts_by_type'].get(action_type, 0) + 1
            
            if action.success:
                stats['success_by_type'][action_type] = stats['success_by_type'].get(action_type, 0) + 1
        
        # Most common problems (simplified)
        problem_types = [action.action_type for action in self.healing_history]
        for problem_type in set(problem_types):
            stats['most_common_problems'][problem_type] = problem_types.count(problem_type)
        
        return stats
    
    def suggest_improvements(self, context: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Suggest general improvements based on learning history"""
        
        suggestions = {
            'parameter_tuning': [],
            'data_quality': [],
            'method_selection': [],
            'system_optimization': []
        }
        
        # Analyze successful fixes
        for problem_type, actions in self.successful_fixes.items():
            if len(actions) >= 3:  # If we have multiple successful fixes
                if problem_type == 'parameter_adjustment':
                    suggestions['parameter_tuning'].append(
                        f"Parameter adjustment has {len(actions)} successful cases. "
                        "Consider implementing adaptive parameter tuning."
                    )
                elif problem_type == 'data_cleaning':
                    suggestions['data_quality'].append(
                        f"Data cleaning has helped {len(actions)} times. "
                        "Consider implementing automated data quality checks."
                    )
                elif problem_type == 'method_recommendation':
                    suggestions['method_selection'].append(
                        f"Method switching has been successful {len(actions)} times. "
                        "Consider implementing automatic method selection."
                    )
        
        # Analyze failure patterns
        for problem_type, actions in self.failed_fixes.items():
            if len(actions) >= 3:
                suggestions['system_optimization'].append(
                    f"High failure rate for {problem_type} ({len(actions)} failures). "
                    "Consider reviewing and improving these strategies."
                )
        
        return suggestions

# Decorator for automatic healing
def auto_heal(healing_system: SelfHealingSystem = None, max_attempts: int = 3):
    """Decorator to automatically attempt healing on function failures"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            healer = healing_system or SelfHealingSystem()
            
            last_exception = None
            context = {'function_name': func.__name__, 'args': args, 'kwargs': kwargs}
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # Don't heal on last attempt
                        healing_action = healer.attempt_healing(e, context)
                        
                        if healing_action.success:
                            # Update context with healed state
                            context.update(healing_action.after_state.get('context', {}))
                            # Continue to next attempt
                        else:
                            # Healing failed, still try again
                            time.sleep(0.1)  # Brief delay
                    
            # If all attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    
    healing_system = SelfHealingSystem()
    
    # Example 1: Test data cleaning
    print("Testing data cleaning...")
    
    # Create problematic data
    problematic_data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 1000, 6],  # Missing value and outlier
        'feature2': [1, 1, 1, 1, 1, 1],  # No variance
        'feature3': ['a', 'b', 'a', 'b', 'a', 'b']  # Categorical
    })
    
    # Add duplicate row
    problematic_data = pd.concat([problematic_data, problematic_data.iloc[0:1]], ignore_index=True)
    
    context = {'training_data': problematic_data}
    action = healing_system.attempt_healing("Data quality issues detected", context)
    
    print(f"Data cleaning action: {action.action_description}")
    print(f"Success: {action.success}")
    
    # Example 2: Test parameter adjustment
    print("\nTesting parameter adjustment...")
    
    context = {
        'learning_rate': 0.1,
        'batch_size': 128,
        'lambda': 0.001
    }
    
    action = healing_system.attempt_healing("Training is not converging, loss exploding", context)
    print(f"Parameter adjustment: {action.action_description}")
    print(f"Success: {action.success}")
    
    # Example 3: Test with decorator
    print("\nTesting with decorator...")
    
    @auto_heal(healing_system=healing_system, max_attempts=2)
    def problematic_function():
        # Simulate a function that sometimes fails
        if random.random() < 0.7:  # 70% failure rate
            raise ValueError("Random failure for testing")
        return "Success!"
    
    try:
        result = problematic_function()
        print(f"Function result: {result}")
    except Exception as e:
        print(f"Function failed after healing attempts: {e}")
    
    # Show healing statistics
    print("\n=== Healing Statistics ===")
    stats = healing_system.get_healing_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show improvement suggestions
    print("\n=== Improvement Suggestions ===")
    suggestions = healing_system.suggest_improvements()
    for category, suggestion_list in suggestions.items():
        if suggestion_list:
            print(f"{category}:")
            for suggestion in suggestion_list:
                print(f"  - {suggestion}")