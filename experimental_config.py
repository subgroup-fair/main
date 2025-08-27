"""
Experimental Configuration for Subgroup Fairness Research
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from enum import Enum

class ExperimentType(Enum):
    ACCURACY_FAIRNESS_TRADEOFF = "exp_1_accuracy_fairness"
    COMPUTATIONAL_EFFICIENCY = "exp_2_computational"
    PARTIAL_VS_COMPLETE = "exp_3_partial_complete"
    MARGINAL_VS_INTERSECTIONAL = "exp_4_marginal_intersectional"
    SCALABILITY_ROBUSTNESS = "exp_5_scalability"

class DatasetConfig:
    """Dataset configurations"""
    UCI_ADULT = {
        'name': 'uci_adult',
        'path': 'data/raw/adult.csv',
        'sensitive_attrs': ['age', 'race', 'sex'],
        'target': 'income',
        'sample_sizes': [5000, 10000, 20000, 32561]  # Full dataset size
    }
    
    COMMUNITIES_CRIME = {
        'name': 'communities_crime',
        'path': 'data/raw/communities.csv',
        'sensitive_attrs': ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'],
        'target': 'ViolentCrimesPerPop',
        'sample_sizes': [1000, 1994]  # Full dataset size
    }
    
    SYNTHETIC = {
        'name': 'synthetic',
        'sensitive_attrs_counts': [2, 4, 6, 8, 10],
        'sample_sizes': [1000, 5000, 10000, 50000],
        'noise_levels': [0.0, 0.05, 0.1, 0.2]
    }

@dataclass
class ExperimentParams:
    """Base parameters for all experiments"""
    # Fairness-accuracy trade-off parameters
    lambda_values: List[float] = None  # [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    # Subgroup thresholds
    n_low_values: List[int] = None  # [10, 50, 100, 200]
    
    # Model parameters
    max_iterations: int = 10000
    learning_rates: Dict[str, float] = None  # {'cls': 0.01, 'disc': 0.01, 'weight': 0.01}
    
    # Evaluation parameters
    cv_folds: int = 5
    random_seeds: List[int] = None  # [42, 123, 456, 789, 999]
    
    def __post_init__(self):
        if self.lambda_values is None:
            self.lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        if self.n_low_values is None:
            self.n_low_values = [10, 50, 100, 200]
        if self.learning_rates is None:
            self.learning_rates = {'cls': 0.01, 'disc': 0.01, 'weight': 0.01}
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456, 789, 999]

class BaselineConfig:
    """Configuration for baseline methods"""
    
    METHODS = {
        'doubly_regressing': {
            'name': 'Doubly Regressing (Your Method)',
            'class': 'DoublyRegressingFairness',
            'params': {'C': 10}
        },
        'kearns_supipm': {
            'name': 'Kearns et al. (supIPM)',
            'class': 'KearnsSubgroupFairness',
            'params': {'method': 'supipm'}
        },
        'molina_bounds': {
            'name': 'Molina & Loiseau (Bounds)',
            'class': 'MolinaBoundedFairness',
            'params': {}
        },
        'agarwal_marginal': {
            'name': 'Agarwal et al. (Marginal)',
            'class': 'AgarwalMarginalFairness',
            'params': {}
        },
        'multicalibration': {
            'name': 'Herbert-Johnson (Multicalibration)',
            'class': 'MulticalibrationFairness',
            'params': {}
        },
        'foulds_intersectional': {
            'name': 'Foulds et al. (Intersectional)',
            'class': 'FouldsIntersectionalFairness',
            'params': {}
        },
        'fams_fairness': {
            'name': 'FAMS (Fairness and Accuracy on Multiple Subgroups)',
            'class': 'FAMSFairness',
            'params': {'n_subtasks': 100, 'training_epochs': 30}
        }
    }

class MetricsConfig:
    """Evaluation metrics configuration"""
    
    FAIRNESS_METRICS = [
        'sup_ipm',           # Maximum IPM violation across subgroups
        'mean_ipm',          # Mean IPM across subgroups  
        'demographic_parity', # Standard demographic parity
        'equal_opportunity',  # Equal opportunity
        'calibration'        # Calibration by group
    ]
    
    ACCURACY_METRICS = [
        'accuracy',
        'auc_roc',
        'f1_score',
        'precision',
        'recall'
    ]
    
    EFFICIENCY_METRICS = [
        'training_time',
        'memory_usage',
        'convergence_iterations',
        'num_discriminators'
    ]

# Experiment-specific configurations
EXPERIMENT_CONFIGS = {
    ExperimentType.ACCURACY_FAIRNESS_TRADEOFF: {
        'datasets': [DatasetConfig.UCI_ADULT, DatasetConfig.COMMUNITIES_CRIME],
        'methods': ['doubly_regressing', 'kearns_supipm', 'fams_fairness', 'agarwal_marginal'],
        'lambda_values': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'primary_metrics': ['accuracy', 'sup_ipm', 'training_time'],
        'output_plots': ['pareto_frontier', 'lambda_vs_metrics']
    },
    
    ExperimentType.COMPUTATIONAL_EFFICIENCY: {
        'datasets': [DatasetConfig.SYNTHETIC],
        'methods': ['doubly_regressing', 'kearns_supipm', 'fams_fairness'],
        'sensitive_attr_counts': [2, 4, 6, 8, 10],
        'primary_metrics': ['training_time', 'memory_usage', 'num_discriminators'],
        'output_plots': ['efficiency_comparison', 'scaling_analysis']
    },
    
    ExperimentType.PARTIAL_VS_COMPLETE: {
        'datasets': [DatasetConfig.UCI_ADULT],
        'methods': ['doubly_regressing'],
        'n_low_values': [10, 25, 50, 100, 200],
        'primary_metrics': ['accuracy', 'sup_ipm', 'subgroup_coverage'],
        'output_plots': ['coverage_vs_fairness', 'threshold_analysis']
    },
    
    ExperimentType.MARGINAL_VS_INTERSECTIONAL: {
        'datasets': [DatasetConfig.UCI_ADULT, DatasetConfig.COMMUNITIES_CRIME],
        'methods': ['doubly_regressing', 'fams_fairness', 'molina_bounds', 'agarwal_marginal'],
        'primary_metrics': ['marginal_fairness', 'intersectional_fairness', 'accuracy'],
        'output_plots': ['marginal_vs_intersectional', 'fairness_gaps']
    },
    
    ExperimentType.SCALABILITY_ROBUSTNESS: {
        'datasets': [DatasetConfig.SYNTHETIC],
        'methods': ['doubly_regressing', 'kearns_supipm', 'fams_fairness', 'agarwal_marginal'],
        'sample_sizes': [1000, 5000, 10000, 25000, 50000],
        'noise_levels': [0.0, 0.05, 0.1, 0.15, 0.2],
        'primary_metrics': ['accuracy', 'sup_ipm', 'convergence_rate'],
        'output_plots': ['scalability_curves', 'robustness_analysis']
    }
}