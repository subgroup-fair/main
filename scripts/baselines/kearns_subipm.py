"""
Kearns et al. Subgroup Fairness Implementation (supIPM baseline)
Based on "Preventing Fairness Gerrymandering" paper
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import itertools
from scipy.optimize import minimize

class KearnsSubgroupFairness(BaseEstimator, ClassifierMixin):
    """
    Implementation of Kearns et al. subgroup fairness approach
    
    Key features:
    - Enforces fairness constraints on all subgroups (not partial)
    - Uses gamma-unfairness formulation
    - Lagrangian optimization approach
    """
    
    def __init__(self,
                 gamma: float = 0.1,
                 fairness_penalty: float = 1.0,
                 max_subgroup_size: int = 3,
                 max_iterations: int = 1000,
                 learning_rate: float = 0.01,
                 random_state: Optional[int] = None,
                 tolerance: float = 1e-6):
        
        self.gamma = gamma  # Fairness tolerance
        self.fairness_penalty = fairness_penalty
        self.max_subgroup_size = max_subgroup_size
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.tolerance = tolerance
        
        # Internal models
        self.base_classifier = None
        self.scaler = StandardScaler()
        self.subgroups = []
        self.lagrange_multipliers = None
        
        # Training history
        self.training_history = {
            'objective': [],
            'fairness_violations': [],
            'accuracy': []
        }
    
    def fit(self, X, y, S):
        """
        Fit the Kearns subgroup fairness model
        
        Args:
            X: Feature matrix
            y: Target labels  
            S: Sensitive attributes DataFrame
        """
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X, y = check_X_y(X, y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Enumerate all subgroups
        self.subgroups = self._enumerate_subgroups(S)
        
        # Initialize base classifier
        self.base_classifier = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Initialize Lagrange multipliers
        self.lagrange_multipliers = np.zeros(len(self.subgroups))
        
        # Optimization loop
        self._optimize_with_constraints(X_scaled, y, S)
        
        return self
    
    def _enumerate_subgroups(self, S, max_combinations: int = 1000):
        """Enumerate all possible subgroups up to max_subgroup_size"""
        subgroups = []
        attr_values = {}
        
        # Get unique values for each attribute
        for col in S.columns:
            attr_values[col] = list(S[col].unique())
        
        attr_names = list(attr_values.keys())
        
        # Generate combinations up to max_subgroup_size
        for r in range(1, min(len(attr_names) + 1, self.max_subgroup_size + 1)):
            for attr_combo in itertools.combinations(attr_names, r):
                value_combinations = itertools.product(*[attr_values[attr] for attr in attr_combo])
                
                for value_combo in value_combinations:
                    subgroup_def = dict(zip(attr_combo, value_combo))
                    
                    # Get mask and check minimum size
                    mask = self._get_subgroup_mask(S, subgroup_def)
                    if mask.sum() >= 10:  # Minimum subgroup size
                        subgroups.append({
                            'definition': subgroup_def,
                            'mask': mask,
                            'size': mask.sum()
                        })
                    
                    if len(subgroups) >= max_combinations:
                        return subgroups
        
        return subgroups
    
    def _get_subgroup_mask(self, S, subgroup_def):
        """Get boolean mask for subgroup"""
        mask = np.ones(len(S), dtype=bool)
        
        for attr, value in subgroup_def.items():
            if attr in S.columns:
                mask = mask & (S[attr] == value)
        
        return mask
    
    def _optimize_with_constraints(self, X, y, S):
        """Optimize classifier with fairness constraints using Lagrangian method"""
        
        n_samples, n_features = X.shape
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Step 1: Update classifier parameters
            self._update_classifier(X, y, S)
            
            # Step 2: Update Lagrange multipliers
            violations = self._compute_constraint_violations(X, y, S)
            self.lagrange_multipliers = np.maximum(
                0, 
                self.lagrange_multipliers + self.learning_rate * violations
            )
            
            # Compute and record metrics
            objective = self._compute_objective(X, y, S)
            accuracy = self._compute_accuracy(X, y)
            max_violation = np.max(np.abs(violations))
            
            self.training_history['objective'].append(objective)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['fairness_violations'].append(max_violation)
            
            # Check convergence
            if max_violation < self.tolerance:
                break
    
    def _update_classifier(self, X, y, S):
        """Update classifier with fairness penalty"""
        
        # Create sample weights based on Lagrange multipliers
        sample_weights = np.ones(len(y))
        
        # Add fairness penalty to samples in violated subgroups
        for i, subgroup in enumerate(self.subgroups):
            if self.lagrange_multipliers[i] > 0:
                # Increase weight for samples in this subgroup
                penalty_weight = 1 + self.fairness_penalty * self.lagrange_multipliers[i]
                sample_weights[subgroup['mask']] *= penalty_weight
        
        # Fit classifier with weighted samples
        self.base_classifier.fit(X, y, sample_weight=sample_weights)
    
    def _compute_constraint_violations(self, X, y, S):
        """Compute fairness constraint violations for each subgroup"""
        
        # Get predictions
        y_pred_proba = self.base_classifier.predict_proba(X)[:, 1]
        overall_rate = y_pred_proba.mean()
        
        violations = []
        
        for subgroup in self.subgroups:
            # Subgroup rate
            subgroup_pred = y_pred_proba[subgroup['mask']]
            subgroup_rate = subgroup_pred.mean()
            
            # Demographic parity violation
            violation = abs(subgroup_rate - overall_rate) - self.gamma
            violations.append(max(0, violation))  # Only positive violations
        
        return np.array(violations)
    
    def _compute_objective(self, X, y, S):
        """Compute Lagrangian objective"""
        
        # Base accuracy loss
        y_pred_proba = self.base_classifier.predict_proba(X)[:, 1]
        accuracy_loss = -np.mean(y * np.log(y_pred_proba + 1e-8) + 
                                (1 - y) * np.log(1 - y_pred_proba + 1e-8))
        
        # Fairness penalty
        violations = self._compute_constraint_violations(X, y, S)
        fairness_penalty = np.sum(self.lagrange_multipliers * violations)
        
        return accuracy_loss + fairness_penalty
    
    def _compute_accuracy(self, X, y):
        """Compute prediction accuracy"""
        y_pred = self.base_classifier.predict(X)
        return np.mean(y_pred == y)
    
    def predict(self, X):
        """Make predictions"""
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        return self.base_classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        return self.base_classifier.predict_proba(X_scaled)
    
    def get_subgroup_violations(self, X, y, S):
        """Get detailed fairness violations for each subgroup"""
        
        y_pred_proba = self.predict_proba(X)[:, 1]
        overall_rate = y_pred_proba.mean()
        
        violations = {}
        
        for i, subgroup in enumerate(self.subgroups):
            subgroup_pred = y_pred_proba[subgroup['mask']]
            subgroup_rate = subgroup_pred.mean()
            
            violations[f"subgroup_{i}"] = {
                'definition': subgroup['definition'],
                'size': subgroup['size'],
                'rate': subgroup_rate,
                'violation': abs(subgroup_rate - overall_rate),
                'multiplier': self.lagrange_multipliers[i]
            }
        
        return violations
    
    def get_training_history(self):
        """Get training history"""
        return self.training_history