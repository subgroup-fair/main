"""
Doubly Regressing Approach for Partial Subgroup Fairness
Implementation of the core method from your research
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DoublyRegressingFairness(BaseEstimator, ClassifierMixin):
    """
    Doubly Regressing approach for partial subgroup fairness
    
    Key innovations:
    1. Partial subgroup fairness: Only enforce fairness on subgroups with size >= n_low
    2. Single discriminator instead of M discriminators for efficiency
    3. supIPM (supremum Integral Probability Metric) as unfairness measure
    """
    
    def __init__(self, 
                 lambda_fair: float = 1.0,
                 n_low: int = 50,
                 max_iterations: int = 10000,
                 learning_rate_cls: float = 0.01,
                 learning_rate_disc: float = 0.01,
                 learning_rate_weight: float = 0.01,
                 C: float = 10.0,
                 random_state: Optional[int] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        
        self.lambda_fair = lambda_fair
        self.n_low = n_low
        self.max_iterations = max_iterations
        self.learning_rate_cls = learning_rate_cls
        self.learning_rate_disc = learning_rate_disc
        self.learning_rate_weight = learning_rate_weight
        self.C = C
        self.random_state = random_state
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Model components
        self.classifier = None
        self.discriminator = None
        self.subgroup_weights = None
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'classifier_loss': [],
            'discriminator_loss': [],
            'fairness_violation': [],
            'total_loss': []
        }
    
    def fit(self, X, y, S):
        """
        Fit the doubly regressing model
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target labels (n_samples,)
            S: Sensitive attributes DataFrame (n_samples x n_sensitive_attrs)
        """
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
        
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = X_scaled.shape
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y)
        
        # Enumerate subgroups and filter by minimum size
        self.covered_subgroups = self._enumerate_covered_subgroups(S)
        
        if self.verbose:
            print(f"Found {len(self.covered_subgroups)} subgroups with size >= {self.n_low}")
        
        # Initialize models
        self._initialize_models(n_features, len(self.covered_subgroups))
        
        # Training loop
        self._train_doubly_regressing(X_tensor, y_tensor, S)
        
        return self
    
    def _enumerate_covered_subgroups(self, S):
        """Enumerate all subgroups with sufficient sample size"""
        from scripts.utils.metrics import SubgroupFairnessMetrics
        
        all_subgroups = SubgroupFairnessMetrics._enumerate_subgroups(S, max_combinations=500)
        covered_subgroups = []
        
        for subgroup_def in all_subgroups:
            mask = SubgroupFairnessMetrics._get_subgroup_mask(S, subgroup_def)
            if mask.sum() >= self.n_low:
                covered_subgroups.append({
                    'definition': subgroup_def,
                    'mask': mask,
                    'size': mask.sum()
                })
        
        return covered_subgroups
    
    def _initialize_models(self, n_features, n_subgroups):
        """Initialize classifier and discriminator networks"""
        
        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Single discriminator network (key innovation)
        self.discriminator = nn.Sequential(
            nn.Linear(1, 32),  # Takes classifier output as input
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_subgroups),  # Outputs for each covered subgroup
            nn.Softmax(dim=1)
        )
        
        # Subgroup importance weights (learned)
        self.subgroup_weights = nn.Parameter(torch.ones(n_subgroups) / n_subgroups)
        
        # Optimizers
        self.optimizer_cls = optim.Adam(self.classifier.parameters(), lr=self.learning_rate_cls)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate_disc)
        self.optimizer_weight = optim.Adam([self.subgroup_weights], lr=self.learning_rate_weight)
    
    def _train_doubly_regressing(self, X_tensor, y_tensor, S):
        """Main training loop implementing doubly regressing algorithm"""
        
        for iteration in range(self.max_iterations):
            # Step 1: Train classifier to minimize prediction loss + fairness penalty
            self.optimizer_cls.zero_grad()
            
            # Forward pass through classifier
            y_pred = self.classifier(X_tensor).squeeze()
            
            # Classification loss
            cls_loss = nn.BCELoss()(y_pred, y_tensor)
            
            # Fairness penalty using discriminator
            fairness_loss = self._compute_fairness_loss(y_pred, S)
            
            # Total classifier loss
            total_cls_loss = cls_loss + self.lambda_fair * fairness_loss
            total_cls_loss.backward(retain_graph=True)
            self.optimizer_cls.step()
            
            # Step 2: Train discriminator to maximize ability to predict subgroup membership
            self.optimizer_disc.zero_grad()
            
            # Discriminator tries to predict subgroup from classifier output
            disc_loss = self._compute_discriminator_loss(y_pred.detach(), S)
            (-disc_loss).backward()  # Maximize discriminator ability
            self.optimizer_disc.step()
            
            # Step 3: Update subgroup weights
            self.optimizer_weight.zero_grad()
            weight_loss = self._compute_weight_loss(y_pred.detach(), S)
            weight_loss.backward()
            self.optimizer_weight.step()
            
            # Project weights to simplex
            with torch.no_grad():
                self.subgroup_weights.data = torch.softmax(self.subgroup_weights.data, dim=0)
            
            # Record training history
            self.training_history['classifier_loss'].append(cls_loss.item())
            self.training_history['discriminator_loss'].append(disc_loss.item())
            self.training_history['fairness_violation'].append(fairness_loss.item())
            self.training_history['total_loss'].append(total_cls_loss.item())
            
            # Check convergence
            if iteration > 100 and self._check_convergence():
                if self.verbose:
                    print(f"Converged after {iteration} iterations")
                break
            
            if self.verbose and iteration % 1000 == 0:
                print(f"Iteration {iteration}: Loss = {total_cls_loss.item():.4f}, "
                      f"Fairness = {fairness_loss.item():.4f}")
    
    def _compute_fairness_loss(self, y_pred, S):
        """Compute fairness loss using supremum IPM"""
        total_fairness_loss = 0.0
        overall_rate = y_pred.mean()
        
        for i, subgroup in enumerate(self.covered_subgroups):
            # Get subgroup predictions
            mask = torch.as_tensor(subgroup["mask"], dtype=torch.bool, device=y_pred.device)
            subgroup_pred = y_pred[mask]
            subgroup_rate = subgroup_pred.mean()
            
            # IPM violation (log ratio as in paper)
            if subgroup_rate > 0 and overall_rate > 0:
                imp_violation = torch.abs(torch.log(subgroup_rate / overall_rate))
                weighted_violation = self.subgroup_weights[i] * imp_violation
                total_fairness_loss += weighted_violation
        
        return total_fairness_loss
    
    def _compute_discriminator_loss(self, y_pred, S):
        """Compute discriminator loss for subgroup detection"""
        
        # Create target labels for subgroup membership
        subgroup_targets = torch.zeros(len(y_pred), len(self.covered_subgroups))
        
        for i, subgroup in enumerate(self.covered_subgroups):
            mask = torch.as_tensor(subgroup['mask'], dtype=torch.bool, device=subgroup_targets.device)
            subgroup_targets[mask, i] = 1.0

        # Discriminator predictions
        disc_pred = self.discriminator(y_pred.unsqueeze(1))
        
        # Cross-entropy loss for subgroup classification
        loss = nn.BCELoss()(disc_pred, subgroup_targets)
        
        return loss
    
    def _compute_weight_loss(self, y_pred, S):
        """Compute loss for updating subgroup importance weights"""
        
        # Weights should be proportional to fairness violations
        fairness_violations = []
        overall_rate = y_pred.mean()
        
        for subgroup in self.covered_subgroups:
            mask = torch.as_tensor(subgroup['mask'], dtype=torch.bool, device=y_pred.device)
            subgroup_pred = y_pred[mask]
            subgroup_rate = subgroup_pred.mean()
            
            if subgroup_rate > 0 and overall_rate > 0:
                violation = torch.abs(torch.log(subgroup_rate / overall_rate))
                fairness_violations.append(violation)
            else:
                fairness_violations.append(torch.tensor(0.0))
        
        # Target weights proportional to violations
        violations_tensor = torch.stack(fairness_violations)
        target_weights = torch.softmax(violations_tensor, dim=0)
        
        # KL divergence loss
        weight_loss = nn.KLDivLoss()(torch.log(self.subgroup_weights), target_weights)
        
        return weight_loss
    
    def _check_convergence(self):
        """Check if training has converged"""
        if len(self.training_history['total_loss']) < 10:
            return False
        
        recent_losses = self.training_history['total_loss'][-10:]
        loss_std = np.std(recent_losses)
        
        return loss_std < self.tolerance
    
    def predict(self, X):
        """Make binary predictions"""
        y_pred_proba = self.predict_proba(X)
        if len(y_pred_proba.shape) == 2:
            return (y_pred_proba[:, 1] > 0.5).astype(int)
        else:
            return (y_pred_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self.classifier.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            y_pred = self.classifier(X_tensor).squeeze()
        
        # Return in sklearn format [P(class=0), P(class=1)]
        proba_positive = y_pred.detach().cpu().numpy()
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def get_training_history(self):
        """Get training history for analysis"""
        return self.training_history
    
    def get_subgroup_importance(self):
        """Get learned subgroup importance weights"""
        if self.subgroup_weights is not None:
            return {
                'weights': self.subgroup_weights.detach().cpu().numpy(),
                'subgroup_definitions': [sg['definition'] for sg in self.covered_subgroups],
                'subgroup_sizes': [sg['size'] for sg in self.covered_subgroups]
            }
        return None