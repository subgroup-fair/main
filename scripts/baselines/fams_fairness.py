"""
FAMS (Fairness and Accuracy on Multiple Subgroups) Implementation
Based on "On Learning Fairness and Accuracy on Multiple Subgroups" (NeurIPS 2022)

This implements a Bayesian meta-learning approach for subgroup fairness
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Any, List, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
import copy
import logging

class StochasticLinear(nn.Module):
    """Stochastic linear layer for Bayesian neural networks"""
    
    def __init__(self, in_features: int, out_features: int, 
                 log_var_init: Dict[str, float], use_bias: bool = True):
        super(StochasticLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.log_var_init_mean = 0.0   # 기본값 (혹은 config에서 받게 수정)
        self.log_var_init_std = 1.0

        
        # Weight parameters (mean and log variance)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.full((out_features, in_features), 
                                                     log_var_init.get('mean', 0.01)))
        
        if use_bias:
            self.bias_mean = nn.Parameter(torch.randn(out_features))
            self.bias_log_var = nn.Parameter(torch.full((out_features,), 
                                                       log_var_init.get('mean', 0.01)))
        
        # Noise standard deviation for reparameterization
        self.eps_std = 0.08
        
        self.weights_count = in_features * out_features
        if use_bias:
            self.weights_count += out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with reparameterization trick"""
        
        # Sample weights using reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_log_var)
        weight_eps = torch.randn_like(self.weight_mean) * self.eps_std
        weight = self.weight_mean + weight_std * weight_eps
        
        if self.use_bias:
            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias_eps = torch.randn_like(self.bias_mean) * self.eps_std
            bias = self.bias_mean + bias_std * bias_eps
        else:
            bias = None
        
        return F.linear(x, weight, bias)
    
    def set_eps_std(self, eps_std: float) -> float:
        """Set noise standard deviation"""
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

class StochasticMLP(nn.Module):
    """Stochastic Multi-Layer Perceptron for FAMS"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 log_var_init: Dict[str, float]):
        super(StochasticMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(StochasticLinear(prev_dim, hidden_dim, log_var_init))
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append(StochasticLinear(prev_dim, output_dim, log_var_init))
        
        # Count total weights
        self.weights_count = sum(layer.weights_count for layer in self.layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        
        # Output layer (no activation for regression/logits)
        x = self.layers[-1](x)
        
        return x
    
    def set_eps_std(self, eps_std: float) -> float:
        """Set noise std for all layers"""
        old_eps_std = None
        for layer in self.layers:
            if isinstance(layer, StochasticLinear):
                old_eps_std = layer.set_eps_std(eps_std)
        return old_eps_std

def get_kld_loss(prior_model: nn.Module, post_model: nn.Module) -> torch.Tensor:
    """Compute KL divergence between posterior and prior models"""
    
    kld_loss = 0.0
    
    for (prior_name, prior_param), (post_name, post_param) in zip(
        prior_model.named_parameters(), post_model.named_parameters()
    ):
        if 'log_var' in prior_name and 'log_var' in post_name:
            # KL divergence for variance parameters
            prior_var = torch.exp(prior_param)
            post_var = torch.exp(post_param)
            
            kld_var = 0.5 * (prior_var / post_var + torch.log(post_var / prior_var) - 1)
            kld_loss += kld_var.sum()
            
        elif 'mean' in prior_name and 'mean' in post_name:
            # KL divergence for mean parameters (assuming unit variance for simplicity)
            kld_mean = 0.5 * (prior_param - post_param).pow(2)
            kld_loss += kld_mean.sum()
    
    return kld_loss

class FAMSFairness(BaseEstimator, ClassifierMixin):
    """
    FAMS: Fairness and Accuracy on Multiple Subgroups
    
    A Bayesian meta-learning approach that learns task-specific posterior models
    for different subgroups while maintaining a shared prior.
    
    Key Features:
    - Bayesian neural networks with stochastic weights
    - Meta-learning across multiple subgroup tasks
    - KL divergence regularization between prior and posterior
    - Monte Carlo sampling for robust predictions
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [64, 32],
                 n_subtasks: int = 200,
                 training_epochs: int = 50,
                 batch_size: int = 50,
                 lr_prior: float = 0.001,
                 lr_post: float = 0.8,
                 weight_kld: float = 0.4,
                 kappa_prior: float = 0.01,
                 kappa_post: float = 0.001,
                 log_var_init_mean: float = 0.01,
                 log_var_init_var: float = 0.01,
                 eps_std: float = 0.08,
                 n_mc_samples: int = 5,
                 max_inner_steps: int = 2,
                 random_state: Optional[int] = None,
                 device: str = 'cpu',
                 verbose: bool = False):
        
        self.hidden_dims = hidden_dims
        self.n_subtasks = n_subtasks
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lr_prior = lr_prior
        self.lr_post = lr_post
        self.weight_kld = weight_kld
        self.kappa_prior = kappa_prior
        self.kappa_post = kappa_post
        self.log_var_init = {
            'mean': log_var_init_mean,
            'std': log_var_init_var
        }
        self.eps_std = eps_std
        self.n_mc_samples = n_mc_samples
        self.max_inner_steps = max_inner_steps
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        
        # Model components
        self.prior_model = None
        self.posterior_models = []
        self.scaler = StandardScaler()
        self.subgroup_tasks = []
        
        # Training history
        self.training_history = {
            'prior_loss': [],
            'posterior_losses': [],
            'kld_losses': [],
            'total_loss': []
        }
    
    def fit(self, X, y, S):
        """
        Fit the FAMS model
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target labels (n_samples,)
            S: Sensitive attributes DataFrame (n_samples x n_sensitive_attrs)
        """
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        X, y = check_X_y(X, y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y).to(self.device)
        
        # Create subgroup tasks
        self.subgroup_tasks = self._create_subgroup_tasks(X_tensor, y_tensor, S)
        
        if self.verbose:
            print(f"Created {len(self.subgroup_tasks)} subgroup tasks")
        
        # Initialize models
        input_dim = X_scaled.shape[1]
        output_dim = 1  # Binary classification
        
        self.prior_model = StochasticMLP(
            input_dim, self.hidden_dims, output_dim, self.log_var_init
        ).to(self.device)
        
        # Create posterior models for each subtask
        self.posterior_models = []
        for _ in range(min(self.n_subtasks, len(self.subgroup_tasks))):
            post_model = StochasticMLP(
                input_dim, self.hidden_dims, output_dim, self.log_var_init
            ).to(self.device)
            self.posterior_models.append(post_model)
        
        # Training
        self._train_meta_learning()
        
        return self
    
    def _create_subgroup_tasks(self, X: torch.Tensor, y: torch.Tensor, 
                               S: pd.DataFrame) -> List[Dict[str, torch.Tensor]]:
        """Create subgroup-specific tasks"""
        
        from scripts.utils.metrics import SubgroupFairnessMetrics
        
        # Enumerate subgroups
        all_subgroups = SubgroupFairnessMetrics._enumerate_subgroups(
            S, max_combinations=self.n_subtasks
        )
        
        tasks = []
        for subgroup_def in all_subgroups:
            mask = SubgroupFairnessMetrics._get_subgroup_mask(S, subgroup_def)
            
            if mask.sum() >= 10:  # Minimum task size
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=X.device)
                task_X = X[mask_t]
                task_y = y[mask_t]

                tasks.append({
                    'X': task_X,
                    'y': task_y,
                    'definition': subgroup_def,
                    'size': mask.sum()
                })
        
        return tasks
    
    def _train_meta_learning(self):
        """Main meta-learning training loop"""
        
        # Prior optimizer
        prior_optimizer = optim.Adam(self.prior_model.parameters(), lr=self.lr_prior)
        
        # Loss criterion
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.training_epochs):
            epoch_prior_loss = 0.0
            epoch_post_losses = []
            epoch_kld_losses = []
            
            # Randomly sample tasks for this epoch
            n_tasks_this_epoch = min(self.n_subtasks, len(self.subgroup_tasks))
            sampled_tasks = np.random.choice(len(self.subgroup_tasks), 
                                           size=n_tasks_this_epoch, 
                                           replace=False)
            
            for task_idx, task_id in enumerate(sampled_tasks):
                task = self.subgroup_tasks[task_id]
                
                # Get corresponding posterior model
                post_model = self.posterior_models[task_idx % len(self.posterior_models)]
                
                # Copy prior to posterior for this task
                post_model.load_state_dict(self.prior_model.state_dict())
                
                # Train task-specific posterior
                post_loss, kld_loss = self._train_one_task(
                    task, post_model, criterion, task_idx, epoch
                )
                
                epoch_post_losses.append(post_loss)
                epoch_kld_losses.append(kld_loss)
                
                # Update prior based on posterior
                prior_loss = self._update_prior(post_model, criterion, task)
                epoch_prior_loss += prior_loss
            
            # Record training history
            avg_post_loss = np.mean(epoch_post_losses) if epoch_post_losses else 0.0
            avg_kld_loss = np.mean(epoch_kld_losses) if epoch_kld_losses else 0.0
            
            self.training_history['prior_loss'].append(epoch_prior_loss / len(sampled_tasks))
            self.training_history['posterior_losses'].append(avg_post_loss)
            self.training_history['kld_losses'].append(avg_kld_loss)
            self.training_history['total_loss'].append(
                epoch_prior_loss + avg_post_loss + self.weight_kld * avg_kld_loss
            )
            
            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Prior Loss={epoch_prior_loss:.4f}, "
                      f"Post Loss={avg_post_loss:.4f}, KLD={avg_kld_loss:.4f}")
    
    def _train_one_task(self, task: Dict, post_model: nn.Module, 
                        criterion: nn.Module, task_idx: int, epoch: int) -> Tuple[float, float]:
        """Train posterior model on one subgroup task"""
        
        # Freeze prior, activate posterior
        for param in self.prior_model.parameters():
            param.requires_grad = False
        
        for param in post_model.parameters():
            param.requires_grad = True
        
        post_model.train()
        
        # Posterior optimizer
        post_optimizer = optim.Adam(post_model.parameters(), lr=self.lr_post)
        
        X_task, y_task = task['X'], task['y']
        
        total_post_loss = 0.0
        total_kld_loss = 0.0
        
        for inner_step in range(self.max_inner_steps):
            # Monte Carlo sampling
            mc_loss = 0.0
            for mc_sample in range(self.n_mc_samples):
                outputs = post_model(X_task).squeeze()
                loss = criterion(outputs, y_task)
                mc_loss += loss / self.n_mc_samples
            
            # KL divergence loss
            kld_loss = get_kld_loss(self.prior_model, post_model)
            
            # Total loss
            total_loss = mc_loss + self.weight_kld * kld_loss
            
            # Optimize
            post_optimizer.zero_grad()
            total_loss.backward()
            post_optimizer.step()
            
            total_post_loss += mc_loss.item()
            total_kld_loss += kld_loss.item()
        
        # Freeze posterior
        for param in post_model.parameters():
            param.requires_grad = False
        
        return total_post_loss / self.max_inner_steps, total_kld_loss / self.max_inner_steps
    
    def _update_prior(self, post_model: nn.Module, criterion: nn.Module, 
                      task: Dict) -> float:
        """Update prior model based on posterior"""
        
        # Activate prior
        for param in self.prior_model.parameters():
            param.requires_grad = True
        
        self.prior_model.train()
        
        # Prior optimizer
        prior_optimizer = optim.Adam(self.prior_model.parameters(), lr=self.lr_prior)
        
        X_task, y_task = task['X'], task['y']
        
        # Prior loss (regularization towards posterior)
        prior_outputs = self.prior_model(X_task).squeeze()
        prior_loss = criterion(prior_outputs, y_task)
        
        # Add regularization to stay close to posterior
        reg_loss = 0.0
        for (prior_name, prior_param), (post_name, post_param) in zip(
            self.prior_model.named_parameters(), post_model.named_parameters()
        ):
            reg_loss += F.mse_loss(prior_param, post_param.detach())
        
        total_prior_loss = prior_loss + self.kappa_prior * reg_loss
        
        # Optimize
        prior_optimizer.zero_grad()
        total_prior_loss.backward()
        prior_optimizer.step()
        
        return total_prior_loss.item()
    
    def predict(self, X):
        """Make binary predictions"""
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities using ensemble of posterior models"""
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.prior_model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            # Use prior model for general prediction
            for _ in range(self.n_mc_samples):
                outputs = self.prior_model(X_tensor).squeeze()
                probs = torch.sigmoid(outputs)
                all_predictions.append(probs.detach().cpu().numpy())
            
            # Average across MC samples
            avg_probs = np.mean(all_predictions, axis=0)
        
        # Return in sklearn format [P(class=0), P(class=1)]
        proba_negative = 1 - avg_probs
        proba_positive = avg_probs
        
        return np.column_stack([proba_negative, proba_positive])
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.training_history
    
    def get_subgroup_info(self) -> Dict[str, Any]:
        """Get information about learned subgroups"""
        return {
            'n_subgroup_tasks': len(self.subgroup_tasks),
            'n_posterior_models': len(self.posterior_models),
            'task_sizes': [task['size'] for task in self.subgroup_tasks],
            'task_definitions': [task['definition'] for task in self.subgroup_tasks[:10]]  # First 10
        }