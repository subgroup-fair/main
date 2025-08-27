"""
Mock data generators for comprehensive testing of subgroup fairness algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.datasets import make_classification
import itertools

@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation"""
    n_samples: int = 1000
    n_features: int = 10
    n_sensitive_attrs: int = 3
    n_classes: int = 2
    bias_strength: float = 0.5
    noise_level: float = 0.1
    imbalance_ratio: float = 0.5
    random_state: Optional[int] = 42

class DatasetGenerator:
    """Comprehensive dataset generator for testing fairness algorithms"""
    
    def __init__(self, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
    
    def generate_biased_dataset(self) -> Dict[str, Any]:
        """Generate dataset with strong bias towards sensitive attributes"""
        
        # Generate base features
        X, y_base = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
            n_informative=max(2, self.config.n_features // 2),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=self.config.random_state
        )
        
        # Generate sensitive attributes
        sensitive_attrs = self._generate_sensitive_attributes()
        
        # Introduce strong bias
        y_biased = self._apply_bias(y_base, sensitive_attrs, self.config.bias_strength)
        
        return {
            'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.config.n_features)]),
            'y': pd.Series(y_biased),
            'S': pd.DataFrame(sensitive_attrs),
            'metadata': {
                'type': 'biased',
                'bias_strength': self.config.bias_strength,
                'n_samples': self.config.n_samples
            }
        }
    
    def generate_fair_dataset(self) -> Dict[str, Any]:
        """Generate dataset with minimal bias (fair baseline)"""
        
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
            n_informative=max(2, self.config.n_features // 2),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=self.config.random_state
        )
        
        # Generate sensitive attributes (independent of target)
        sensitive_attrs = self._generate_sensitive_attributes()
        
        # Add minimal random bias (< 5%)
        bias_mask = np.random.random(self.config.n_samples) < 0.05
        y[bias_mask] = 1 - y[bias_mask]  # Flip small fraction randomly
        
        return {
            'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.config.n_features)]),
            'y': pd.Series(y),
            'S': pd.DataFrame(sensitive_attrs),
            'metadata': {
                'type': 'fair',
                'bias_strength': 0.05,
                'n_samples': self.config.n_samples
            }
        }
    
    def generate_noisy_dataset(self) -> Dict[str, Any]:
        """Generate dataset with added noise for robustness testing"""
        
        # Start with biased dataset
        dataset = self.generate_biased_dataset()
        
        # Add noise to features
        noise = np.random.normal(0, self.config.noise_level, 
                               (self.config.n_samples, self.config.n_features))
        dataset['X'] += noise
        
        # Add label noise
        label_noise_mask = np.random.random(self.config.n_samples) < self.config.noise_level
        dataset['y'][label_noise_mask] = 1 - dataset['y'][label_noise_mask]
        
        dataset['metadata']['type'] = 'noisy'
        dataset['metadata']['noise_level'] = self.config.noise_level
        
        return dataset
    
    def generate_extreme_imbalance_dataset(self) -> Dict[str, Any]:
        """Generate dataset with extreme class imbalance"""
        
        # Calculate number of samples for minority class
        minority_samples = int(self.config.n_samples * self.config.imbalance_ratio)
        majority_samples = self.config.n_samples - minority_samples
        
        # Generate features for each class separately
        X_majority = np.random.randn(majority_samples, self.config.n_features)
        X_minority = np.random.randn(minority_samples, self.config.n_features) + 1  # Slight shift
        
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([np.zeros(majority_samples), np.ones(minority_samples)])
        
        # Generate sensitive attributes
        sensitive_attrs = self._generate_sensitive_attributes()
        
        # Introduce bias correlated with imbalance
        bias_strength = self.config.bias_strength * 2  # Amplify bias for imbalanced case
        y_biased = self._apply_bias(y, sensitive_attrs, bias_strength)
        
        return {
            'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.config.n_features)]),
            'y': pd.Series(y_biased.astype(int)),
            'S': pd.DataFrame(sensitive_attrs),
            'metadata': {
                'type': 'imbalanced',
                'imbalance_ratio': self.config.imbalance_ratio,
                'bias_strength': bias_strength,
                'n_samples': self.config.n_samples
            }
        }
    
    def generate_high_dimensional_dataset(self) -> Dict[str, Any]:
        """Generate high-dimensional dataset for scalability testing"""
        
        # Increase feature count significantly
        high_dim_features = self.config.n_features * 10
        
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=high_dim_features,
            n_informative=max(5, high_dim_features // 5),
            n_redundant=high_dim_features // 10,
            n_clusters_per_class=2,
            random_state=self.config.random_state
        )
        
        # Generate many sensitive attributes with interactions
        sensitive_attrs = {}
        for i in range(self.config.n_sensitive_attrs * 2):  # Double sensitive attrs
            n_categories = np.random.choice([2, 3, 4, 5])
            categories = [f'attr{i}_cat{j}' for j in range(n_categories)]
            sensitive_attrs[f'sensitive_{i}'] = np.random.choice(categories, self.config.n_samples)
        
        # Apply complex bias with interactions
        y_biased = self._apply_complex_bias(y, sensitive_attrs)
        
        return {
            'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(high_dim_features)]),
            'y': pd.Series(y_biased),
            'S': pd.DataFrame(sensitive_attrs),
            'metadata': {
                'type': 'high_dimensional',
                'n_features': high_dim_features,
                'n_sensitive_attrs': len(sensitive_attrs),
                'n_samples': self.config.n_samples
            }
        }
    
    def generate_intersectional_bias_dataset(self) -> Dict[str, Any]:
        """Generate dataset with complex intersectional bias patterns"""
        
        X, y_base = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
            random_state=self.config.random_state
        )
        
        # Create sensitive attributes with known intersections
        sensitive_attrs = {
            'gender': np.random.choice(['male', 'female'], self.config.n_samples, p=[0.5, 0.5]),
            'race': np.random.choice(['white', 'black', 'asian', 'hispanic'], 
                                   self.config.n_samples, p=[0.6, 0.2, 0.1, 0.1]),
            'age': np.random.choice(['young', 'middle', 'old'], 
                                  self.config.n_samples, p=[0.3, 0.4, 0.3])
        }
        
        # Apply intersectional bias
        y_biased = y_base.copy()
        
        # Specific bias for intersections
        intersectional_biases = [
            (['male', 'white', 'young'], 0.8),    # Highly favored
            (['female', 'black', 'old'], -0.6),   # Highly disadvantaged
            (['female', 'asian', 'middle'], 0.3), # Moderately favored
        ]
        
        for attributes, bias in intersectional_biases:
            mask = (
                (sensitive_attrs['gender'] == attributes[0]) &
                (sensitive_attrs['race'] == attributes[1]) &
                (sensitive_attrs['age'] == attributes[2])
            )
            
            # Apply bias by flipping labels probabilistically
            flip_prob = abs(bias) * 0.5
            flip_mask = np.random.random(self.config.n_samples) < flip_prob
            intersection_mask = mask & flip_mask
            
            if bias > 0:  # Favor this group
                y_biased[intersection_mask] = 1
            else:  # Disadvantage this group
                y_biased[intersection_mask] = 0
        
        return {
            'X': pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.config.n_features)]),
            'y': pd.Series(y_biased),
            'S': pd.DataFrame(sensitive_attrs),
            'metadata': {
                'type': 'intersectional_bias',
                'intersectional_patterns': intersectional_biases,
                'n_samples': self.config.n_samples
            }
        }
    
    def _generate_sensitive_attributes(self) -> Dict[str, np.ndarray]:
        """Generate synthetic sensitive attributes"""
        sensitive_attrs = {}
        
        for i in range(self.config.n_sensitive_attrs):
            # Randomly choose number of categories (2-5)
            n_categories = np.random.choice([2, 3, 4, 5])
            categories = [f'attr{i}_cat{j}' for j in range(n_categories)]
            
            # Generate random probabilities for categories
            probs = np.random.dirichlet(np.ones(n_categories))
            sensitive_attrs[f'sensitive_{i}'] = np.random.choice(
                categories, self.config.n_samples, p=probs
            )
        
        return sensitive_attrs
    
    def _apply_bias(self, y: np.ndarray, sensitive_attrs: Dict[str, np.ndarray], 
                   bias_strength: float) -> np.ndarray:
        """Apply bias based on sensitive attributes"""
        y_biased = y.copy()
        
        for attr_name, attr_values in sensitive_attrs.items():
            unique_values = np.unique(attr_values)
            
            # Randomly choose which category gets biased
            biased_category = np.random.choice(unique_values)
            mask = attr_values == biased_category
            
            # Apply bias by flipping labels probabilistically
            flip_prob = bias_strength * 0.5
            flip_mask = np.random.random(len(y)) < flip_prob
            bias_mask = mask & flip_mask
            
            # Favor or disadvantage based on coin flip
            favor = np.random.choice([True, False])
            if favor:
                y_biased[bias_mask] = 1
            else:
                y_biased[bias_mask] = 0
        
        return y_biased
    
    def _apply_complex_bias(self, y: np.ndarray, 
                          sensitive_attrs: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply complex bias with interactions between sensitive attributes"""
        y_biased = y.copy()
        
        # Get all possible pairs of sensitive attributes for interactions
        attr_names = list(sensitive_attrs.keys())
        
        for attr1, attr2 in itertools.combinations(attr_names, 2):
            # Create interaction bias for random category combinations
            unique_vals1 = np.unique(sensitive_attrs[attr1])
            unique_vals2 = np.unique(sensitive_attrs[attr2])
            
            # Pick random combination
            val1 = np.random.choice(unique_vals1)
            val2 = np.random.choice(unique_vals2)
            
            mask = (sensitive_attrs[attr1] == val1) & (sensitive_attrs[attr2] == val2)
            
            # Apply interaction bias
            interaction_strength = np.random.uniform(0.2, 0.8)
            flip_prob = interaction_strength * 0.3
            flip_mask = np.random.random(len(y)) < flip_prob
            bias_mask = mask & flip_mask
            
            # Random direction of bias
            if np.random.choice([True, False]):
                y_biased[bias_mask] = 1
            else:
                y_biased[bias_mask] = 0
        
        return y_biased

# Convenience functions for quick data generation
def generate_biased_dataset(n_samples: int = 1000, n_features: int = 10, 
                          bias_strength: float = 0.5, random_state: int = 42) -> Dict[str, Any]:
    """Quick biased dataset generation"""
    config = DatasetConfig(
        n_samples=n_samples, n_features=n_features, 
        bias_strength=bias_strength, random_state=random_state
    )
    generator = DatasetGenerator(config)
    return generator.generate_biased_dataset()

def generate_fair_dataset(n_samples: int = 1000, n_features: int = 10, 
                         random_state: int = 42) -> Dict[str, Any]:
    """Quick fair dataset generation"""
    config = DatasetConfig(n_samples=n_samples, n_features=n_features, random_state=random_state)
    generator = DatasetGenerator(config)
    return generator.generate_fair_dataset()

def generate_noisy_dataset(n_samples: int = 1000, n_features: int = 10,
                          noise_level: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """Quick noisy dataset generation"""
    config = DatasetConfig(
        n_samples=n_samples, n_features=n_features,
        noise_level=noise_level, random_state=random_state
    )
    generator = DatasetGenerator(config)
    return generator.generate_noisy_dataset()

def generate_extreme_imbalance_dataset(n_samples: int = 1000, imbalance_ratio: float = 0.1,
                                     random_state: int = 42) -> Dict[str, Any]:
    """Quick imbalanced dataset generation"""
    config = DatasetConfig(
        n_samples=n_samples, imbalance_ratio=imbalance_ratio, random_state=random_state
    )
    generator = DatasetGenerator(config)
    return generator.generate_extreme_imbalance_dataset()

def generate_high_dimensional_dataset(n_samples: int = 1000, n_features: int = 100,
                                    random_state: int = 42) -> Dict[str, Any]:
    """Quick high-dimensional dataset generation"""
    config = DatasetConfig(n_samples=n_samples, n_features=n_features, random_state=random_state)
    generator = DatasetGenerator(config)
    return generator.generate_high_dimensional_dataset()