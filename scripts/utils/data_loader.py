"""
Data loading utilities for subgroup fairness experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load and preprocess dataset based on configuration
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Dictionary with train/test splits and metadata
    """
    
    dataset_name = config['name']
    
    if dataset_name == 'uci_adult':
        return load_uci_adult_dataset(config)
    elif dataset_name == 'communities_crime':
        return load_communities_crime_dataset(config)
    elif dataset_name == 'synthetic':
        return generate_synthetic_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_uci_adult_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and preprocess UCI Adult Income dataset"""
    
    # For now, generate synthetic data resembling Adult dataset
    # TODO: Implement actual Adult dataset loading
    np.random.seed(42)
    
    n_samples = config.get('sample_size', 10000)
    
    # Generate synthetic adult-like data
    data = {}
    
    # Age (continuous)
    data['age'] = np.random.normal(40, 12, n_samples)
    data['age'] = np.clip(data['age'], 18, 90).astype(int)
    
    # Create age groups for sensitive attribute
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 40, 65, 100], 
                              labels=['young', 'middle_young', 'middle_old', 'old'])
    
    # Sex (binary sensitive attribute)
    data['sex'] = np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4])
    
    # Race (categorical sensitive attribute)
    data['race'] = np.random.choice(['white', 'black', 'asian', 'hispanic', 'other'], 
                                   n_samples, p=[0.7, 0.12, 0.08, 0.06, 0.04])
    
    # Education level
    data['education'] = np.random.choice(['high_school', 'some_college', 'bachelors', 'masters', 'doctorate'],
                                        n_samples, p=[0.3, 0.25, 0.25, 0.15, 0.05])
    
    # Work hours
    data['hours_per_week'] = np.random.normal(40, 10, n_samples)
    data['hours_per_week'] = np.clip(data['hours_per_week'], 10, 80)
    
    # Create correlated features for realism
    # Higher education and more hours -> higher income probability
    education_scores = {'high_school': 0, 'some_college': 1, 'bachelors': 2, 'masters': 3, 'doctorate': 4}
    edu_numeric = np.array([education_scores[edu] for edu in data['education']])
    
    # Income probability based on features with some bias
    income_logits = (
        0.1 * (data['age'] - 40) +
        0.3 * edu_numeric +
        0.02 * (data['hours_per_week'] - 40) +
        0.2 * (data['sex'] == 'male').astype(int) +  # Gender bias
        0.1 * (data['race'] == 'white').astype(int)   # Racial bias
    )
    
    income_probs = 1 / (1 + np.exp(-income_logits))
    data['target'] = np.random.binomial(1, income_probs, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_features = ['education', 'race', 'sex', 'age_group']
    
    for feature in categorical_features:
        if feature in df.columns:
            df[f'{feature}_encoded'] = le.fit_transform(df[feature])
    
    # Feature columns (exclude original categorical and target)
    feature_cols = ['age', 'hours_per_week', 'education_encoded', 'race_encoded', 'sex_encoded']
    sensitive_attrs = config.get('sensitive_attrs', ['age_group', 'race', 'sex'])
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    return {
        'name': 'uci_adult',
        'train': train_df[feature_cols + ['target']],
        'test': test_df[feature_cols + ['target']],
        'sensitive_attrs': sensitive_attrs,
        'feature_names': feature_cols,
        'n_samples': n_samples,
        'metadata': {
            'description': 'UCI Adult Income Dataset (synthetic version)',
            'target': 'Income >50K',
            'sensitive_attributes': sensitive_attrs
        }
    }

def load_communities_crime_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load and preprocess Communities and Crime dataset"""
    
    # Generate synthetic communities crime data
    np.random.seed(42)
    
    n_samples = config.get('sample_size', 2000)
    
    data = {}
    
    # Racial composition (sensitive attributes)
    data['racepctblack'] = np.random.beta(2, 8, n_samples)  # Skewed towards lower values
    data['racePctWhite'] = np.clip(1 - data['racepctblack'] - np.random.beta(2, 10, n_samples), 0, 1)
    data['racePctAsian'] = np.random.beta(1, 20, n_samples)
    data['racePctHisp'] = np.clip(1 - data['racepctblack'] - data['racePctWhite'] - data['racePctAsian'], 0, 1)
    
    # Socioeconomic features
    data['medIncome'] = np.random.lognormal(10, 0.5, n_samples)
    data['pctUnemploy'] = np.random.beta(2, 8, n_samples)
    data['pctPopUnderPov'] = np.random.beta(3, 7, n_samples)
    
    # Create target with bias towards racial composition
    crime_logits = (
        2 * data['racepctblack'] +
        1 * data['racePctHisp'] +
        -0.5 * np.log(data['medIncome'] / 10000) +
        3 * data['pctUnemploy'] +
        2 * data['pctPopUnderPov'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Binary high crime indicator
    data['target'] = (crime_logits > np.median(crime_logits)).astype(int)
    
    df = pd.DataFrame(data)
    
    # Feature columns
    feature_cols = ['medIncome', 'pctUnemploy', 'pctPopUnderPov', 'racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    sensitive_attrs = config.get('sensitive_attrs', ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp'])
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    return {
        'name': 'communities_crime',
        'train': train_df[feature_cols + ['target']],
        'test': test_df[feature_cols + ['target']],
        'sensitive_attrs': sensitive_attrs,
        'feature_names': feature_cols,
        'n_samples': n_samples,
        'metadata': {
            'description': 'Communities and Crime Dataset (synthetic version)',
            'target': 'High Crime Rate',
            'sensitive_attributes': sensitive_attrs
        }
    }

def generate_synthetic_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic dataset with specified parameters"""
    
    np.random.seed(42)
    
    n_samples = config.get('sample_size', 10000)
    n_sensitive_attrs = config.get('n_sensitive_attrs', 4)
    noise_level = config.get('noise_level', 0.0)
    
    data = {}
    
    # Generate sensitive attributes
    sensitive_attr_names = [f'sensitive_{i}' for i in range(n_sensitive_attrs)]
    
    for i, attr_name in enumerate(sensitive_attr_names):
        # Each sensitive attribute has 2-4 categories
        n_categories = np.random.choice([2, 3, 4])
        categories = [f'{attr_name}_cat_{j}' for j in range(n_categories)]
        data[attr_name] = np.random.choice(categories, n_samples)
    
    # Generate regular features
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Add feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    for i, feature_name in enumerate(feature_names):
        data[feature_name] = X[:, i]
    
    # Generate target with bias towards sensitive attributes
    target_logits = np.sum(X, axis=1) * 0.1  # Base signal from features
    
    # Add bias from sensitive attributes
    for i, attr_name in enumerate(sensitive_attr_names):
        # Create bias based on category
        le = LabelEncoder()
        attr_encoded = le.fit_transform(data[attr_name])
        bias_strength = np.random.uniform(0.2, 0.8)
        target_logits += bias_strength * attr_encoded
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, n_samples)
        target_logits += noise
    
    # Convert to binary target
    target_probs = 1 / (1 + np.exp(-target_logits))
    data['target'] = np.random.binomial(1, target_probs, n_samples)
    
    df = pd.DataFrame(data)
    
    # Encode sensitive attributes
    for attr_name in sensitive_attr_names:
        le = LabelEncoder()
        df[f'{attr_name}_encoded'] = le.fit_transform(df[attr_name])
    
    # Feature columns
    feature_cols = feature_names + [f'{attr}_encoded' for attr in sensitive_attr_names]
    
    # Split data  
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    return {
        'name': 'synthetic',
        'train': train_df[feature_cols + ['target']],
        'test': test_df[feature_cols + ['target']],
        'sensitive_attrs': sensitive_attr_names,
        'feature_names': feature_cols,
        'n_samples': n_samples,
        'metadata': {
            'description': f'Synthetic dataset with {n_sensitive_attrs} sensitive attributes',
            'target': 'Synthetic binary target',
            'sensitive_attributes': sensitive_attr_names,
            'noise_level': noise_level
        }
    }

def preprocess_sensitive_attributes(df: pd.DataFrame, sensitive_attrs: List[str]) -> pd.DataFrame:
    """
    Preprocess sensitive attributes for fairness analysis
    
    Args:
        df: DataFrame containing data
        sensitive_attrs: List of sensitive attribute column names
        
    Returns:
        DataFrame with processed sensitive attributes
    """
    
    sensitive_df = df[sensitive_attrs].copy()
    
    # For continuous sensitive attributes, bin them
    for attr in sensitive_attrs:
        if sensitive_df[attr].dtype in ['float64', 'float32']:
            # Create bins for continuous attributes
            sensitive_df[attr] = pd.qcut(sensitive_df[attr], q=4, labels=['low', 'medium_low', 'medium_high', 'high'])
    
    return sensitive_df

def create_train_test_split(df: pd.DataFrame, 
                           target_col: str = 'target',
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train-test split
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    
    return train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[target_col]
    )