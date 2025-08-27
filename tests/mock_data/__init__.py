"""Mock data generators for testing subgroup fairness experiments"""

from .generators import (
    generate_biased_dataset,
    generate_fair_dataset, 
    generate_noisy_dataset,
    generate_extreme_imbalance_dataset,
    generate_high_dimensional_dataset,
    DatasetGenerator
)

__all__ = [
    'generate_biased_dataset',
    'generate_fair_dataset',
    'generate_noisy_dataset', 
    'generate_extreme_imbalance_dataset',
    'generate_high_dimensional_dataset',
    'DatasetGenerator'
]