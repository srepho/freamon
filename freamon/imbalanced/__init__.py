"""
Imbalanced dataset handling module for freamon.

This module provides tools and utilities for handling imbalanced datasets in machine learning tasks.
"""

from .samplers import (
    RandomOverSampler, 
    RandomUnderSampler,
    SMOTE, 
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
    KMeansSMOTE,
    TomekLinks,
    NeighbourhoodCleaningRule,
    EditedNearestNeighbours,
    CondensedNearestNeighbour,
    OneSidedSelection,
    BalancedRandomForestSampler,
    SMOTEBoostSampler
)

from .pipeline import (
    ImbalancedPipeline,
    ImbalancedSamplingStep
)

from .evaluators import (
    ImbalancedEvaluator,
    geometric_mean_score,
    balanced_accuracy_score
)

from .metrics import (
    specificity_score,
    sensitivity_score,
    balanced_fbeta_score,
    balanced_precision_score,
    balanced_recall_score
)

from .utils import (
    calculate_class_weights,
    create_balanced_batches,
    summarize_class_distribution
)

__all__ = [
    # Samplers
    'RandomOverSampler', 
    'RandomUnderSampler',
    'SMOTE', 
    'ADASYN',
    'BorderlineSMOTE',
    'SVMSMOTE',
    'KMeansSMOTE',
    'TomekLinks',
    'NeighbourhoodCleaningRule',
    'EditedNearestNeighbours',
    'CondensedNearestNeighbour',
    'OneSidedSelection',
    'BalancedRandomForestSampler',
    'SMOTEBoostSampler',
    
    # Pipeline integration
    'ImbalancedPipeline',
    'ImbalancedSamplingStep',
    
    # Evaluators
    'ImbalancedEvaluator',
    'geometric_mean_score',
    'balanced_accuracy_score',
    
    # Metrics
    'specificity_score',
    'sensitivity_score',
    'balanced_fbeta_score',
    'balanced_precision_score',
    'balanced_recall_score',
    
    # Utilities
    'calculate_class_weights',
    'create_balanced_batches',
    'summarize_class_distribution'
]