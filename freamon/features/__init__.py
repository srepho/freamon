"""
Module for feature engineering and feature selection.

This module contains functions and classes for feature engineering and selection,
including polynomial features, interaction terms, datetime-based features, and
various feature selection methods.
"""

from freamon.features.engineer import (
    FeatureEngineer,
    create_polynomial_features,
    create_interaction_features,
    create_datetime_features,
    create_binned_features,
    create_lagged_features,
)
from freamon.features.selector import (
    # Basic feature selection methods
    select_features,
    select_by_correlation,
    select_by_importance,
    select_by_variance,
    select_by_mutual_info,
    select_by_kbest,
    select_by_percentile,
    
    # Advanced feature selection classes
    RecursiveFeatureEliminationCV,
    StabilitySelector,
    GeneticFeatureSelector,
    MultiObjectiveFeatureSelector,
    TimeSeriesFeatureSelector,
    
    # Advanced feature selection wrapper functions
    select_features_rfecv,
    select_features_stability,
    select_features_genetic,
    select_features_multi_objective,
    select_features_time_series,
)

# Import ShapIQ feature engineering for automatic interaction detection
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer

__all__ = [
    # Feature engineering
    "FeatureEngineer",
    "create_polynomial_features",
    "create_interaction_features",
    "create_datetime_features",
    "create_binned_features",
    "create_lagged_features",
    "ShapIQFeatureEngineer",
    
    # Basic feature selection
    "select_features",
    "select_by_correlation",
    "select_by_importance",
    "select_by_variance",
    "select_by_mutual_info",
    "select_by_kbest",
    "select_by_percentile",
    
    # Advanced feature selection classes
    "RecursiveFeatureEliminationCV",
    "StabilitySelector",
    "GeneticFeatureSelector",
    "MultiObjectiveFeatureSelector",
    "TimeSeriesFeatureSelector",
    
    # Advanced feature selection wrapper functions
    "select_features_rfecv",
    "select_features_stability",
    "select_features_genetic",
    "select_features_multi_objective",
    "select_features_time_series",
]