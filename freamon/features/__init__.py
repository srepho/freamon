"""
Module for feature engineering and feature selection.
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
    select_features,
    select_by_correlation,
    select_by_importance,
    select_by_variance,
    select_by_mutual_info,
)

# Import ShapIQ feature engineering for automatic interaction detection
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer

__all__ = [
    "FeatureEngineer",
    "create_polynomial_features",
    "create_interaction_features",
    "create_datetime_features",
    "create_binned_features",
    "create_lagged_features",
    "select_features",
    "select_by_correlation",
    "select_by_importance",
    "select_by_variance",
    "select_by_mutual_info",
    "ShapIQFeatureEngineer",
]