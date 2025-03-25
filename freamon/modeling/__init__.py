"""
Module for model training, prediction, and evaluation.
"""

from freamon.modeling.factory import create_model
from freamon.modeling.metrics import calculate_metrics
from freamon.modeling.model import Model
from freamon.modeling.trainer import ModelTrainer
from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.helpers import (
    create_lightgbm_regressor,
    create_lightgbm_classifier,
    create_sklearn_model,
)
from freamon.modeling.visualization import (
    plot_cv_metrics,
    plot_feature_importance,
    summarize_feature_importance_by_groups,
    plot_importance_by_groups,
    plot_time_series_predictions,
    plot_cv_predictions_over_time,
)
# Import new autoflow functionality
from freamon.modeling.autoflow import AutoModelFlow, auto_model

# Problem type constants for easier usage
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

# Model type constants
LIGHTGBM = 'lightgbm'
SKLEARN = 'sklearn'
XGBOOST = 'xgboost'
CATBOOST = 'catboost'

# Common feature group patterns for analysis
TEXT_FEATURE_GROUPS = {
    'Text Statistics': ['text_stat_'],
    'Text Readability': ['text_read_'],
    'Text Sentiment': ['text_sent_'],
    'Bag-of-Words': ['bow_'],
    'TF-IDF': ['tfidf_'],
    'Topic Model': ['_Topic_']
}

TIME_SERIES_FEATURE_GROUPS = {
    'Lag Features': ['_lag_'],
    'Rolling Features': ['_rolling_'],
    'Difference Features': ['_diff_'],
    'Seasonal Features': ['_seasonal_'],
    'Date Features': ['_year', '_month', '_day', '_dayofweek', '_quarter']
}

# Combined feature groups for typical use cases
DEFAULT_FEATURE_GROUPS = {
    **TEXT_FEATURE_GROUPS,
    **TIME_SERIES_FEATURE_GROUPS,
}

__all__ = [
    # Functions
    "create_model",
    "calculate_metrics",
    "create_lightgbm_regressor",
    "create_lightgbm_classifier", 
    "create_sklearn_model",
    "auto_model",  # New automated modeling function
    
    # Visualization functions
    "plot_cv_metrics",
    "plot_feature_importance",
    "summarize_feature_importance_by_groups",
    "plot_importance_by_groups",
    "plot_time_series_predictions",
    "plot_cv_predictions_over_time",
    
    # Classes
    "Model",
    "ModelTrainer",
    "LightGBMModel",
    "AutoModelFlow",  # New automated modeling class
    
    # Constants
    "REGRESSION",
    "CLASSIFICATION",
    "LIGHTGBM",
    "SKLEARN",
    "XGBOOST",
    "CATBOOST",
    
    # Feature groups
    "TEXT_FEATURE_GROUPS",
    "TIME_SERIES_FEATURE_GROUPS",
    "DEFAULT_FEATURE_GROUPS",
]