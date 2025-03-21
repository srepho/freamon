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

# Problem type constants for easier usage
REGRESSION = 'regression'
CLASSIFICATION = 'classification'

# Model type constants
LIGHTGBM = 'lightgbm'
SKLEARN = 'sklearn'
XGBOOST = 'xgboost'
CATBOOST = 'catboost'

__all__ = [
    # Functions
    "create_model",
    "calculate_metrics",
    "create_lightgbm_regressor",
    "create_lightgbm_classifier", 
    "create_sklearn_model",
    
    # Classes
    "Model",
    "ModelTrainer",
    "LightGBMModel",
    
    # Constants
    "REGRESSION",
    "CLASSIFICATION",
    "LIGHTGBM",
    "SKLEARN",
    "XGBOOST",
    "CATBOOST",
]