"""
Module for model training, prediction, and evaluation.
"""

from freamon.modeling.factory import create_model
from freamon.modeling.metrics import calculate_metrics
from freamon.modeling.model import Model
from freamon.modeling.trainer import ModelTrainer

__all__ = [
    "create_model",
    "calculate_metrics",
    "Model",
    "ModelTrainer",
]