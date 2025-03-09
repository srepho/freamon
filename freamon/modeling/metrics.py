"""
Module for evaluating model performance.
"""
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    problem_type: Literal['classification', 'regression'],
    average: Optional[str] = 'macro',
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate performance metrics for a model.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        The true target values.
    y_pred : Union[pd.Series, np.ndarray]
        The predicted target values.
    problem_type : Literal['classification', 'regression']
        The type of problem (classification or regression).
    average : Optional[str], default='macro'
        The averaging method for classification metrics.
        Options: 'micro', 'macro', 'weighted', 'samples', None.
        Only used for multiclass classification.
    y_proba : Optional[Union[pd.Series, np.ndarray]], default=None
        The predicted probabilities for classification problems.
        Required for metrics like ROC AUC and log loss.
    
    Returns
    -------
    Dict[str, float]
        A dictionary of metric names and values.
    
    Examples
    --------
    >>> # Classification metrics
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1]
    >>> metrics = calculate_metrics(y_true, y_pred, 'classification')
    >>> 
    >>> # Regression metrics
    >>> y_true = [1.5, 2.1, 3.3, 4.7]
    >>> y_pred = [1.7, 1.9, 3.2, 4.9]
    >>> metrics = calculate_metrics(y_true, y_pred, 'regression')
    """
    if problem_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred, average, y_proba)
    elif problem_type == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    average: Optional[str] = 'macro',
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        The true class labels.
    y_pred : Union[pd.Series, np.ndarray]
        The predicted class labels.
    average : Optional[str], default='macro'
        The averaging method for multiclass metrics.
    y_proba : Optional[Union[pd.Series, np.ndarray]], default=None
        The predicted probabilities for ROC AUC and log loss.
    
    Returns
    -------
    Dict[str, float]
        A dictionary of classification metrics.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss,
            confusion_matrix,
            classification_report,
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_proba is not None and isinstance(y_proba, pd.Series):
        y_proba = y_proba.values
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                # If y_proba is 2D, use the second column (probability of the positive class)
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
            # For multiclass classification
            else:
                # For ROC AUC, we need probabilities for each class
                if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, average=average, multi_class='ovr'
                    )
        except ValueError:
            # ROC AUC may fail if there is only one class in y_true
            metrics['roc_auc'] = float('nan')
        
        # Calculate log loss
        try:
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except ValueError:
            metrics['log_loss'] = float('nan')
    
    return metrics


def calculate_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        The true target values.
    y_pred : Union[pd.Series, np.ndarray]
        The predicted target values.
    
    Returns
    -------
    Dict[str, float]
        A dictionary of regression metrics.
    """
    try:
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            explained_variance_score,
            median_absolute_error,
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'median_ae': median_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100,
    }
    
    return metrics