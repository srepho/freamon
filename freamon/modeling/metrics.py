"""
Module for evaluating model performance.
"""
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from scipy import interpolate


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


def find_optimal_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    metric: Union[str, Callable] = 'f1',
    thresholds: Optional[Union[int, List[float], np.ndarray]] = 100,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find the optimal probability threshold for a binary classification problem.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        The true class labels (0 or 1).
    y_proba : Union[pd.Series, np.ndarray]
        The predicted probabilities for the positive class.
    metric : Union[str, Callable], default='f1'
        The metric to optimize for. Available options:
        - 'f1': F1 score (harmonic mean of precision and recall)
        - 'precision': Precision score
        - 'recall': Recall score
        - 'accuracy': Accuracy score
        - 'balanced_accuracy': Balanced accuracy score
        - 'f_beta': F-beta score (harmonic mean of precision and recall weighted by beta)
        - 'precision_recall_product': Product of precision and recall
        - 'younden_j': Younden's J statistic (sensitivity + specificity - 1)
        - 'kappa': Cohen's kappa score
        - 'mcc': Matthews correlation coefficient
        - A custom callable function that takes (y_true, y_pred) and returns a score to maximize
    thresholds : Optional[Union[int, List[float], np.ndarray]], default=100
        The thresholds to evaluate. If an integer, generates that many thresholds linearly spaced
        between 0 and 1. If a list or array, uses those specific threshold values.
    
    Returns
    -------
    Tuple[float, float, pd.DataFrame]
        A tuple containing:
        - The optimal threshold value
        - The score achieved at the optimal threshold
        - A DataFrame with threshold values and resulting metric scores
    
    Examples
    --------
    >>> # Find optimal threshold to maximize F1 score
    >>> from freamon.modeling.metrics import find_optimal_threshold
    >>> threshold, score, results = find_optimal_threshold(y_true, y_proba)
    >>> 
    >>> # Find optimal threshold to maximize precision
    >>> threshold, score, results = find_optimal_threshold(y_true, y_proba, metric='precision')
    >>> 
    >>> # Find optimal threshold to maximize a custom metric
    >>> def custom_metric(y_true, y_pred):
    >>>     # Return a score to maximize
    >>>     return 0.3 * precision + 0.7 * recall
    >>> threshold, score, results = find_optimal_threshold(y_true, y_proba, metric=custom_metric)
    """
    try:
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, accuracy_score,
            balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
            fbeta_score, confusion_matrix
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_proba, pd.Series):
        y_proba = y_proba.values
    
    # Generate thresholds if an integer is provided
    if isinstance(thresholds, int):
        thresholds = np.linspace(0.01, 0.99, thresholds)
    
    # Define metric function based on input
    if metric == 'f1':
        metric_func = lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0)
    elif metric == 'precision':
        metric_func = lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0)
    elif metric == 'recall':
        metric_func = lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    elif metric == 'accuracy':
        metric_func = accuracy_score
    elif metric == 'balanced_accuracy':
        metric_func = balanced_accuracy_score
    elif metric == 'f_beta':
        # Default to f_beta with beta=0.5 (more weight on precision)
        metric_func = lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    elif metric == 'precision_recall_product':
        metric_func = lambda y_true, y_pred: (
            precision_score(y_true, y_pred, zero_division=0) *
            recall_score(y_true, y_pred, zero_division=0)
        )
    elif metric == 'younden_j':
        def younden_j_metric(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return sensitivity + specificity - 1
        metric_func = younden_j_metric
    elif metric == 'kappa':
        metric_func = cohen_kappa_score
    elif metric == 'mcc':
        metric_func = matthews_corrcoef
    elif callable(metric):
        metric_func = metric
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Evaluate each threshold
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = metric_func(y_true, y_pred)
        results.append({'threshold': threshold, 'score': score})
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find the optimal threshold
    best_idx = results_df['score'].argmax()
    optimal_threshold = results_df.iloc[best_idx]['threshold']
    optimal_score = results_df.iloc[best_idx]['score']
    
    return optimal_threshold, optimal_score, results_df


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