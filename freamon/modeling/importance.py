"""
Feature importance calculation methods.

This module provides functions for calculating feature importance
using different methods, including permutation importance, and
tools for intelligent feature selection based on importance.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def calculate_permutation_importance(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    scoring: Optional[Union[str, Callable]] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate permutation feature importance for a fitted model.
    
    Permutation importance is a model inspection technique that measures the
    increase in the prediction error after we permute the feature's values,
    which breaks the relationship between the feature and the target.
    A feature is important if shuffling its values increases the model error,
    because the model relied on the feature for the prediction.
    
    Parameters
    ----------
    model : Any
        A fitted model object with a predict method.
    X : Union[pd.DataFrame, np.ndarray]
        The feature data used for importance calculation.
    y : Union[pd.Series, np.ndarray]
        The target values.
    n_repeats : int, default=10
        Number of times to permute a feature.
    random_state : Optional[int], default=None
        Controls the randomness of the permutation.
    scoring : Optional[Union[str, Callable]], default=None
        Scoring metric to use. If None, uses r2 for regression and accuracy for
        classification. If string, must be one of:
        - 'r2', 'mse', 'rmse' for regression
        - 'accuracy', 'f1', 'roc_auc', 'log_loss' for classification
        If callable, must take (y_true, y_pred) as input and return a scalar.
    n_jobs : Optional[int], default=None
        Number of jobs to run in parallel. None means 1.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'importances_mean': Mean importance for each feature
        - 'importances_std': Standard deviation of importance for each feature
        - 'importances': Raw importance data for each feature
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(random_state=0)
    >>> model = RandomForestClassifier(random_state=0).fit(X, y)
    >>> result = calculate_permutation_importance(model, X, y)
    >>> print(result['importances_mean'])
    [0.1  0.05 0.01 0.02 0.15 ...]
    """
    # Determine if it's classification or regression
    is_classifier = hasattr(model, 'classes_')
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_np = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_np = X
    
    if isinstance(y, pd.Series):
        y_np = y.values
    else:
        y_np = y
    
    # Handle scoring parameter
    if scoring is None:
        scoring = 'r2' if not is_classifier else 'accuracy'
    
    if isinstance(scoring, str):
        if is_classifier:
            if scoring == 'accuracy':
                scoring_func = accuracy_score
            elif scoring == 'f1':
                scoring_func = f1_score
            elif scoring == 'roc_auc':
                def scoring_func(y_true, y_pred):
                    try:
                        # For probability predictions
                        return roc_auc_score(y_true, y_pred)
                    except ValueError:
                        # For class predictions, try to get proba if available
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_np)
                            if y_proba.shape[1] == 2:
                                return roc_auc_score(y_true, y_proba[:, 1])
                            else:
                                return roc_auc_score(y_true, y_proba, multi_class='ovr')
                        else:
                            raise ValueError("Model doesn't support predict_proba for ROC AUC")
            elif scoring == 'log_loss':
                def scoring_func(y_true, y_pred):
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_np)
                        return -log_loss(y_true, y_proba)  # Negative for consistent interpretation
                    else:
                        raise ValueError("Model doesn't support predict_proba for log loss")
            else:
                raise ValueError(f"Unknown scoring for classification: {scoring}")
        else:  # Regression
            if scoring == 'r2':
                scoring_func = r2_score
            elif scoring == 'mse':
                scoring_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
            elif scoring == 'rmse':
                scoring_func = lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
            else:
                raise ValueError(f"Unknown scoring for regression: {scoring}")
    else:
        # Assume scoring is a callable
        scoring_func = scoring

    # Calculate permutation importance
    result = permutation_importance(
        model, X_np, y_np,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring_func,
        n_jobs=n_jobs
    )
    
    # Create the return dictionary with feature names
    importance_dict = {
        'importances_mean': result.importances_mean,
        'importances_std': result.importances_std,
        'importances': result.importances,
        'feature_names': feature_names
    }
    
    return importance_dict


def get_permutation_importance_df(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    scoring: Optional[Union[str, Callable]] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance and return as a DataFrame.
    
    Parameters
    ----------
    model : Any
        A fitted model object with a predict method.
    X : Union[pd.DataFrame, np.ndarray]
        The feature data used for importance calculation.
    y : Union[pd.Series, np.ndarray]
        The target values.
    n_repeats : int, default=10
        Number of times to permute a feature.
    random_state : Optional[int], default=None
        Controls the randomness of the permutation.
    scoring : Optional[Union[str, Callable]], default=None
        Scoring metric to use. If None, uses r2 for regression and accuracy for
        classification. See calculate_permutation_importance for details.
    n_jobs : Optional[int], default=None
        Number of jobs to run in parallel. None means 1.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the feature importance results, sorted by importance.
        Columns: 'feature', 'importance', 'std'
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = RandomForestRegressor(random_state=0).fit(X, y)
    >>> df = get_permutation_importance_df(model, X, y)
    >>> print(df.head())
       feature  importance       std
    0  feature_3    0.62341  0.11235
    1  feature_8    0.48291  0.09872
    ...
    """
    result = calculate_permutation_importance(
        model, X, y, n_repeats, random_state, scoring, n_jobs
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': result['feature_names'],
        'importance': result['importances_mean'],
        'std': result['importances_std']
    })
    
    # Sort by importance
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return df


def plot_permutation_importance(
    result: Union[Dict[str, np.ndarray], pd.DataFrame],
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    title: str = "Permutation Feature Importance",
    return_fig: bool = False,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot permutation feature importance.
    
    Parameters
    ----------
    result : Union[Dict[str, np.ndarray], pd.DataFrame]
        The result from calculate_permutation_importance or get_permutation_importance_df.
    top_n : Optional[int], default=None
        Number of top features to plot. If None, all features are plotted.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    show : bool, default=True
        Whether to show the plot.
    title : str, default="Permutation Feature Importance"
        Plot title.
    return_fig : bool, default=False
        Whether to return the figure.
    save_path : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.
        
    Returns
    -------
    Optional[Any]
        The figure if return_fig is True, otherwise None.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = RandomForestRegressor(random_state=0).fit(X, y)
    >>> result = calculate_permutation_importance(model, X, y)
    >>> plot_permutation_importance(result, top_n=5)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with 'pip install matplotlib'."
        )
    
    # Convert dictionary to DataFrame if needed
    if isinstance(result, dict):
        df = pd.DataFrame({
            'feature': result['feature_names'],
            'importance': result['importances_mean'],
            'std': result['importances_std']
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    else:
        df = result.copy()
    
    # Select top N features if specified
    if top_n is not None:
        df = df.head(top_n)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort for better visualization (lowest to highest for horizontal bar)
    df = df.sort_values('importance', ascending=True)
    
    # Plot horizontal bars
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['importance'], xerr=df['std'], 
            align='center', alpha=0.8, ecolor='black', capsize=5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add grid lines for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if return_fig:
        return fig
    return None


def select_features_by_importance(
    importance_df: pd.DataFrame,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    min_features: int = 1,
    max_features: Optional[int] = None,
    cumulative_importance: Optional[float] = None,
    features_to_keep: Optional[List[str]] = None,
) -> List[str]:
    """
    Select features based on importance values using multiple strategies.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame containing feature importance values.
        Must have columns 'feature' and 'importance'.
    threshold : Optional[float], default=None
        Minimum importance threshold to keep a feature.
    top_n : Optional[int], default=None
        Number of top features to keep.
    min_features : int, default=1
        Minimum number of features to select, regardless of other criteria.
    max_features : Optional[int], default=None
        Maximum number of features to select.
    cumulative_importance : Optional[float], default=None
        Select features until their cumulative importance reaches this threshold.
        Value should be between 0 and 1 (e.g., 0.95 for 95% of total importance).
    features_to_keep : Optional[List[str]], default=None
        Features to always keep, regardless of importance.
    
    Returns
    -------
    List[str]
        List of selected feature names.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = RandomForestRegressor(random_state=0).fit(X, y)
    >>> imp_df = get_permutation_importance_df(model, X, y)
    >>> # Select top 5 features
    >>> select_features_by_importance(imp_df, top_n=5)
    ['feature_3', 'feature_8', 'feature_1', ...]
    >>> # Select features with importance > 0.05
    >>> select_features_by_importance(imp_df, threshold=0.05)
    ['feature_3', 'feature_8', 'feature_1', ...]
    >>> # Select features accounting for 90% of total importance
    >>> select_features_by_importance(imp_df, cumulative_importance=0.9)
    ['feature_3', 'feature_8', 'feature_1', ...]
    """
    # Verify the DataFrame has the required columns
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        raise ValueError("importance_df must have 'feature' and 'importance' columns")
    
    # Ensure importance values are non-negative
    if (importance_df['importance'] < 0).any():
        # Handle negative importance values (e.g., from negative scoring functions)
        importance_df = importance_df.copy()
        importance_df['importance'] = np.abs(importance_df['importance'])
    
    # Sort by importance in descending order
    sorted_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Initialize with all features
    all_features = sorted_df['feature'].tolist()
    selected_features = all_features.copy()
    
    # First apply threshold if specified
    if threshold is not None:
        selected_features = sorted_df[sorted_df['importance'] >= threshold]['feature'].tolist()
    
    # Apply top_n if specified
    if top_n is not None:
        top_n_features = sorted_df.head(top_n)['feature'].tolist()
        if threshold is not None:
            # Intersection with threshold-selected features
            selected_features = [f for f in top_n_features if f in selected_features]
        else:
            selected_features = top_n_features
    
    # Apply cumulative_importance if specified
    if cumulative_importance is not None:
        if cumulative_importance <= 0 or cumulative_importance > 1:
            raise ValueError("cumulative_importance must be between 0 and 1")
        
        # Calculate total importance
        total_importance = sorted_df['importance'].sum()
        
        # Calculate cumulative importance
        sorted_df['cumulative'] = sorted_df['importance'].cumsum() / total_importance
        
        # Select features up to cumulative threshold
        cum_features = sorted_df[sorted_df['cumulative'] <= cumulative_importance]['feature'].tolist()
        
        # Add one more feature to exceed the threshold
        if cum_features and len(cum_features) < len(all_features):
            next_feature_idx = len(cum_features)
            if next_feature_idx < len(sorted_df):
                cum_features.append(sorted_df.iloc[next_feature_idx]['feature'])
        
        if threshold is not None or top_n is not None:
            # Intersection with previously selected features
            selected_features = [f for f in cum_features if f in selected_features]
        else:
            selected_features = cum_features
    
    # Ensure min_features is respected
    if len(selected_features) < min_features:
        # Add the highest importance features that weren't selected
        missing = min_features - len(selected_features)
        additional_features = [f for f in all_features if f not in selected_features]
        selected_features.extend(additional_features[:missing])
    
    # Apply max_features if specified
    if max_features is not None and len(selected_features) > max_features:
        # Keep only the top max_features
        selected_features = selected_features[:max_features]
    
    # Add features_to_keep
    if features_to_keep:
        # Add only the ones not already in the list
        for feat in features_to_keep:
            if feat not in selected_features:
                selected_features.append(feat)
    
    return selected_features


def auto_select_features(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    selection_method: str = 'auto',
    importance_method: str = 'permutation',
    n_repeats: int = 10,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    cumulative_importance: Optional[float] = 0.95,
    scoring: Optional[Union[str, Callable]] = None,
    features_to_keep: Optional[List[str]] = None,
    min_features: int = 1,
    max_features: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Automatically select features using model importance.
    
    This function combines feature importance calculation and selection
    into a single convenient method.
    
    Parameters
    ----------
    model : Any
        A fitted model with feature importance capabilities.
    X : Union[pd.DataFrame, np.ndarray]
        The feature data.
    y : Union[pd.Series, np.ndarray]
        The target values.
    selection_method : str, default='auto'
        Method to use for feature selection:
        - 'auto': Use cumulative importance with default 95% threshold
        - 'threshold': Select features above the importance threshold
        - 'top_n': Select top n features
        - 'cumulative': Select features until cumulative importance threshold is reached
    importance_method : str, default='permutation'
        Method to use for calculating feature importance:
        - 'permutation': Use permutation importance
        - 'native': Use model's native feature importance (fallback if not available)
    n_repeats : int, default=10
        Number of times to permute a feature (for permutation importance).
    threshold : Optional[float], default=None
        Minimum importance threshold (for 'threshold' selection_method).
    top_n : Optional[int], default=None
        Number of top features to keep (for 'top_n' selection_method).
    cumulative_importance : Optional[float], default=0.95
        Cumulative importance threshold (for 'cumulative' or 'auto' selection_method).
    scoring : Optional[Union[str, Callable]], default=None
        Scoring metric for permutation importance.
    features_to_keep : Optional[List[str]], default=None
        Features to always keep, regardless of importance.
    min_features : int, default=1
        Minimum number of features to select.
    max_features : Optional[int], default=None
        Maximum number of features to select.
    random_state : Optional[int], default=None
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    
    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        A tuple containing:
        - List of selected feature names
        - DataFrame with feature importance results
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=20, random_state=0)
    >>> model = RandomForestRegressor(random_state=0).fit(X, y)
    >>> features, importances = auto_select_features(model, X, y, selection_method='top_n', top_n=10)
    >>> print(f"Selected {len(features)} features")
    Selected 10 features
    """
    # Ensure X is a DataFrame for feature names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    # Calculate feature importance
    if importance_method == 'permutation':
        # Use permutation importance
        imp_result = calculate_permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs
        )
        imp_df = pd.DataFrame({
            'feature': imp_result['feature_names'],
            'importance': imp_result['importances_mean'],
            'std': imp_result['importances_std']
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    elif importance_method == 'native':
        # Try to use model's native feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        elif hasattr(model, 'get_feature_importance'):
            # For LightGBM models wrapped by Freamon
            importances = model.get_feature_importance()
        else:
            raise ValueError(f"Model doesn't have feature importance capabilities")
        
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown importance_method: {importance_method}")
    
    # Set selection parameters based on selection_method
    select_params = {
        'importance_df': imp_df,
        'features_to_keep': features_to_keep,
        'min_features': min_features,
        'max_features': max_features
    }
    
    if selection_method == 'auto':
        select_params['cumulative_importance'] = cumulative_importance
    elif selection_method == 'threshold':
        if threshold is None:
            raise ValueError("threshold must be specified when selection_method='threshold'")
        select_params['threshold'] = threshold
    elif selection_method == 'top_n':
        if top_n is None:
            raise ValueError("top_n must be specified when selection_method='top_n'")
        select_params['top_n'] = top_n
    elif selection_method == 'cumulative':
        if cumulative_importance is None:
            raise ValueError("cumulative_importance must be specified when selection_method='cumulative'")
        select_params['cumulative_importance'] = cumulative_importance
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    
    # Select features
    selected_features = select_features_by_importance(**select_params)
    
    return selected_features, imp_df