"""
Visualization utilities for model analysis.

This module provides functions for creating informative visualizations
of model performance, feature importance, and predictions.
"""
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def plot_cv_metrics(
    cv_results: Dict[str, List],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Cross-Validation Results",
    sort_metrics: bool = True,
) -> Figure:
    """
    Plot cross-validation metrics with error bars.
    
    Parameters
    ----------
    cv_results : Dict[str, List]
        Dictionary of cross-validation results from time_series_cross_validate
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : str, default="Cross-Validation Results"
        Title of the plot
    sort_metrics : bool, default=True
        Whether to sort metrics by their mean value
        
    Returns
    -------
    Figure
        Matplotlib Figure containing the plot
    
    Examples
    --------
    >>> cv_results = time_series_cross_validate(...)
    >>> fig = plot_cv_metrics(cv_results)
    >>> plt.show()
    """
    # Extract metrics (skip non-metric fields)
    metrics = {}
    non_metrics = ['fold', 'train_size', 'test_size', 'train_start_date', 
                   'train_end_date', 'test_start_date', 'test_end_date',
                   'predictions', 'test_targets', 'test_dates']
    
    for metric, values in cv_results.items():
        if metric not in non_metrics and isinstance(values[0], (int, float)):
            metrics[metric] = values
    
    # Calculate mean and std for each metric
    means = {k: np.mean(v) for k, v in metrics.items()}
    stds = {k: np.std(v) for k, v in metrics.items()}
    
    # Create dataframe for plotting
    metrics_df = pd.DataFrame({
        'metric': list(means.keys()),
        'mean': list(means.values()),
        'std': list(stds.values())
    })
    
    # Sort metrics if requested
    if sort_metrics:
        metrics_df = metrics_df.sort_values('mean', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars with error bars
    ax.bar(metrics_df['metric'], metrics_df['mean'], yerr=metrics_df['std'],
           capsize=10, alpha=0.7, color='steelblue')
    
    # Add value labels
    for i, v in enumerate(metrics_df['mean']):
        ax.text(i, v + metrics_df['std'].iloc[i] + 0.02, f"{v:.4f}",
                ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance",
    color_map: str = "viridis",
    method: str = 'native',
    X: Optional[pd.DataFrame] = None,
) -> Figure:
    """
    Plot feature importance for a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model (LightGBM, XGBoost, scikit-learn, or Freamon model)
    feature_names : Optional[List[str]], default=None
        List of feature names. If None, uses model.feature_names_ if available
    top_n : int, default=20
        Number of top features to show
    figsize : Tuple[int, int], default=(10, 8)
        Figure size
    title : str, default="Feature Importance"
        Title of the plot
    color_map : str, default="viridis"
        Matplotlib colormap to use
    method : str, default='native'
        Method to compute feature importance ('native', 'shap', 'shapiq')
    X : Optional[pd.DataFrame], default=None
        Input data, required for SHAP methods
        
    Returns
    -------
    Figure
        Matplotlib Figure containing the plot
        
    Examples
    --------
    >>> model = LGBMRegressor()
    >>> model.fit(X_train, y_train)
    >>> fig = plot_feature_importance(model, X_train.columns)
    >>> plt.show()
    """
    # Get feature importance
    if hasattr(model, 'get_feature_importance'):
        # Freamon model
        importance = model.get_feature_importance(method=method, X=X)
    elif hasattr(model, 'feature_importances_'):
        # scikit-learn, LightGBM, XGBoost
        importances = model.feature_importances_
        if feature_names is None:
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance = pd.Series(importances, index=feature_names)
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_).flatten()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance = pd.Series(importances, index=feature_names)
    else:
        raise ValueError("Model does not support feature importance")
    
    # Sort and select top N features
    importance = importance.sort_values(ascending=False)
    if top_n is not None and len(importance) > top_n:
        importance = importance.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    colormap = plt.cm.get_cmap(color_map)
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(importance))]
    
    importance.sort_values().plot(
        kind='barh',
        ax=ax,
        color=colors,
    )
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    
    return fig


def summarize_feature_importance_by_groups(
    importance: pd.Series,
    feature_groups: Dict[str, List[str]],
    sort: bool = True,
) -> pd.Series:
    """
    Summarize feature importance by predefined groups.
    
    Parameters
    ----------
    importance : pd.Series
        Feature importance series with feature names as index and importance values
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to lists of feature prefixes or full names
    sort : bool, default=True
        Whether to sort the result by importance
        
    Returns
    -------
    pd.Series
        Summary of importance by group
    
    Examples
    --------
    >>> importance = model.get_feature_importance()
    >>> feature_groups = {
    ...     'Text statistics': ['text_stat_'],
    ...     'Text sentiment': ['text_sent_'],
    ...     'Time series': ['lag_', 'rolling_']
    ... }
    >>> summary = summarize_feature_importance_by_groups(importance, feature_groups)
    """
    # Initialize results
    result = {}
    
    # Process each group
    for group_name, patterns in feature_groups.items():
        # Find features matching any pattern in the group
        group_features = []
        for pattern in patterns:
            matches = [f for f in importance.index if pattern in f]
            group_features.extend(matches)
        
        # Calculate total importance for the group
        if group_features:
            group_importance = importance[group_features].sum()
        else:
            group_importance = 0
        
        result[group_name] = group_importance
    
    # Convert to Series
    result_series = pd.Series(result)
    
    # Sort if requested
    if sort:
        result_series = result_series.sort_values(ascending=False)
    
    return result_series


def plot_importance_by_groups(
    importance: pd.Series,
    feature_groups: Dict[str, List[str]],
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Feature Importance by Group",
    color_map: str = "plasma",
    plot_type: str = "bar",
) -> Figure:
    """
    Plot feature importance summarized by groups.
    
    Parameters
    ----------
    importance : pd.Series
        Feature importance series with feature names as index and importance values
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to lists of feature prefixes or full names
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    title : str, default="Feature Importance by Group"
        Title of the plot
    color_map : str, default="plasma"
        Matplotlib colormap to use
    plot_type : str, default="bar"
        Type of plot ('bar', 'pie')
        
    Returns
    -------
    Figure
        Matplotlib Figure containing the plot
        
    Examples
    --------
    >>> importance = model.get_feature_importance()
    >>> feature_groups = {
    ...     'Text statistics': ['text_stat_'],
    ...     'Text sentiment': ['text_sent_'],
    ...     'Time series': ['lag_', 'rolling_']
    ... }
    >>> fig = plot_importance_by_groups(importance, feature_groups)
    >>> plt.show()
    """
    # Summarize importance by groups
    summary = summarize_feature_importance_by_groups(importance, feature_groups)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'bar':
        # Bar plot
        colormap = plt.cm.get_cmap(color_map)
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(summary))]
        
        summary.plot(
            kind='bar',
            ax=ax,
            color=colors,
            rot=45,
        )
        
        # Add value labels
        for i, v in enumerate(summary):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
            
        ax.set_title(title)
        ax.set_ylabel('Importance')
        ax.set_xlabel('Feature Group')
        
    elif plot_type == 'pie':
        # Pie chart
        colormap = plt.cm.get_cmap(color_map)
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(summary))]
        
        # Only keep positive values for pie chart
        pie_data = summary[summary > 0]
        
        wedges, texts, autotexts = ax.pie(
            pie_data,
            labels=pie_data.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
        )
        
        # Ensure equal aspect ratio
        ax.axis('equal')
        ax.set_title(title)
        
        # Make labels more readable
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
            
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'bar' or 'pie'")
    
    plt.tight_layout()
    return fig


def plot_time_series_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    dates: Optional[Union[pd.Series, List, np.ndarray]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Time Series Predictions",
    show_residuals: bool = True,
    plot_band: bool = True,
) -> Figure:
    """
    Plot time series predictions against actual values.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        Actual target values
    y_pred : Union[pd.Series, np.ndarray]
        Predicted values
    dates : Optional[Union[pd.Series, List, np.ndarray]], default=None
        Date values for x-axis. If None, uses indices
    figsize : Tuple[int, int], default=(12, 6)
        Figure size
    title : str, default="Time Series Predictions"
        Title of the plot
    show_residuals : bool, default=True
        Whether to show residuals plot
    plot_band : bool, default=True
        Whether to plot error bands around predictions
        
    Returns
    -------
    Figure
        Matplotlib Figure containing the plot
        
    Examples
    --------
    >>> model = LGBMRegressor()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> fig = plot_time_series_predictions(y_test, y_pred, X_test['date'])
    >>> plt.show()
    """
    # Convert inputs to numpy arrays if they're pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Create dates if not provided
    if dates is None:
        dates = list(range(len(y_true)))
    elif isinstance(dates, pd.Series):
        dates = dates.values
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot actual vs predicted
    ax1.plot(dates, y_true, 'o-', label='Actual', color='steelblue', markersize=4)
    ax1.plot(dates, y_pred, 'o-', label='Predicted', color='coral', markersize=4)
    
    # Add error bands if requested
    if plot_band:
        std_err = np.std(residuals)
        ax1.fill_between(
            dates,
            y_pred - std_err,
            y_pred + std_err,
            color='coral',
            alpha=0.2,
            label=f'±1σ ({std_err:.2f})'
        )
    
    # Add labels
    ax1.set_title(title)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot residuals if requested
    if show_residuals:
        ax2.plot(dates, residuals, 'o-', color='darkred', markersize=3)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Residuals')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Date')
    
    # Format dates if they are datetime objects
    if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_cv_predictions_over_time(
    cv_results: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 8),
    title: str = "Cross-Validation Predictions Over Time",
) -> Figure:
    """
    Plot time series cross-validation predictions.
    
    Parameters
    ----------
    cv_results : Dict[str, Any]
        Results from time_series_cross_validate with save_predictions=True
    figsize : Tuple[int, int], default=(15, 8)
        Figure size
    title : str, default="Cross-Validation Predictions Over Time"
        Title of the plot
        
    Returns
    -------
    Figure
        Matplotlib Figure containing the plot
        
    Examples
    --------
    >>> cv_results = time_series_cross_validate(..., save_predictions=True)
    >>> fig = plot_cv_predictions_over_time(cv_results)
    >>> plt.show()
    """
    # Check if predictions are saved
    if 'predictions' not in cv_results:
        raise ValueError("cv_results does not contain predictions. "
                         "Run time_series_cross_validate with save_predictions=True")
    
    predictions = cv_results['predictions']
    dates = cv_results['test_dates']
    actuals = cv_results['test_targets']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values as continuous line
    all_dates = []
    all_actuals = []
    for fold_dates, fold_actuals in zip(dates, actuals):
        all_dates.extend(fold_dates)
        all_actuals.extend(fold_actuals)
    
    # Convert to pandas Series for easier handling
    if len(all_dates) > 0:
        df_combined = pd.DataFrame({
            'date': all_dates,
            'actual': all_actuals
        })
        df_combined = df_combined.sort_values('date')
        all_dates = df_combined['date'].tolist()
        all_actuals = df_combined['actual'].tolist()
    
    # Plot the continuous actual line
    ax.plot(all_dates, all_actuals, '-', color='black', alpha=0.5, 
            label='Actual', zorder=0)
    
    # Plot individual fold predictions with different colors
    n_folds = len(predictions)
    colormap = plt.cm.get_cmap('tab10')
    colors = [colormap(i) for i in range(n_folds)]
    
    for i, (fold_dates, fold_preds, fold_actuals) in enumerate(zip(dates, predictions, actuals)):
        fold_name = f"Fold {i+1}"
        ax.scatter(fold_dates, fold_actuals, color=colors[i], alpha=0.6, s=30,
                   label=f"Actual ({fold_name})")
        ax.scatter(fold_dates, fold_preds, color=colors[i], marker='x', s=30,
                   label=f"Predicted ({fold_name})")
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    # Format date ticks
    fig.autofmt_xdate()
    
    # Create a better legend (group by fold)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    return fig