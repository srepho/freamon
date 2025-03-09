"""
Functions for cross-validation of models.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from freamon.modeling.metrics import calculate_metrics


def cross_validate(
    df: pd.DataFrame,
    target_column: str,
    model_fn: Callable[..., Any],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratify_by: Optional[str] = None,
    problem_type: Literal['classification', 'regression'] = 'classification',
    feature_columns: Optional[List[str]] = None,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform cross-validation on a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to use for cross-validation.
    target_column : str
        The name of the column containing the target values.
    model_fn : Callable[..., Any]
        A function that returns a model object. The function should take
        keyword arguments specified in model_kwargs.
    n_splits : int, default=5
        The number of folds for cross-validation.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : Optional[int], default=None
        Controls the randomness of the CV splitter.
    stratify_by : Optional[str], default=None
        If not None, data is split in a stratified fashion using this column.
        Only applicable for classification problems.
    problem_type : Literal['classification', 'regression'], default='classification'
        The type of problem (classification or regression).
    feature_columns : Optional[List[str]], default=None
        The names of the columns to use as features. If None, uses all columns
        except the target column.
    **model_kwargs
        Additional keyword arguments to pass to the model function.
    
    Returns
    -------
    Dict[str, List[float]]
        A dictionary of metric names mapped to lists of values for each fold.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> df = pd.DataFrame({
    ...     'feature1': range(100),
    ...     'feature2': range(100, 200),
    ...     'target': [0, 1] * 50
    ... })
    >>> def model_fn(**kwargs):
    ...     return RandomForestClassifier(**kwargs)
    >>> results = cross_validate(
    ...     df, 'target', model_fn, n_splits=3, random_state=42,
    ...     n_estimators=100, max_depth=5
    ... )
    >>> list(results.keys())
    ['accuracy', 'precision', 'recall', 'f1']
    """
    try:
        from sklearn.model_selection import StratifiedKFold, KFold
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Validate input
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataframe")
    
    # Determine feature columns
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    else:
        # Validate feature columns
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in dataframe")
    
    # Prepare the CV splitter
    if problem_type == 'classification' and stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"Column '{stratify_by}' not found in dataframe")
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        split_args = (df.index, df[stratify_by])
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        split_args = (df.index,)
    
    # Initialize metrics dictionary
    metrics_dict: Dict[str, List[float]] = {}
    
    # Perform cross-validation
    for train_idx, test_idx in splitter.split(*split_args):
        # Split data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Extract features and target
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]
        
        # Create and train the model
        model = model_fn(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For classification, get predicted probabilities if available
        y_proba = None
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except (ValueError, AttributeError):
                pass
        
        # Calculate metrics
        fold_metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            problem_type=problem_type,
            y_proba=y_proba,
        )
        
        # Add metrics to the dictionary
        for metric, value in fold_metrics.items():
            if metric not in metrics_dict:
                metrics_dict[metric] = []
            metrics_dict[metric].append(value)
    
    return metrics_dict


def time_series_cross_validate(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    model_fn: Callable[..., Any],
    n_splits: int = 5,
    gap: Optional[Union[str, pd.Timedelta]] = None,
    problem_type: Literal['classification', 'regression'] = 'regression',
    feature_columns: Optional[List[str]] = None,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform time series cross-validation on a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to use for cross-validation.
    target_column : str
        The name of the column containing the target values.
    date_column : str
        The name of the column containing the date/time values.
    model_fn : Callable[..., Any]
        A function that returns a model object. The function should take
        keyword arguments specified in model_kwargs.
    n_splits : int, default=5
        The number of folds for cross-validation.
    gap : Optional[Union[str, pd.Timedelta]], default=None
        If not None, leaves a gap between train and test sets.
    problem_type : Literal['classification', 'regression'], default='regression'
        The type of problem (classification or regression).
    feature_columns : Optional[List[str]], default=None
        The names of the columns to use as features. If None, uses all columns
        except the target and date columns.
    **model_kwargs
        Additional keyword arguments to pass to the model function.
    
    Returns
    -------
    Dict[str, List[float]]
        A dictionary of metric names mapped to lists of values for each fold.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2020-01-01', periods=100),
    ...     'feature1': range(100),
    ...     'feature2': range(100, 200),
    ...     'target': range(200, 300)
    ... })
    >>> def model_fn(**kwargs):
    ...     return LinearRegression(**kwargs)
    >>> results = time_series_cross_validate(
    ...     df, 'target', 'date', model_fn, n_splits=3, gap='1D'
    ... )
    >>> list(results.keys())
    ['mse', 'rmse', 'mae', 'r2', 'explained_variance', 'median_ae', 'mape']
    """
    try:
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Validate input
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataframe")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    # Sort the DataFrame by date
    df = df.sort_values(by=date_column).reset_index(drop=True)
    
    # Determine feature columns
    if feature_columns is None:
        feature_columns = [
            col for col in df.columns 
            if col not in [target_column, date_column]
        ]
    else:
        # Validate feature columns
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in dataframe")
    
    # Create a TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize metrics dictionary
    metrics_dict: Dict[str, List[float]] = {}
    
    # Perform cross-validation
    for train_idx, test_idx in tscv.split(df):
        # Apply gap if specified
        if gap is not None:
            # Convert string to Timedelta if needed
            if isinstance(gap, str):
                gap = pd.Timedelta(gap)
            
            # Calculate the last date in the training set
            last_train_date = df.iloc[train_idx[-1]][date_column]
            
            # Get the indices where the date is greater than last_train_date + gap
            test_idx = df.index[
                (df.index.isin(test_idx)) & 
                (df[date_column] > (last_train_date + gap))
            ].tolist()
            
            # If no test indices remain after applying the gap, skip this fold
            if not test_idx:
                continue
        
        # Split data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Extract features and target
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]
        
        # Create and train the model
        model = model_fn(**model_kwargs)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For classification, get predicted probabilities if available
        y_proba = None
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except (ValueError, AttributeError):
                pass
        
        # Calculate metrics
        fold_metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            problem_type=problem_type,
            y_proba=y_proba,
        )
        
        # Add metrics to the dictionary
        for metric, value in fold_metrics.items():
            if metric not in metrics_dict:
                metrics_dict[metric] = []
            metrics_dict[metric].append(value)
    
    return metrics_dict