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
    expanding_window: bool = False,
    save_predictions: bool = False,
    **model_kwargs
) -> Dict[str, List[Any]]:
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
    expanding_window : bool, default=False
        If True, uses an expanding window approach where each new split includes
        all previously seen data plus the new fold. This simulates production scenarios
        where you retrain with all available historical data.
    save_predictions : bool, default=False
        If True, saves the predictions for each fold for later analysis and visualization.
        This allows plotting predictions over time.
    **model_kwargs
        Additional keyword arguments to pass to the model function.
    
    Returns
    -------
    Dict[str, List[Any]]
        A dictionary of metric names mapped to lists of values for each fold.
        If save_predictions=True, also includes 'predictions', 'test_targets', and 'test_dates'.
    
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
    ...     df, 'target', 'date', model_fn, n_splits=3, gap='1D', expanding_window=True
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
    metrics_dict: Dict[str, List[Any]] = {}
    
    # If saving predictions, initialize lists
    if save_predictions:
        metrics_dict['predictions'] = []
        metrics_dict['test_targets'] = []
        metrics_dict['test_dates'] = []
    
    # For expanding window approach, we need to track all previous train indices
    all_previous_indices = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
        # For expanding window, include all previous train data
        if expanding_window and fold_idx > 0:
            # Combine with previous train indices
            train_idx = np.concatenate([all_previous_indices, train_idx])
        
        # Store current train indices for next iteration if using expanding window
        if expanding_window:
            # Store a copy of the current train indices
            all_previous_indices = train_idx.copy()
        
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
        
        # Save predictions if requested
        if save_predictions:
            metrics_dict['predictions'].append(y_pred)
            metrics_dict['test_targets'].append(y_test.values)
            metrics_dict['test_dates'].append(test_df[date_column].values)
        
        # Save fold information for debugging and analysis
        if 'fold' not in metrics_dict:
            metrics_dict['fold'] = []
        if 'train_size' not in metrics_dict:
            metrics_dict['train_size'] = []
        if 'test_size' not in metrics_dict:
            metrics_dict['test_size'] = []
        if 'train_start_date' not in metrics_dict:
            metrics_dict['train_start_date'] = []
        if 'train_end_date' not in metrics_dict:
            metrics_dict['train_end_date'] = []
        if 'test_start_date' not in metrics_dict:
            metrics_dict['test_start_date'] = []
        if 'test_end_date' not in metrics_dict:
            metrics_dict['test_end_date'] = []
        
        metrics_dict['fold'].append(fold_idx)
        metrics_dict['train_size'].append(len(train_df))
        metrics_dict['test_size'].append(len(test_df))
        metrics_dict['train_start_date'].append(train_df[date_column].min())
        metrics_dict['train_end_date'].append(train_df[date_column].max())
        metrics_dict['test_start_date'].append(test_df[date_column].min())
        metrics_dict['test_end_date'].append(test_df[date_column].max())
    
    return metrics_dict


def walk_forward_validation(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    model_fn: Callable[..., Any],
    initial_train_size: Union[int, float, str] = 0.5,
    test_size: Union[int, float, str] = '1M',
    step_size: Union[int, float, str] = '1M',
    max_test_sets: Optional[int] = None,
    gap: Optional[Union[str, pd.Timedelta]] = None,
    problem_type: Literal['classification', 'regression'] = 'regression',
    feature_columns: Optional[List[str]] = None,
    expanding_window: bool = True,
    **model_kwargs
) -> Dict[str, List[float]]:
    """
    Perform walk-forward validation on time series data.
    
    This is an advanced form of time series cross-validation where the model is 
    iteratively trained on expanding windows of data and evaluated on subsequent periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to use for validation.
    target_column : str
        The name of the column containing the target values.
    date_column : str
        The name of the column containing the date/time values.
    model_fn : Callable[..., Any]
        A function that returns a model object. The function should take
        keyword arguments specified in model_kwargs.
    initial_train_size : Union[int, float, str], default=0.5
        The size of the initial training set:
        - int: Number of rows
        - float: Fraction of total rows (between 0 and 1)
        - str: Time duration (e.g., '1Y', '6M', '180D')
    test_size : Union[int, float, str], default='1M'
        The size of each test set:
        - int: Number of rows
        - float: Fraction of remaining rows
        - str: Time duration (e.g., '1M', '30D')
    step_size : Union[int, float, str], default='1M'
        The step size between consecutive test sets:
        - int: Number of rows
        - float: Not supported, raises an error
        - str: Time duration (e.g., '1M', '30D')
    max_test_sets : Optional[int], default=None
        Maximum number of test sets to create. If None, creates as many as possible.
    gap : Optional[Union[str, pd.Timedelta]], default=None
        If not None, leaves a gap between train and test sets.
    problem_type : Literal['classification', 'regression'], default='regression'
        The type of problem (classification or regression).
    feature_columns : Optional[List[str]], default=None
        The names of the columns to use as features. If None, uses all columns
        except the target and date columns.
    expanding_window : bool, default=True
        If True, training set expands to include all previous data.
        If False, uses a sliding window of size initial_train_size.
    **model_kwargs
        Additional keyword arguments to pass to the model function.
    
    Returns
    -------
    Dict[str, List[Any]]
        A dictionary containing metric values for each test period, as well as
        information about the train/test splits.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2020-01-01', periods=365),
    ...     'feature1': range(365),
    ...     'feature2': range(365, 730),
    ...     'target': range(730, 1095)
    ... })
    >>> def model_fn(**kwargs):
    ...     return LinearRegression(**kwargs)
    >>> results = walk_forward_validation(
    ...     df, 'target', 'date', model_fn, 
    ...     initial_train_size='6M', test_size='1M', step_size='1M'
    ... )
    >>> list(results.keys())
    ['mse', 'rmse', 'mae', 'r2', 'fold', 'train_size', 'test_size', 
     'train_start_date', 'train_end_date', 'test_start_date', 'test_end_date']
    """
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
    
    # Determine the initial train split point
    if isinstance(initial_train_size, int):
        if initial_train_size <= 0 or initial_train_size >= len(df):
            raise ValueError(f"initial_train_size must be between 1 and {len(df)-1}")
        train_end_idx = initial_train_size
    elif isinstance(initial_train_size, float):
        if initial_train_size <= 0 or initial_train_size >= 1:
            raise ValueError("initial_train_size as a fraction must be between 0 and 1 exclusive")
        train_end_idx = int(len(df) * initial_train_size)
    elif isinstance(initial_train_size, str):
        # String represents a time duration
        start_date = df[date_column].min()
        end_date = start_date + pd.Timedelta(initial_train_size)
        train_end_idx = df[df[date_column] <= end_date].index[-1] + 1
    else:
        raise ValueError("initial_train_size must be an int, float, or str")

    # Initialize results dictionary
    metrics_dict: Dict[str, List[Any]] = {
        'fold': [],
        'train_size': [],
        'test_size': [],
        'train_start_date': [],
        'train_end_date': [],
        'test_start_date': [],
        'test_end_date': []
    }
    
    # Create the test sets
    test_sets = []
    current_idx = train_end_idx
    fold_idx = 0
    
    while current_idx < len(df):
        # Determine the end of the test set
        if isinstance(test_size, int):
            test_end_idx = min(current_idx + test_size, len(df))
        elif isinstance(test_size, float):
            remaining = len(df) - current_idx
            test_end_idx = min(current_idx + int(remaining * test_size), len(df))
        elif isinstance(test_size, str):
            # String represents a time duration
            start_date = df.iloc[current_idx][date_column]
            end_date = start_date + pd.Timedelta(test_size)
            candidates = df[df[date_column] <= end_date].index
            test_end_idx = candidates[-1] + 1 if len(candidates) > 0 else len(df)
        else:
            raise ValueError("test_size must be an int, float, or str")
        
        # Apply gap if specified
        test_start_idx = current_idx
        if gap is not None:
            if isinstance(gap, str):
                gap = pd.Timedelta(gap)
            
            # Get the last date in the training set
            last_train_date = df.iloc[current_idx - 1][date_column]
            
            # Find the first index after the gap
            gap_end_date = last_train_date + gap
            gap_indices = df[(df[date_column] > gap_end_date) & 
                             (df.index >= current_idx)].index
            
            if len(gap_indices) > 0:
                test_start_idx = gap_indices[0]
            else:
                # Skip this fold if no data after the gap
                current_idx = test_end_idx
                continue
        
        # Add the test set if there's data
        if test_start_idx < test_end_idx:
            test_sets.append((test_start_idx, test_end_idx, fold_idx))
            fold_idx += 1
        
        # Move to the next position
        if isinstance(step_size, int):
            current_idx += step_size
        elif isinstance(step_size, float):
            raise ValueError("step_size as a fraction is not supported")
        elif isinstance(step_size, str):
            # String represents a time duration
            start_date = df.iloc[current_idx][date_column]
            next_date = start_date + pd.Timedelta(step_size)
            candidates = df[df[date_column] >= next_date].index
            if len(candidates) > 0:
                current_idx = candidates[0]
            else:
                break
        else:
            raise ValueError("step_size must be an int or str")
        
        # Break if we've reached the maximum number of test sets
        if max_test_sets is not None and len(test_sets) >= max_test_sets:
            break
    
    # Perform walk-forward validation
    for test_start_idx, test_end_idx, fold_idx in test_sets:
        # Determine the training window
        if expanding_window:
            # Use all data from the beginning to the test start
            train_start_idx = 0
            train_end_idx = test_start_idx
        else:
            # Use a sliding window of size initial_train_size
            if isinstance(initial_train_size, int):
                train_start_idx = max(0, test_start_idx - initial_train_size)
            elif isinstance(initial_train_size, float):
                # Recalculate the size for each fold to maintain proportion
                size = int(test_start_idx * initial_train_size)
                train_start_idx = max(0, test_start_idx - size)
            elif isinstance(initial_train_size, str):
                duration = pd.Timedelta(initial_train_size)
                test_start_date = df.iloc[test_start_idx][date_column]
                train_start_date = test_start_date - duration
                train_start_candidates = df[df[date_column] >= train_start_date].index
                train_start_idx = train_start_candidates[0] if len(train_start_candidates) > 0 else 0
            train_end_idx = test_start_idx
        
        # Split data
        train_df = df.iloc[train_start_idx:train_end_idx]
        test_df = df.iloc[test_start_idx:test_end_idx]
        
        # Skip fold if train or test sets are too small
        if len(train_df) < 2 or len(test_df) < 1:
            continue
        
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
        
        # Save fold information for debugging and analysis
        metrics_dict['fold'].append(fold_idx)
        metrics_dict['train_size'].append(len(train_df))
        metrics_dict['test_size'].append(len(test_df))
        metrics_dict['train_start_date'].append(train_df[date_column].min())
        metrics_dict['train_end_date'].append(train_df[date_column].max())
        metrics_dict['test_start_date'].append(test_df[date_column].min())
        metrics_dict['test_end_date'].append(test_df[date_column].max())
    
    return metrics_dict