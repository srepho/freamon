"""
Automated time series feature engineering module.

This module provides tools for automatically detecting and generating optimal time series
features based on data characteristics.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from sklearn.feature_selection import mutual_info_regression

from freamon.utils import check_dataframe_type, convert_dataframe


def auto_detect_lags(
    df: Any,
    target_col: str,
    date_col: str,
    max_lags: int = 10,
    significance_threshold: float = 0.05,
    group_col: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Automatically detect significant lags for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target_col : str
        The target column to analyze for lag importance.
    date_col : str
        The datetime column used for ordering.
    max_lags : int, default=10
        Maximum number of lags to consider.
    significance_threshold : float, default=0.05
        Threshold for considering a lag significant.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    
    Returns
    -------
    Dict[str, List[int]]
        Dictionary with types of detected lags: 'acf' (autocorrelation),
        'pacf' (partial autocorrelation), and 'mutual_info' (based on 
        mutual information with target).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    ... })
    >>> result = auto_detect_lags(df, 'value', 'date', max_lags=5)
    >>> result
    {'acf': [1, 3], 'pacf': [1], 'mutual_info': [1, 2, 3]}
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df.copy()
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df_pandas[date_col]):
        try:
            df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Sort by date (and group if provided)
    if group_col is not None:
        df_pandas = df_pandas.sort_values([group_col, date_col])
    else:
        df_pandas = df_pandas.sort_values(date_col)
    
    result = {
        'acf': [],
        'pacf': [],
        'mutual_info': [],
    }
    
    # Process each group separately if group_col is provided
    if group_col is not None:
        groups = df_pandas[group_col].unique()
        acf_lags_by_group = []
        pacf_lags_by_group = []
        mi_lags_by_group = []
        
        for group in groups:
            group_data = df_pandas[df_pandas[group_col] == group][target_col].dropna()
            if len(group_data) <= max_lags:
                continue
                
            # Calculate ACF and PACF
            try:
                acf_values = acf(group_data, nlags=max_lags, alpha=significance_threshold)
                pacf_values = pacf(group_data, nlags=max_lags, alpha=significance_threshold)
                
                # ACF - Get significant lags based on confidence intervals
                acf_significant = [i for i in range(1, max_lags + 1) 
                                   if acf_values[i] > acf_values[0] * significance_threshold]
                acf_lags_by_group.append(acf_significant)
                
                # PACF - Get significant lags
                pacf_significant = [i for i in range(1, max_lags + 1) 
                                    if abs(pacf_values[i]) > significance_threshold]
                pacf_lags_by_group.append(pacf_significant)
                
                # Mutual Information for lags
                mi_significant = []
                for lag in range(1, max_lags + 1):
                    # Create lag feature
                    lagged_series = group_data.shift(lag)
                    # Remove rows with NaN from shifting
                    valid_idx = ~lagged_series.isna()
                    if sum(valid_idx) > 10:  # Need enough data
                        X = lagged_series[valid_idx].values.reshape(-1, 1)
                        y = group_data[valid_idx].values
                        mi = mutual_info_regression(X, y)[0]
                        if mi > significance_threshold:
                            mi_significant.append(lag)
                
                mi_lags_by_group.append(mi_significant)
            except:
                # Skip if calculations fail
                continue
        
        # Aggregate lags across groups
        if acf_lags_by_group:
            # Count frequency of each lag across groups
            acf_lag_counts = {}
            for lags in acf_lags_by_group:
                for lag in lags:
                    acf_lag_counts[lag] = acf_lag_counts.get(lag, 0) + 1
            
            # Keep lags that appear in at least 50% of groups
            min_count = max(1, len(groups) // 2)
            result['acf'] = sorted([lag for lag, count in acf_lag_counts.items() 
                                   if count >= min_count])
        
        if pacf_lags_by_group:
            pacf_lag_counts = {}
            for lags in pacf_lags_by_group:
                for lag in lags:
                    pacf_lag_counts[lag] = pacf_lag_counts.get(lag, 0) + 1
            
            min_count = max(1, len(groups) // 2)
            result['pacf'] = sorted([lag for lag, count in pacf_lag_counts.items() 
                                    if count >= min_count])
        
        if mi_lags_by_group:
            mi_lag_counts = {}
            for lags in mi_lags_by_group:
                for lag in lags:
                    mi_lag_counts[lag] = mi_lag_counts.get(lag, 0) + 1
            
            min_count = max(1, len(groups) // 2)
            result['mutual_info'] = sorted([lag for lag, count in mi_lag_counts.items() 
                                          if count >= min_count])
    else:
        # Process the entire dataset as a single time series
        series = df_pandas[target_col].dropna()
        if len(series) > max_lags:
            try:
                # Calculate ACF and PACF
                acf_values = acf(series, nlags=max_lags, alpha=significance_threshold)
                pacf_values = pacf(series, nlags=max_lags, alpha=significance_threshold)
                
                # ACF - Get significant lags
                confidence_threshold = 2 / np.sqrt(len(series))  # Rule of thumb
                result['acf'] = [i for i in range(1, max_lags + 1) 
                                if abs(acf_values[i]) > confidence_threshold]
                
                # PACF - Get significant lags
                result['pacf'] = [i for i in range(1, max_lags + 1) 
                                 if abs(pacf_values[i]) > confidence_threshold]
                
                # Mutual Information for lags
                mi_significant = []
                for lag in range(1, max_lags + 1):
                    # Create lag feature
                    lagged_series = series.shift(lag)
                    # Remove rows with NaN from shifting
                    valid_idx = ~lagged_series.isna()
                    if sum(valid_idx) > 10:  # Need enough data
                        X = lagged_series[valid_idx].values.reshape(-1, 1)
                        y = series[valid_idx].values
                        mi = mutual_info_regression(X, y)[0]
                        if mi > significance_threshold:
                            mi_significant.append(lag)
                
                result['mutual_info'] = mi_significant
            except Exception:
                # If calculations fail, leave empty lists
                pass
    
    # Return detected lags
    return result


def auto_detect_rolling_windows(
    df: Any,
    target_col: str,
    date_col: str,
    max_window_size: int = 10,
    metrics: List[str] = ['mean', 'std'],
    group_col: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Automatically detect optimal rolling window sizes for time series features.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target_col : str
        The target column to analyze.
    date_col : str
        The datetime column used for ordering.
    max_window_size : int, default=10
        Maximum window size to consider.
    metrics : List[str], default=['mean', 'std']
        Metrics to calculate on rolling windows. Options: 'mean', 'std', 'min', 'max', 'sum'.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    
    Returns
    -------
    Dict[str, List[int]]
        Dictionary with optimal window sizes for each metric.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    ... })
    >>> result = auto_detect_rolling_windows(df, 'value', 'date', max_window_size=5)
    >>> result
    {'mean': [3, 5], 'std': [2]}
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df.copy()
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df_pandas[date_col]):
        try:
            df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Sort by date (and group if provided)
    if group_col is not None:
        df_pandas = df_pandas.sort_values([group_col, date_col])
    else:
        df_pandas = df_pandas.sort_values(date_col)
    
    # Validate metrics
    valid_metrics = ['mean', 'std', 'min', 'max', 'sum']
    metrics = [m for m in metrics if m in valid_metrics]
    
    if not metrics:
        return {}
    
    result = {metric: [] for metric in metrics}
    
    # Helper function to calculate MI between rolling features and target
    def calculate_mi_for_windows(series, target, windows, metric_func):
        window_scores = {}
        for window in windows:
            if len(series) <= window:
                continue
                
            # Calculate rolling metric
            rolling_values = metric_func(series.rolling(window=window, min_periods=1))
            
            # Remove rows with NaN
            valid_idx = ~rolling_values.isna()
            if sum(valid_idx) > 10:  # Need enough data
                X = rolling_values[valid_idx].values.reshape(-1, 1)
                y = target[valid_idx].values
                try:
                    mi = mutual_info_regression(X, y)[0]
                    window_scores[window] = mi
                except:
                    pass
        
        # Return windows with top 2 MI scores
        if window_scores:
            sorted_windows = sorted(window_scores.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_windows[:2]]  # Top 2 windows
        return []
    
    # Define metric functions
    metric_funcs = {
        'mean': lambda x: x.mean(),
        'std': lambda x: x.std(),
        'min': lambda x: x.min(),
        'max': lambda x: x.max(),
        'sum': lambda x: x.sum(),
    }
    
    # Process each group separately if group_col is provided
    if group_col is not None:
        groups = df_pandas[group_col].unique()
        windows_by_group = {metric: [] for metric in metrics}
        
        for group in groups:
            group_data = df_pandas[df_pandas[group_col] == group]
            series = group_data[target_col]
            target = group_data[target_col].shift(-1)  # Predict next value
            
            # Skip if not enough data
            if len(series) <= max_window_size or len(series) < 20:
                continue
                
            windows = range(2, min(max_window_size + 1, len(series) // 2))
            
            for metric in metrics:
                metric_func = metric_funcs[metric]
                optimal_windows = calculate_mi_for_windows(series, target, windows, metric_func)
                windows_by_group[metric].append(optimal_windows)
        
        # Aggregate windows across groups
        for metric in metrics:
            if windows_by_group[metric]:
                # Count frequency of each window size across groups
                window_counts = {}
                for windows in windows_by_group[metric]:
                    for window in windows:
                        window_counts[window] = window_counts.get(window, 0) + 1
                
                # Keep window sizes that appear frequently
                min_count = max(1, len(groups) // 3)
                result[metric] = sorted([w for w, count in window_counts.items() 
                                       if count >= min_count])
    else:
        # Process the entire dataset as a single time series
        series = df_pandas[target_col]
        target = df_pandas[target_col].shift(-1)  # Predict next value
        
        # Skip if not enough data
        if len(series) > max_window_size and len(series) >= 20:
            windows = range(2, min(max_window_size + 1, len(series) // 2))
            
            for metric in metrics:
                metric_func = metric_funcs[metric]
                result[metric] = calculate_mi_for_windows(series, target, windows, metric_func)
    
    return result


def create_auto_lag_features(
    df: Any,
    target_cols: Union[str, List[str]],
    date_col: str,
    group_col: Optional[str] = None,
    max_lags: int = 10,
    strategy: str = 'auto',
    prefix: Optional[str] = None,
) -> Any:
    """
    Automatically create optimal lag features for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target_cols : Union[str, List[str]]
        Column(s) to create lag features for.
    date_col : str
        The datetime column used for ordering.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    max_lags : int, default=10
        Maximum lag to consider.
    strategy : str, default='auto'
        Strategy for selecting lags. Options:
        - 'auto': Automatically detect optimal lags
        - 'all': Use all lags up to max_lags
        - 'fibonacci': Use Fibonacci sequence lags (1, 2, 3, 5, 8, ...)
        - 'exponential': Use exponentially increasing lags (1, 2, 4, 8, ...)
    prefix : Optional[str], default=None
        Prefix for new feature names. If None, target column name is used.
    
    Returns
    -------
    Any
        Dataframe with added lag features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    ... })
    >>> result = create_auto_lag_features(df, 'value', 'date', strategy='auto')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df.copy()
    
    # Handle single vs. list of target columns
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # Validate columns
    for col in target_cols:
        if col not in df_pandas.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    if date_col not in df_pandas.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    
    if group_col is not None and group_col not in df_pandas.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df_pandas[date_col]):
        try:
            df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Sort by date (and group if provided)
    if group_col is not None:
        df_pandas = df_pandas.sort_values([group_col, date_col])
    else:
        df_pandas = df_pandas.sort_values(date_col)
    
    # Process each target column
    for col in target_cols:
        # Set prefix for this column
        col_prefix = prefix if prefix is not None else col
        
        # Determine lags based on strategy
        if strategy == 'auto':
            # Detect optimal lags
            detected_lags = auto_detect_lags(df_pandas, col, date_col, max_lags, group_col=group_col)
            
            # Combine detected lags from different methods and take unique values
            optimal_lags = sorted(set(
                detected_lags['acf'] + 
                detected_lags['pacf'] + 
                detected_lags['mutual_info']
            ))
            
            # If no lags detected, fall back to defaults
            if not optimal_lags:
                optimal_lags = [1, 2, 3]
            
            lags = optimal_lags
        
        elif strategy == 'all':
            # Use all lags up to max_lags
            lags = list(range(1, max_lags + 1))
        
        elif strategy == 'fibonacci':
            # Use Fibonacci sequence lags
            lags = [1, 2]
            while lags[-1] + lags[-2] <= max_lags:
                lags.append(lags[-1] + lags[-2])
        
        elif strategy == 'exponential':
            # Use exponentially increasing lags
            lags = [1]
            while lags[-1] * 2 <= max_lags:
                lags.append(lags[-1] * 2)
        
        else:
            # Default to [1, 2, 3]
            lags = [1, 2, 3]
        
        # Create lag features
        for lag in lags:
            if group_col is not None:
                # Create lags within each group
                df_pandas[f'{col_prefix}_lag_{lag}'] = df_pandas.groupby(group_col)[col].shift(lag)
            else:
                # Create lags for the whole dataset
                df_pandas[f'{col_prefix}_lag_{lag}'] = df_pandas[col].shift(lag)
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(df_pandas, df_type)
    
    return df_pandas


def create_auto_rolling_features(
    df: Any,
    target_cols: Union[str, List[str]],
    date_col: str,
    group_col: Optional[str] = None,
    windows: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    metrics: List[str] = ['mean', 'std', 'min', 'max'],
    auto_detect: bool = True,
    max_window: int = 10,
    prefix: Optional[str] = None,
) -> Any:
    """
    Automatically create optimal rolling window features for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target_cols : Union[str, List[str]]
        Column(s) to create rolling features for.
    date_col : str
        The datetime column used for ordering.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    windows : Optional[Union[List[int], Dict[str, List[int]]]], default=None
        Window sizes to use. Can be:
        - None: Auto-detect optimal windows
        - List[int]: Use these window sizes for all metrics
        - Dict[str, List[int]]: Map metric names to window sizes
    metrics : List[str], default=['mean', 'std', 'min', 'max']
        Metrics to calculate on rolling windows.
    auto_detect : bool, default=True
        Whether to auto-detect optimal window sizes if windows=None.
    max_window : int, default=10
        Maximum window size to consider when auto-detecting.
    prefix : Optional[str], default=None
        Prefix for new feature names. If None, target column name is used.
    
    Returns
    -------
    Any
        Dataframe with added rolling features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    ... })
    >>> result = create_auto_rolling_features(df, 'value', 'date')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df.copy()
    
    # Handle single vs. list of target columns
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # Validate columns
    for col in target_cols:
        if col not in df_pandas.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    if date_col not in df_pandas.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    
    if group_col is not None and group_col not in df_pandas.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df_pandas[date_col]):
        try:
            df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Sort by date (and group if provided)
    if group_col is not None:
        df_pandas = df_pandas.sort_values([group_col, date_col])
    else:
        df_pandas = df_pandas.sort_values(date_col)
    
    # Validate metrics
    valid_metrics = ['mean', 'std', 'min', 'max', 'sum', 'count']
    metrics = [m for m in metrics if m in valid_metrics]
    
    if not metrics:
        return df_pandas
    
    # Process each target column
    for col in target_cols:
        # Set prefix for this column
        col_prefix = prefix if prefix is not None else col
        
        # Determine window sizes
        if windows is None:
            if auto_detect:
                # Try to auto-detect optimal windows
                try:
                    detected_windows = auto_detect_rolling_windows(
                        df_pandas, col, date_col, 
                        max_window_size=max_window,
                        metrics=metrics,
                        group_col=group_col
                    )
                    
                    # If no windows detected for some metrics, use defaults
                    for metric in metrics:
                        if metric not in detected_windows or not detected_windows[metric]:
                            detected_windows[metric] = [3, 7]
                    
                    windows = detected_windows
                except:
                    # Fall back to defaults
                    windows = {metric: [3, 7] for metric in metrics}
            else:
                # Use default windows
                windows = {metric: [3, 7] for metric in metrics}
        
        elif isinstance(windows, list):
            # Same windows for all metrics
            windows = {metric: windows for metric in metrics}
        
        # Create rolling features
        for metric in metrics:
            metric_windows = windows.get(metric, [3, 7])
            
            for window in metric_windows:
                feature_name = f'{col_prefix}_rolling_{metric}_{window}'
                
                # Different handling based on whether we have groups
                if group_col is not None:
                    # Create rolling features within each group
                    for group in df_pandas[group_col].unique():
                        mask = df_pandas[group_col] == group
                        rolled = df_pandas.loc[mask, col].rolling(window=window, min_periods=1)
                        
                        if metric == 'mean':
                            df_pandas.loc[mask, feature_name] = rolled.mean()
                        elif metric == 'std':
                            df_pandas.loc[mask, feature_name] = rolled.std()
                        elif metric == 'min':
                            df_pandas.loc[mask, feature_name] = rolled.min()
                        elif metric == 'max':
                            df_pandas.loc[mask, feature_name] = rolled.max()
                        elif metric == 'sum':
                            df_pandas.loc[mask, feature_name] = rolled.sum()
                        elif metric == 'count':
                            df_pandas.loc[mask, feature_name] = rolled.count()
                else:
                    # Create rolling features for the whole dataset
                    rolled = df_pandas[col].rolling(window=window, min_periods=1)
                    
                    if metric == 'mean':
                        df_pandas[feature_name] = rolled.mean()
                    elif metric == 'std':
                        df_pandas[feature_name] = rolled.std()
                    elif metric == 'min':
                        df_pandas[feature_name] = rolled.min()
                    elif metric == 'max':
                        df_pandas[feature_name] = rolled.max()
                    elif metric == 'sum':
                        df_pandas[feature_name] = rolled.sum()
                    elif metric == 'count':
                        df_pandas[feature_name] = rolled.count()
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(df_pandas, df_type)
    
    return df_pandas


def create_auto_differential_features(
    df: Any,
    target_cols: Union[str, List[str]],
    date_col: str,
    group_col: Optional[str] = None,
    periods: List[int] = [1],
    prefix: Optional[str] = None,
) -> Any:
    """
    Create differencing features for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target_cols : Union[str, List[str]]
        Column(s) to create differencing features for.
    date_col : str
        The datetime column used for ordering.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    periods : List[int], default=[1]
        Periods to difference over.
    prefix : Optional[str], default=None
        Prefix for new feature names. If None, target column name is used.
    
    Returns
    -------
    Any
        Dataframe with added differencing features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=10),
    ...     'value': [1, 2, 4, 7, 11, 16, 22, 29, 37, 46]
    ... })
    >>> result = create_auto_differential_features(df, 'value', 'date')
    >>> result['value_diff_1']
    0     NaN
    1     1.0
    2     2.0
    3     3.0
    4     4.0
    5     5.0
    6     6.0
    7     7.0
    8     8.0
    9     9.0
    Name: value_diff_1, dtype: float64
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df.copy()
    
    # Handle single vs. list of target columns
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # Validate columns
    for col in target_cols:
        if col not in df_pandas.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    if date_col not in df_pandas.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    
    if group_col is not None and group_col not in df_pandas.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df_pandas[date_col]):
        try:
            df_pandas[date_col] = pd.to_datetime(df_pandas[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Sort by date (and group if provided)
    if group_col is not None:
        df_pandas = df_pandas.sort_values([group_col, date_col])
    else:
        df_pandas = df_pandas.sort_values(date_col)
    
    # Process each target column
    for col in target_cols:
        # Set prefix for this column
        col_prefix = prefix if prefix is not None else col
        
        # Create differencing features
        for period in periods:
            feature_name = f'{col_prefix}_diff_{period}'
            
            if group_col is not None:
                # Create differences within each group
                df_pandas[feature_name] = df_pandas.groupby(group_col)[col].diff(period)
            else:
                # Create differences for the whole dataset
                df_pandas[feature_name] = df_pandas[col].diff(period)
            
            # Also create percentage change
            pct_name = f'{col_prefix}_pct_change_{period}'
            
            if group_col is not None:
                # Create percentage changes within each group
                df_pandas[pct_name] = df_pandas.groupby(group_col)[col].pct_change(period)
            else:
                # Create percentage changes for the whole dataset
                df_pandas[pct_name] = df_pandas[col].pct_change(period)
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(df_pandas, df_type)
    
    return df_pandas


class TimeSeriesFeatureEngineer:
    """
    Automated feature engineering for time series data.
    
    This class analyzes time series data to automatically detect and create optimal
    features for time series forecasting and classification tasks.
    
    Parameters
    ----------
    df : Optional[Any], default=None
        The dataframe to process. Can be pandas, polars, or dask.
        If not provided during initialization, must be provided during fit().
    date_col : Optional[str], default=None
        The datetime column used for ordering.
    target_cols : Optional[Union[str, List[str]]], default=None
        Column(s) to create features for.
    group_col : Optional[str], default=None
        Column to group by (for panel data with multiple time series).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    ... })
    >>> engineer = TimeSeriesFeatureEngineer(df, 'date', 'value')
    >>> result = (engineer
    ...     .create_lag_features(max_lags=5)
    ...     .create_rolling_features()
    ...     .create_differential_features()
    ...     .transform()
    ... )
    """
    
    def __init__(
        self,
        df: Optional[Any] = None,
        date_col: Optional[str] = None,
        target_cols: Optional[Union[str, List[str]]] = None,
        group_col: Optional[str] = None,
    ):
        """Initialize the TimeSeriesFeatureEngineer."""
        self.df = None
        self.df_type = None
        self.date_col = date_col
        self.target_cols = target_cols
        self.group_col = group_col
        self.fitted = False
        
        if df is not None:
            self.df_type = check_dataframe_type(df)
            
            # Convert to pandas internally if needed
            if self.df_type != 'pandas':
                self.df = convert_dataframe(df, 'pandas')
            else:
                self.df = df.copy()
        
        # Initialize transformations list
        self.transformations = []
    
    def create_lag_features(
        self,
        target_cols: Optional[Union[str, List[str]]] = None,
        max_lags: int = 10,
        strategy: str = 'auto',
        prefix: Optional[str] = None,
    ) -> 'TimeSeriesFeatureEngineer':
        """
        Add automatic lag feature creation to the transformation pipeline.
        
        Parameters
        ----------
        target_cols : Optional[Union[str, List[str]]], default=None
            Column(s) to create lag features for. If None, use the target_cols from initialization.
        max_lags : int, default=10
            Maximum lag to consider.
        strategy : str, default='auto'
            Strategy for selecting lags. Options: 'auto', 'all', 'fibonacci', 'exponential'.
        prefix : Optional[str], default=None
            Prefix for new feature names.
        
        Returns
        -------
        TimeSeriesFeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'lag',
            'target_cols': target_cols,
            'max_lags': max_lags,
            'strategy': strategy,
            'prefix': prefix,
        })
        
        return self
    
    def create_rolling_features(
        self,
        target_cols: Optional[Union[str, List[str]]] = None,
        windows: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        metrics: List[str] = ['mean', 'std', 'min', 'max'],
        auto_detect: bool = True,
        max_window: int = 10,
        prefix: Optional[str] = None,
    ) -> 'TimeSeriesFeatureEngineer':
        """
        Add automatic rolling window feature creation to the transformation pipeline.
        
        Parameters
        ----------
        target_cols : Optional[Union[str, List[str]]], default=None
            Column(s) to create rolling features for. If None, use the target_cols from initialization.
        windows : Optional[Union[List[int], Dict[str, List[int]]]], default=None
            Window sizes to use. If None, auto-detect optimal windows.
        metrics : List[str], default=['mean', 'std', 'min', 'max']
            Metrics to calculate on rolling windows.
        auto_detect : bool, default=True
            Whether to auto-detect optimal window sizes if windows=None.
        max_window : int, default=10
            Maximum window size to consider when auto-detecting.
        prefix : Optional[str], default=None
            Prefix for new feature names.
        
        Returns
        -------
        TimeSeriesFeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'rolling',
            'target_cols': target_cols,
            'windows': windows,
            'metrics': metrics,
            'auto_detect': auto_detect,
            'max_window': max_window,
            'prefix': prefix,
        })
        
        return self
    
    def create_differential_features(
        self,
        target_cols: Optional[Union[str, List[str]]] = None,
        periods: List[int] = [1],
        prefix: Optional[str] = None,
    ) -> 'TimeSeriesFeatureEngineer':
        """
        Add differencing feature creation to the transformation pipeline.
        
        Parameters
        ----------
        target_cols : Optional[Union[str, List[str]]], default=None
            Column(s) to create differencing features for. If None, use the target_cols from initialization.
        periods : List[int], default=[1]
            Periods to difference over.
        prefix : Optional[str], default=None
            Prefix for new feature names.
        
        Returns
        -------
        TimeSeriesFeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'diff',
            'target_cols': target_cols,
            'periods': periods,
            'prefix': prefix,
        })
        
        return self
    
    def fit(
        self,
        df: Optional[Any] = None,
        date_col: Optional[str] = None,
        target_cols: Optional[Union[str, List[str]]] = None,
        group_col: Optional[str] = None,
    ) -> 'TimeSeriesFeatureEngineer':
        """
        Fit the time series feature engineer on the given dataframe.
        
        Parameters
        ----------
        df : Optional[Any], default=None
            The dataframe to fit on. If None, use the dataframe from initialization.
        date_col : Optional[str], default=None
            The datetime column. If None, use the date_col from initialization.
        target_cols : Optional[Union[str, List[str]]], default=None
            Column(s) to create features for. If None, use the target_cols from initialization.
        group_col : Optional[str], default=None
            Column to group by. If None, use the group_col from initialization.
            
        Returns
        -------
        TimeSeriesFeatureEngineer
            The fitted engineer.
            
        Raises
        ------
        ValueError
            If required parameters are missing.
        """
        # Update parameters if provided
        if df is not None:
            self.df_type = check_dataframe_type(df)
            
            # Convert to pandas internally if needed
            if self.df_type != 'pandas':
                self.df = convert_dataframe(df, 'pandas')
            else:
                self.df = df.copy()
        
        if date_col is not None:
            self.date_col = date_col
        
        if target_cols is not None:
            self.target_cols = target_cols
        
        if group_col is not None:
            self.group_col = group_col
        
        # Validate parameters
        if self.df is None:
            raise ValueError("A dataframe must be provided to fit")
        
        if self.date_col is None:
            raise ValueError("A date column must be provided")
        
        if self.target_cols is None:
            raise ValueError("Target column(s) must be provided")
        
        # Handle string vs. list for target_cols
        if isinstance(self.target_cols, str):
            self.target_cols = [self.target_cols]
        
        # Validate columns
        if self.date_col not in self.df.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in dataframe")
        
        for col in self.target_cols:
            if col not in self.df.columns:
                raise ValueError(f"Target column '{col}' not found in dataframe")
        
        if self.group_col is not None and self.group_col not in self.df.columns:
            raise ValueError(f"Group column '{self.group_col}' not found in dataframe")
        
        self.fitted = True
        return self
    
    def transform(self, df: Optional[Any] = None) -> Any:
        """
        Apply all transformations to the dataframe.
        
        Parameters
        ----------
        df : Optional[Any], default=None
            The dataframe to transform. If None, use the fitted dataframe.
            
        Returns
        -------
        Any
            Transformed dataframe in the original type.
            
        Raises
        ------
        ValueError
            If the engineer has not been fitted.
        """
        if not self.fitted and df is None:
            raise ValueError("The TimeSeriesFeatureEngineer must be fitted before transform()")
        
        # Use provided dataframe or fitted dataframe
        if df is not None:
            input_df_type = check_dataframe_type(df)
            
            # Convert to pandas internally if needed
            if input_df_type != 'pandas':
                result = convert_dataframe(df, 'pandas')
            else:
                result = df.copy()
                
            output_df_type = input_df_type
        else:
            # Use the fitted dataframe
            result = self.df
            output_df_type = self.df_type
        
        # Apply each transformation in sequence
        for transform in self.transformations:
            transform_type = transform['type']
            
            # Get target columns (from transform or from initialization)
            target_cols = transform.get('target_cols', self.target_cols)
            
            if transform_type == 'lag':
                result = create_auto_lag_features(
                    result,
                    target_cols=target_cols,
                    date_col=self.date_col,
                    group_col=self.group_col,
                    max_lags=transform.get('max_lags', 10),
                    strategy=transform.get('strategy', 'auto'),
                    prefix=transform.get('prefix'),
                )
            
            elif transform_type == 'rolling':
                result = create_auto_rolling_features(
                    result,
                    target_cols=target_cols,
                    date_col=self.date_col,
                    group_col=self.group_col,
                    windows=transform.get('windows'),
                    metrics=transform.get('metrics', ['mean', 'std', 'min', 'max']),
                    auto_detect=transform.get('auto_detect', True),
                    max_window=transform.get('max_window', 10),
                    prefix=transform.get('prefix'),
                )
            
            elif transform_type == 'diff':
                result = create_auto_differential_features(
                    result,
                    target_cols=target_cols,
                    date_col=self.date_col,
                    group_col=self.group_col,
                    periods=transform.get('periods', [1]),
                    prefix=transform.get('prefix'),
                )
        
        # Convert back to original type if needed
        if output_df_type != 'pandas':
            return convert_dataframe(result, output_df_type)
        
        return result