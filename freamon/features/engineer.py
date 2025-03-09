"""
Feature engineering module for creating new features from existing ones.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures as SklearnPolyFeatures

from freamon.utils import check_dataframe_type, convert_dataframe


def create_polynomial_features(
    df: Any,
    columns: Optional[List[str]] = None,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    prefix: str = 'poly',
) -> Any:
    """
    Create polynomial features from the specified columns.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    columns : Optional[List[str]], default=None
        The columns to use for creating polynomial features. If None, all numeric columns are used.
    degree : int, default=2
        The degree of the polynomial features. Must be >= 1.
    include_bias : bool, default=False
        If True, include a bias column (all 1s).
    interaction_only : bool, default=False
        If True, only include interaction features, not pure polynomial features.
    prefix : str, default='poly'
        Prefix for the new feature names.
    
    Returns
    -------
    Any
        Dataframe with added polynomial features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> result = create_polynomial_features(df, degree=2)
    >>> result.columns
    Index(['A', 'B', 'poly_A^2', 'poly_A*B', 'poly_B^2'], dtype='object')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Create a copy to avoid modifying the original
    result = df_pandas.copy()
    
    # Determine which columns to process
    if columns is None:
        # Use all numeric columns
        numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verify all columns exist and are numeric
        numeric_cols = []
        for col in columns:
            if col not in result.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(f"Column '{col}' is not numeric")
            numeric_cols.append(col)
    
    if not numeric_cols:
        return result
    
    # Create polynomial features
    poly = SklearnPolyFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    
    # Get the polynomial features
    poly_features = poly.fit_transform(result[numeric_cols])
    
    # Generate feature names
    if hasattr(poly, 'get_feature_names_out'):
        # For scikit-learn >= 1.0
        feature_names = poly.get_feature_names_out(numeric_cols)
    else:
        # For scikit-learn < 1.0
        feature_names = poly.get_feature_names(numeric_cols)
    
    # Convert simple feature names to more intuitive ones
    # e.g., 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2' -> 'A', 'B', 'A^2', 'A*B', 'B^2'
    intuitive_names = []
    for name in feature_names:
        # Skip the original feature names
        if name in numeric_cols:
            continue
        
        # Replace x0, x1, etc. with actual column names
        intuitive_name = name
        for i, col in enumerate(numeric_cols):
            intuitive_name = intuitive_name.replace(f'x{i}', col)
        
        # Replace spaces with * to indicate multiplication
        intuitive_name = intuitive_name.replace(' ', '*')
        
        # Add prefix
        intuitive_name = f"{prefix}_{intuitive_name}"
        
        intuitive_names.append(intuitive_name)
    
    # Add polynomial features to result dataframe, skip the original features
    for i, name in enumerate(intuitive_names):
        # The first len(numeric_cols) features are the original ones
        result[name] = poly_features[:, i + len(numeric_cols)]
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(result, df_type)
    
    return result


def create_interaction_features(
    df: Any,
    feature_columns: List[str],
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    operations: List[str] = ['multiply'],
    prefix: str = 'interaction',
) -> Any:
    """
    Create interaction features between pairs of columns.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    feature_columns : List[str]
        The columns to consider for interactions.
    interaction_pairs : Optional[List[Tuple[str, str]]], default=None
        Specific pairs of columns to create interactions for. If None, all pairwise 
        combinations of feature_columns will be used.
    operations : List[str], default=['multiply']
        Types of interactions to create. Options: 'multiply', 'divide', 'add', 'subtract', 'ratio'.
        Note: 'ratio' creates both ratio and inverse ratio features.
    prefix : str, default='interaction'
        Prefix for the new feature names.
    
    Returns
    -------
    Any
        Dataframe with added interaction features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> result = create_interaction_features(df, ['A', 'B'], operations=['multiply', 'add'])
    >>> result.columns
    Index(['A', 'B', 'interaction_A*B', 'interaction_A+B'], dtype='object')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Create a copy to avoid modifying the original
    result = df_pandas.copy()
    
    # Verify all columns exist
    for col in feature_columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Determine interaction pairs
    if interaction_pairs is None:
        # Create all pairwise combinations
        pairs = []
        for i, col1 in enumerate(feature_columns):
            for col2 in feature_columns[i+1:]:
                pairs.append((col1, col2))
        interaction_pairs = pairs
    
    # Create interaction features
    for col1, col2 in interaction_pairs:
        # Verify columns exist
        if col1 not in result.columns or col2 not in result.columns:
            continue
        
        # Verify both columns are numeric
        if not (pd.api.types.is_numeric_dtype(result[col1]) and 
                pd.api.types.is_numeric_dtype(result[col2])):
            continue
        
        # Create features based on specified operations
        for op in operations:
            if op == 'multiply':
                result[f'{prefix}_{col1}*{col2}'] = result[col1] * result[col2]
            elif op == 'divide' and 'ratio' not in operations:
                # Only add divide if ratio is not requested (to avoid duplication)
                # Add epsilon to avoid division by zero
                epsilon = 1e-10
                result[f'{prefix}_{col1}/{col2}'] = result[col1] / (result[col2] + epsilon)
            elif op == 'add':
                result[f'{prefix}_{col1}+{col2}'] = result[col1] + result[col2]
            elif op == 'subtract':
                result[f'{prefix}_{col1}-{col2}'] = result[col1] - result[col2]
            elif op == 'ratio':
                # Create both ratio and its inverse
                epsilon = 1e-10
                result[f'{prefix}_{col1}/{col2}'] = result[col1] / (result[col2] + epsilon)
                result[f'{prefix}_{col2}/{col1}'] = result[col2] / (result[col1] + epsilon)
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(result, df_type)
    
    return result


def create_datetime_features(
    df: Any,
    datetime_column: str,
    features: Optional[List[str]] = None,
    drop_original: bool = False,
    prefix: Optional[str] = None,
    date_format: Optional[str] = None,
) -> Any:
    """
    Create features from a datetime column.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    datetime_column : str
        The datetime column to extract features from.
    features : Optional[List[str]], default=None
        The features to create. Options: 'year', 'quarter', 'month', 'day', 'dayofweek', 
        'dayofyear', 'weekofyear', 'hour', 'minute', 'second', 'is_weekend', 'is_month_start', 
        'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end',
        'days_in_month', 'sin_cos_month', 'sin_cos_day', 'sin_cos_weekday', 'sin_cos_hour'.
        If None, all appropriate features will be created.
    drop_original : bool, default=False
        Whether to drop the original datetime column.
    prefix : Optional[str], default=None
        Prefix for the new feature names. If None, the datetime_column name is used.
    date_format : Optional[str], default=None
        The format of the datetime column, if it needs to be parsed from strings.
    
    Returns
    -------
    Any
        Dataframe with added datetime features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3)})
    >>> result = create_datetime_features(df, 'date', features=['year', 'month', 'day'])
    >>> result.columns
    Index(['date', 'date_year', 'date_month', 'date_day'], dtype='object')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Create a copy to avoid modifying the original
    result = df_pandas.copy()
    
    # Verify datetime column exists
    if datetime_column not in result.columns:
        raise ValueError(f"Column '{datetime_column}' not found in dataframe")
    
    # Set the prefix
    if prefix is None:
        prefix = datetime_column
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_dtype(result[datetime_column]):
        result[datetime_column] = pd.to_datetime(
            result[datetime_column], format=date_format, errors='coerce'
        )
    
    # Determine which features to create
    dt_attributes = {
        'year': lambda x: x.dt.year,
        'quarter': lambda x: x.dt.quarter,
        'month': lambda x: x.dt.month,
        'day': lambda x: x.dt.day,
        'dayofweek': lambda x: x.dt.dayofweek,
        'dayofyear': lambda x: x.dt.dayofyear,
        'weekofyear': lambda x: x.dt.isocalendar().week,
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'second': lambda x: x.dt.second,
        'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int),
        'is_month_start': lambda x: x.dt.is_month_start.astype(int),
        'is_month_end': lambda x: x.dt.is_month_end.astype(int),
        'is_quarter_start': lambda x: x.dt.is_quarter_start.astype(int),
        'is_quarter_end': lambda x: x.dt.is_quarter_end.astype(int),
        'is_year_start': lambda x: x.dt.is_year_start.astype(int),
        'is_year_end': lambda x: x.dt.is_year_end.astype(int),
        'days_in_month': lambda x: x.dt.days_in_month,
    }
    
    # Cyclic features (sin/cos encoding)
    cyclic_features = {
        'sin_cos_month': (lambda x: np.sin(2 * np.pi * x.dt.month / 12),
                         lambda x: np.cos(2 * np.pi * x.dt.month / 12)),
        'sin_cos_day': (lambda x: np.sin(2 * np.pi * x.dt.day / 31),
                       lambda x: np.cos(2 * np.pi * x.dt.day / 31)),
        'sin_cos_weekday': (lambda x: np.sin(2 * np.pi * x.dt.dayofweek / 7),
                           lambda x: np.cos(2 * np.pi * x.dt.dayofweek / 7)),
        'sin_cos_hour': (lambda x: np.sin(2 * np.pi * x.dt.hour / 24),
                        lambda x: np.cos(2 * np.pi * x.dt.hour / 24)),
    }
    
    # If no features specified, use all appropriate ones
    if features is None:
        # Check if the data has time components (not just dates)
        has_time = False
        if result[datetime_column].dt.hour.nunique() > 1:
            has_time = True
        
        # Select features based on whether time components are present
        if has_time:
            features = list(dt_attributes.keys()) + list(cyclic_features.keys())
        else:
            # Exclude time-based features
            features = [f for f in dt_attributes.keys() if f not in 
                      ['hour', 'minute', 'second', 'sin_cos_hour']]
            features += [f for f in cyclic_features.keys() if f not in 
                        ['sin_cos_hour']]
    
    # Create features
    for feature in features:
        if feature in dt_attributes:
            result[f'{prefix}_{feature}'] = dt_attributes[feature](result[datetime_column])
        elif feature in cyclic_features:
            sin_func, cos_func = cyclic_features[feature]
            result[f'{prefix}_{feature}_sin'] = sin_func(result[datetime_column])
            result[f'{prefix}_{feature}_cos'] = cos_func(result[datetime_column])
    
    # Drop original column if requested
    if drop_original:
        result = result.drop(columns=[datetime_column])
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(result, df_type)
    
    return result


def create_binned_features(
    df: Any,
    columns: List[str],
    n_bins: Union[int, Dict[str, int]] = 5,
    strategy: Union[str, Dict[str, str]] = 'quantile',
    labels: Optional[Union[bool, Dict[str, bool]]] = None,
    prefix: str = 'bin',
) -> Any:
    """
    Create binned features from numeric columns.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    columns : List[str]
        The columns to bin.
    n_bins : Union[int, Dict[str, int]], default=5
        Number of bins to create. Either a single value for all columns or a dict
        mapping column names to numbers of bins.
    strategy : Union[str, Dict[str, str]], default='quantile'
        Strategy to use for binning. Options: 'uniform', 'quantile', 'kmeans'.
        Either a single value for all columns or a dict mapping column names to strategies.
    labels : Optional[Union[bool, Dict[str, bool]]], default=None
        Whether to use labels for the bins. If True, label bins with integers.
        If False, use the bin interval values. If None, defaults to True.
        Either a single value for all columns or a dict mapping column names to label options.
    prefix : str, default='bin'
        Prefix for the new feature names.
    
    Returns
    -------
    Any
        Dataframe with added binned features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 5, 10, 15, 20]})
    >>> result = create_binned_features(df, ['A'], n_bins=3, strategy='uniform')
    >>> result.columns
    Index(['A', 'bin_A'], dtype='object')
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Create a copy to avoid modifying the original
    result = df_pandas.copy()
    
    # Process labels default
    if labels is None:
        labels = True
    
    # Verify all columns exist and are numeric
    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(result[col]):
            raise ValueError(f"Column '{col}' is not numeric")
    
    # Create binned features
    for col in columns:
        # Determine n_bins for this column
        if isinstance(n_bins, dict):
            col_n_bins = n_bins.get(col, 5)
        else:
            col_n_bins = n_bins
        
        # Determine strategy for this column
        if isinstance(strategy, dict):
            col_strategy = strategy.get(col, 'quantile')
        else:
            col_strategy = strategy
        
        # Determine labels for this column
        if isinstance(labels, dict):
            col_labels = labels.get(col, True)
        else:
            col_labels = labels
        
        # Create bins
        if col_strategy == 'uniform':
            result[f'{prefix}_{col}'] = pd.cut(
                result[col], bins=col_n_bins, labels=col_labels if col_labels is True else False
            )
        elif col_strategy == 'quantile':
            result[f'{prefix}_{col}'] = pd.qcut(
                result[col], q=col_n_bins, labels=col_labels if col_labels is True else False,
                duplicates='drop'
            )
        elif col_strategy == 'kmeans':
            from sklearn.cluster import KMeans
            
            # Reshape for KMeans
            X = result[col].values.reshape(-1, 1)
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=col_n_bins, random_state=0).fit(X)
            
            # Get cluster centers and sort them
            centers = kmeans.cluster_centers_.flatten()
            centers_idx = np.argsort(centers)
            
            # Get sorted cluster labels
            labels_sorted = np.zeros_like(kmeans.labels_)
            for i, idx in enumerate(centers_idx):
                labels_sorted[kmeans.labels_ == idx] = i
            
            # Add to result
            if col_labels is True:
                result[f'{prefix}_{col}'] = labels_sorted
            else:
                # Create interval labels
                bins = np.append(
                    -np.inf, 
                    [(centers[i] + centers[j]) / 2 for i, j in zip(centers_idx[:-1], centers_idx[1:])],
                    np.inf
                )
                result[f'{prefix}_{col}'] = pd.cut(
                    result[col], bins=bins, labels=False
                )
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(result, df_type)
    
    return result


def create_lagged_features(
    df: Any,
    column: str,
    lags: List[int],
    group_column: Optional[str] = None,
    date_column: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Any:
    """
    Create lagged features for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    column : str
        The column to create lags for.
    lags : List[int]
        List of lag values to create.
    group_column : Optional[str], default=None
        Column to group by when creating lags. Useful for panel data with multiple series.
    date_column : Optional[str], default=None
        Date column to sort by. If provided, the data will be sorted by this column within groups.
    prefix : Optional[str], default=None
        Prefix for the new feature names. If None, the column name is used.
    
    Returns
    -------
    Any
        Dataframe with added lag features.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=5),
    ...     'value': [1, 2, 3, 4, 5]
    ... })
    >>> result = create_lagged_features(df, 'value', lags=[1, 2], date_column='date')
    >>> result
              date  value  value_lag_1  value_lag_2
    0   2020-01-01      1          NaN          NaN
    1   2020-01-02      2          1.0          NaN
    2   2020-01-03      3          2.0          1.0
    3   2020-01-04      4          3.0          2.0
    4   2020-01-05      5          4.0          3.0
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Create a copy to avoid modifying the original
    result = df_pandas.copy()
    
    # Verify columns exist
    if column not in result.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    if group_column is not None and group_column not in result.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe")
    if date_column is not None and date_column not in result.columns:
        raise ValueError(f"Date column '{date_column}' not found in dataframe")
    
    # Set the prefix
    if prefix is None:
        prefix = column
    
    # Sort by date if provided
    if date_column is not None:
        if group_column is not None:
            result = result.sort_values([group_column, date_column])
        else:
            result = result.sort_values(date_column)
    
    # Create lagged features
    for lag in lags:
        if lag <= 0:
            continue
        
        if group_column is not None:
            # Create lags within each group
            result[f'{prefix}_lag_{lag}'] = result.groupby(group_column)[column].shift(lag)
        else:
            # Create lags for the whole dataset
            result[f'{prefix}_lag_{lag}'] = result[column].shift(lag)
    
    # Convert back to original type if needed
    if df_type != 'pandas':
        return convert_dataframe(result, df_type)
    
    return result


class FeatureEngineer:
    """
    Class for engineering new features from existing ones.
    
    This class provides a fluent interface for applying various feature engineering
    transformations to a dataframe. It supports method chaining for better readability.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'date': pd.date_range('2020-01-01', periods=3)
    ... })
    >>> engineer = FeatureEngineer(df)
    >>> result = (engineer
    ...     .create_polynomial_features(['A', 'B'])
    ...     .create_datetime_features('date')
    ...     .transform()
    ... )
    """
    
    def __init__(self, df: Any):
        """Initialize the FeatureEngineer with a dataframe."""
        self.df_type = check_dataframe_type(df)
        
        # Convert to pandas internally if needed
        if self.df_type != 'pandas':
            self.df = convert_dataframe(df, 'pandas')
        else:
            self.df = df.copy()
        
        # Initialize transformations list
        self.transformations = []
    
    def create_polynomial_features(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        prefix: str = 'poly',
    ) -> 'FeatureEngineer':
        """
        Add polynomial feature creation to the transformation pipeline.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            The columns to use for creating polynomial features. If None, all numeric columns are used.
        degree : int, default=2
            The degree of the polynomial features. Must be >= 1.
        include_bias : bool, default=False
            If True, include a bias column (all 1s).
        interaction_only : bool, default=False
            If True, only include interaction features, not pure polynomial features.
        prefix : str, default='poly'
            Prefix for the new feature names.
        
        Returns
        -------
        FeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'polynomial',
            'columns': columns,
            'degree': degree,
            'include_bias': include_bias,
            'interaction_only': interaction_only,
            'prefix': prefix,
        })
        
        return self
    
    def create_interaction_features(
        self,
        feature_columns: List[str],
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        operations: List[str] = ['multiply'],
        prefix: str = 'interaction',
    ) -> 'FeatureEngineer':
        """
        Add interaction feature creation to the transformation pipeline.
        
        Parameters
        ----------
        feature_columns : List[str]
            The columns to consider for interactions.
        interaction_pairs : Optional[List[Tuple[str, str]]], default=None
            Specific pairs of columns to create interactions for. If None, all pairwise 
            combinations of feature_columns will be used.
        operations : List[str], default=['multiply']
            Types of interactions to create. Options: 'multiply', 'divide', 'add', 'subtract', 'ratio'.
        prefix : str, default='interaction'
            Prefix for the new feature names.
        
        Returns
        -------
        FeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'interaction',
            'feature_columns': feature_columns,
            'interaction_pairs': interaction_pairs,
            'operations': operations,
            'prefix': prefix,
        })
        
        return self
    
    def create_datetime_features(
        self,
        datetime_column: str,
        features: Optional[List[str]] = None,
        drop_original: bool = False,
        prefix: Optional[str] = None,
        date_format: Optional[str] = None,
    ) -> 'FeatureEngineer':
        """
        Add datetime feature creation to the transformation pipeline.
        
        Parameters
        ----------
        datetime_column : str
            The datetime column to extract features from.
        features : Optional[List[str]], default=None
            The features to create. If None, all appropriate features will be created.
        drop_original : bool, default=False
            Whether to drop the original datetime column.
        prefix : Optional[str], default=None
            Prefix for the new feature names. If None, the datetime_column name is used.
        date_format : Optional[str], default=None
            The format of the datetime column, if it needs to be parsed from strings.
        
        Returns
        -------
        FeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'datetime',
            'datetime_column': datetime_column,
            'features': features,
            'drop_original': drop_original,
            'prefix': prefix,
            'date_format': date_format,
        })
        
        return self
    
    def create_binned_features(
        self,
        columns: List[str],
        n_bins: Union[int, Dict[str, int]] = 5,
        strategy: Union[str, Dict[str, str]] = 'quantile',
        labels: Optional[Union[bool, Dict[str, bool]]] = None,
        prefix: str = 'bin',
    ) -> 'FeatureEngineer':
        """
        Add binned feature creation to the transformation pipeline.
        
        Parameters
        ----------
        columns : List[str]
            The columns to bin.
        n_bins : Union[int, Dict[str, int]], default=5
            Number of bins to create. Either a single value for all columns or a dict
            mapping column names to numbers of bins.
        strategy : Union[str, Dict[str, str]], default='quantile'
            Strategy to use for binning. Options: 'uniform', 'quantile', 'kmeans'.
        labels : Optional[Union[bool, Dict[str, bool]]], default=None
            Whether to use labels for the bins. If True, label bins with integers.
        prefix : str, default='bin'
            Prefix for the new feature names.
        
        Returns
        -------
        FeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'binned',
            'columns': columns,
            'n_bins': n_bins,
            'strategy': strategy,
            'labels': labels,
            'prefix': prefix,
        })
        
        return self
    
    def create_lagged_features(
        self,
        column: str,
        lags: List[int],
        group_column: Optional[str] = None,
        date_column: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> 'FeatureEngineer':
        """
        Add lagged feature creation to the transformation pipeline.
        
        Parameters
        ----------
        column : str
            The column to create lags for.
        lags : List[int]
            List of lag values to create.
        group_column : Optional[str], default=None
            Column to group by when creating lags.
        date_column : Optional[str], default=None
            Date column to sort by.
        prefix : Optional[str], default=None
            Prefix for the new feature names. If None, the column name is used.
        
        Returns
        -------
        FeatureEngineer
            Self for method chaining.
        """
        self.transformations.append({
            'type': 'lagged',
            'column': column,
            'lags': lags,
            'group_column': group_column,
            'date_column': date_column,
            'prefix': prefix,
        })
        
        return self
    
    def transform(self) -> Any:
        """
        Apply all transformations to the dataframe.
        
        Returns
        -------
        Any
            Transformed dataframe in the original type.
        """
        result = self.df
        
        # Apply each transformation in sequence
        for transform in self.transformations:
            transform_type = transform.pop('type')
            
            if transform_type == 'polynomial':
                result = create_polynomial_features(result, **transform)
            elif transform_type == 'interaction':
                result = create_interaction_features(result, **transform)
            elif transform_type == 'datetime':
                result = create_datetime_features(result, **transform)
            elif transform_type == 'binned':
                result = create_binned_features(result, **transform)
            elif transform_type == 'lagged':
                result = create_lagged_features(result, **transform)
        
        # Convert back to original type if needed
        if self.df_type != 'pandas':
            return convert_dataframe(result, self.df_type)
        
        return result