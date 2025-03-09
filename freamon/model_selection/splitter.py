"""
Functions for splitting data into training and testing sets.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify_by: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into training and testing sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    test_size : float, default=0.2
        The proportion of the dataframe to include in the test split.
    random_state : Optional[int], default=None
        Controls the shuffling applied to the data before applying the split.
    stratify_by : Optional[str], default=None
        If not None, data is split in a stratified fashion using this column.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training and testing dataframes.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': range(10), 'B': range(10)})
    >>> train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    >>> len(train_df), len(test_df)
    (7, 3)
    """
    try:
        from sklearn.model_selection import train_test_split as sklearn_split
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    if stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"Column '{stratify_by}' not found in dataframe")
        stratify = df[stratify_by]
    else:
        stratify = None
    
    train_df, test_df = sklearn_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    
    return train_df, test_df


def time_series_split(
    df: pd.DataFrame,
    date_column: str,
    test_size: Union[float, str, pd.Timedelta] = 0.2,
    gap: Optional[Union[str, pd.Timedelta]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into training and testing sets based on a time column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    date_column : str
        The name of the column containing the date/time values.
    test_size : Union[float, str, pd.Timedelta], default=0.2
        The proportion of the dataframe to include in the test split,
        or a string/Timedelta specifying the test period.
    gap : Optional[Union[str, pd.Timedelta]], default=None
        If not None, leaves a gap between train and test sets.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training and testing dataframes.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2020-01-01', periods=10),
    ...     'value': range(10)
    ... })
    >>> train_df, test_df = time_series_split(df, 'date', test_size=0.3)
    >>> len(train_df), len(test_df)
    (7, 3)
    
    >>> # Using a specific time period for the test set
    >>> train_df, test_df = time_series_split(df, 'date', test_size='2D')
    >>> len(train_df), len(test_df)
    (8, 2)
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    # Ensure df is sorted by date_column
    df = df.sort_values(by=date_column).reset_index(drop=True)
    
    # Convert gap to Timedelta if provided
    if gap is not None:
        if isinstance(gap, str):
            gap = pd.Timedelta(gap)
    
    # Determine the test start date
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("If test_size is a float, it must be between 0 and 1")
        
        # Calculate the number of records for test
        n_test = int(len(df) * test_size)
        
        # Get the date where test starts
        test_start_idx = len(df) - n_test
        test_start_date = df.iloc[test_start_idx][date_column]
    else:
        # Convert string to Timedelta if needed
        if isinstance(test_size, str):
            test_size = pd.Timedelta(test_size)
        
        # Calculate the test start date
        test_start_date = df[date_column].max() - test_size
    
    # If gap is specified, adjust the train end date to be gap days before test start
    if gap is not None:
        train_end_date = test_start_date - gap
    else:
        train_end_date = test_start_date
    
    # Split based on the dates
    train_df = df[df[date_column] < train_end_date].copy()
    test_df = df[df[date_column] >= test_start_date].copy()
    
    return train_df, test_df


def stratified_time_series_split(
    df: pd.DataFrame,
    date_column: str,
    group_column: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into training and testing sets based on time and groups.
    
    For each group, the most recent observations are placed in the test set.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to split.
    date_column : str
        The name of the column containing the date/time values.
    group_column : str
        The name of the column containing the group identifiers.
    test_size : float, default=0.2
        The proportion of each group to include in the test split.
    random_state : Optional[int], default=None
        Controls the shuffling applied to the data before applying the split.
        Only used if there are groups with only one observation.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training and testing dataframes.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2020-01-01', periods=10).repeat(3),
    ...     'group': [1, 2, 3] * 10,
    ...     'value': range(30)
    ... })
    >>> train_df, test_df = stratified_time_series_split(
    ...     df, 'date', 'group', test_size=0.3
    ... )
    >>> len(train_df), len(test_df)
    (21, 9)
    """
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in dataframe")
    
    # Ensure test_size is a float between 0 and 1
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Initialize empty DataFrames for train and test
    train_dfs = []
    test_dfs = []
    
    # Get unique groups
    groups = df[group_column].unique()
    
    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Split each group
    for group in groups:
        # Get data for this group
        group_df = df[df[group_column] == group].sort_values(by=date_column)
        
        # Calculate the number of rows for the test set - at least 1 if possible
        n_test = max(1, int(len(group_df) * test_size))
        
        # Need to ensure we have at least one observation in both train and test sets
        # when a group has enough observations
        if len(group_df) > 1:
            # Ensure we don't use all observations for test
            n_test = min(n_test, len(group_df) - 1)
            
            # Use the latest observations for test
            train_group = group_df.iloc[:-n_test].copy()
            test_group = group_df.iloc[-n_test:].copy()
            
            # Append to the lists
            train_dfs.append(train_group)
            test_dfs.append(test_group)
        else:
            # Handle the case of a single observation group
            # Use random assignment with the specified test_size probability
            if np.random.random() < test_size:
                test_dfs.append(group_df)
            else:
                train_dfs.append(group_df)
    
    # Combine all groups
    train_df = pd.concat(train_dfs) if train_dfs else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame(columns=df.columns)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, test_df