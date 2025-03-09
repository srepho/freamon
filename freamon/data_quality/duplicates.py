"""
Module for detecting and handling duplicate rows in dataframes.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from freamon.utils import check_dataframe_type, convert_dataframe


def detect_duplicates(
    df: Any,
    subset: Optional[Union[str, List[str]]] = None,
    keep: str = 'first',
    return_counts: bool = False,
    return_detailed: bool = False,
) -> Union[Dict[str, Any], Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Detect duplicate rows in a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe to analyze. Can be pandas, polars, or dask.
    subset : Optional[Union[str, List[str]]], default=None
        Column(s) to consider for identifying duplicates. If None, uses all columns.
    keep : str, default='first'
        Which duplicates to mark as False. Options: 'first', 'last', False.
        If 'first', mark all duplicates except the first occurrence as True.
        If 'last', mark all duplicates except the last occurrence as True.
        If False, mark all duplicates as True.
    return_counts : bool, default=False
        If True, include duplicate value counts in the results.
    return_detailed : bool, default=False
        If True, return a tuple of (duplicate_df, info_dict) where duplicate_df
        contains only the duplicate rows, otherwise return just info_dict.
    
    Returns
    -------
    Union[Dict[str, Any], Tuple[pd.DataFrame, Dict[str, Any]]]
        If return_detailed is False, returns a dictionary with duplicate statistics.
        If return_detailed is True, returns a tuple of (duplicate_df, info_dict).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 1, 3, 4],
    ...     'B': ['x', 'y', 'x', 'z', 'x'],
    ... })
    >>> result = detect_duplicates(df)
    >>> result['has_duplicates']
    True
    >>> result['duplicate_count']
    1
    >>> result['duplicate_percentage']
    20.0
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Set subset to list if it's a string
    if isinstance(subset, str):
        subset = [subset]
    
    # Initial dataframe shape
    n_rows, n_cols = df_pandas.shape
    
    # Find duplicated rows
    duplicated = df_pandas.duplicated(subset=subset, keep=keep)
    
    # Create result dictionary
    duplicate_count = duplicated.sum()
    info_dict = {
        'has_duplicates': duplicate_count > 0,
        'duplicate_count': int(duplicate_count),
        'duplicate_percentage': float(duplicate_count / n_rows * 100),
        'total_rows': n_rows,
        'unique_rows': int(n_rows - duplicate_count),
    }
    
    # If subset is specified, add to info
    if subset is not None:
        info_dict['subset'] = subset
    
    # Get counts of duplicate values if requested
    if return_counts and duplicate_count > 0:
        if subset is None:
            # Count by all columns
            counts = df_pandas.groupby(list(df_pandas.columns)).size()
            dup_counts = counts[counts > 1].sort_values(ascending=False)
        else:
            # Count by subset columns
            counts = df_pandas.groupby(subset).size()
            dup_counts = counts[counts > 1].sort_values(ascending=False)
        
        # Add to info dict
        value_counts = []
        for idx, count in dup_counts.items():
            if isinstance(idx, tuple):
                # Convert values to strings for JSON serialization
                values = [str(x) if not pd.isna(x) else 'NaN' for x in idx]
                key = dict(zip(dup_counts.index.names, values)) if dup_counts.index.names else dict(enumerate(values))
            else:
                key = {subset[0] if subset else 'value': str(idx) if not pd.isna(idx) else 'NaN'}
            
            value_counts.append({
                'values': key,
                'count': int(count)
            })
        
        info_dict['value_counts'] = value_counts
    
    # Return detailed result if requested
    if return_detailed and duplicate_count > 0:
        # Get duplicate rows
        duplicate_df = df_pandas[duplicated].copy()
        return duplicate_df, info_dict
    
    return info_dict


def remove_duplicates(
    df: Any,
    subset: Optional[Union[str, List[str]]] = None,
    keep: str = 'first',
    inplace: bool = False,
) -> Any:
    """
    Remove duplicate rows from a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    subset : Optional[Union[str, List[str]]], default=None
        Column(s) to consider for identifying duplicates. If None, uses all columns.
    keep : str, default='first'
        Which occurrence to keep. Options: 'first', 'last', False.
        If 'first', keep the first occurrence of each duplicate.
        If 'last', keep the last occurrence of each duplicate.
        If False, remove all duplicates.
    inplace : bool, default=False
        If True, perform operation in-place on the dataframe.
    
    Returns
    -------
    Any
        DataFrame with duplicates removed.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 1, 3, 4],
    ...     'B': ['x', 'y', 'x', 'z', 'x'],
    ... })
    >>> result = remove_duplicates(df, subset=['A', 'B'])
    >>> len(result)
    4
    """
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Set subset to list if it's a string
    if isinstance(subset, str):
        subset = [subset]
    
    if df_type == 'pandas':
        if inplace:
            result = df.drop_duplicates(subset=subset, keep=keep, inplace=True)
            return df
        else:
            return df.drop_duplicates(subset=subset, keep=keep, inplace=False)
    
    elif df_type == 'polars':
        # Convert arguments to polars-compatible format
        if keep == 'first':
            keep_strategy = 'first'
        elif keep == 'last':
            keep_strategy = 'last'
        else:  # keep == False
            keep_strategy = None
        
        # Process with polars
        if subset is None:
            # Use all columns if subset is None
            if inplace:
                df = df.unique(keep=keep_strategy)
                return df
            else:
                return df.unique(keep=keep_strategy)
        else:
            # Use specified subset
            if inplace:
                df = df.unique(subset=subset, keep=keep_strategy)
                return df
            else:
                return df.unique(subset=subset, keep=keep_strategy)
    
    elif df_type == 'dask':
        if inplace:
            # Dask doesn't support inplace operations, so we'll assign back
            result = df.drop_duplicates(subset=subset, keep=keep)
            # This is a no-op for dask, but we'll keep it for consistency
            df = result
            return df
        else:
            return df.drop_duplicates(subset=subset, keep=keep)
    
    else:
        # Convert to pandas, process, and convert back if not supported
        df_pandas = convert_dataframe(df, 'pandas')
        result_pandas = df_pandas.drop_duplicates(subset=subset, keep=keep, inplace=False)
        
        # Convert back to original type if needed
        if df_type != 'unknown':
            return convert_dataframe(result_pandas, df_type)
        else:
            return result_pandas