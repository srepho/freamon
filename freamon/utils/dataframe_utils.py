"""
Utilities for working with different types of dataframes.
"""
from typing import Any, Dict, Literal, Optional, Union

import pandas as pd


def check_dataframe_type(df: Any) -> str:
    """
    Check the type of dataframe.

    Parameters
    ----------
    df : Any
        The dataframe to check.

    Returns
    -------
    str
        The type of dataframe: 'pandas', 'polars', 'dask', or 'unknown'.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> check_dataframe_type(df)
    'pandas'
    """
    if isinstance(df, pd.DataFrame):
        return "pandas"
    # We'll add more dataframe type checks as needed
    return "unknown"


def convert_dataframe(
    df: Any, to_type: Literal["pandas", "polars", "dask"]
) -> Any:
    """
    Convert a dataframe to the specified type.

    Parameters
    ----------
    df : Any
        The dataframe to convert.
    to_type : Literal["pandas", "polars", "dask"]
        The type to convert to.

    Returns
    -------
    Any
        The converted dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> converted_df = convert_dataframe(df, "pandas")
    >>> isinstance(converted_df, pd.DataFrame)
    True
    """
    df_type = check_dataframe_type(df)
    
    if df_type == to_type:
        return df
    
    if to_type == "pandas":
        if df_type == "unknown":
            raise ValueError("Cannot convert unknown dataframe type to pandas")
        # Handle other conversions to pandas
        
    # Add more conversion logic as needed
    
    raise NotImplementedError(f"Conversion from {df_type} to {to_type} not implemented")


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize dataframe memory usage by adjusting data types.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe to optimize.

    Returns
    -------
    pd.DataFrame
        The optimized dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0]})
    >>> optimized_df = optimize_dtypes(df)
    >>> optimized_df.dtypes  # doctest: +SKIP
    A    int8
    B    float32
    dtype: object
    """
    result = df.copy()
    
    # Optimize integers
    int_cols = result.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_min, col_max = result[col].min(), result[col].max()
        
        # Choose smallest int type that can represent the data
        if col_min >= -128 and col_max <= 127:
            result[col] = result[col].astype("int8")
        elif col_min >= -32768 and col_max <= 32767:
            result[col] = result[col].astype("int16")
        elif col_min >= -2147483648 and col_max <= 2147483647:
            result[col] = result[col].astype("int32")
    
    # Optimize floats
    float_cols = result.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        result[col] = result[col].astype("float32")
    
    # Optimize objects/strings (future enhancement)
    
    return result


def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Estimate memory usage of a dataframe, both total and per column.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe to analyze.

    Returns
    -------
    Dict[str, Union[float, Dict[str, float]]]
        Dictionary with total memory usage in MB and per-column breakdown.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> memory_info = estimate_memory_usage(df)
    >>> 'total_mb' in memory_info
    True
    >>> 'columns' in memory_info
    True
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory_mb = memory_usage.sum() / (1024 * 1024)
    
    column_memory = {}
    for col in df.columns:
        column_memory[col] = memory_usage[col] / (1024 * 1024)
    
    return {
        "total_mb": total_memory_mb,
        "columns": column_memory
    }