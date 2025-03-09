"""
Module for handling missing values in dataframes.
"""
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from freamon.utils import check_dataframe_type


def handle_missing_values(
    df: Any,
    strategy: Literal["drop", "mean", "median", "mode", "constant", "interpolate"] = "drop",
    fill_value: Optional[Any] = None,
    subset: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> Any:
    """
    Handle missing values in a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe with missing values to handle.
    strategy : Literal["drop", "mean", "median", "mode", "constant", "interpolate"]
        The strategy to use for handling missing values.
    fill_value : Optional[Any]
        The value to use for the 'constant' strategy.
    subset : Optional[Union[str, List[str]]]
        The column(s) to apply the strategy to. If None, applies to all columns.
    **kwargs
        Additional keyword arguments for specific strategies.
    
    Returns
    -------
    Any
        The dataframe with missing values handled.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 5, 6]})
    >>> result = handle_missing_values(df, strategy="mean")
    >>> result
       A    B
    0  1.0  5.5
    1  2.0  5.0
    2  1.5  6.0
    """
    df_type = check_dataframe_type(df)
    
    if df_type == "pandas":
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Determine columns to process
        if subset is None:
            cols = df.columns
        elif isinstance(subset, str):
            cols = [subset]
        else:
            cols = subset
        
        # Apply the strategy
        if strategy == "drop":
            # Drop rows with any missing value in the specified columns
            # If subset is None, drops rows with any missing values across all columns
            if subset is None:
                return result.dropna()
            else:
                return result.dropna(subset=cols)
        
        elif strategy == "mean":
            # Fill missing values with column means (numeric columns only)
            for col in cols:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())
            return result
        
        elif strategy == "median":
            # Fill missing values with column medians (numeric columns only)
            for col in cols:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
            return result
        
        elif strategy == "mode":
            # Fill missing values with column modes
            for col in cols:
                mode_value = result[col].mode()
                if not mode_value.empty:
                    result[col] = result[col].fillna(mode_value[0])
            return result
        
        elif strategy == "constant":
            # Fill missing values with a constant value
            if fill_value is None:
                raise ValueError("fill_value must be provided for constant strategy")
            return result.fillna(value={col: fill_value for col in cols})
        
        elif strategy == "interpolate":
            # Interpolate missing values
            method = kwargs.get("method", "linear")
            for col in cols:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].interpolate(method=method)
            return result
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    # Add support for other dataframe types as needed
    raise NotImplementedError(f"Missing value handling not implemented for {df_type}")