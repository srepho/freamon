"""
Module for analyzing cardinality of columns in dataframes.
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from freamon.utils import check_dataframe_type, convert_dataframe


def analyze_cardinality(
    df: Any,
    columns: Optional[List[str]] = None,
    max_unique_to_list: int = 20,
    include_plots: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze the cardinality (number of unique values) of columns in a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe to analyze. Can be pandas, polars, or dask.
    columns : Optional[List[str]], default=None
        List of column names to analyze. If None, analyzes all columns.
    max_unique_to_list : int, default=20
        Maximum number of unique values to include in the result for each column.
        If a column has more unique values than this, only the most frequent ones
        will be included.
    include_plots : bool, default=True
        Whether to include plot images (as base64) in the results.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping column names to dictionaries of cardinality statistics.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': ['x', 'y', 'x', 'y', 'z'],
    ...     'C': ['a', 'a', 'a', 'a', 'a'],
    ... })
    >>> result = analyze_cardinality(df)
    >>> result['A']['unique_count']
    5
    >>> result['B']['unique_count']
    3
    >>> result['C']['unique_count']
    1
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Determine which columns to analyze
    if columns is None:
        columns = df_pandas.columns.tolist()
    else:
        # Verify all columns exist
        for col in columns:
            if col not in df_pandas.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        col_result = _analyze_column_cardinality(
            df_pandas, col, max_unique_to_list, include_plots
        )
        result[col] = col_result
    
    return result


def _analyze_column_cardinality(
    df: pd.DataFrame,
    column: str,
    max_unique_to_list: int,
    include_plots: bool,
) -> Dict[str, Any]:
    """Analyze the cardinality of a single column."""
    series = df[column]
    total_count = len(series)
    missing_count = series.isna().sum()
    valid_count = total_count - missing_count
    
    # Calculate unique count, excluding nulls
    unique_count = series.nunique(dropna=True)
    
    # Cardinality ratio (excluding nulls)
    cardinality_ratio = unique_count / valid_count if valid_count > 0 else 0
    
    # Determine column type
    if pd.api.types.is_numeric_dtype(series):
        col_type = 'numeric'
    elif pd.api.types.is_datetime64_dtype(series):
        col_type = 'datetime'
    else:
        col_type = 'categorical'
    
    # Classify column based on cardinality
    if cardinality_ratio == 0:
        cardinality_type = 'empty'
    elif cardinality_ratio < 0.01:
        cardinality_type = 'very_low'
    elif cardinality_ratio < 0.05:
        cardinality_type = 'low'
    elif cardinality_ratio < 0.2:
        cardinality_type = 'medium'
    elif cardinality_ratio < 0.5:
        cardinality_type = 'high'
    elif cardinality_ratio < 1.0:
        cardinality_type = 'very_high'
    else:
        cardinality_type = 'unique'
    
    # Create result dict
    result = {
        'unique_count': unique_count,
        'total_count': total_count,
        'missing_count': missing_count,
        'valid_count': valid_count,
        'cardinality_ratio': cardinality_ratio,
        'cardinality_type': cardinality_type,
        'column_type': col_type,
    }
    
    # Get value counts
    value_counts = series.value_counts(dropna=False).sort_values(ascending=False)
    
    # Handle nulls for consistent output
    renamed_counts = {}
    for val, count in value_counts.items():
        if pd.isna(val):
            key = 'null'
        else:
            # Convert to string for JSON compatibility
            key = str(val)
        
        renamed_counts[key] = int(count)
    
    # Limit the number of values to include
    if len(renamed_counts) > max_unique_to_list:
        limited_counts = {}
        i = 0
        total_other = 0
        
        for key, count in renamed_counts.items():
            if i < max_unique_to_list:
                limited_counts[key] = count
                i += 1
            else:
                total_other += count
        
        if total_other > 0:
            limited_counts['other'] = total_other
        
        result['value_counts'] = limited_counts
        result['value_counts_limited'] = True
    else:
        result['value_counts'] = renamed_counts
        result['value_counts_limited'] = False
    
    # Add plot if requested
    if include_plots and col_type in ['categorical', 'numeric'] and unique_count > 0:
        # Create a plot of the value distribution
        plt.figure(figsize=(10, 6))
        
        # For categorical columns with few unique values, use a bar chart
        if col_type == 'categorical' or (col_type == 'numeric' and unique_count <= 20):
            # Get top values
            top_values = list(result['value_counts'].keys())[:max_unique_to_list]
            top_counts = [result['value_counts'][key] for key in top_values]
            
            # Create bar chart
            plt.bar(range(len(top_values)), top_counts)
            plt.xticks(range(len(top_values)), top_values, rotation=45, ha='right')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.title(f'Value Distribution for {column}')
            
            # Add count labels
            for i, count in enumerate(top_counts):
                plt.text(i, count + 0.5, str(count), ha='center')
            
        # For numeric columns with many unique values, use a histogram
        elif col_type == 'numeric' and unique_count > 20:
            # Use only non-null values for the histogram
            valid_series = series.dropna()
            plt.hist(valid_series, bins=min(30, unique_count))
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.title(f'Value Distribution for {column}')
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        result['plot'] = f"data:image/png;base64,{img_str}"
    
    return result