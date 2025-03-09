"""
Module for detecting outliers in dataframes.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from freamon.utils import check_dataframe_type


def detect_outliers(
    df: Any,
    method: Literal["iqr", "zscore", "modified_zscore"] = "iqr",
    columns: Optional[Union[str, List[str]]] = None,
    threshold: float = 1.5,
    return_mask: bool = False,
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Detect outliers in a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe to analyze.
    method : Literal["iqr", "zscore", "modified_zscore"]
        The method to use for outlier detection.
    columns : Optional[Union[str, List[str]]]
        The column(s) to check for outliers. If None, checks all numeric columns.
    threshold : float
        The threshold for outlier detection. Default is 1.5 for IQR method.
        For z-score methods, default is 3.0.
    return_mask : bool
        If True, returns a dictionary of boolean masks for each column.
        If False, returns the dataframe with outliers removed.
    
    Returns
    -------
    Union[pd.DataFrame, Dict[str, np.ndarray]]
        Either the dataframe with outliers removed, or a dictionary of boolean masks.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [5, 6, 7, 8]})
    >>> result = detect_outliers(df, method="iqr", return_mask=True)
    >>> result['A']
    array([False, False, False,  True])
    """
    df_type = check_dataframe_type(df)
    
    if df_type == "pandas":
        # Determine columns to process
        if columns is None:
            # Get all numeric columns
            cols = df.select_dtypes(include=np.number).columns.tolist()
        elif isinstance(columns, str):
            cols = [columns]
        else:
            cols = columns
        
        outlier_masks = {}
        
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_masks[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Avoid division by zero
                    outlier_masks[col] = np.zeros(len(df), dtype=bool)
                    continue
                z_scores = np.abs((df[col] - mean) / std)
                outlier_masks[col] = z_scores > threshold
            
            elif method == "modified_zscore":
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                if mad == 0:  # Avoid division by zero
                    outlier_masks[col] = np.zeros(len(df), dtype=bool)
                    continue
                modified_z_scores = 0.6745 * np.abs(df[col] - median) / mad
                outlier_masks[col] = modified_z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        if return_mask:
            return outlier_masks
        else:
            # Create a mask for rows to keep (not outliers in any column)
            combined_mask = np.zeros(len(df), dtype=bool)
            for col_mask in outlier_masks.values():
                combined_mask |= col_mask
            
            # Return dataframe with outliers removed
            return df[~combined_mask]
    
    # Add support for other dataframe types as needed
    raise NotImplementedError(f"Outlier detection not implemented for {df_type}")