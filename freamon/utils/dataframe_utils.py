"""
Utilities for working with different types of dataframes.
"""
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
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
    
    # Check for polars DataFrame
    try:
        import polars
        if isinstance(df, polars.DataFrame):
            return "polars"
    except ImportError:
        pass
    
    # Check for dask DataFrame
    try:
        import dask.dataframe
        if isinstance(df, dask.dataframe.DataFrame):
            return "dask"
    except ImportError:
        pass
    
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
    >>> # Convert pandas to pandas (no-op)
    >>> converted_df = convert_dataframe(df, "pandas")
    >>> isinstance(converted_df, pd.DataFrame)
    True
    
    >>> # Convert to polars (if installed)
    >>> try:
    ...     import polars
    ...     polars_df = convert_dataframe(df, "polars")
    ...     isinstance(polars_df, polars.DataFrame)
    ... except ImportError:
    ...     True
    True
    """
    df_type = check_dataframe_type(df)
    
    if df_type == to_type:
        return df
    
    if to_type == "pandas":
        if df_type == "polars":
            # Polars to pandas
            return df.to_pandas()
        elif df_type == "dask":
            # Dask to pandas
            return df.compute()
        elif df_type == "unknown":
            raise ValueError("Cannot convert unknown dataframe type to pandas")
    
    elif to_type == "polars":
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "polars is not installed. Install it with 'pip install polars'."
            )
        
        if df_type == "pandas":
            # Pandas to polars
            return pl.from_pandas(df)
        elif df_type == "dask":
            # Dask to polars (via pandas)
            return pl.from_pandas(df.compute())
        elif df_type == "unknown":
            raise ValueError("Cannot convert unknown dataframe type to polars")
    
    elif to_type == "dask":
        try:
            import dask.dataframe as dd
        except ImportError:
            raise ImportError(
                "dask is not installed. Install it with 'pip install dask[dataframe]'."
            )
        
        if df_type == "pandas":
            # Pandas to dask
            return dd.from_pandas(df, npartitions=1)
        elif df_type == "polars":
            # Polars to dask (via pandas)
            return dd.from_pandas(df.to_pandas(), npartitions=1)
        elif df_type == "unknown":
            raise ValueError("Cannot convert unknown dataframe type to dask")
    
    raise NotImplementedError(f"Conversion from {df_type} to {to_type} not implemented")


def optimize_dtypes(df: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """
    Optimize dataframe memory usage by adjusting data types.

    Parameters
    ----------
    df : Union[pd.DataFrame, Any]
        The dataframe to optimize. Can be pandas or polars.

    Returns
    -------
    Union[pd.DataFrame, Any]
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
    df_type = check_dataframe_type(df)
    
    if df_type == "pandas":
        return _optimize_pandas_dtypes(df)
    elif df_type == "polars":
        return _optimize_polars_dtypes(df)
    else:
        raise ValueError(f"Optimize dtypes not supported for {df_type} dataframes")


def _optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize pandas dataframe memory usage by adjusting data types.
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe to optimize.
        
    Returns
    -------
    pd.DataFrame
        The optimized dataframe.
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
    
    # Optimize objects/strings (categories)
    obj_cols = result.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        num_unique = result[col].nunique()
        num_total = len(result)
        
        # If the column has relatively few unique values, convert to category
        if num_unique / num_total < 0.5:
            result[col] = result[col].astype("category")
    
    return result


def _optimize_polars_dtypes(df: Any) -> Any:
    """
    Optimize polars dataframe memory usage by adjusting data types.
    
    Parameters
    ----------
    df : Any
        The polars dataframe to optimize.
        
    Returns
    -------
    Any
        The optimized dataframe.
    """
    import polars as pl
    
    # Create the optimization schema
    schema = {}
    
    # Get the current schema
    current_schema = df.schema
    
    for name, dtype in current_schema.items():
        # Optimize integers
        if dtype == pl.Int64:
            # Sample the column to find min/max (avoid processing entire column)
            sample = df.select(pl.col(name).min().alias("min"), pl.col(name).max().alias("max")).row(0)
            col_min, col_max = sample['min'], sample['max']
            
            if col_min >= -128 and col_max <= 127:
                schema[name] = pl.Int8
            elif col_min >= -32768 and col_max <= 32767:
                schema[name] = pl.Int16
            elif col_min >= -2147483648 and col_max <= 2147483647:
                schema[name] = pl.Int32
        
        # Optimize floats
        elif dtype == pl.Float64:
            schema[name] = pl.Float32
        
        # Optimize strings conditionally
        elif dtype == pl.Utf8:
            # Count unique values
            n_unique = df.select(pl.col(name).n_unique()).row(0)[0]
            n_total = df.height
            
            # If the column has relatively few unique values, convert to category
            if n_unique / n_total < 0.5:
                schema[name] = pl.Categorical
    
    # Apply the optimized schema
    if schema:
        return df.with_columns([pl.col(name).cast(dtype) for name, dtype in schema.items()])
    
    return df


def estimate_memory_usage(df: Union[pd.DataFrame, Any]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Estimate memory usage of a dataframe, both total and per column.

    Parameters
    ----------
    df : Union[pd.DataFrame, Any]
        The dataframe to analyze. Can be pandas or polars.

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
    df_type = check_dataframe_type(df)
    
    if df_type == "pandas":
        return _estimate_pandas_memory_usage(df)
    elif df_type == "polars":
        return _estimate_polars_memory_usage(df)
    else:
        raise ValueError(f"Memory usage estimation not supported for {df_type} dataframes")


def _estimate_pandas_memory_usage(df: pd.DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
    """Estimate memory usage for pandas dataframes."""
    memory_usage = df.memory_usage(deep=True)
    total_memory_mb = memory_usage.sum() / (1024 * 1024)
    
    column_memory = {}
    for col in df.columns:
        column_memory[col] = memory_usage[col] / (1024 * 1024)
    
    return {
        "total_mb": total_memory_mb,
        "columns": column_memory
    }


def _estimate_polars_memory_usage(df: Any) -> Dict[str, Union[float, Dict[str, float]]]:
    """Estimate memory usage for polars dataframes."""
    import polars as pl
    
    # Polars doesn't have a direct equivalent to pandas memory_usage(deep=True)
    # We'll approximate using known type sizes
    
    column_memory = {}
    total_memory = 0
    
    type_sizes = {
        pl.Int8: 1,
        pl.Int16: 2,
        pl.Int32: 4,
        pl.Int64: 8,
        pl.UInt8: 1,
        pl.UInt16: 2,
        pl.UInt32: 4,
        pl.UInt64: 8,
        pl.Float32: 4,
        pl.Float64: 8,
        pl.Boolean: 1,
        pl.Date: 4,
        pl.Datetime: 8,
        pl.Time: 8,
        pl.Duration: 8,
    }
    
    n_rows = df.height
    for col_name, dtype in df.schema.items():
        if dtype in type_sizes:
            # For fixed-size types
            col_size = type_sizes[dtype] * n_rows
        elif dtype == pl.Categorical:
            # For categorical, estimate as 4 bytes per row plus dictionary
            n_unique = df.select(pl.col(col_name).n_unique()).row(0)[0]
            col_size = 4 * n_rows + 32 * n_unique  # Rough estimate
        elif dtype == pl.Utf8:
            # For strings, sample average length
            sample_size = min(1000, n_rows)
            if sample_size > 0:
                # Take a sample to estimate average string length
                sample = df.select(pl.col(col_name)).sample(sample_size, seed=42)
                avg_len = sample.select(pl.col(col_name).str.len_bytes().mean()).row(0)[0]
                if pd.isna(avg_len):
                    avg_len = 8  # Default if we can't calculate
                col_size = avg_len * n_rows
            else:
                col_size = 0
        else:
            # Default estimate for other types
            col_size = 8 * n_rows
        
        # Convert to MB
        col_size_mb = col_size / (1024 * 1024)
        column_memory[col_name] = col_size_mb
        total_memory += col_size
    
    # Convert total to MB
    total_memory_mb = total_memory / (1024 * 1024)
    
    return {
        "total_mb": total_memory_mb,
        "columns": column_memory
    }


def detect_datetime_columns(
    df: Union[pd.DataFrame, Any],
    threshold: float = 0.9,
    inplace: bool = False,
    sample_size: int = 1000,
    date_formats: Optional[List[str]] = None,
) -> Union[pd.DataFrame, Any]:
    """
    Detect and convert potential datetime columns in a dataframe.

    Parameters
    ----------
    df : Union[pd.DataFrame, Any]
        The dataframe to process. Can be pandas or polars.
    threshold : float, default=0.9
        The minimum proportion of values that must be parseable as dates for a
        column to be considered a datetime column.
    inplace : bool, default=False
        Whether to modify the dataframe in place.
    sample_size : int, default=1000
        The number of values to sample for date detection. Use -1 to check all values.
    date_formats : Optional[List[str]], default=None
        List of date formats to check. If None, a default list will be used.

    Returns
    -------
    Union[pd.DataFrame, Any]
        The dataframe with datetime columns converted.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'date_str': ['2020-01-01', '2020-02-01', '2020-03-01'],
    ...     'timestamp': [1577836800, 1580515200, 1583020800]
    ... })
    >>> result = detect_datetime_columns(df)
    >>> result['date_str'].dtype
    dtype('<M8[ns]')
    """
    df_type = check_dataframe_type(df)
    
    # Use default date formats if none provided
    if date_formats is None:
        date_formats = [
            # ISO formats
            "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z",
            
            # Common formats
            "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
            "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
            
            # Month name formats
            "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
            
            # Other formats
            "%Y%m%d", "%Y%m%d%H%M%S"
        ]
    
    if df_type == "pandas":
        return _detect_pandas_datetime(df, threshold, inplace, sample_size, date_formats)
    elif df_type == "polars":
        return _detect_polars_datetime(df, threshold, inplace, sample_size, date_formats)
    else:
        raise ValueError(f"Datetime detection not supported for {df_type} dataframes")


def _detect_pandas_datetime(
    df: pd.DataFrame,
    threshold: float,
    inplace: bool,
    sample_size: int,
    date_formats: List[str],
) -> pd.DataFrame:
    """Detect datetime columns in pandas dataframes."""
    if not inplace:
        df = df.copy()
    
    # Only check string/object columns that aren't already datetime
    cols_to_check = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cols_to_check:
        # Skip columns that have a significant number of empty values
        if df[col].isna().mean() > 0.5:
            continue
        
        # Get a sample of values
        values = df[col].dropna()
        if sample_size > 0 and len(values) > sample_size:
            values = values.sample(sample_size, random_state=42)
        
        # Check if the column contains timestamps (all numeric values)
        if values.str.isnumeric().all():
            # If all values are numeric and within a reasonable timestamp range
            numeric_values = values.astype(float)
            min_timestamp = datetime(1970, 1, 1).timestamp()
            max_timestamp = datetime(2050, 1, 1).timestamp()
            if (
                (numeric_values >= min_timestamp) & 
                (numeric_values <= max_timestamp)
            ).mean() >= threshold:
                # Convert to datetime
                try:
                    df[col] = pd.to_datetime(df[col].astype(float), unit='s')
                    continue
                except (ValueError, OverflowError):
                    pass
        
        # Check if the column follows a date pattern
        is_date = False
        
        # First, try pandas' automatic date parsing
        try:
            parsed = pd.to_datetime(values, errors='coerce')
            if parsed.notna().mean() >= threshold:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                is_date = True
        except (ValueError, TypeError):
            pass
        
        # If automatic parsing fails, try explicit formats
        if not is_date:
            for date_format in date_formats:
                try:
                    parsed = pd.to_datetime(values, format=date_format, errors='coerce')
                    if parsed.notna().mean() >= threshold:
                        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                        is_date = True
                        break
                except (ValueError, TypeError):
                    continue
    
    # Also look for integer columns that might be unix timestamps
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    
    for col in int_cols:
        values = df[col].dropna()
        if sample_size > 0 and len(values) > sample_size:
            values = values.sample(sample_size, random_state=42)
        
        # Check if values are in a reasonable unix timestamp range
        min_timestamp = datetime(1970, 1, 1).timestamp()
        max_timestamp = datetime(2050, 1, 1).timestamp()
        
        if (
            (values >= min_timestamp) & 
            (values <= max_timestamp)
        ).mean() >= threshold:
            # Try converting to datetime
            try:
                parsed = pd.to_datetime(values, unit='s', errors='coerce')
                if parsed.notna().mean() >= threshold:
                    df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
            except (ValueError, OverflowError, TypeError):
                pass
    
    return df


def _detect_polars_datetime(
    df: Any,
    threshold: float,
    inplace: bool,
    sample_size: int,
    date_formats: List[str],
) -> Any:
    """Detect datetime columns in polars dataframes."""
    import polars as pl
    
    # If not inplace, clone the dataframe
    if not inplace:
        df = df.clone()
    
    # Get schema and identify string columns
    schema = df.schema
    str_cols = [name for name, dtype in schema.items() if dtype == pl.Utf8]
    
    for col in str_cols:
        # Skip columns with too many null values
        null_ratio = df.select(pl.col(col).is_null().mean()).row(0)[0]
        if null_ratio > 0.5:
            continue
        
        # Get a sample of values
        if sample_size > 0 and df.height > sample_size:
            sample = df.select(pl.col(col)).sample(sample_size, seed=42)
        else:
            sample = df.select(pl.col(col))
        
        # Try each date format
        for date_format in date_formats:
            # Convert strptime format to polars strptime format
            pl_format = _convert_to_polars_dateformat(date_format)
            
            # Try parsing with this format
            parsed = sample.select(
                pl.col(col).str.strptime(pl.Datetime, pl_format, strict=False)
            )
            
            # Check if enough values parsed successfully
            success_ratio = parsed.select(pl.col(col).is_not_null().mean()).row(0)[0]
            
            if success_ratio >= threshold:
                # Convert the entire column
                df = df.with_columns([
                    pl.col(col).str.strptime(pl.Datetime, pl_format, strict=False)
                ])
                break
    
    # Check integer columns for unix timestamps
    int_cols = [name for name, dtype in schema.items() 
                if dtype in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)]
    
    for col in int_cols:
        # Get a sample of values
        if sample_size > 0 and df.height > sample_size:
            sample = df.select(pl.col(col)).sample(sample_size, seed=42)
        else:
            sample = df.select(pl.col(col))
        
        # Calculate min and max of the sample
        min_max = sample.select([
            pl.col(col).min().alias('min'),
            pl.col(col).max().alias('max')
        ]).row(0)
        
        min_val, max_val = min_max['min'], min_max['max']
        
        # Check if in reasonable unix timestamp range
        min_timestamp = int(datetime(1970, 1, 1).timestamp())
        max_timestamp = int(datetime(2050, 1, 1).timestamp())
        
        if min_val is not None and max_val is not None and min_val >= min_timestamp and max_val <= max_timestamp:
            # Try converting to datetime
            df = df.with_columns([
                pl.col(col).cast(pl.Int64).cast(pl.Datetime, time_unit='s')
            ])
    
    return df


def _convert_to_polars_dateformat(strptime_format: str) -> str:
    """
    Convert a datetime.strptime format string to a polars strptime format string.
    
    Parameters
    ----------
    strptime_format : str
        The strptime format string.
        
    Returns
    -------
    str
        The polars strptime format string.
    """
    # Mapping of Python's strptime directives to Polars strptime directives
    # This is a simplified mapping and may not cover all cases
    mapping = {
        "%Y": "%Y",  # Year with century
        "%m": "%m",  # Month as zero-padded decimal
        "%d": "%d",  # Day of the month as zero-padded decimal
        "%H": "%H",  # Hour (24-hour clock) as zero-padded decimal
        "%I": "%I",  # Hour (12-hour clock) as zero-padded decimal
        "%M": "%M",  # Minute as zero-padded decimal
        "%S": "%S",  # Second as zero-padded decimal
        "%f": "%f",  # Microsecond as decimal
        "%z": "%z",  # UTC offset
        "%Z": "%Z",  # Time zone name
        "%p": "%p",  # AM/PM
        "%b": "%b",  # Month as abbreviated name
        "%B": "%B",  # Month as full name
        "%a": "%a",  # Weekday as abbreviated name
        "%A": "%A",  # Weekday as full name
        "%j": "%j",  # Day of the year as zero-padded decimal
        "%U": "%U",  # Week number of the year (Sunday as first day)
        "%W": "%W",  # Week number of the year (Monday as first day)
        "%c": "%c",  # Locale's appropriate date and time representation
        "%x": "%x",  # Locale's appropriate date representation
        "%X": "%X",  # Locale's appropriate time representation
        "%G": "%G",  # ISO 8601 year
        "%u": "%u",  # ISO 8601 weekday (1-7)
        "%V": "%V",  # ISO 8601 week number
    }
    
    # Apply the mapping
    for python_fmt, polars_fmt in mapping.items():
        strptime_format = strptime_format.replace(python_fmt, polars_fmt)
    
    return strptime_format