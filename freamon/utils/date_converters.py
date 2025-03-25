"""
Date conversion utilities for special datetime formats like month-year.
"""
import re
from typing import List, Union

import numpy as np
import pandas as pd


def convert_month_year_format(values: Union[pd.Series, List, np.ndarray]) -> pd.Series:
    """
    Convert various month-year formats to datetime objects.
    
    This function handles multiple month-year formats like:
    - 'Aug-24', 'Sep-2024' (Month abbreviation with hyphen and year)
    - 'October-23', 'November-2023' (Full month with hyphen and year)
    - 'Dec 24' (Month abbreviation with space and year)
    - '01-25', '02/25', '03.25' (Numeric month with various separators and year)
    - 'Jan/22', 'Apr.22', 'May22' (Month name with various separators or no separator)
    
    Parameters
    ----------
    values : Union[pd.Series, List, np.ndarray]
        The values to convert to datetime
        
    Returns
    -------
    pd.Series
        Series of datetime objects with NaT for values that couldn't be converted
        
    Examples
    --------
    >>> convert_month_year_format(['Aug-24', 'Sep-2024', 'Dec 24', '01-25'])
    0   2024-08-01
    1   2024-09-01
    2   2024-12-01
    3   2025-01-01
    dtype: datetime64[ns]
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Convert to pandas Series if not already
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Convert everything to Python strings for consistency
    # This avoids issues with numpy.str_ and other types
    # Replace NaN with empty strings to avoid errors
    string_values = values.fillna('').astype(str)
    string_values = string_values.replace('', np.nan)
    
    # Try a direct conversion with the most common format first
    # Most month-year formats use the 3-letter abbreviation with 2-digit year
    try:
        # This will handle values like 'Jun-24', 'Jul-24', etc.
        result = pd.to_datetime(string_values, format='%b-%y', errors='coerce')
        logger.debug(f"Direct conversion with '%b-%y' format successful for {result.notna().sum()} values")
        
        # If all values were successfully parsed, return the result
        if result.notna().all():
            return result
    except Exception as e:
        logger.debug(f"Direct conversion with '%b-%y' format failed: {e}")
        # Initialize result series with NaT if direct conversion failed
        result = pd.Series([pd.NaT] * len(values), index=values.index)
    
    # Define format patterns and try each one for remaining NaT values
    format_patterns = [
        # Month abbreviation with 2-digit year (Aug-24)
        ('%b-%y', r'^[A-Za-z]{3}-\d{2}$'),
        # Month abbreviation with 4-digit year (Sep-2024)
        ('%b-%Y', r'^[A-Za-z]{3}-\d{4}$'),
        # Full month with 2-digit year (October-23)
        ('%B-%y', r'^[A-Za-z]{4,}-\d{2}$'),
        # Full month with 4-digit year (November-2023)
        ('%B-%Y', r'^[A-Za-z]{4,}-\d{4}$'),
        # Month abbreviation with space and 2-digit year (Dec 24)
        ('%b %y', r'^[A-Za-z]{3}\s+\d{2}$'),
        # Month abbreviation with slash and 2-digit year (Jan/22)
        ('%b/%y', r'^[A-Za-z]{3}/\d{2}$'),
        # Month abbreviation with dot and 2-digit year (Apr.22)
        ('%b.%y', r'^[A-Za-z]{3}\.\d{2}$'),
        # Month abbreviation without separator and 2-digit year (May22)
        ('%b%y', r'^[A-Za-z]{3}\d{2}$'),
        # Full month with space and 2-digit year (March 22)
        ('%B %y', r'^[A-Za-z]{4,}\s+\d{2}$'),
        # Numeric month with hyphen and 2-digit year (01-25)
        ('%m-%y', r'^\d{1,2}-\d{2}$'),
        # Numeric month with slash and 2-digit year (02/25)
        ('%m/%y', r'^\d{1,2}/\d{2}$'),
        # Numeric month with dot and 2-digit year (03.25)
        ('%m.%y', r'^\d{1,2}\.\d{2}$'),
    ]
    
    # Process each pattern for remaining NaT values
    for fmt, pattern in format_patterns:
        # Skip if all values have been parsed
        if result.notna().all():
            break
            
        # Only process values that are still NaT
        nat_mask = result.isna()
        if not nat_mask.any():
            continue
            
        # Check which values match the pattern (skip NaN values)
        pattern_mask = string_values[nat_mask].str.match(pattern, case=False, na=False)
        if not pattern_mask.any():
            continue
            
        # Combine masks to get values that are both NaT and match the pattern
        combined_mask_indices = nat_mask[nat_mask].index[pattern_mask]
        
        try:
            # Convert these specific values
            parsed = pd.to_datetime(string_values.loc[combined_mask_indices], format=fmt, errors='coerce')
            
            # Update the result series
            result.loc[combined_mask_indices] = parsed
            logger.debug(f"Parsed {parsed.notna().sum()} values with format '{fmt}'")
        except Exception as e:
            logger.debug(f"Error parsing with format '{fmt}': {e}")
    
    return result


def is_month_year_format(values: Union[pd.Series, List, np.ndarray], threshold: float = 0.9) -> bool:
    """
    Check if a series of values matches month-year format patterns.
    
    Parameters
    ----------
    values : Union[pd.Series, List, np.ndarray]
        The values to check
    threshold : float, default=0.9
        Minimum ratio of matching values to consider the entire series as month-year format
        
    Returns
    -------
    bool
        True if the series is likely month-year format, False otherwise
    
    Examples
    --------
    >>> is_month_year_format(['Aug-24', 'Sep-2024', 'Dec 24', '01-25'])
    True
    >>> is_month_year_format(['2023-01-01', '2023-12-31'])
    False
    """
    # Convert to pandas Series if not already
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Handle empty series
    if len(values) == 0:
        return False
    
    # Compile the month-year pattern regex - more flexible to match various formats
    month_year_pattern = re.compile(
        r'^([A-Za-z]{3,9})[-/\. ]?(20\d{2}|\d{2})$|^(0?\d|1[0-2])[-/\.](20\d{2}|\d{2})$', 
        re.IGNORECASE
    )
    
    # Count matches - handle numpy.str_ and other string-like types
    match_count = 0
    for x in values:
        if pd.notna(x):  # Skip NaN/None values
            x_str = str(x).strip()
            if month_year_pattern.match(x_str):
                match_count += 1
    
    match_ratio = match_count / len(values)
    
    # Return True if the ratio exceeds the threshold
    return match_ratio >= threshold