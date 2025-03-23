"""
Utilities for advanced data type detection and inference.
"""
import re
import os
import math
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Iterator, TypeVar

import numpy as np
import pandas as pd

from freamon.utils.date_converters import convert_month_year_format, is_month_year_format

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter common pandas warnings about date parsing
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually")


class DataTypeDetector:
    """
    Advanced data type detection and inference.
    
    This class provides methods to detect logical data types beyond the
    basic storage types, including IDs, zip codes, phone numbers, addresses,
    Excel dates, and distinguishing between categorical and continuous numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to analyze
    sample_size : int, default=1000
        Number of values to sample when detecting patterns
    threshold : float, default=0.9
        Minimum ratio of matching values to consider a pattern valid
    detect_semantic_types : bool, default=True
        Whether to detect semantic types like IDs, codes, etc.
    categorize_numeric : bool, default=True
        Whether to distinguish between categorical and continuous numeric columns
    custom_patterns : Optional[Dict[str, str]], default=None
        Dictionary of custom regex patterns to use for semantic type detection
        
    Examples
    --------
    >>> import pandas as pd
    >>> from freamon.utils.datatype_detector import DataTypeDetector
    >>> df = pd.DataFrame({
    ...     'id': range(100),
    ...     'category': ['A', 'B', 'C'] * 34,
    ...     'date_str': ['2023-01-01', '2023-02-01'] * 50,
    ...     'month_year': ['Jan-23', 'Feb-23'] * 50
    ... })
    >>> detector = DataTypeDetector(df)
    >>> results = detector.detect_all_types()
    >>> # For Jupyter notebooks, display the results as a styled DataFrame
    >>> detector.display_detection_report()
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        sample_size: int = 1000,
        threshold: float = 0.9,
        detect_semantic_types: bool = True,
        categorize_numeric: bool = True,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the DataTypeDetector with a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame to analyze
        sample_size : int, default=1000
            Number of values to sample when detecting patterns
        threshold : float, default=0.9
            Minimum ratio of matching values to consider a pattern valid
        detect_semantic_types : bool, default=True
            Whether to detect semantic types like IDs, codes, etc.
        categorize_numeric : bool, default=True
            Whether to distinguish between categorical and continuous numeric columns
        custom_patterns : Optional[Dict[str, str]], default=None
            Dictionary of custom regex patterns to use for semantic type detection
            in the format {'type_name': 'regex_pattern'}
        """
        self.df = df
        self.sample_size = sample_size
        self.threshold = threshold
        self.detect_semantic_types = detect_semantic_types
        self.categorize_numeric = categorize_numeric
        
        # Results storage
        self.column_types = {}
        self.conversion_suggestions = {}
        self.semantic_types = {}
        
        # Default regex patterns for detecting semantic types
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^(https?|ftp)://[^\s/$.?#].[^\s]*$',
            'ip_address': r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'phone_number': r'^\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'zip_code': r'^[0-9]{5}(-[0-9]{4})?$',
            'credit_card': r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$',
            'ssn': r'^[0-9]{3}-[0-9]{2}-[0-9]{4}$',
            'uuid': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
            'path': r'^(/[^/ ]*)+/?$',
            'currency': r'^\$?\s?[+-]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{2})?$',
            'isbn': r'^(?:ISBN(?:-1[03])?:?\s)?(?=[0-9X]{10}$|(?=(?:[0-9]+[-\s]){3})[-\s0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[-\s]){4})[-\s0-9]{17}$)(?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9X]$',
            'latitude': r'^[-+]?([1-8]?[0-9](\.[0-9]+)?|90(\.0+)?)$',
            'longitude': r'^[-+]?(180(\.0+)?|((1[0-7][0-9])|([1-9]?[0-9]))(\.[0-9]+)?)$',
            
            # Australian-specific patterns
            'au_postcode': r'^(0[0-9]{3}|[0-9]{4})$',  # Handles 4 digits, including leading zeros
            'au_phone': r'^(?:\+?61|0)[2-478](?:[ -]?[0-9]){8}$',  # Australian phone numbers
            'au_mobile': r'^(?:\+?61|0)[4](?:[ -]?[0-9]){8}$',     # Australian mobile numbers
            'au_abn': r'^([0-9][ ]?){11}$',                           # Australian Business Number
            'au_acn': r'^([0-9][ ]?){9}$',                            # Australian Company Number
            'au_tfn': r'^([0-9][ ]?){8,9}$',                          # Australian Tax File Number
        }
        
        # Add custom patterns if provided
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Date patterns for date detection
        self.date_formats = [
            # ISO formats
            "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z",
            
            # Common formats
            "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
            "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
            
            # Month name formats
            "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y",
            
            # Month-Year formats
            "%b-%y", "%b-%Y", "%B-%y", "%B-%Y",  # e.g., Aug-24, Aug-2024
            "%y-%b", "%Y-%b", "%y-%B", "%Y-%B",  # e.g., 24-Aug, 2024-Aug
            "%b %y", "%b %Y", "%B %y", "%B %Y",  # e.g., Aug 24, Aug 2024
            "%m-%y", "%m/%y", "%m.%y",           # e.g., 08-24, 08/24, 08.24
            "%y-%m", "%y/%m", "%y.%m",           # e.g., 24-08, 24/08, 24.08
            
            # Other formats
            "%Y%m%d", "%Y%m%d%H%M%S"
        ]
    
    def detect_all_types(self, use_pyarrow: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Detect and categorize all column types in the dataframe.
        
        Parameters
        ----------
        use_pyarrow : bool, default=True
            Whether to use PyArrow for faster preprocessing when possible
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with column type information
        """
        # Try to use PyArrow for initial type inference when dealing with large dataframes
        if use_pyarrow and len(self.df) > 10000:
            try:
                import pyarrow as pa
                import pyarrow.compute as pc
                
                logger.info(f"Using PyArrow for preprocessing large dataframe with {len(self.df)} rows")
                
                # Convert to arrow table for faster processing
                arrow_table = pa.Table.from_pandas(self.df)
                
                # Pre-compute some statistics that we'll need later
                # This is much faster in Arrow for large datasets
                column_stats = {}
                for col in self.df.columns:
                    try:
                        # Get arrow column
                        arrow_col = arrow_table[col]
                        
                        # Count nulls more efficiently
                        null_count = pc.sum(pc.is_null(arrow_col)).as_py()
                        
                        # Count unique values more efficiently
                        # Note: This might still be expensive for very large columns with many unique values
                        if len(self.df) > 100000:  # For extremely large datasets, use sampling
                            sample_indices = np.random.choice(len(self.df), 
                                                           min(50000, len(self.df)), 
                                                           replace=False)
                            sample_indices.sort()  # Keep indices sorted
                            sampled_col = pc.take(arrow_col, pa.array(sample_indices))
                            unique_count = len(pc.unique(sampled_col))
                            is_sampled = True
                        else:
                            unique_count = len(pc.unique(arrow_col))
                            is_sampled = False
                            
                        column_stats[col] = {
                            'null_count': null_count,
                            'null_ratio': null_count / len(self.df),
                            'unique_count': unique_count,
                            'unique_is_sampled': is_sampled,
                        }
                    except Exception as e:
                        logger.warning(f"PyArrow statistics failed for column {col}: {str(e)}")
                
                # Store this for later use
                self._column_stats = column_stats
                
            except (ImportError, Exception) as e:
                logger.warning(f"PyArrow preprocessing failed, falling back to standard method: {str(e)}")
                self._column_stats = None
        else:
            self._column_stats = None
        
        # First, detect basic types
        self._detect_basic_types()
        
        # Then detect datetime columns
        if 'object' in self.df.dtypes.values or 'int64' in self.df.dtypes.values:
            self._detect_datetime_columns()
        
        # Detect categorical vs continuous for numeric columns
        if self.categorize_numeric:
            self._detect_categorical_numeric()
        
        # Detect semantic types if requested
        if self.detect_semantic_types:
            self._detect_semantic_types()
        
        # Generate conversion suggestions
        self._generate_conversion_suggestions()
        
        # Compile the final result
        result = {}
        for col in self.df.columns:
            result[col] = {
                'storage_type': str(self.df[col].dtype),
                'logical_type': self.column_types.get(col, 'unknown'),
            }
            
            if col in self.semantic_types:
                result[col]['semantic_type'] = self.semantic_types[col]
            
            if col in self.conversion_suggestions:
                result[col]['suggested_conversion'] = self.conversion_suggestions[col]
        
        return result
    
    def _detect_basic_types(self) -> None:
        """
        Detect basic column types based on pandas dtypes.
        """
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    self.column_types[col] = 'integer'
                else:
                    self.column_types[col] = 'float'
            elif pd.api.types.is_datetime64_dtype(dtype):
                self.column_types[col] = 'datetime'
            elif isinstance(dtype, pd.CategoricalDtype):
                self.column_types[col] = 'categorical'
            elif pd.api.types.is_bool_dtype(dtype):
                self.column_types[col] = 'boolean'
            elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
                self.column_types[col] = 'string'
            else:
                self.column_types[col] = 'unknown'
    
    def _detect_datetime_columns(self) -> None:
        """
        Detect and convert potential datetime columns.
        """
        # Look for datetime patterns in string columns
        for col in self.df.columns:
            if self.column_types.get(col) == 'string':
                # Get sample of values for efficiency
                values = self.df[col].dropna()
                if len(values) == 0:
                    continue
                    
                if self.sample_size > 0 and len(values) > self.sample_size:
                    values = values.sample(self.sample_size, random_state=42)
                
                # PART 1: First try automatic parsing which handles some mixed formats
                try:
                    parsed = pd.to_datetime(values, errors='coerce')
                    if parsed.notna().mean() >= self.threshold:
                        self.column_types[col] = 'datetime'
                        self.conversion_suggestions[col] = {
                            'convert_to': 'datetime',
                            'method': 'pd.to_datetime'
                        }
                except (ValueError, TypeError):
                    pass
                
                # PART 2: Try explicit date formats if automatic parsing failed
                if self.column_types.get(col) != 'datetime':
                    # Use the specialized month-year format detection function
                    if is_month_year_format(values, threshold=self.threshold):
                        # It's likely a month-year format
                        self.column_types[col] = 'datetime'
                        self.semantic_types[col] = 'month_year_format'
                        
                        # Try to determine the best format
                        month_name_formats = ["%b-%y", "%b-%Y", "%B-%y", "%B-%Y", "%b %y", "%b %Y", "%B %y", "%B %Y"]
                        numeric_formats = ["%m-%y", "%m/%y", "%m.%y"]
                        
                        # Check for month name formats vs numeric formats
                        month_name_pattern = re.compile(r'^([A-Za-z]{3,9})[- ]', re.IGNORECASE)
                        uses_month_names = any(month_name_pattern.match(str(x).strip()) for x in values if not pd.isna(x))
                        
                        if uses_month_names:
                            self.conversion_suggestions[col] = {
                                'convert_to': 'datetime',
                                'method': 'convert_month_year_format',
                                'note': 'Contains month-year format with month names',
                                'detected_formats': month_name_formats
                            }
                        else:
                            self.conversion_suggestions[col] = {
                                'convert_to': 'datetime',
                                'method': 'convert_month_year_format',
                                'note': 'Contains month-year format with numeric months',
                                'detected_formats': numeric_formats
                            }
                    else:
                        # First, check if pandas can parse at least some dates automatically
                        try:
                            parsed = pd.to_datetime(values, errors='coerce')
                            auto_parse_ratio = parsed.notna().mean()
                            
                            # If some but not all dates can be parsed automatically, it might be mixed formats
                            if auto_parse_ratio > 0.1 and auto_parse_ratio < self.threshold:
                                # Try individual formats to see if we can improve coverage
                                format_success = {}
                                for date_format in self.date_formats:
                                    try:
                                        mask = parsed.isna()
                                        # Only try to parse values that weren't automatically parsed
                                        if mask.any():
                                            values_to_try = values[mask]
                                            parsed_with_format = pd.to_datetime(values_to_try, format=date_format, errors='coerce')
                                            success_ratio = parsed_with_format.notna().mean()
                                            if success_ratio > 0:
                                                format_success[date_format] = success_ratio
                                    except (ValueError, TypeError):
                                        continue
                                
                                # If we found additional successful formats
                                if format_success:
                                    self.column_types[col] = 'datetime'
                                    self.semantic_types[col] = 'mixed_date_formats'
                                    detected_formats = sorted(format_success.items(), key=lambda x: x[1], reverse=True)
                                    detected_format_list = ['auto'] + [fmt for fmt, _ in detected_formats]
                                    
                                    self.conversion_suggestions[col] = {
                                        'convert_to': 'datetime',
                                        'method': 'pd.to_datetime(errors="coerce")',
                                        'detected_formats': detected_format_list
                                    }
                        except:
                            pass
                        
                        # If still not detected as datetime, try each format individually
                        if self.column_types.get(col) != 'datetime':
                            for date_format in self.date_formats:
                                try:
                                    parsed = pd.to_datetime(values, format=date_format, errors='coerce')
                                    if parsed.notna().mean() >= self.threshold:
                                        self.column_types[col] = 'datetime'
                                        self.conversion_suggestions[col] = {
                                            'convert_to': 'datetime',
                                            'method': f'pd.to_datetime(format="{date_format}")'
                                        }
                                        break
                                except (ValueError, TypeError):
                                    continue
        
        # Look for timestamps in integer columns and for Excel dates in numeric columns
        for col in self.df.columns:
            if self.column_types.get(col) in ['integer', 'float']:
                # Check for column names that suggest it might be an ID
                # Skip non-string column names (e.g., integers)
                if not isinstance(col, str):
                    continue
                    
                if col.lower() in ['id', 'key', 'index', 'code']:
                    continue
                
                values = self.df[col].dropna()
                if len(values) == 0:
                    continue
                    
                if self.sample_size > 0 and len(values) > self.sample_size:
                    values = values.sample(self.sample_size, random_state=42)
                
                # PART 1: Check for Unix timestamps (large integer values)
                if self.column_types.get(col) == 'integer' and values.min() >= 1000000:
                    # Check if values are in a reasonable unix timestamp range
                    min_timestamp = datetime(1970, 1, 1).timestamp()
                    max_timestamp = datetime(2050, 1, 1).timestamp()
                    
                    if ((values >= min_timestamp) & (values <= max_timestamp)).mean() >= self.threshold:
                        # Try converting to datetime
                        try:
                            parsed = pd.to_datetime(values, unit='s', errors='coerce')
                            if parsed.notna().mean() >= self.threshold:
                                self.column_types[col] = 'datetime'
                                self.conversion_suggestions[col] = {
                                    'convert_to': 'datetime',
                                    'method': 'pd.to_datetime(unit="s")'
                                }
                        except (ValueError, OverflowError, TypeError):
                            pass
                
                # PART 2: Check for scientific notation in numeric columns
                # This is done before Excel date detection because scientific notation is more common
                if self.column_types.get(col) in ['float'] and 'datetime' not in self.column_types.get(col, ''):
                    # Convert to strings to check for scientific notation pattern
                    values_str = values.astype(str)
                    # Pattern for scientific notation (e.g., 1.23e-10, 4.56E+5)
                    sci_pattern = re.compile(r"^[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+$")
                    
                    # Check how many values match the scientific notation pattern
                    sci_matches = [bool(sci_pattern.match(val)) for val in values_str]
                    sci_ratio = sum(sci_matches) / len(values_str) if len(values_str) > 0 else 0
                    
                    if sci_ratio >= self.threshold:
                        # It's likely scientific notation
                        self.semantic_types[col] = 'scientific_notation'
                        
                        # Don't suggest conversion since this is already the right format,
                        # but add a note in the detection results
                        if col not in self.conversion_suggestions:
                            self.conversion_suggestions[col] = {
                                'convert_to': 'float',
                                'method': 'astype("float")',
                                'note': 'Contains scientific notation'
                            }
                
                # PART 3: Check for Excel dates (numerical representation of dates)
                # Excel dates are stored as days since 1899-12-30 (with some adjustments)
                # Typically in the range of ~25000-50000 for modern dates (2000-2050)
                if self.column_types.get(col) in ['integer', 'float'] and 'datetime' not in self.column_types.get(col, ''):
                    # Skip large values (likely not Excel dates)
                    if values.min() < 0 or values.max() > 50000:
                        continue
                    
                    # Excel dates are typically in this range for dates after 1970
                    excel_date_min = 25569  # 1970-01-01 in Excel date format
                    excel_date_max = 47482  # 2030-01-01 in Excel date format
                    
                    # Check if the values are within typical Excel date range
                    values_in_range = ((values >= excel_date_min) & (values <= excel_date_max))
                    ratio_in_range = values_in_range.mean()
                    
                    if ratio_in_range >= self.threshold:
                        # Try to convert a sample to dates to verify they're valid
                        try:
                            # Convert to Excel dates (origin='1899-12-30')
                            sample_dates = pd.to_datetime(values.head(min(20, len(values))), unit='D', origin='1899-12-30')
                            
                            # Check for realistic date distribution
                            years = sample_dates.dt.year
                            # Modern dates should have years after 1970 in most cases
                            modern_years_ratio = (years >= 1970).mean()
                            # Years should be reasonable (not in future by much)
                            reasonable_years_ratio = (years <= datetime.now().year + 5).mean()
                            
                            if modern_years_ratio >= 0.8 and reasonable_years_ratio >= 0.8:
                                # It's very likely an Excel date
                                self.column_types[col] = 'datetime'
                                self.semantic_types[col] = 'excel_date'
                                self.conversion_suggestions[col] = {
                                    'convert_to': 'datetime',
                                    'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'
                                }
                        except (ValueError, OverflowError, TypeError):
                            pass
                            
    def _detect_mixed_date_formats(self, values: pd.Series) -> List[str]:
        """
        Detect multiple date formats within a single column.
        
        Parameters
        ----------
        values : pd.Series
            Series of string values to check for date formats
            
        Returns
        -------
        List[str]
            List of detected date formats
        """
        if len(values) == 0:
            return []
            
        # Convert to strings if needed
        values = values.astype(str)
        
        # Dictionary to track formats and their match counts
        format_counts = {}
        
        # Try each format and count successful parses
        for date_format in self.date_formats:
            successful_formats = []
            
            # Test each value with the current format
            for val in values:
                try:
                    if pd.isna(val) or val.strip() == '':
                        continue
                        
                    # Attempt to parse with the current format
                    dt = datetime.strptime(val, date_format)
                    successful_formats.append(val)
                except ValueError:
                    continue
            
            # Calculate match ratio for this format
            match_ratio = len(successful_formats) / len(values.dropna())
            
            # If this format matches some values but not all, track it
            if match_ratio > 0.1 and match_ratio < self.threshold:
                format_counts[date_format] = len(successful_formats)
                
        # Also count how many parse with automatic detection
        try:
            parsed = pd.to_datetime(values, errors='coerce')
            auto_parse_count = parsed.notna().sum()
            
            # Check if auto-parsing works better than any single format
            if auto_parse_count > max(format_counts.values(), default=0):
                return ['auto']  # Auto-parsing is the best option
        except:
            pass
            
        # Sort formats by count in descending order
        sorted_formats = sorted(format_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return formats that have significant matches (at least 10% of values)
        return [fmt for fmt, count in sorted_formats if count / len(values.dropna()) >= 0.1]
    
    def _detect_categorical_numeric(self) -> None:
        """
        Detect categorical vs continuous numeric columns.
        """
        for col in self.df.columns:
            # Only analyze columns detected as numeric
            if self.column_types.get(col) in ['integer', 'float']:
                values = self.df[col].dropna()
                
                if len(values) == 0:
                    continue
                
                # Check number of unique values relative to total values
                n_unique = values.nunique()
                n_total = len(values)
                unique_ratio = n_unique / n_total
                
                # Check if values are integers even if stored as float
                all_integers = np.all(np.equal(np.mod(values, 1), 0))
                
                # Check if values are evenly distributed or clustered
                # (categorical values tend to be clustered)
                try:
                    # For smaller datasets, use all values
                    if n_total <= 5000:
                        value_counts = values.value_counts(normalize=True)
                        max_freq = value_counts.max()
                    else:
                        # For larger datasets, sample for efficiency
                        sample = values.sample(5000, random_state=42)
                        value_counts = sample.value_counts(normalize=True)
                        max_freq = value_counts.max()
                except:
                    max_freq = 0
                
                # Rule-based classification criteria
                # 1. Low cardinality relative to total (typically categorical)
                # 2. Small number of unique values (typically categorical)
                # 3. Integer values (more likely categorical)
                # 4. High frequency of most common value (typically categorical)
                is_likely_categorical = False
                
                if n_unique <= 20:
                    # Small number of distinct values suggests categorical
                    is_likely_categorical = True
                elif unique_ratio < 0.05:
                    # Low cardinality ratio suggests categorical
                    is_likely_categorical = True
                elif all_integers and n_unique <= 100 and max_freq > 0.01:
                    # Integer values with clustering suggests categorical
                    is_likely_categorical = True
                elif max_freq > 0.2:
                    # High concentration of values suggests categorical
                    is_likely_categorical = True
                
                # Update column type based on our findings
                if is_likely_categorical:
                    if self.column_types[col] == 'integer':
                        self.column_types[col] = 'categorical_integer'
                        if n_unique <= 20:  # Only suggest conversion for low cardinality
                            self.conversion_suggestions[col] = {
                                'convert_to': 'category',
                                'method': 'astype("category")'
                            }
                    else:  # float
                        if all_integers:
                            self.column_types[col] = 'categorical_integer'
                            if n_unique <= 20:  # Only suggest conversion for low cardinality
                                self.conversion_suggestions[col] = {
                                    'convert_to': 'category',
                                    'method': 'astype("category")'
                                }
                        else:
                            self.column_types[col] = 'categorical_float'
                else:
                    if self.column_types[col] == 'integer':
                        self.column_types[col] = 'continuous_integer'
                    else:  # float
                        self.column_types[col] = 'continuous_float'
    
    def _detect_semantic_types(self) -> None:
        """
        Detect semantic types for string columns (email, URL, etc.).
        """
        # First, check for ID-like columns based on name conventions
        for col in self.df.columns:
            # Skip non-string column names (e.g., integers)
            if not isinstance(col, str):
                continue
                
            col_lower = col.lower()
            
            # Check for ID-like column names
            if (col_lower.endswith('id') or col_lower.startswith('id') or 
                col_lower == 'key' or col_lower.endswith('_key') or
                col_lower == 'index' or col_lower.endswith('_index')):
                
                if self.column_types.get(col) in ['integer', 'categorical_integer', 'continuous_integer']:
                    self.semantic_types[col] = 'id'
                elif self.column_types.get(col) in ['string', 'categorical']:
                    # Check if it's a UUID pattern for string IDs
                    uuid_pattern = re.compile(self.patterns['uuid'])
                    sample = self.df[col].dropna().sample(
                        min(self.sample_size, len(self.df[col].dropna())),
                        random_state=42
                    ) if len(self.df[col].dropna()) > 0 else []
                    
                    if all(bool(uuid_pattern.match(str(x))) for x in sample if not pd.isna(x)):
                        self.semantic_types[col] = 'uuid'
                    else:
                        self.semantic_types[col] = 'id'
                        
            # Check for Australian postcode columns
            elif (col_lower in ['postcode', 'post_code', 'postal_code', 'postal', 'zip', 'zipcode']):
                # If it's an integer column, check if it's likely an Australian postcode that lost leading zeros
                if self.column_types.get(col) in ['integer', 'categorical_integer', 'continuous_integer']:
                    # Get values and check if they're in the valid AU postcode range (0-9999)
                    values = self.df[col].dropna()
                    if len(values) > 0:
                        min_val = values.min()
                        max_val = values.max()
                        if min_val >= 0 and max_val <= 9999:
                            # Check if it's specifically in Australian postcode ranges
                            in_au_range = False
                            for val in values.sample(min(self.sample_size, len(values)), random_state=42):
                                # Check if value is in valid AU postcode ranges
                                val_str = f"{val:04d}"  # Format with leading zeros
                                if (val >= 800 and val <= 899) or \
                                   (val >= 200 and val <= 299) or (val >= 2600 and val <= 2639) or \
                                   (val >= 1000 and val <= 2999) or \
                                   (val >= 3000 and val <= 3999) or \
                                   (val >= 4000 and val <= 4999) or \
                                   (val >= 5000 and val <= 5999) or \
                                   (val >= 6000 and val <= 6999) or \
                                   (val >= 7000 and val <= 7999):
                                    in_au_range = True
                                    break
                            
                            if in_au_range:
                                self.semantic_types[col] = 'au_postcode'
                                # Add suggestion to convert to string with zero-padding
                                self.conversion_suggestions[col] = {
                                    'convert_to': 'str_padded',
                                    'method': 'lambda x: f"{x:04d}" if pd.notna(x) else pd.NA'
                                }
        
        # Check string columns for common patterns
        for col in self.df.columns:
            if self.column_types.get(col) in ['string', 'categorical']:
                # Skip if already identified as an ID
                if col in self.semantic_types:
                    continue
                
                # Get a sample of non-null values
                sample = self.df[col].dropna()
                if len(sample) == 0:
                    continue
                
                if self.sample_size > 0 and len(sample) > self.sample_size:
                    sample = sample.sample(self.sample_size, random_state=42)
                
                # Convert all values to strings for pattern matching
                sample = sample.astype(str)
                
                # First check custom patterns (if any were provided during initialization)
                custom_patterns = {}
                built_in_patterns = {}
                
                # Built-in pattern names
                built_in_names = [
                    'email', 'url', 'ip_address', 'phone_number', 
                    'zip_code', 'credit_card', 'ssn', 'uuid', 'path', 
                    'currency', 'isbn', 'latitude', 'longitude',
                    'au_postcode', 'au_phone', 'au_mobile', 'au_abn', 
                    'au_acn', 'au_tfn'
                ]
                
                # Split patterns into custom and built-in
                for pattern_name, pattern_regex in self.patterns.items():
                    if pattern_name not in built_in_names:
                        custom_patterns[pattern_name] = pattern_regex
                    else:
                        built_in_patterns[pattern_name] = pattern_regex
                
                # Check custom patterns first (higher priority)
                pattern_matched = False
                for pattern_name, pattern_regex in custom_patterns.items():
                    pattern = re.compile(pattern_regex)
                    match_ratio = sum(1 for x in sample if bool(pattern.match(x))) / len(sample)
                    
                    if match_ratio >= self.threshold:
                        self.semantic_types[col] = pattern_name
                        pattern_matched = True
                        break
                
                # If no custom pattern matched, check built-in patterns
                if not pattern_matched:
                    for pattern_name, pattern_regex in built_in_patterns.items():
                        pattern = re.compile(pattern_regex)
                        match_ratio = sum(1 for x in sample if bool(pattern.match(x))) / len(sample)
                        
                        if match_ratio >= self.threshold:
                            self.semantic_types[col] = pattern_name
                            break
                
                # Special check for postal codes or phone numbers with varied formats
                if col not in self.semantic_types:
                    col_lower = col.lower()
                    if 'zip' in col_lower or 'postal' in col_lower or 'postcode' in col_lower:
                        # More flexible zip code check
                        zip_like = sum(1 for x in sample if len(x) <= 10 and re.search(r'[0-9]{4,5}', x)) / len(sample)
                        if zip_like >= self.threshold:
                            self.semantic_types[col] = 'zip_code'
                    
                    elif 'phone' in col_lower or 'tel' in col_lower or 'mobile' in col_lower:
                        # More flexible phone number check
                        phone_like = sum(1 for x in sample if re.search(r'[0-9]{3}.*[0-9]{3}.*[0-9]{4}', x)) / len(sample)
                        if phone_like >= self.threshold:
                            self.semantic_types[col] = 'phone_number'
                    
                    elif 'address' in col_lower or 'street' in col_lower or 'addr' in col_lower:
                        self.semantic_types[col] = 'address'
                    
                    elif 'name' in col_lower or 'first' in col_lower or 'last' in col_lower:
                        self.semantic_types[col] = 'name'
    
    def _generate_conversion_suggestions(self) -> None:
        """
        Generate suggestions for optimizing column data types.
        """
        for col in self.df.columns:
            # Skip if we already have a suggestion
            if col in self.conversion_suggestions:
                continue
            
            # Get current type info
            dtype = self.df[col].dtype
            logical_type = self.column_types.get(col, 'unknown')
            
            # Suggest appropriate conversions based on current type
            if dtype == 'object':
                if logical_type == 'string':
                    # For string columns with low cardinality, suggest category
                    n_unique = self.df[col].nunique()
                    n_total = len(self.df[col])
                    if n_unique > 0 and n_unique / n_total < 0.5:
                        self.conversion_suggestions[col] = {
                            'convert_to': 'category',
                            'method': 'astype("category")'
                        }
            
            elif pd.api.types.is_integer_dtype(dtype):
                # For large integer values, check if they need the full int64 range
                try:
                    col_min, col_max = self.df[col].min(), self.df[col].max()
                    
                    # Suggest smaller integer types if possible
                    if col_min >= -128 and col_max <= 127:
                        self.conversion_suggestions[col] = {
                            'convert_to': 'int8',
                            'method': 'astype("int8")'
                        }
                    elif col_min >= -32768 and col_max <= 32767:
                        self.conversion_suggestions[col] = {
                            'convert_to': 'int16',
                            'method': 'astype("int16")'
                        }
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        self.conversion_suggestions[col] = {
                            'convert_to': 'int32',
                            'method': 'astype("int32")'
                        }
                except (TypeError, ValueError):
                    pass
            
            elif pd.api.types.is_float_dtype(dtype) and dtype == 'float64':
                # Suggest float32 for float64 columns if they don't need high precision
                self.conversion_suggestions[col] = {
                    'convert_to': 'float32',
                    'method': 'astype("float32")'
                }
    
    def convert_types(self, columns: Optional[List[str]] = None, use_pyarrow: bool = True) -> pd.DataFrame:
        """
        Convert column types based on suggestions.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            List of columns to convert. If None, convert all columns with suggestions.
        use_pyarrow : bool, default=True
            Whether to use PyArrow for faster conversions when possible
        
        Returns
        -------
        pd.DataFrame
            DataFrame with converted types
        """
        if not hasattr(self, 'conversion_suggestions') or not self.conversion_suggestions:
            # Run detection if not already done
            self.detect_all_types()
        
        # Check if we can use PyArrow for faster batch conversions
        if use_pyarrow and len(self.df) > 10000:
            try:
                import pyarrow as pa
                
                # Group columns by conversion type for batch processing
                batch_conversions = {}
                for col in self.conversion_suggestions:
                    if columns is not None and col not in columns:
                        continue
                    suggestion = self.conversion_suggestions[col]
                    target_type = suggestion['convert_to']
                    
                    # Group by target type
                    if target_type not in batch_conversions:
                        batch_conversions[target_type] = []
                    batch_conversions[target_type].append((col, suggestion))
                
                # Check if we have enough columns for batch processing to be worth it
                if any(len(cols) > 1 for cols in batch_conversions.values()):
                    logger.info("Using PyArrow for batch type conversions")
                    
                    # Convert to arrow table for batch processing
                    arrow_table = pa.Table.from_pandas(self.df)
                    schema = arrow_table.schema
                    
                    # Process each type in batch
                    for target_type, col_suggestions in batch_conversions.items():
                        if target_type in ('int8', 'int16', 'int32', 'float32'):
                            # These are simple type casts that can be done efficiently with PyArrow
                            for col, suggestion in col_suggestions:
                                # Get corresponding arrow type
                                if target_type == 'int8':
                                    arrow_type = pa.int8()
                                elif target_type == 'int16':
                                    arrow_type = pa.int16()
                                elif target_type == 'int32':
                                    arrow_type = pa.int32()
                                elif target_type == 'float32':
                                    arrow_type = pa.float32()
                                    
                                # Update schema with new type
                                field_index = schema.get_field_index(col)
                                if field_index >= 0:  # Column exists
                                    fields = list(schema)
                                    fields[field_index] = pa.field(col, arrow_type)
                                    schema = pa.schema(fields)
                    
                    # Create new table with updated schema and convert back to pandas
                    try:
                        new_table = arrow_table.cast(schema)
                        # Convert back to pandas (will use original schema for columns not in the batch)
                        df_converted = new_table.to_pandas()
                        
                        # We've converted the simple types, now process any special conversions
                        remaining_columns = []
                        for target_type, col_suggestions in batch_conversions.items():
                            if target_type not in ('int8', 'int16', 'int32', 'float32'):
                                # These need special handling
                                for col, suggestion in col_suggestions:
                                    remaining_columns.append(col)
                        
                        # Skip PyArrow and fall back to normal processing for the remaining columns
                        if remaining_columns:
                            return self._convert_specific_columns(df_converted, remaining_columns)
                        
                        return df_converted
                    except Exception as e:
                        logger.warning(f"PyArrow batch conversion failed: {str(e)}")
                        # Fall back to standard processing
            except (ImportError, Exception) as e:
                logger.warning(f"PyArrow batch conversion failed, falling back to standard method: {str(e)}")
        
        # Make a copy of the dataframe
        df_converted = self.df.copy()
        
        # Get the list of columns to convert
        if columns is None:
            columns_to_convert = list(self.conversion_suggestions.keys())
        else:
            columns_to_convert = [col for col in columns if col in self.conversion_suggestions]
        
        # Apply conversions
        for col in columns_to_convert:
            suggestion = self.conversion_suggestions[col]
            target_type = suggestion['convert_to']
            
            try:
                if target_type == 'datetime':
                    if 'unit=' in suggestion['method'] and 'origin=' in suggestion['method']:
                        # Excel date conversion
                        # First convert the column to numeric, forcing non-numeric values to NaN
                        numeric_values = pd.to_numeric(df_converted[col], errors='coerce')
                        
                        # Create a mask for finite values to avoid overflow errors
                        mask = np.isfinite(numeric_values)
                        
                        # Initialize result Series with NaT values
                        result = pd.Series([pd.NaT] * len(df_converted), index=df_converted.index)
                        
                        # Extract parameters from the method string
                        unit_match = re.search(r'unit="([^"]+)"', suggestion['method'])
                        origin_match = re.search(r'origin="([^"]+)"', suggestion['method'])
                        
                        unit = unit_match.group(1) if unit_match else 'D'
                        origin = origin_match.group(1) if origin_match else '1899-12-30'
                        
                        # Apply the conversion only to finite values, converting them one by one to avoid issues
                        if mask.any():
                            try:
                                # Process each finite value individually to avoid issues with mixed types
                                result_dates = []
                                finite_indices = mask[mask].index
                                
                                for idx in finite_indices:
                                    try:
                                        val = numeric_values.loc[idx]
                                        # Convert this single value
                                        date_val = pd.to_datetime(val, unit=unit, origin=origin)
                                        # Store in our results array
                                        result.loc[idx] = date_val
                                    except Exception as e:
                                        # If a specific value fails, leave it as NaT
                                        logger.warning(f"Value {val} at index {idx} could not be converted: {str(e)}")
                                
                                # Set the converted column
                                df_converted[col] = result
                            except Exception as e:
                                logger.warning(f"Error during Excel date conversion for column '{col}': {str(e)}")
                                # Keep the column as is
                        else:
                            # No finite values to convert, keep the column as is
                            logger.warning(f"No finite values to convert in column '{col}'")
                            df_converted[col] = result
                    elif 'unit=' in suggestion['method']:
                        # Unix timestamp conversion - use same approach as Excel dates to handle NaN and infinity values
                        # First convert the column to numeric, forcing non-numeric values to NaN
                        numeric_values = pd.to_numeric(df_converted[col], errors='coerce')
                        
                        # Create a mask for finite values to avoid overflow errors
                        mask = np.isfinite(numeric_values)
                        
                        # Initialize result Series with NaT values
                        result = pd.Series([pd.NaT] * len(df_converted), index=df_converted.index)
                        
                        # Apply the conversion only to finite values, converting them one by one to avoid issues
                        if mask.any():
                            try:
                                # Process each finite value individually to avoid issues with mixed types
                                result_dates = []
                                finite_indices = mask[mask].index
                                
                                for idx in finite_indices:
                                    try:
                                        val = numeric_values.loc[idx]
                                        # Convert this single value
                                        date_val = pd.to_datetime(val, unit='s')
                                        # Store in our results array
                                        result.loc[idx] = date_val
                                    except Exception as e:
                                        # If a specific value fails, leave it as NaT
                                        logger.warning(f"Value {val} at index {idx} could not be converted: {str(e)}")
                                
                                # Set the converted column
                                df_converted[col] = result
                            except Exception as e:
                                logger.warning(f"Error during Unix timestamp conversion for column '{col}': {str(e)}")
                                # Keep the column as is
                        else:
                            # No finite values to convert, keep the column as is
                            logger.warning(f"No finite values to convert in column '{col}'")
                            df_converted[col] = result
                    elif 'format=' in suggestion['method']:
                        # String date with specific format
                        format_match = re.search(r'format="([^"]+)"', suggestion['method'])
                        if format_match:
                            date_format = format_match.group(1)
                            df_converted[col] = pd.to_datetime(df_converted[col], format=date_format, errors='coerce')
                        else:
                            df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                    elif col in self.semantic_types and self.semantic_types[col] == 'mixed_date_formats':
                        # Handle mixed date formats - first try automatic parsing
                        df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                        
                        # If automatic parsing left any NaT values and we have detected formats,
                        # try each detected format for the remaining values
                        if 'detected_formats' in suggestion and df_converted[col].isna().any():
                            # Create a mask of values that didn't parse
                            mask = df_converted[col].isna()
                            
                            # Try each detected format on the unparsed values
                            for fmt in suggestion['detected_formats']:
                                if fmt == 'auto':
                                    continue  # Skip 'auto' since we already tried it
                                
                                # Apply this format only to values that haven't been parsed yet
                                try:
                                    parsed = pd.to_datetime(df_converted.loc[mask, col], format=fmt, errors='coerce')
                                    
                                    # Update only the values that were successfully parsed with this format
                                    newly_parsed = ~parsed.isna()
                                    if newly_parsed.any():
                                        df_converted.loc[mask & newly_parsed.values, col] = parsed.loc[newly_parsed]
                                    
                                    # Update the mask to exclude newly parsed values
                                    mask = df_converted[col].isna()
                                except:
                                    continue
                    elif col in self.semantic_types and self.semantic_types[col] == 'month_year_format':
                        # Use our specialized month-year format converter
                        if suggestion['method'] == 'convert_month_year_format':
                            # Convert to string if needed to ensure format detection works properly
                            str_vals = df_converted[col].astype(str)
                            df_converted[col] = convert_month_year_format(str_vals)
                            logger.info(f"Converted column '{col}' using month-year format converter")
                        else:
                            # Fallback to standard approach if method is different
                            # First convert to string if needed
                            str_vals = df_converted[col].astype(str)
                            
                            # First try automatic parsing
                            df_converted[col] = pd.to_datetime(str_vals, errors='coerce')
                            
                            # If some values didn't parse and we have detected formats, try them
                            if 'detected_formats' in suggestion and df_converted[col].isna().any():
                                mask = df_converted[col].isna()
                                
                                for fmt in suggestion['detected_formats']:
                                    try:
                                        # For month-year formats, we need to handle day=1 appropriately
                                        parsed = pd.to_datetime(str_vals[mask], format=fmt, errors='coerce')
                                        
                                        # Update only the values that were successfully parsed
                                        newly_parsed = ~parsed.isna()
                                        if newly_parsed.any():
                                            df_converted.loc[mask & newly_parsed.values, col] = parsed.loc[newly_parsed]
                                        
                                        # Update the mask
                                        mask = df_converted[col].isna()
                                        
                                        # Break if all values are parsed
                                        if not mask.any():
                                            break
                                    except:
                                        continue
                    else:
                        # Default conversion
                        df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                elif target_type == 'str_padded':
                    # For Australian postcodes or similar that need zero-padding
                    if 'lambda' in suggestion['method']:
                        # Execute the lambda expression provided in the method
                        # Note: Using eval is generally not recommended, but in this controlled context for
                        # simple lambda expressions that we've created ourselves, it's acceptable
                        lambda_expr = eval(suggestion['method'])
                        df_converted[col] = df_converted[col].apply(lambda_expr)
                    else:
                        # Default padding to 4 digits for postcodes
                        df_converted[col] = df_converted[col].apply(
                            lambda x: f"{int(x):04d}" if pd.notna(x) and isinstance(x, (int, float)) else x
                        )
                else:
                    df_converted[col] = df_converted[col].astype(target_type)
                
                logger.info(f"Converted column '{col}' to {target_type}")
            except Exception as e:
                logger.warning(f"Failed to convert column '{col}': {str(e)}")
        
        return df_converted
    
    def _convert_specific_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Helper method to convert specific columns using the standard conversion logic.
        Used by the PyArrow-optimized convert_types method to handle special conversions.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with some columns already converted by PyArrow
        columns : List[str]
            The list of columns that need special conversion
            
        Returns
        -------
        pd.DataFrame
            The dataframe with all conversions applied
        """
        df_converted = df.copy()
        
        # Apply conversions to the specified columns
        for col in columns:
            if col in self.conversion_suggestions:
                suggestion = self.conversion_suggestions[col]
                target_type = suggestion['convert_to']
                
                try:
                    if target_type == 'datetime':
                        if 'unit=' in suggestion['method'] and 'origin=' in suggestion['method']:
                            # Excel date conversion
                            # First convert the column to numeric, forcing non-numeric values to NaN
                            numeric_values = pd.to_numeric(df_converted[col], errors='coerce')
                            
                            # Create a mask for finite values to avoid overflow errors
                            mask = np.isfinite(numeric_values)
                            
                            # Initialize result Series with NaT values
                            result = pd.Series([pd.NaT] * len(df_converted), index=df_converted.index)
                            
                            # Extract parameters from the method string
                            unit_match = re.search(r'unit="([^"]+)"', suggestion['method'])
                            origin_match = re.search(r'origin="([^"]+)"', suggestion['method'])
                            
                            unit = unit_match.group(1) if unit_match else 'D'
                            origin = origin_match.group(1) if origin_match else '1899-12-30'
                            
                            # Apply the conversion only to finite values, converting them one by one to avoid issues
                            if mask.any():
                                try:
                                    # Process each finite value individually to avoid issues with mixed types
                                    result_dates = []
                                    finite_indices = mask[mask].index
                                    
                                    for idx in finite_indices:
                                        try:
                                            val = numeric_values.loc[idx]
                                            # Convert this single value
                                            date_val = pd.to_datetime(val, unit=unit, origin=origin)
                                            # Store in our results array
                                            result.loc[idx] = date_val
                                        except Exception as e:
                                            # If a specific value fails, leave it as NaT
                                            logger.warning(f"Value {val} at index {idx} could not be converted: {str(e)}")
                                    
                                    # Set the converted column
                                    df_converted[col] = result
                                except Exception as e:
                                    logger.warning(f"Error during Excel date conversion for column '{col}': {str(e)}")
                                    # Keep the column as is
                            else:
                                # No finite values to convert, keep the column as is
                                logger.warning(f"No finite values to convert in column '{col}'")
                                df_converted[col] = result
                        elif 'unit=' in suggestion['method']:
                            # Unix timestamp conversion - use same approach as Excel dates to handle NaN and infinity values
                            # First convert the column to numeric, forcing non-numeric values to NaN
                            numeric_values = pd.to_numeric(df_converted[col], errors='coerce')
                            
                            # Create a mask for finite values to avoid overflow errors
                            mask = np.isfinite(numeric_values)
                            
                            # Initialize result Series with NaT values
                            result = pd.Series([pd.NaT] * len(df_converted), index=df_converted.index)
                            
                            # Apply the conversion only to finite values, converting them one by one to avoid issues
                            if mask.any():
                                try:
                                    # Process each finite value individually to avoid issues with mixed types
                                    result_dates = []
                                    finite_indices = mask[mask].index
                                    
                                    for idx in finite_indices:
                                        try:
                                            val = numeric_values.loc[idx]
                                            # Convert this single value
                                            date_val = pd.to_datetime(val, unit='s')
                                            # Store in our results array
                                            result.loc[idx] = date_val
                                        except Exception as e:
                                            # If a specific value fails, leave it as NaT
                                            logger.warning(f"Value {val} at index {idx} could not be converted: {str(e)}")
                                    
                                    # Set the converted column
                                    df_converted[col] = result
                                except Exception as e:
                                    logger.warning(f"Error during Unix timestamp conversion for column '{col}': {str(e)}")
                                    # Keep the column as is
                            else:
                                # No finite values to convert, keep the column as is
                                logger.warning(f"No finite values to convert in column '{col}'")
                                df_converted[col] = result
                        elif 'format=' in suggestion['method']:
                            # String date with specific format
                            format_match = re.search(r'format="([^"]+)"', suggestion['method'])
                            if format_match:
                                date_format = format_match.group(1)
                                df_converted[col] = pd.to_datetime(df_converted[col], format=date_format, errors='coerce')
                            else:
                                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                        elif col in self.semantic_types and self.semantic_types[col] == 'mixed_date_formats':
                            # Handle mixed date formats - first try automatic parsing
                            df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                            
                            # If automatic parsing left any NaT values and we have detected formats,
                            # try each detected format for the remaining values
                            if 'detected_formats' in suggestion and df_converted[col].isna().any():
                                # Create a mask of values that didn't parse
                                mask = df_converted[col].isna()
                                
                                # Try each detected format on the unparsed values
                                for fmt in suggestion['detected_formats']:
                                    if fmt == 'auto':
                                        continue  # Skip 'auto' since we already tried it
                                    
                                    # Apply this format only to values that haven't been parsed yet
                                    try:
                                        parsed = pd.to_datetime(df_converted.loc[mask, col], format=fmt, errors='coerce')
                                        
                                        # Update only the values that were successfully parsed with this format
                                        newly_parsed = ~parsed.isna()
                                        if newly_parsed.any():
                                            df_converted.loc[mask & newly_parsed.values, col] = parsed.loc[newly_parsed]
                                        
                                        # Update the mask to exclude newly parsed values
                                        mask = df_converted[col].isna()
                                    except:
                                        continue
                        elif col in self.semantic_types and self.semantic_types[col] == 'month_year_format':
                            # Use our specialized month-year format converter
                            if suggestion['method'] == 'convert_month_year_format':
                                # Convert to string if needed to ensure format detection works properly
                                str_vals = df_converted[col].astype(str)
                                df_converted[col] = convert_month_year_format(str_vals)
                                logger.info(f"Converted column '{col}' using month-year format converter")
                            else:
                                # Fallback to standard approach if method is different
                                # First convert to string if needed
                                str_vals = df_converted[col].astype(str)
                                
                                # First try automatic parsing
                                df_converted[col] = pd.to_datetime(str_vals, errors='coerce')
                                
                                # If some values didn't parse and we have detected formats, try them
                                if 'detected_formats' in suggestion and df_converted[col].isna().any():
                                    mask = df_converted[col].isna()
                                    
                                    for fmt in suggestion['detected_formats']:
                                        try:
                                            # For month-year formats, we need to handle day=1 appropriately
                                            parsed = pd.to_datetime(str_vals[mask], format=fmt, errors='coerce')
                                            
                                            # Update only the values that were successfully parsed
                                            newly_parsed = ~parsed.isna()
                                            if newly_parsed.any():
                                                df_converted.loc[mask & newly_parsed.values, col] = parsed.loc[newly_parsed]
                                            
                                            # Update the mask
                                            mask = df_converted[col].isna()
                                            
                                            # Break if all values are parsed
                                            if not mask.any():
                                                break
                                        except:
                                            continue
                        else:
                            # Default conversion
                            df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                    elif target_type == 'str_padded':
                        # For Australian postcodes or similar that need zero-padding
                        if 'lambda' in suggestion['method']:
                            # Execute the lambda expression provided in the method
                            # Note: Using eval is generally not recommended, but in this controlled context for
                            # simple lambda expressions that we've created ourselves, it's acceptable
                            lambda_expr = eval(suggestion['method'])
                            df_converted[col] = df_converted[col].apply(lambda_expr)
                        else:
                            # Default padding to 4 digits for postcodes
                            df_converted[col] = df_converted[col].apply(
                                lambda x: f"{int(x):04d}" if pd.notna(x) and isinstance(x, (int, float)) else x
                            )
                    else:
                        df_converted[col] = df_converted[col].astype(target_type)
                    
                    logger.info(f"Converted column '{col}' to {target_type}")
                except Exception as e:
                    logger.warning(f"Failed to convert column '{col}': {str(e)}")
        
        return df_converted
        
    def get_column_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a detailed report about each column.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with detailed column information
        """
        if not self.column_types:
            # Run detection if not already done
            self.detect_all_types()
        
        report = {}
        
        for col in self.df.columns:
            # Get basic info
            report[col] = {
                'storage_type': str(self.df[col].dtype),
                'logical_type': self.column_types.get(col, 'unknown'),
                'null_count': int(self.df[col].isna().sum()),
                'null_percentage': float(self.df[col].isna().mean() * 100),
                'unique_count': int(self.df[col].nunique()),
            }
            
            # Add semantic type if available
            if col in self.semantic_types:
                report[col]['semantic_type'] = self.semantic_types[col]
            
            # Add conversion suggestion if available
            if col in self.conversion_suggestions:
                report[col]['suggested_conversion'] = self.conversion_suggestions[col]
            
            # Add additional type-specific metrics
            try:
                if pd.api.types.is_numeric_dtype(self.df[col].dtype):
                    report[col]['min'] = float(self.df[col].min())
                    report[col]['max'] = float(self.df[col].max())
                    report[col]['mean'] = float(self.df[col].mean())
                    report[col]['std'] = float(self.df[col].std())
                
                elif self.column_types.get(col) == 'datetime':
                    if pd.api.types.is_datetime64_dtype(self.df[col].dtype):
                        report[col]['min'] = str(self.df[col].min())
                        report[col]['max'] = str(self.df[col].max())
                    else:
                        # Try to parse to get date range
                        try:
                            parsed = pd.to_datetime(self.df[col], errors='coerce')
                            report[col]['min'] = str(parsed.min())
                            report[col]['max'] = str(parsed.max())
                        except:
                            pass
                
                elif report[col]['logical_type'] in ['string', 'categorical']:
                    # Get average string length
                    str_len = self.df[col].astype(str).str.len()
                    report[col]['avg_length'] = float(str_len.mean())
                    report[col]['max_length'] = int(str_len.max())
                    
                    # Add top value counts for categorical
                    if report[col]['unique_count'] <= 20:
                        value_counts = self.df[col].value_counts().head(10).to_dict()
                        report[col]['top_values'] = value_counts
            except:
                # Skip if any calculations fail
                pass
        
        return report
        
    def display_detection_report(self, include_stats=True):
        """
        Display a styled DataFrame with detection results, suitable for Jupyter notebooks.
        
        This method creates a visually enhanced report of the data type detection
        results, with color highlighting for different data types and semantic information.
        
        Parameters
        ----------
        include_stats : bool, default=True
            Whether to include detailed statistics for each column
            
        Returns
        -------
        pd.DataFrame
            A styled pandas DataFrame that will render nicely in Jupyter notebooks
            
        Notes
        -----
        This method uses pandas' DataFrame styling capabilities, which are
        specifically designed for Jupyter notebook display. The output may
        not render correctly in other environments.
        
        Examples
        --------
        >>> detector = DataTypeDetector(df)
        >>> detector.detect_all_types()
        >>> detector.display_detection_report()
        """
        # Make sure we have detection results
        if not self.column_types:
            self.detect_all_types()
            
        # Get the detailed report
        report = self.get_column_report()
        
        # Create a DataFrame for display - we'll use different subsets of data
        # Basic info
        basic_data = []
        for col, info in report.items():
            # Determine semantic type info 
            semantic = info.get('semantic_type', '')
            
            # Create suggestion info
            if 'suggested_conversion' in info:
                suggestion = info['suggested_conversion'].get('convert_to', '')
                if 'note' in info['suggested_conversion']:
                    suggestion += f" ({info['suggested_conversion']['note']})"
            else:
                suggestion = ''
                
            # Create basic info row    
            row = {
                'Column': col,
                'Storage Type': info['storage_type'],
                'Logical Type': info['logical_type'],
                'Semantic Type': semantic,
                'Null Count': info['null_count'],
                'Null %': f"{info['null_percentage']:.1f}%",
                'Unique Values': info['unique_count'],
                'Suggested Conversion': suggestion
            }
            
            # Add statistics if requested
            if include_stats:
                # Add numeric stats if available
                if 'min' in info:
                    if info['logical_type'] == 'datetime':
                        row['Range'] = f"{info['min']} to {info['max']}"
                    else:  
                        row['Min'] = f"{info['min']:.2f}"
                        row['Max'] = f"{info['max']:.2f}"
                        row['Mean'] = f"{info['mean']:.2f}"
                        row['Std Dev'] = f"{info['std']:.2f}"
                        
                # Add string/categorical stats if available
                if 'avg_length' in info:
                    row['Avg Length'] = f"{info['avg_length']:.1f}"
                    row['Max Length'] = info['max_length']
                    
            basic_data.append(row)
            
        # Create DataFrame
        report_df = pd.DataFrame(basic_data)
        
        # Apply styling
        try:
            # Function to apply colors based on data type
            def style_cells(val, column):
                # Color coding by type
                if column == 'Logical Type':
                    if 'datetime' in str(val).lower():
                        return 'background-color: #BBDEFB'  # Light blue for dates
                    elif 'categorical' in str(val).lower():
                        return 'background-color: #C8E6C9'  # Light green for categorical
                    elif 'continuous' in str(val).lower() or 'float' in str(val).lower():
                        return 'background-color: #FFF9C4'  # Light yellow for numeric
                    elif 'string' in str(val).lower():
                        return 'background-color: #F8BBD0'  # Light pink for strings
                    else:
                        return ''
                        
                # Color coding for semantic types
                elif column == 'Semantic Type':
                    if 'date' in str(val).lower() or 'month_year' in str(val).lower():
                        return 'background-color: #BBDEFB'  # Light blue for dates
                    elif 'id' in str(val).lower() or 'uuid' in str(val).lower():
                        return 'background-color: #D1C4E9'  # Light purple for IDs
                    elif val:  # Any other semantic type that's non-empty
                        return 'background-color: #E1BEE7'  # Light purple for other semantic types
                    else:
                        return ''
                
                # Color coding for null percentage
                elif column == 'Null %':
                    if val == '0.0%':
                        return 'color: green'
                    null_value = float(val.replace('%', ''))
                    if null_value > 50:
                        return 'color: red; font-weight: bold'
                    elif null_value > 20:
                        return 'color: orange'
                    else:
                        return ''
                        
                # Color coding for suggested conversions
                elif column == 'Suggested Conversion':
                    if val:
                        return 'background-color: #FFE0B2'  # Light orange for suggestions
                    else:
                        return ''
                        
                return ''
            
            # Apply style to DataFrame - using newer 'map' method instead of deprecated 'applymap'
            styled_df = report_df.style.map(style_cells, subset=pd.IndexSlice[:, ['Logical Type', 'Semantic Type', 'Null %', 'Suggested Conversion']])
            
            # Format the table
            styled_df = styled_df.set_caption('Data Type Detection Report')
            
            # Add a title
            display_text = f"<h3>DataTypeDetector Analysis for DataFrame with {len(self.df.columns)} columns and {len(self.df)} rows</h3>"
            
            # Display the styled DataFrame in Jupyter
            try:
                from IPython.display import display, HTML
                display(HTML(display_text))
                return styled_df
            except ImportError:
                # If not in Jupyter/IPython environment, return the styled DataFrame
                return styled_df
                
        except Exception as e:
            logger.warning(f"Error styling detection report: {str(e)}")
            # Fall back to returning the unstyled DataFrame
            return report_df


def detect_column_types(
    df: pd.DataFrame,
    sample_size: int = 1000,
    threshold: float = 0.9,
    detect_semantic_types: bool = True,
    categorize_numeric: bool = True,
    custom_patterns: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Detect and categorize column types in a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to analyze
    sample_size : int, default=1000
        Number of values to sample when detecting patterns
    threshold : float, default=0.9
        Minimum ratio of matching values to consider a pattern valid
    detect_semantic_types : bool, default=True
        Whether to detect semantic types like IDs, codes, etc.
    categorize_numeric : bool, default=True
        Whether to distinguish between categorical and continuous numeric columns
    custom_patterns : Optional[Dict[str, str]], default=None
        Dictionary of custom regex patterns to use for semantic type detection
        in the format {'type_name': 'regex_pattern'}
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with column type information
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'id': range(100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100),
    ...     'number': np.random.randn(100),
    ...     'int_category': np.random.choice([1, 2, 3, 4, 5], 100),
    ...     'date_str': pd.date_range('2020-01-01', periods=100).astype(str),
    ...     'timestamp': pd.date_range('2020-01-01', periods=100).astype(int) // 10**9
    ... })
    >>> types = detect_column_types(df)
    >>> types['id']['logical_type']
    'categorical_integer'
    >>> types['id']['semantic_type']
    'id'
    >>> types['date_str']['logical_type']
    'datetime'
    
    # Using custom patterns
    >>> custom = {'product_code': r'^[A-Z]{2}-[0-9]{4}$'}
    >>> df['product'] = ['AB-1234', 'CD-5678', 'EF-9012']
    >>> types = detect_column_types(df, custom_patterns=custom)
    >>> types['product']['semantic_type']
    'product_code'
    """
    detector = DataTypeDetector(
        df,
        sample_size=sample_size,
        threshold=threshold,
        detect_semantic_types=detect_semantic_types,
        categorize_numeric=categorize_numeric,
        custom_patterns=custom_patterns,
    )
    return detector.detect_all_types()


def convert_month_year_format(values: Union[pd.Series, List, np.ndarray]) -> pd.Series:
    """
    Convert various month-year formats to datetime objects.
    
    This function handles multiple month-year formats like:
    - 'Aug-24', 'Sep-2024' (Month abbreviation with hyphen and year)
    - 'October-23', 'November-2023' (Full month with hyphen and year)
    - 'Dec 24' (Month abbreviation with space and year)
    - '01-25', '02/25', '03.25' (Numeric month with various separators and year)
    
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
    # Convert to pandas Series if not already
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    # Initialize result series with NaT
    result = pd.Series([pd.NaT] * len(values), index=values.index)
    
    # Define format patterns and try each one
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
        # Numeric month with hyphen and 2-digit year (01-25)
        ('%m-%y', r'^\d{1,2}-\d{2}$'),
        # Numeric month with slash and 2-digit year (02/25)
        ('%m/%y', r'^\d{1,2}/\d{2}$'),
        # Numeric month with dot and 2-digit year (03.25)
        ('%m.%y', r'^\d{1,2}\.\d{2}$'),
    ]
    
    # Try each format pattern
    for fmt, pattern in format_patterns:
        mask = values.astype(str).str.match(pattern, case=False)
        if mask.any():
            try:
                parsed = pd.to_datetime(values[mask], format=fmt, errors='coerce')
                result[mask] = parsed
            except:
                pass
    
    return result


def optimize_dataframe_types(
    df: pd.DataFrame,
    sample_size: int = 1000,
    threshold: float = 0.9,
    columns: Optional[List[str]] = None,
    custom_patterns: Optional[Dict[str, str]] = None,
    use_pyarrow: bool = True,
) -> pd.DataFrame:
    """
    Optimize a dataframe's column types based on content analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to optimize
    sample_size : int, default=1000
        Number of values to sample when detecting patterns
    threshold : float, default=0.9
        Minimum ratio of matching values to consider a pattern valid
    columns : Optional[List[str]], default=None
        List of columns to optimize. If None, optimize all columns.
    custom_patterns : Optional[Dict[str, str]], default=None
        Dictionary of custom regex patterns to use for semantic type detection
        in the format {'type_name': 'regex_pattern'}
    use_pyarrow : bool, default=True
        Whether to use PyArrow for faster type optimizations when possible.
        This can significantly improve performance for large dataframes.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized types
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'id': range(100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100),
    ...     'number': np.random.randn(100),
    ...     'int_category': np.random.choice([1, 2, 3, 4, 5], 100),
    ...     'date_str': pd.date_range('2020-01-01', periods=100).astype(str),
    ...     'timestamp': pd.date_range('2020-01-01', periods=100).astype(int) // 10**9
    ... })
    >>> optimized_df = optimize_dataframe_types(df)
    >>> # Categories are converted to 'category' type
    >>> str(optimized_df['category'].dtype).startswith('category')
    True
    >>> # Datetime strings are parsed
    >>> pd.api.types.is_datetime64_dtype(optimized_df['date_str'].dtype)
    True
    """
    try:
        # Fast path using PyArrow for initial type optimization if requested
        # This will handle simple type conversions without the need for semantic analysis
        if use_pyarrow:
            try:
                import pyarrow as pa
                
                # Start with a PyArrow-based optimization for basic types
                # This handles numeric, string, and basic datetime conversions efficiently
                logger.info("Using PyArrow for initial type optimization")
                
                # Convert to arrow table
                arrow_table = pa.Table.from_pandas(df)
                
                # Check if we need to do specialized conversions (semantic types, excel dates, etc.)
                if custom_patterns or any(col.lower().endswith(('date', 'time', 'dt')) for col in df.columns):
                    # Still need DataTypeDetector for specialized conversions
                    # But we can start with the PyArrow optimized version
                    optimized_base_df = arrow_table.to_pandas()
                    detector = DataTypeDetector(
                        optimized_base_df,
                        sample_size=sample_size,
                        threshold=threshold,
                        custom_patterns=custom_patterns,
                    )
                    detector.detect_all_types()
                    return detector.convert_types(columns=columns)
                else:
                    # Simple case, just return the PyArrow optimized dataframe
                    return arrow_table.to_pandas()
            
            except (ImportError, Exception) as e:
                logger.warning(f"PyArrow optimization failed, falling back to standard method: {str(e)}")
    except Exception as e:
        logger.warning(f"Error in fast path optimization, falling back to standard method: {str(e)}")
        
    # Fall back to standard detector-based optimization
    detector = DataTypeDetector(
        df,
        sample_size=sample_size,
        threshold=threshold,
        custom_patterns=custom_patterns,
    )
    detector.detect_all_types()
    return detector.convert_types(columns=columns)