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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter common pandas warnings about date parsing
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually")


class DataTypeDetector:
    """
    Advanced data type detection and inference.
    
    This class provides methods to detect logical data types beyond the
    basic storage types, including IDs, zip codes, phone numbers, addresses,
    and distinguishing between categorical and continuous numeric features.
    
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
            
            # Other formats
            "%Y%m%d", "%Y%m%d%H%M%S"
        ]
    
    def detect_all_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect and categorize all column types in the dataframe.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with column type information
        """
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
                # Check if column contains valid datetime strings
                try:
                    # Get sample of values for efficiency
                    values = self.df[col].dropna()
                    if self.sample_size > 0 and len(values) > self.sample_size:
                        values = values.sample(self.sample_size, random_state=42)
                    
                    # Try pandas automatic parsing
                    parsed = pd.to_datetime(values, errors='coerce')
                    if parsed.notna().mean() >= self.threshold:
                        self.column_types[col] = 'datetime'
                        self.conversion_suggestions[col] = {
                            'convert_to': 'datetime',
                            'method': 'pd.to_datetime'
                        }
                except (ValueError, TypeError):
                    pass
                
                # Try explicit date formats if automatic parsing failed
                if self.column_types.get(col) != 'datetime':
                    for date_format in self.date_formats:
                        try:
                            values = self.df[col].dropna()
                            if self.sample_size > 0 and len(values) > self.sample_size:
                                values = values.sample(self.sample_size, random_state=42)
                            
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
        
        # Look for timestamps in integer columns
        for col in self.df.columns:
            if self.column_types.get(col) == 'integer':
                # Check for column names that suggest it might be an ID
                # Skip non-string column names (e.g., integers)
                if not isinstance(col, str):
                    continue
                    
                if col.lower() in ['id', 'key', 'index', 'code']:
                    continue
                
                values = self.df[col].dropna()
                if self.sample_size > 0 and len(values) > self.sample_size:
                    values = values.sample(self.sample_size, random_state=42)
                
                # Skip very small values (likely IDs, not timestamps)
                if values.min() < 1000000:  # Timestamp for 1970-01-12
                    continue
                
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
    
    def convert_types(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert column types based on suggestions.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            List of columns to convert. If None, convert all columns with suggestions.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with converted types
        """
        if not hasattr(self, 'conversion_suggestions') or not self.conversion_suggestions:
            # Run detection if not already done
            self.detect_all_types()
        
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
                    if 'unit=' in suggestion['method']:
                        df_converted[col] = pd.to_datetime(df_converted[col], unit='s', errors='coerce')
                    elif 'format=' in suggestion['method']:
                        # Extract format from the suggestion
                        format_match = re.search(r'format="([^"]+)"', suggestion['method'])
                        if format_match:
                            date_format = format_match.group(1)
                            df_converted[col] = pd.to_datetime(df_converted[col], format=date_format, errors='coerce')
                        else:
                            df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                    else:
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


def optimize_dataframe_types(
    df: pd.DataFrame,
    sample_size: int = 1000,
    threshold: float = 0.9,
    columns: Optional[List[str]] = None,
    custom_patterns: Optional[Dict[str, str]] = None,
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
    detector = DataTypeDetector(
        df,
        sample_size=sample_size,
        threshold=threshold,
        custom_patterns=custom_patterns,
    )
    detector.detect_all_types()
    return detector.convert_types(columns=columns)