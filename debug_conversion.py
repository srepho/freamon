"""
Test script to debug month-year format conversion.
"""
import pandas as pd
import numpy as np
import logging
from freamon.utils.date_converters import convert_month_year_format

# Set up logging to debug the detection process
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create test data
data = {
    'month_year': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24']
}
df = pd.DataFrame(data)

print(f"Original data: {df['month_year'].tolist()}")

# Test direct conversion
print("\nTesting direct conversion:")
converted = convert_month_year_format(df['month_year'])
print(f"Converted data: {converted.tolist()}")
print(f"Dtype after conversion: {converted.dtype}")

# Test with different dtypes
print("\nTesting with different data types:")

# Convert to numpy array and back to pandas to get numpy.str_ type
np_arr = np.array(df['month_year'].tolist())
np_series = pd.Series(np_arr)
print(f"NumPy array dtype: {np_arr.dtype}")
print(f"Series from NumPy array dtype: {np_series.dtype}")

converted_np = convert_month_year_format(np_series)
print(f"Converted NumPy-based data: {converted_np.tolist()}")

# Test with mixed data
print("\nTesting with mixed/missing data:")
mixed_data = pd.Series(['Jun-24', np.nan, 'Jul-24', None, 'Aug-24'])
print(f"Mixed data: {mixed_data.tolist()}")
converted_mixed = convert_month_year_format(mixed_data)
print(f"Converted mixed data: {converted_mixed.tolist()}")

# Detailed debugging
print("\nDetailed debugging of format patterns:")
from freamon.utils.date_converters import convert_month_year_format

# Define manual format patterns for testing
format_patterns = [
    # Month abbreviation with 2-digit year (Aug-24)
    ('%b-%y', r'^[A-Za-z]{3}-\d{2}$'),
]

# Test each pattern directly
for fmt, pattern in format_patterns:
    mask = df['month_year'].astype(str).str.match(pattern, case=False)
    print(f"Pattern '{pattern}' matches: {mask.tolist()}")
    
    # Try parsing with the format
    try:
        parsed = pd.to_datetime(df['month_year'], format=fmt, errors='coerce')
        print(f"Parsed with format '{fmt}': {parsed.tolist()}")
    except Exception as e:
        print(f"Error parsing with format '{fmt}': {e}")
        
    # Try str conversion first
    try:
        str_values = df['month_year'].astype(str)
        parsed = pd.to_datetime(str_values, format=fmt, errors='coerce')
        print(f"Parsed with format '{fmt}' after str conversion: {parsed.tolist()}")
    except Exception as e:
        print(f"Error parsing with str conversion and format '{fmt}': {e}")
    
    # Try apply
    try:
        str_values = df['month_year'].apply(lambda x: str(x) if pd.notna(x) else x)
        parsed = pd.to_datetime(str_values, format=fmt, errors='coerce')
        print(f"Parsed with format '{fmt}' after apply str conversion: {parsed.tolist()}")
    except Exception as e:
        print(f"Error parsing with apply str conversion and format '{fmt}': {e}")
        
print("\nDirect conversion using pandas to_datetime:")
try:
    # Try direct conversion with common formats
    for test_fmt in ['%b-%y', '%b-%Y', '%B-%y', '%B-%Y']:
        parsed = pd.to_datetime(df['month_year'], format=test_fmt, errors='coerce')
        print(f"Format '{test_fmt}': {parsed.tolist()}")
except Exception as e:
    print(f"Error with direct conversion: {e}")