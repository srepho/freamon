"""
Final test script to verify month-year format detection and conversion.
"""
import pandas as pd
import numpy as np
import logging
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.date_converters import convert_month_year_format, is_month_year_format

# Set up logging to debug the detection process
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('freamon.utils.datatype_detector')
logger.setLevel(logging.DEBUG)

# Create a simple test dataframe with month-year data
data = {
    'month_year': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24']
}
df = pd.DataFrame(data)

print(f"Original data: {df['month_year'].tolist()}")

# Test direct conversion function
print("\nTesting direct conversion with convert_month_year_format:")
converted = convert_month_year_format(df['month_year'])
print(f"Converted data: {converted.tolist()}")
print(f"Converted dtype: {converted.dtype}")

# Test with DataTypeDetector
print("\nTesting with DataTypeDetector:")
detector = DataTypeDetector(df)
detector.detect_all_types()

print(f"Detected column types: {detector.column_types}")
print(f"Detected semantic types: {detector.semantic_types}")
print(f"Conversion suggestions: {detector.conversion_suggestions}")

# Test conversion through DataTypeDetector
print("\nTesting conversion through DataTypeDetector:")
converted_df = detector.convert_types()
print(f"Converted data: {converted_df['month_year'].tolist()}")
print(f"Converted dtype: {converted_df['month_year'].dtype}")

# Test with a real-world example (with missing values)
print("\nTesting with a more complex dataset:")
complex_data = {
    'month_col': ['Jun-24', None, 'Aug-24', np.nan, 'Oct-24'],
    'numeric': [1, 2, 3, 4, 5]
}
complex_df = pd.DataFrame(complex_data)

print(f"Original complex data: {complex_df['month_col'].tolist()}")

# Check direct month-year detection
print("\nDirect month-year format check for complex data:")
valid_values = complex_df['month_col'].dropna()
print(f"Non-missing values: {valid_values.tolist()}")
is_my = is_month_year_format(valid_values)
print(f"is_month_year_format result: {is_my}")

# Test direct conversion
print("\nTesting direct conversion with convert_month_year_format for complex data:")
complex_converted = convert_month_year_format(complex_df['month_col'])
print(f"Converted complex data: {complex_converted.tolist()}")
print(f"Converted complex dtype: {complex_converted.dtype}")

# Test with DataTypeDetector
print("\nTesting complex data with DataTypeDetector:")
complex_detector = DataTypeDetector(complex_df)
complex_detector.detect_all_types()

print(f"Detected column types: {complex_detector.column_types}")
print(f"Detected semantic types: {complex_detector.semantic_types}")
print(f"Conversion suggestions: {complex_detector.conversion_suggestions}")

# Test conversion through DataTypeDetector
print("\nTesting conversion of complex data through DataTypeDetector:")
complex_converted_df = complex_detector.convert_types()
print(f"Converted complex data: {complex_converted_df['month_col'].tolist()}")
print(f"Converted complex dtype: {complex_converted_df['month_col'].dtype}")