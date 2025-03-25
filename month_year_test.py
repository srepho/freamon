"""
Test script to debug month-year format detection issue.
"""
import pandas as pd
import logging
import re
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.date_converters import is_month_year_format, convert_month_year_format

# Set up logging to debug the detection process
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('freamon.utils.datatype_detector')
logger.setLevel(logging.DEBUG)

# Create a sample DataFrame with various month-year formats
data = {
    'month_year_column': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    'month_with_missing': ['Jun-24', None, 'Aug-24', 'Sep-24', None],
    'mixed_data': ['Jun-24', 'Other text', 'Aug-24', 'Not a month', 'Oct-24'],
    'numeric': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\nDataFrame dtypes:")
print(df.dtypes)

# First, test the direct pattern matching function
for col in ['month_year_column', 'month_with_missing', 'mixed_data']:
    values = df[col].dropna()
    is_month_year = is_month_year_format(values)
    print(f"\nDirect check for {col} with is_month_year_format: {is_month_year}")

# Now run through the DataTypeDetector
print("\nRunning DataTypeDetector:")
detector = DataTypeDetector(df)
detector.detect_all_types()

# Print detection results
print("\nDetected column types:")
for col, type_info in detector.column_types.items():
    print(f"  {col}: {type_info}")

print("\nDetected semantic types:")
for col, sem_type in detector.semantic_types.items():
    sem_type_str = sem_type if sem_type else 'None'
    print(f"  {col}: {sem_type_str}")

print("\nConversion suggestions:")
for col, suggestion in detector.conversion_suggestions.items():
    print(f"  {col}: {suggestion}")

# Test conversion
print("\nTesting conversion:")
converted_df = detector.convert_types()
print(converted_df)