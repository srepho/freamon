"""
Test script to debug month-year format detection with a more complex dataset.
"""
import pandas as pd
import numpy as np
import logging
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.date_converters import is_month_year_format, convert_month_year_format

# Set up logging to debug the detection process
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('freamon.utils.datatype_detector')
logger.setLevel(logging.DEBUG)

# Create a sample DataFrame that better matches real-world data
# Include some columns with month-year data and some with other types
data = {
    # Month-year columns with normal pattern
    'month_1': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    'month_2': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    
    # Other columns that might affect detection
    'id': [1, 2, 3, 4, 5],
    'value': [100.5, 200.7, 150.3, 300.2, 250.1],
    'category': ['A', 'B', 'A', 'C', 'B'],
    
    # Month-year with some missing values
    'month_with_missing': ['Jun-24', np.nan, 'Aug-24', np.nan, 'Oct-24'],
    
    # Date column in a different format
    'full_date': ['2024-06-15', '2024-07-20', '2024-08-10', '2024-09-05', '2024-10-30'],
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df.head())
print("\nDataFrame dtypes:")
print(df.dtypes)

# Run the full detector
print("\nRunning DataTypeDetector:")
detector = DataTypeDetector(df)
type_report = detector.detect_all_types()

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
print(f"Converted DataFrame (first row):")
print(converted_df.head(1))
print(f"Converted DataFrame dtypes:")
print(converted_df.dtypes)

# Check if any month-year columns were missed
print("\nChecking for missed month-year columns:")
missed_columns = []
for col in df.columns:
    if 'month' in col.lower() and (col not in detector.semantic_types or detector.semantic_types[col] != 'month_year_format'):
        print(f"  Missed column: {col}")
        missed_columns.append(col)
        
        # Check direct pattern matching
        is_month_year = is_month_year_format(df[col].dropna())
        print(f"    Direct is_month_year_format check: {is_month_year}")
        
        # Test direct conversion 
        print(f"    Original values: {df[col].head(3).tolist()}")
        converted = convert_month_year_format(df[col])
        print(f"    Direct conversion result: {converted.head(3).tolist()}")

if not missed_columns:
    print("  All month-year columns were detected correctly!")