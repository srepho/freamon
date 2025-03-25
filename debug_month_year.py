"""
Test script to debug month-year format detection for columns pandas reads as objects.
"""
import pandas as pd
import logging
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.date_converters import is_month_year_format, convert_month_year_format

# Set up logging to debug the detection process
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('freamon.utils.datatype_detector')
logger.setLevel(logging.DEBUG)

# Create a sample DataFrame with month-year formats that pandas keeps as object type
data = {
    'month_1': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    'month_2': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    'month_3': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df.head())
print("\nDataFrame dtypes:")
print(df.dtypes)

# Check if the pattern matches these values
print("\nChecking pattern matching:")
for col in df.columns:
    is_month_year = is_month_year_format(df[col])
    print(f"  Column {col}: {is_month_year}")

# Run through the DataTypeDetector
print("\nRunning DataTypeDetector:")
detector = DataTypeDetector(df)

# Debug the basic type detection
print("\nRunning basic type detection:")
detector._detect_basic_types()
print("\nBasic detected column types:")
for col, type_info in detector.column_types.items():
    print(f"  {col}: {type_info}")

# Debug semantic type detection
print("\nRunning semantic type detection:")
detector._detect_semantic_types()
print("\nDetected semantic types:")
for col, sem_type in detector.semantic_types.items():
    sem_type_str = sem_type if sem_type else 'None'
    print(f"  {col}: {sem_type_str}")

# Run full detection
print("\nRunning full detection:")
detector = DataTypeDetector(df)
detector.detect_all_types()

# Print detection results
print("\nFull detection - column types:")
for col, type_info in detector.column_types.items():
    print(f"  {col}: {type_info}")

print("\nFull detection - semantic types:")
for col, sem_type in detector.semantic_types.items():
    sem_type_str = sem_type if sem_type else 'None'
    print(f"  {col}: {sem_type_str}")

print("\nFull detection - conversion suggestions:")
for col, suggestion in detector.conversion_suggestions.items():
    print(f"  {col}: {suggestion}")

# Test conversion
print("\nTesting conversion:")
converted_df = detector.convert_types()
print(f"Converted DataFrame:")
print(converted_df.head())
print(f"Converted DataFrame dtypes:")
print(converted_df.dtypes)