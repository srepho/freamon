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

# Create a sample DataFrame with the problematic format
data = {
    'month_year_column': ['Aug-24', 'Oct-24', 'Nov-24', 'Dec-24', 'Jan-25']
}
df = pd.DataFrame(data)

# First, test the direct pattern matching function
values = df['month_year_column']
is_month_year = is_month_year_format(values)
print(f"Direct check with is_month_year_format: {is_month_year}")

# Check each value against the regex pattern to see if it matches
import re
month_year_pattern = re.compile(
    r'^([A-Za-z]{3,9})[-/\. ]?(20\d{2}|\d{2})$|^(0?\d|1[0-2])[-/\.](20\d{2}|\d{2})$', 
    re.IGNORECASE
)

print("\nChecking each value against the pattern:")
for val in values:
    match = month_year_pattern.match(str(val).strip())
    print(f"  '{val}': {'✓' if match else '✗'} {match.groups() if match else ''}")

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

# If the detection didn't work as expected, print detailed debug info
if df.columns[0] not in detector.semantic_types or detector.semantic_types[df.columns[0]] != 'month_year_format':
    print("\nDetailed debug info:")
    # Try the exact code path that's used in DataTypeDetector
    column = df.columns[0]
    sample = detector._get_column_sample(column)
    
    # Check if any values are strings
    print(f"  Sample data types: {sample.apply(type).value_counts().to_dict()}")
    
    # For string columns, check if they're likely month-year format
    if 'object' in detector.column_types.get(column, ''):
        print(f"  Column is string type: {detector.column_types.get(column)}")
        
        # Check if it's a month-year format
        print(f"  Checking for month-year format...")
        # Get non-null values for analysis
        valid_values = sample.dropna()
        if len(valid_values) > 0 and is_month_year_format(valid_values, threshold=detector.threshold):
            print("  ✓ Month-year format detected!")
        else:
            print("  ✗ Month-year format not detected")
else:
    # Test the conversion if it was detected correctly
    print("\nTesting conversion functionality:")
    print(f"Original values: {df['month_year_column'].tolist()}")
    
    # Try converting with date_converters.convert_month_year_format directly
    converted = convert_month_year_format(df['month_year_column'])
    print(f"Converted with convert_month_year_format: {converted.tolist()}")
    
    # Try converting with the detector
    converted_df = detector.convert_types()
    print(f"Converted with detector.convert_types(): {converted_df['month_year_column'].tolist()}")
    print(f"Data type after conversion: {converted_df['month_year_column'].dtype}")
    
    # Show resulting year and month values
    if pd.api.types.is_datetime64_dtype(converted_df['month_year_column'].dtype):
        years = [d.year if not pd.isna(d) else None for d in converted_df['month_year_column']]
        months = [d.month if not pd.isna(d) else None for d in converted_df['month_year_column']]
        print(f"Years after conversion: {years}")
        print(f"Months after conversion: {months}")