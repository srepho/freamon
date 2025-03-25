"""
Test script to simulate the real-world scenario with month-year columns that aren't being detected.
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

# Create a larger, more realistic DataFrame with multiple columns
# Some will have missing values, which is often the case in real datasets
np.random.seed(42)

# Create date range for test data
months = ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24']
numeric_cols = {}

# Create 10 numeric columns for context
for i in range(1, 11):
    numeric_cols[f'metric_{i}'] = np.random.randn(50) * 100

# Create month-year columns with varying amounts of missing data
month_cols = {}
for i in range(1, 6):
    # Introduce some missing values with increasing probability
    missing_prob = i * 0.05
    column_data = []
    for _ in range(50):
        if np.random.random() < missing_prob:
            column_data.append(np.nan)
        else:
            column_data.append(np.random.choice(months))
    month_cols[f'month_col_{i}'] = column_data

# Combine all data
all_data = {**numeric_cols, **month_cols}
df = pd.DataFrame(all_data)

print(f"Created test DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"DataFrame contains {df.isna().sum().sum()} missing values")
print("\nSample of DataFrame:")
print(df.head())
print("\nDataFrame dtypes:")
print(df.dtypes)

# Check pattern matching for month columns directly
print("\nDirect pattern matching check for month columns:")
for col in df.columns:
    if 'month' in col:
        # Count non-missing values
        non_missing = df[col].dropna()
        is_month_year = is_month_year_format(non_missing)
        print(f"  {col}: is_month_year_format = {is_month_year} (non-missing values: {len(non_missing)})")

# Run DataTypeDetector
print("\nRunning DataTypeDetector:")
detector = DataTypeDetector(df)
type_report = detector.detect_all_types()

# Print summary of detection results for month columns
print("\nMonth column detection results:")
print("-" * 50)
print(f"{'Column':<15} {'Type':<15} {'Semantic Type':<20} {'Conversion':<20}")
print("-" * 50)
for col in df.columns:
    if 'month' in col:
        col_type = detector.column_types.get(col, 'Not detected')
        semantic = detector.semantic_types.get(col, 'Not detected')
        conversion = 'Yes' if col in detector.conversion_suggestions else 'No'
        print(f"{col:<15} {col_type:<15} {semantic:<20} {conversion:<20}")

# Test conversion of detected month-year columns
print("\nTest conversion for detected month-year columns:")
converted_df = detector.convert_types()
for col in df.columns:
    if 'month' in col:
        if col in detector.semantic_types and detector.semantic_types[col] == 'month_year_format':
            print(f"\n{col} - Original values (first 3):")
            print(df[col].head(3))
            print(f"{col} - Converted values (first 3):")
            print(converted_df[col].head(3))
            print(f"{col} - Dtype after conversion: {converted_df[col].dtype}")
        else:
            print(f"\n{col} - NOT detected as month-year format")
            print(f"Original values (first 3): {df[col].head(3).tolist()}")

# Check if any month columns were missed and diagnose why
print("\nMissed month-year columns diagnosis:")
missed_columns = []
for col in df.columns:
    if 'month' in col and (col not in detector.semantic_types or detector.semantic_types[col] != 'month_year_format'):
        missed_columns.append(col)
        print(f"\n  Missed column: {col}")
        
        # Get non-missing values for analysis
        non_missing = df[col].dropna()
        print(f"    Non-missing values: {len(non_missing)} out of {len(df[col])}")
        
        # Check if it's detected as something else first
        print(f"    Detected as: {detector.column_types.get(col, 'Not detected')}")
        
        # Check direct pattern matching
        is_month_year = is_month_year_format(non_missing)
        print(f"    Direct is_month_year_format check: {is_month_year}")
        
        # Test direct conversion 
        if len(non_missing) > 0:
            print(f"    Original sample: {non_missing.head(3).tolist()}")
            converted = convert_month_year_format(df[col])
            print(f"    Direct conversion result: {converted.head(3).tolist()}")

if not missed_columns:
    print("  All month columns were detected correctly!")

# Modify the DataTypeDetector class to fix the issue (add the fix recommendation)
print("\nRecommended fix for the issue:")
print("""
The issue is that month_year_format detection is not working reliably for columns with missing values.
The fix should ensure that:

1. Columns are checked for month-year format AFTER basic type detection even if they're object type
2. When checking for month-year format, only non-missing values are considered 
3. Lowering the threshold for columns with higher missing value ratios

Here's the recommended code change to fix this issue:
```python
# In the detect_all_types method of DataTypeDetector:

# After _detect_basic_types and before _detect_semantic_types, add:
# Check for month-year format in all string/object columns
for col in self.df.columns:
    # Only check columns that are still typed as 'object' or 'string' at this point
    if self.column_types.get(col) in ['object', 'string'] and col not in self.semantic_types:
        # Get non-missing values
        sample = self._get_column_sample(col)
        valid_values = sample.dropna()
        
        # Skip if we don't have enough valid values
        if len(valid_values) < 5:
            continue
            
        # Calculate adaptive threshold based on missing ratio
        missing_ratio = sample.isna().mean()
        adaptive_threshold = max(0.7, self.threshold - missing_ratio/2)
        
        # Check for month-year format using valid values only
        if is_month_year_format(valid_values, threshold=adaptive_threshold):
            self.semantic_types[col] = 'month_year_format'
            self.conversion_suggestions[col] = {
                'convert_to': 'datetime',
                'method': 'convert_month_year_format',
                'note': 'Contains month-year format'
            }
            logger.debug(f"Month-year format detected for column '{col}' with adaptive threshold {adaptive_threshold:.2f}")
```
""")