import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

"""
This script tests our improved Excel date conversion implementation 
with mixed data types
"""

# Create a DataFrame with a mix of value types
df = pd.DataFrame({
    'date': [
        43831.0,                 # Normal Excel date (2020-01-01)
        44196.0,                 # Normal Excel date (2020-12-31)
        np.nan,                  # NaN value
        'not a number',          # String value
        44562.0,                 # Normal Excel date (2022-01-01)
        0,                       # Zero
        1000000000000,           # Very large number
        -1000,                   # Negative number
    ],
    'description': [
        'New Year 2020', 
        'New Year 2021', 
        'Missing', 
        'Invalid',
        'New Year 2022', 
        'Zero', 
        'Large', 
        'Negative'
    ]
})

print("Original DataFrame with mixed values:")
print(df)
print(f"Data types: {df.dtypes}")

# Test with the DataTypeDetector
print("\nForcing Excel date detection to test the fix...")
detector = DataTypeDetector(df)
detector.column_types['date'] = 'datetime'
detector.semantic_types['date'] = 'excel_date'
detector.conversion_suggestions['date'] = {
    'convert_to': 'datetime',
    'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'
}

# Now try the conversion with our fixed implementation
print("\nRunning conversion with the fixed implementation...")
try:
    converted_df = detector.convert_types()
    print("\nConversion succeeded! Results:")
    print(converted_df)
    print(f"Converted data types: {converted_df.dtypes}")
    
    # Check which values were properly converted to datetime
    # Note: pd.to_datetime will return NaT for values that aren't already datetime or can't be parsed
    if pd.api.types.is_datetime64_dtype(converted_df['date'].dtype):
        print("\nColumn was successfully converted to datetime!")
    else:
        # Try coercing the values to see which ones are datetime
        print("\nTrying to identify which values are properly formatted as datetime")
        converted_values = pd.to_datetime(converted_df['date'], errors='coerce')
        valid_mask = converted_values.notna()
        
        print(f"Valid datetime values: {valid_mask.sum()} of {len(valid_mask)}")
        if valid_mask.any():
            print("Valid datetime values:")
            for idx in valid_mask[valid_mask].index:
                print(f"  Index {idx}: {converted_df['date'].iloc[idx]} -> {converted_values.iloc[idx]}")
        
except Exception as e:
    print(f"\nConversion failed: {e}")