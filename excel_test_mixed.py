import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

"""
This test script tests the DataTypeDetector's Excel date conversion with a mixed column 
that contains various types of values - numbers, NaN, strings, and potentially problematic values.
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

# Let's manually check what happens with numeric conversion first
print("\nManual numeric conversion test:")
numeric_values = pd.to_numeric(df['date'], errors='coerce')
print(numeric_values)
print(f"After numeric conversion dtypes: {numeric_values.dtype}")

# Check which values are finite
mask = np.isfinite(numeric_values)
print(f"\nFinite values mask: {mask.values}")

# Try direct pandas to_datetime on the numeric values
try:
    dates = pd.to_datetime(numeric_values, unit='D', origin='1899-12-30')
    print(f"\nDirect pandas conversion result: {dates}")
except Exception as e:
    print(f"\nDirect pandas conversion failed: {e}")

# Try our approach
print("\nTesting our approach manually:")
result = pd.Series([pd.NaT] * len(df), index=df.index)
if mask.any():
    try:
        finite_dates = pd.to_datetime(numeric_values[mask], unit='D', origin='1899-12-30')
        print(f"Converted finite values: {finite_dates}")
        # Assign back to result
        result[mask] = finite_dates
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Our approach failed: {e}")

# Now let's test with the actual DataTypeDetector
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
    
    # Check which values are datetime and which aren't
    datetime_mask = pd.to_datetime(converted_df['date'], errors='coerce').notna()
    print(f"\nConverted to datetime mask: {datetime_mask.values}")
    
    if datetime_mask.any():
        print("\nSuccessfully converted values:")
        print(converted_df[datetime_mask])
    else:
        print("\nNo values were successfully converted to datetime!")
        
except Exception as e:
    print(f"\nConversion failed: {e}")