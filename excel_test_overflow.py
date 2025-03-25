import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

# Create a dataframe with Excel dates but add some problematic values
df = pd.DataFrame({
    'date': [
        43831.0,                 # Normal Excel date (2020-01-01)
        np.nan,                  # NaN value
        'not a number',         # String value
        44196.0,                 # Normal Excel date (2020-12-31)
        0,                       # Zero (edge case)
        44562.0,                 # Normal Excel date (2022-01-01)
        -1000,                   # Negative value (edge case)
        1000000000000000000.0,   # Very large number that could cause overflow
        np.inf,                  # Infinity
        -np.inf,                # Negative infinity
    ],
    'description': [
        'Normal date', 'NaN', 'String', 'Normal date', 'Zero', 
        'Normal date', 'Negative', 'Very large', 'Infinity', 'Negative infinity'
    ]
})

print("Original DataFrame with problematic values:")
print(df)

# First, try direct pandas conversion to see how it handles various cases
print("\nTrying direct pandas conversion:")
try:
    direct_result = pd.to_datetime(df['date'], unit='D', origin='1899-12-30', errors='coerce')
    print("Direct conversion result:")
    print(direct_result)
except Exception as e:
    print(f"Direct conversion failed: {e}")

# Test the conversion through the DataTypeDetector
detector = DataTypeDetector(df)

# Force excel_date detection
detector._detect_basic_types()
detector.column_types['date'] = 'datetime'
detector.semantic_types['date'] = 'excel_date'
detector.conversion_suggestions['date'] = {
    'convert_to': 'datetime',
    'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'
}

print("\nAttempting conversion through DataTypeDetector:")
try:
    converted_df = detector.convert_types()
    print("Conversion succeeded!")
    print(converted_df)
except Exception as e:
    print(f"Conversion failed: {e}")
    
    # Let's look at the code that might be causing issues
    print("\nAnalyzing the problematic line from convert_types method:")
    print("Looking at values being used...")
    
    # Get the column values
    values = df['date']
    print(f"Values:\n{values}")
    
    # Check types
    print(f"\nValue types: {values.dtype}")
    
    # Try to identify which value is causing the problem
    print("\nTesting values one by one:")
    for i, val in enumerate(values):
        try:
            if pd.notna(val):
                result = pd.to_datetime(val, unit='D', origin='1899-12-30')
                print(f"Value at index {i} ({val}) converted successfully: {result}")
        except Exception as e2:
            print(f"Value at index {i} ({val}) failed: {e2}")
            
    # Try to convert with various error handling approaches
    print("\nTrying alternative approaches:")
    
    # Approach 1: First convert to float and handle exceptions
    print("Approach 1 - Convert to float first:")
    try:
        float_values = pd.to_numeric(values, errors='coerce')
        print(f"After coercing to numeric:\n{float_values}")
        datetime_values = pd.to_datetime(float_values, unit='D', origin='1899-12-30', errors='coerce')
        print(f"After converting to datetime:\n{datetime_values}")
    except Exception as e3:
        print(f"Approach 1 failed: {e3}")
    
    # Approach 2: Filter out non-finite values first
    print("\nApproach 2 - Filter non-finite values:")
    try:
        # Convert to numeric, then mask out non-finite values
        numeric_values = pd.to_numeric(values, errors='coerce')
        mask = np.isfinite(numeric_values)
        masked_values = numeric_values.copy()
        masked_values[~mask] = np.nan
        
        print(f"After masking non-finite values:\n{masked_values}")
        datetime_values = pd.to_datetime(masked_values, unit='D', origin='1899-12-30', errors='coerce')
        print(f"After converting to datetime:\n{datetime_values}")
    except Exception as e4:
        print(f"Approach 2 failed: {e4}")