import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

# Create a realistic test case with Excel dates and NaN values
df = pd.DataFrame({
    'date': [43831.0, 44196.0, np.nan, 44562.0, np.nan, 44927.0, 45292.0, np.nan],  # 2020 through 2024 with NaN values
    'price': [100.0, 105.0, np.nan, 110.0, np.nan, 115.0, 120.0, np.nan]
})

print("Original DataFrame:")
print(df)

# Set up the detector
detector = DataTypeDetector(df)

# Force excel_date detection for the date column
detector._detect_basic_types()
detector.column_types['date'] = 'datetime'
detector.semantic_types['date'] = 'excel_date'
detector.conversion_suggestions['date'] = {
    'convert_to': 'datetime',
    'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'
}

print("\nForced Excel date detection for testing")

# Define a modified implementation of convert_types method to fix the issue
def fix_excel_date_conversion(df, column):
    """
    Fixed implementation for Excel date conversion that properly handles NaN values
    """
    print(f"\nApplying fixed Excel date conversion for column: {column}")
    
    # Step 1: Convert the column to numeric, forcing non-numeric values to NaN
    numeric_values = pd.to_numeric(df[column], errors='coerce')
    print(f"After converting to numeric:\n{numeric_values}")
    
    # Step 2: Apply the conversion only to finite values
    # Create a mask for finite values
    mask = np.isfinite(numeric_values)
    print(f"Finite values mask:\n{mask}")
    
    # Step 3: Initialize result Series with NaT values
    result = pd.Series([pd.NaT] * len(df), index=df.index)
    
    # Step 4: Convert only the finite values
    if mask.any():
        try:
            finite_dates = pd.to_datetime(
                numeric_values[mask], 
                unit='D', 
                origin='1899-12-30'
            )
            # Assign the converted dates back to the result Series
            result[mask] = finite_dates
            print(f"Converted dates:\n{result}")
            return result
        except Exception as e:
            print(f"Error during conversion: {e}")
            return df[column]  # Return original if conversion fails
    else:
        print("No finite values to convert")
        return df[column]  # Return original if no finite values

# Now try our fixed implementation
try:
    # Apply the fixed conversion function
    fixed_result = fix_excel_date_conversion(df, 'date')
    
    # Create a modified DataFrame with the fixed conversion
    fixed_df = df.copy()
    fixed_df['date'] = fixed_result
    
    print("\nFixed conversion result:")
    print(fixed_df)
    
    # Compare with the original detector's conversion (which might fail)
    print("\nNow trying the original conversion method for comparison:")
    try:
        converted_df = detector.convert_types()
        print("\nOriginal conversion succeeded:")
        print(converted_df)
    except Exception as e:
        print(f"\nOriginal conversion failed: {e}")
        
except Exception as e:
    print(f"Fixed implementation failed: {e}")