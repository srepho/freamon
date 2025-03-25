import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

# Create a dataframe with Excel dates that will clearly be detected as Excel dates
# Excel dates are days since 1899-12-30 (with some adjustments)
df = pd.DataFrame({
    'date': [43831, 44196, np.nan, 44562, np.nan, 44927, 45292, np.nan],  # 2020 through 2024 with NaN values
    'description': ['New Year 2020', 'New Year 2021', 'Missing', 'New Year 2022', 'Missing', 'New Year 2023', 'New Year 2024', 'Missing']
})

print("Original DataFrame:")
print(df)

# Set up the detector
detector = DataTypeDetector(df)

# Force excel_date detection to ensure it's detected correctly
# This simulates what would happen in your real code
detector._detect_basic_types()
detector.column_types['date'] = 'datetime'
detector.semantic_types['date'] = 'excel_date'
detector.conversion_suggestions['date'] = {
    'convert_to': 'datetime',
    'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'
}

print("\nDetection results forced to Excel date:")
print(f"Column type: {detector.column_types['date']}")
print(f"Semantic type: {detector.semantic_types['date']}")
print(f"Conversion suggestion: {detector.conversion_suggestions['date']}")

# Test the conversion with NaN values
print("\nConverting Excel dates to datetime...")
try:
    converted_df = detector.convert_types()
    print("\nConverted DataFrame:")
    print(converted_df)
except Exception as e:
    print(f"\nError during conversion: {e}")
    
    # Print the values we're trying to convert
    print("\nValues being converted:")
    values = df['date'].dropna()
    print(values)
    
    # Try manual conversion directly for comparison
    print("\nTrying manual conversion:")
    try:
        manual_converted = pd.to_datetime(df['date'], unit='D', origin='1899-12-30', errors='coerce')
        print("Manual conversion succeeded:")
        print(manual_converted)
    except Exception as e2:
        print(f"Manual conversion also failed: {e2}")