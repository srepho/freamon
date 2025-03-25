import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

# Create a test dataframe with Excel dates and NaN values
df = pd.DataFrame({
    'date': [43831, 44196, np.nan, 44562, np.nan, 44927, 45292, np.nan],  # 2020 through 2024 with NaN values
})

print("Original DataFrame:")
print(df)

# Test the DataTypeDetector
detector = DataTypeDetector(df)
results = detector.detect_all_types()

print("\nDetection Results:")
for col, info in results.items():
    print(f"\n{col}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# Test the conversion with NaN values
print("\nConverting Excel dates to datetime...")
try:
    converted_df = detector.convert_types()
    print("\nConverted DataFrame:")
    print(converted_df)
except Exception as e:
    print(f"\nError during conversion: {e}")
    
    # Try to troubleshoot by inspecting what's happening during conversion
    print("\nTroubleshooting:")
    column = 'date'
    if column in detector.conversion_suggestions:
        suggestion = detector.conversion_suggestions[column]
        print(f"Suggestion for '{column}': {suggestion}")
        
        # Try with NaN values dropped
        print("\nTrying with NaN values dropped:")
        cleaned_df = df.copy()
        cleaned_df[column] = cleaned_df[column].dropna()
        cleaned_detector = DataTypeDetector(cleaned_df)
        cleaned_detector.detect_all_types()
        try:
            cleaned_converted = cleaned_detector.convert_types()
            print("Success with NaN values dropped!")
            print(cleaned_converted)
        except Exception as e2:
            print(f"Still failed: {e2}")
