import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector

"""
This script simulates a real-world scenario with Excel dates in a DataFrame
that has many missing values, similar to what the user is experiencing.
"""

# Create a realistic DataFrame with Excel dates and many missing values
np.random.seed(42)  # For reproducibility

# Create dates (Excel dates from 2020-01-01 to 2022-12-31 with ~50% missing)
n_rows = 1000
excel_base_date = 43831  # 2020-01-01
date_range = 1095        # ~3 years in days

dates = []
for i in range(n_rows):
    # 50% chance of missing value
    if np.random.random() < 0.5:
        dates.append(np.nan)
    else:
        # Random date in the range
        date_val = excel_base_date + np.random.randint(0, date_range)
        dates.append(date_val)

# Create DataFrame
df = pd.DataFrame({
    'id': range(1, n_rows + 1),
    'date': dates,
    'value': np.random.normal(100, 15, n_rows)
})

print(f"Created DataFrame with {n_rows} rows, {df['date'].isna().sum()} missing dates ({df['date'].isna().mean()*100:.1f}%)")
print(df.head(10))

# Apply datatype detection
print("\nRunning DataTypeDetector...")
detector = DataTypeDetector(df)
results = detector.detect_all_types()

# Check if Excel dates were detected
print("\nDetection Results for date column:")
if 'date' in results:
    for key, value in results['date'].items():
        print(f"  {key}: {value}")

# Run the conversion
print("\nConverting types...")
try:
    converted_df = detector.convert_types()
    
    # Check if the date column was successfully converted
    print("\nConversion results:")
    print(f"Date column dtype: {converted_df['date'].dtype}")
    print(f"Non-null values: {converted_df['date'].notna().sum()} of {len(converted_df)}")
    
    # Show some sample values
    print("\nSample of converted dates:")
    mask = converted_df['date'].notna()
    print(converted_df[mask].head())
    
except Exception as e:
    print(f"\nConversion failed: {e}")