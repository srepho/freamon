"""
Mixed Date Formats Detection Example

This example demonstrates how the Freamon library can detect and convert
columns with mixed date formats. This is particularly useful when working with
real-world datasets where date representations may be inconsistent.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types

# Create sample data with mixed date formats
def create_mixed_date_df():
    """Create a sample dataframe with mixed date formats."""
    # Create a dataframe with dates in multiple formats
    return pd.DataFrame({
        'mixed_dates': [
            '2020-01-01',          # ISO format
            '01/15/2020',          # MM/DD/YYYY
            '15/01/2020',          # DD/MM/YYYY
            'January 20, 2020',    # Month name format
            '2020/02/01',          # YYYY/MM/DD
            '01-Mar-2020',         # DD-Mon-YYYY
            '20200401',            # YYYYMMDD
            '2020.05.01',          # YYYY.MM.DD
            '06-2020-01',          # MM-YYYY-DD
            '2020-07-01 14:30:00'  # ISO with time
        ],
        'dates_iso': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
                      '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10'],
        'values': np.random.normal(0, 1, 10)
    })

# Create and display the dataframe
print("Creating sample dataframe with mixed date formats...")
df = create_mixed_date_df()
print("\nOriginal DataFrame:")
print(df.to_string())
print("\nDataFrame Info (original):")
print(df.dtypes)

# Use the DataTypeDetector to identify date formats
print("\nDetecting data types...")
detector = DataTypeDetector(df)
results = detector.detect_all_types()

# Print the detection results
print("\nDetection Results:")
for col, info in results.items():
    print(f"\n{col}:")
    print(f"  Storage type: {info['storage_type']}")
    print(f"  Logical type: {info['logical_type']}")
    if 'semantic_type' in info:
        print(f"  Semantic type: {info['semantic_type']}")
    if 'suggested_conversion' in info:
        print(f"  Suggested conversion: {info['suggested_conversion']['convert_to']} using {info['suggested_conversion']['method']}")
        if 'detected_formats' in info['suggested_conversion']:
            print(f"  Detected formats: {info['suggested_conversion']['detected_formats']}")

# Convert the dates using the detector
print("\nConverting mixed date formats to proper datetime objects...")
converted_df = detector.convert_types()

print("\nConverted DataFrame:")
print(converted_df.to_string())
print("\nDataFrame Info (after conversion):")
print(converted_df.dtypes)

# Generate a plot to visualize the time series data
print("\nCreating a time series plot with the ISO dates...")
plt.figure(figsize=(12, 6))

# Plot only the ISO dates since we know they're correctly parsed
plt.plot(pd.to_datetime(converted_df['dates_iso']), converted_df['values'], 
         marker='o', linestyle='-', alpha=0.6, color='blue', label='Time Series Data')

plt.title('Time Series Data with Date Values')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mixed_date_formats_example.png')
print("Plot saved as 'mixed_date_formats_example.png'")

# Explanation of the mixed date format detection process
print("\nHow Mixed Date Format Detection Works:")
print("1. First, automatic parsing is attempted with pd.to_datetime()")
print("2. If that fails to parse all dates, individual format detection begins")
print("3. Each date format is tested on the sample values")
print("4. If multiple formats are needed to parse all values, the column is identified as having mixed formats")
print("5. During conversion, a multi-pass approach is used:")
print("   - First automatic parsing is applied to handle the easily parsable dates")
print("   - Then each detected format is applied to remaining unparsed values")
print("   - This process continues until all dates are converted or no more formats are available")

print("\nExample complete!")