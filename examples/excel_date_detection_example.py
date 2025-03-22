"""
Excel Date Detection Example

This example demonstrates how the Freamon library can detect Excel dates in data frames
and convert them to proper datetime objects. This is particularly useful when working
with data exported from Excel to CSV, where dates are often saved as numbers.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types

# Create sample data with Excel dates
# Excel dates are days since 1899-12-30 (with some adjustments)
# Common values:
# 43831 = 2020-01-01
# 44196 = 2021-01-01
# 44562 = 2022-01-01

def create_excel_date_df():
    """Create a sample dataframe with Excel dates."""
    # Generate some realistic Excel dates for the past 5 years
    # 43831 = 2020-01-01 in Excel
    # Add some random days to generate dates in a range
    dates = [43831 + i * 30 + np.random.randint(0, 10) for i in range(60)]
    
    # For some dates, add time component (decimal part)
    # 0.5 = 12:00 PM, 0.25 = 6:00 AM, 0.75 = 6:00 PM
    time_vals = [0.0, 0.25, 0.5, 0.75]
    dates_with_time = [d + np.random.choice(time_vals) for d in dates]
    
    # Create some values that correlate with the dates
    values = [100 + (d - 43831) * 0.5 + np.random.normal(0, 10) for d in dates]
    
    # Set up a dataframe with different types of Excel dates
    df = pd.DataFrame({
        'excel_date': dates,
        'excel_datetime': dates_with_time,
        'normal_numbers': np.random.randint(1000, 9999, len(dates)),
        'string_values': [f"Item {i}" for i in range(len(dates))],
        'values': values
    })
    
    # Add some metadata columns with date-sounding names but with normal numbers
    df['item_id'] = np.arange(1, len(dates) + 1)
    df['price'] = np.random.uniform(10, 1000, len(dates))
    
    return df

# Create and display the dataframe
print("Creating sample dataframe with Excel dates...")
df = create_excel_date_df()
print("\nOriginal DataFrame (first 5 rows):")
print(df.head().to_string())
print("\nDataFrame Info (original):")
print(df.dtypes)

# Use the DataTypeDetector to identify the Excel dates
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

# Convert the Excel dates using the detector
print("\nConverting Excel dates to proper datetime objects...")
converted_df = detector.convert_types()

print("\nConverted DataFrame (first 5 rows):")
print(converted_df.head().to_string())
print("\nDataFrame Info (after conversion):")
print(converted_df.dtypes)

# Generate a plot to visualize the time series data
print("\nCreating a time series plot with the converted dates...")
plt.figure(figsize=(12, 6))
plt.plot(converted_df['excel_date'], converted_df['values'], marker='o', linestyle='-', alpha=0.6)
plt.title('Time Series Data After Excel Date Conversion')
plt.xlabel('Date')
plt.ylabel('Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('excel_date_conversion_example.png')
print("Plot saved as 'excel_date_conversion_example.png'")

# Demonstration of manual conversion for comparison
print("\nFor comparison, here's how to manually convert Excel dates:")
manual_conversion = pd.to_datetime(df['excel_date'], unit='D', origin='1899-12-30')
print(manual_conversion.head())

# Demonstration of Excel date with time components
print("\nExcel dates with time components (decimal parts):")
print(pd.to_datetime(df['excel_datetime'], unit='D', origin='1899-12-30').head())

# Show how to create an Excel date from a Python datetime
print("\nConverting Python datetime to Excel date number:")
today = datetime.now()
excel_date = (today - datetime(1899, 12, 30)).days + (today.hour * 3600 + today.minute * 60 + today.second) / 86400
print(f"Today ({today}) as Excel date: {excel_date}")
print(f"Converting back: {pd.to_datetime(excel_date, unit='D', origin='1899-12-30')}")

print("\nExample complete!")