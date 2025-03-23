import pandas as pd
import numpy as np
import time
from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types

"""
Test script to measure performance improvements in the optimized DataTypeDetector
"""

def create_large_test_df(rows=100000, mixed_excel_dates=True):
    """Create a large test dataframe with various data types"""
    np.random.seed(42)  # For reproducibility
    
    # Create a dictionary to hold column data
    data = {
        'id': np.arange(1, rows+1),
        'float_col': np.random.normal(100, 15, rows),
        'int_col': np.random.randint(1, 1000, rows),
        'categorical_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
        'low_cardinality_int': np.random.choice(range(1, 11), rows),
        'string_col': [f'Item-{i:05d}' for i in range(rows)],
        'date_str': [f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}' for _ in range(rows)]
    }
    
    # Add Excel dates if requested
    if mixed_excel_dates:
        # Create Excel dates with ~30% missing values
        excel_base_date = 43831  # 2020-01-01
        date_range = 1095        # ~3 years in days
        
        dates = []
        for i in range(rows):
            # 30% chance of missing value
            if np.random.random() < 0.3:
                dates.append(np.nan)
            else:
                # Random date in the range
                date_val = excel_base_date + np.random.randint(0, date_range)
                dates.append(date_val)
                
        data['excel_date'] = dates
    
    # Create DataFrame
    return pd.DataFrame(data)

def time_function(func, *args, **kwargs):
    """Time a function call and return the result and elapsed time"""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

# Create a dataframe for testing
print(f"Creating test dataframe...")
df = create_large_test_df(rows=100000)
print(f"Created dataframe with {len(df)} rows and {len(df.columns)} columns")
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"DataFrame dtypes:\n{df.dtypes}")

# Test standard DataTypeDetector
print("\n1. Testing standard DataTypeDetector (without PyArrow)...")
detector = DataTypeDetector(df)
results, time_detection = time_function(detector.detect_all_types, use_pyarrow=False)
print(f"Detection time: {time_detection:.2f} seconds")

converted_df, time_conversion = time_function(detector.convert_types, use_pyarrow=False)
print(f"Conversion time: {time_conversion:.2f} seconds")
print(f"Total time (detection + conversion): {time_detection + time_conversion:.2f} seconds")
print(f"Converted DataFrame memory usage: {converted_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"Converted DataFrame dtypes:\n{converted_df.dtypes}")

# Test optimized DataTypeDetector
print("\n2. Testing optimized DataTypeDetector (with PyArrow)...")
detector = DataTypeDetector(df)
results, time_detection = time_function(detector.detect_all_types, use_pyarrow=True)
print(f"Detection time with PyArrow: {time_detection:.2f} seconds")

converted_df, time_conversion = time_function(detector.convert_types, use_pyarrow=True)
print(f"Conversion time with PyArrow: {time_conversion:.2f} seconds")
print(f"Total time (detection + conversion): {time_detection + time_conversion:.2f} seconds")
print(f"Converted DataFrame memory usage: {converted_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"Converted DataFrame dtypes:\n{converted_df.dtypes}")

# Test optimize_dataframe_types function
print("\n3. Testing optimize_dataframe_types function...")
optimized_df, time_optimize = time_function(optimize_dataframe_types, df, use_pyarrow=True)
print(f"Optimization time with optimize_dataframe_types: {time_optimize:.2f} seconds")
print(f"Optimized DataFrame memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
print(f"Optimized DataFrame dtypes:\n{optimized_df.dtypes}")

# Test Excel date fix with missing values
print("\n4. Testing Excel date fix with missing values...")
excel_dates = df['excel_date'].head(10)
print(f"Sample Excel dates:\n{excel_dates}")

excel_converted, time_excel = time_function(lambda: pd.to_datetime(
    pd.to_numeric(excel_dates, errors='coerce'), 
    unit='D', 
    origin='1899-12-30', 
    errors='coerce'
))
print(f"Excel date conversion time: {time_excel:.5f} seconds")
print(f"Converted dates:\n{excel_converted}")