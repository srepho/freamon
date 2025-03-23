"""
Example demonstrating DataTypeDetector functionality with performance optimizations.

This example shows how to:
1. Create a sample dataset with various data types
2. Detect and analyze column types using DataTypeDetector
3. Use optimizations for large datasets
4. Generate a visual report of data types
5. Convert columns to appropriate types
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import time
from freamon.utils.datatype_detector import DataTypeDetector

# Create synthetic dataset with various data types
def create_sample_dataset(n_rows=1000):
    """Create a dataset with various data types for demonstration."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate dates with various formats
    start_date = datetime(2022, 1, 1)
    dates_iso = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_rows)]
    dates_us = [(start_date + timedelta(days=i)).strftime('%m/%d/%Y') for i in range(n_rows)]
    dates_uk = [(start_date + timedelta(days=i)).strftime('%d/%m/%Y') for i in range(n_rows)]
    dates_mixed = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') if i % 3 == 0 else
                   (start_date + timedelta(days=i)).strftime('%m/%d/%Y') if i % 3 == 1 else
                   (start_date + timedelta(days=i)).strftime('%d/%m/%Y') 
                   for i in range(n_rows)]
    
    # Generate month_year formats
    month_years = [(start_date + timedelta(days=30*i)).strftime('%b %Y') for i in range(n_rows // 30 + 1)]
    month_years = month_years[:n_rows]
    
    # Generate Excel dates (numeric days since 1899-12-30)
    excel_dates = [(start_date - datetime(1899, 12, 30)).days + i for i in range(n_rows)]
    excel_dates_with_nan = excel_dates.copy()
    # Add some NaN values
    for i in range(0, n_rows, 10):
        excel_dates_with_nan[i] = np.nan
    
    # Generate Australian postcodes (4 digits, between 0800-7999)
    postcodes = [f"{random.randint(800, 7999):04d}" for _ in range(n_rows)]
    
    # Generate Australian phone numbers
    phone_prefixes = ['02', '03', '04', '07', '08']
    phone_numbers = [f"{random.choice(phone_prefixes)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}" 
                     for _ in range(n_rows)]
    
    # Generate Australian ABNs (11 digits)
    abns = [f"{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}" 
            for _ in range(n_rows)]
    
    # Generate IDs
    ids = [f"ID-{i:06d}" for i in range(n_rows)]
    
    # Generate numeric data (continuous and categorical)
    continuous_data = np.random.normal(100, 25, n_rows)
    categorical_numeric = np.random.choice([1, 2, 3, 4, 5], n_rows)
    
    # Generate scientific notation values
    scientific_values = np.random.uniform(0.000001, 0.1, n_rows)
    scientific_strings = [f"{val:.2e}" for val in scientific_values]
    
    # Generate text data
    text_data = [''.join(random.choices(string.ascii_letters + ' ', k=random.randint(20, 100))) 
                for _ in range(n_rows)]
    
    # Create the DataFrame
    df = pd.DataFrame({
        'id': ids,
        'iso_date': dates_iso,
        'us_date': dates_us,
        'uk_date': dates_uk,
        'mixed_date_formats': dates_mixed,
        'month_year': month_years,
        'excel_date': excel_dates,
        'excel_date_with_nan': excel_dates_with_nan,
        'australian_postcode': postcodes,
        'australian_phone': phone_numbers,
        'australian_abn': abns,
        'continuous_numeric': continuous_data,
        'categorical_numeric': categorical_numeric,
        'scientific_notation': scientific_strings,
        'text': text_data
    })
    
    return df

def main():
    # Generate sample data
    print("Generating sample dataset...")
    df = create_sample_dataset(10000)  # Create a larger dataset to demonstrate optimization
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Display the first few rows
    print("\nSample data:")
    print(df.head(3))
    
    # Without optimization
    print("\nDetecting data types without optimization...")
    start_time = time.time()
    detector = DataTypeDetector(df, optimized=False)
    detector.detect_all_types()
    standard_time = time.time() - start_time
    print(f"Standard detection time: {standard_time:.2f} seconds")
    
    # With optimization
    print("\nDetecting data types with optimization...")
    start_time = time.time()
    detector_optimized = DataTypeDetector(df, optimized=True)
    detector_optimized.detect_all_types()
    optimized_time = time.time() - start_time
    print(f"Optimized detection time: {optimized_time:.2f} seconds")
    print(f"Speed improvement: {(standard_time/optimized_time):.2f}x faster")
    
    # Show detected types
    print("\nDetected column types:")
    for col, type_info in detector_optimized.column_types.items():
        print(f"{col}: {type_info}")
    
    print("\nSemantic types:")
    for col, sem_type in detector_optimized.semantic_types.items():
        if sem_type:
            print(f"{col}: {sem_type}")
    
    # Get conversion suggestions
    print("\nSuggested conversions:")
    conversions = detector_optimized.generate_conversion_suggestions()
    for col, suggestion in conversions.items():
        print(f"{col}: {suggestion}")
    
    # Convert columns based on suggestions
    print("\nConverting columns...")
    converted_df = detector_optimized.convert_types()
    
    # Show resulting datatypes
    print("\nDataframe info after conversion:")
    print(converted_df.dtypes)
    
    # Generate HTML report if in Jupyter
    print("\nTo see a visual report in Jupyter notebook, run:")
    print("detector_optimized.display_detection_report()")
    
    # Save the HTML report
    report_html = detector_optimized.get_column_report_html()
    with open("datatype_detection_report.html", "w") as f:
        f.write(report_html)
    print("\nSaved HTML report to 'datatype_detection_report.html'")

if __name__ == "__main__":
    main()