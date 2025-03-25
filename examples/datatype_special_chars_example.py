"""
Example demonstrating the HTML report generation functionality for DataTypeDetector.

This example shows how to:
1. Create a sample dataset with various data types
2. Detect and analyze data types using DataTypeDetector
3. Generate HTML reports with the new save_html_report method
4. Customize HTML reports using get_column_report_html
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import os
from freamon.utils.datatype_detector import DataTypeDetector

def create_sample_dataset(n_rows=1000):
    """Create a dataset with various data types for demonstration."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate dates with various formats
    start_date = datetime(2022, 1, 1)
    dates_iso = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_rows)]
    dates_us = [(start_date + timedelta(days=i)).strftime('%m/%d/%Y') for i in range(n_rows)]
    
    # Generate month_year formats
    month_years = [(start_date + timedelta(days=30*i)).strftime('%b %Y') for i in range(n_rows // 30 + 1)]
    month_years = month_years[:n_rows]
    
    # Generate Australian postcodes and finance related data with currency symbols
    postcodes = [f"{random.randint(800, 7999):04d}" for _ in range(n_rows)]
    prices = [f"${random.uniform(10, 1000):.2f}" for _ in range(n_rows)]
    percents = [f"{random.uniform(0, 100):.2f}%" for _ in range(n_rows)]
    
    # Generate IDs and emails
    ids = [f"ID-{i:06d}" for i in range(n_rows)]
    emails = [f"user{i}@example.com" for i in range(n_rows)]
    
    # Generate scientific notation values
    scientific_values = np.random.uniform(0.000001, 0.1, n_rows)
    scientific_strings = [f"{val:.2e}" for val in scientific_values]
    
    # Generate null values for some columns
    null_mask = np.random.choice([True, False], n_rows, p=[0.2, 0.8])
    
    # Create the DataFrame with special characters in column names
    df = pd.DataFrame({
        'id': ids,
        'iso_date': dates_iso,
        'us_date': dates_us,
        'month_year': month_years,
        'australian_postcode': postcodes,
        'price_$': prices,  # Special character in column name
        'growth_%': percents,  # Special character in column name
        'email_address': emails,
        'scientific_notation': scientific_strings,
        'with_underscores_in_name': np.random.normal(100, 25, n_rows),
        'with_nulls': np.where(null_mask, np.nan, np.random.randint(1, 100, n_rows)),
        'categorical_values': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'boolean_values': np.random.choice([True, False], n_rows),
    })
    
    return df

def main():
    """Run the example demonstrating HTML report generation."""
    print("DataTypeDetector HTML Report Generation Example")
    print("==============================================")
    
    # Create a sample dataset
    print("\nGenerating sample dataset...")
    df = create_sample_dataset(1000)
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Print the first few rows to show the data
    print("\nSample data:")
    print(df.head(3))
    
    # Create the DataTypeDetector and run detection
    print("\nRunning data type detection...")
    detector = DataTypeDetector(df)
    results = detector.detect_all_types()
    
    # Print a summary of detected types
    print("\nDetected types summary:")
    for col, info in results.items():
        print(f"{col}: {info['logical_type']}", end="")
        if 'semantic_type' in info:
            print(f" ({info['semantic_type']})", end="")
        print()
    
    # Method 1: Save HTML report directly using save_html_report method
    print("\nMethod 1: Using save_html_report() to generate and save HTML report...")
    report_path = detector.save_html_report("datatype_report_example.html", include_stats=True)
    print(f"Report saved to: {report_path}")
    
    # Method 2: Get HTML content and customize it before saving
    print("\nMethod 2: Using get_column_report_html() to customize the report...")
    html_content = detector.get_column_report_html()
    
    # Customize the HTML content (for example, add a custom header)
    custom_report_path = "custom_datatype_report_example.html"
    
    # Here we could modify the HTML content before saving it
    # For demonstration, we'll just save it as is
    with open(custom_report_path, "w") as f:
        f.write(html_content)
    print(f"Custom report saved to: {custom_report_path}")
    
    # Print next steps for the user
    print("\nNext steps:")
    print(f"1. Open {report_path} in your browser to view the report")
    print(f"2. Open {custom_report_path} to view the customizable report")
    print("\nYou can use these methods in your own code to generate reports as follows:")
    print("```python")
    print("from freamon.utils.datatype_detector import DataTypeDetector")
    print("detector = DataTypeDetector(df)")
    print("detector.detect_all_types()")
    print("# Method 1: Direct to file")
    print("detector.save_html_report('my_report.html')")
    print("# Method 2: Get content for customization")
    print("html_content = detector.get_column_report_html()")
    print("# Customize html_content...")
    print("with open('custom_report.html', 'w') as f:")
    print("    f.write(html_content)")
    print("```")

if __name__ == "__main__":
    main()