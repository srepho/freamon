"""
Example script demonstrating a test suite for the enhanced EDA reporting functionality.

This script tests the report generation with various types of data:
1. Numeric data (different distributions)
2. Categorical data (different cardinalities)
3. Date/time data (different formats and frequencies)
4. Text data
5. Currency data with special characters
6. Missing data scenarios
7. Various column names (including spaces, special characters)
8. Different sized datasets
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import string
import re
import random

# Import from freamon
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches

# Apply matplotlib patches to handle currency symbols properly
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)

def generate_test_dataset(size=1000):
    """Generate a dataset with various data types and edge cases."""
    print(f"Generating test dataset with {size} rows...")
    
    # Basic numeric columns with different distributions
    data = {
        # Numeric columns with different distributions
        'normal_dist': np.random.normal(100, 15, size),
        'exponential_dist': np.random.exponential(10, size),
        'uniform_dist': np.random.uniform(0, 1000, size),
        'lognormal_dist': np.random.lognormal(0, 1, size),
        'integers': np.random.randint(1, 100, size),
        
        # Categorical columns with different cardinalities
        'low_cardinality': np.random.choice(['A', 'B', 'C'], size, p=[0.7, 0.2, 0.1]),
        'medium_cardinality': np.random.choice(list(string.ascii_uppercase[:10]), size),
        'high_cardinality': [f"CAT_{i % 50}" for i in range(size)],
        'boolean': np.random.choice([True, False], size),
        
        # Column with special characters in name
        'price_$_€': [f"${x:.2f}" for x in np.random.uniform(10, 1000, size)],
        'percent%': [f"{x:.1f}%" for x in np.random.uniform(0, 100, size)],
        
        # Column with spaces in name
        'column with spaces': np.random.normal(50, 10, size),
    }
    
    # Date columns with different formats
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i % 365) for i in range(size)]
    data['date_standard'] = dates
    
    # Month-year format
    data['month_year'] = [d.strftime('%b %Y') for d in dates]
    
    # Text data column
    sentences = [
        "This is a test sentence with normal text.",
        "Here's another example with some punctuation!",
        "Numbers like 123 and 456 should be handled properly.",
        "Special characters like $, %, and € should render correctly.",
        "Some longer text that goes into more detail about a particular topic with multiple clauses.",
        "Very short text.",
        "This column tests how the report handles text data and wrapping of longer content in tables and visualizations."
    ]
    data['text_data'] = [random.choice(sentences) for _ in range(size)]
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in df.columns:
        if col not in ['date_standard']:  # Keep date column complete for time series
            mask = np.random.choice([True, False], size, p=[0.05, 0.95])
            df.loc[mask, col] = np.nan
    
    # Add some correlations
    df['correlated_positive'] = df['normal_dist'] * 0.7 + np.random.normal(0, 5, size)
    df['correlated_negative'] = df['normal_dist'] * -0.7 + np.random.normal(0, 5, size)
    
    # Add a target column
    df['target'] = np.where(df['normal_dist'] > 100, 1, 0)
    
    return df

def test_report_with_different_sizes():
    """Generate and test reports with different dataset sizes."""
    output_dir = Path("eda_test_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Test different sizes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        print(f"\nTesting with dataset size: {size}")
        df = generate_test_dataset(size=size)
        
        # Create an EDA analyzer
        analyzer = EDAAnalyzer(
            df,
            date_column='date_standard',
            target_column='target'
        )
        
        # Run analysis with all features enabled
        analyzer.run_full_analysis(
            output_path=str(output_dir / f"report_size_{size}.html"),
            title=f"Test Report (Size: {size})",
            include_multivariate=True,
            include_feature_importance=True,
            include_time_series=True,
            lazy_loading=True,
            include_export_button=True,
        )
        
        print(f"Report generated for size {size}")

def test_report_with_column_subsets():
    """Test reports with different subsets of columns."""
    output_dir = Path("eda_test_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a dataset
    df = generate_test_dataset(size=1000)
    
    # Test cases
    test_cases = [
        {
            'name': 'numeric_only',
            'columns': ['normal_dist', 'exponential_dist', 'uniform_dist', 'lognormal_dist', 'integers', 'target'],
            'title': 'Numeric Columns Only'
        },
        {
            'name': 'categorical_only',
            'columns': ['low_cardinality', 'medium_cardinality', 'high_cardinality', 'boolean', 'target'],
            'title': 'Categorical Columns Only'
        },
        {
            'name': 'datetime_only',
            'columns': ['date_standard', 'month_year', 'target'],
            'title': 'Date/Time Columns Only'
        },
        {
            'name': 'special_chars',
            'columns': ['price_$_€', 'percent%', 'column with spaces', 'target'],
            'title': 'Columns with Special Characters'
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting with column subset: {case['name']}")
        
        # Create a subset dataframe
        df_subset = df[case['columns']].copy()
        
        # Create an EDA analyzer
        analyzer = EDAAnalyzer(
            df_subset,
            date_column='date_standard' if 'date_standard' in case['columns'] else None,
            target_column='target'
        )
        
        # Run analysis
        analyzer.run_full_analysis(
            output_path=str(output_dir / f"report_{case['name']}.html"),
            title=f"Test Report - {case['title']}",
            include_multivariate=True,
            lazy_loading=True,
            include_export_button=True,
        )
        
        print(f"Report generated for {case['name']}")

def test_report_with_edge_cases():
    """Test reports with various edge cases."""
    output_dir = Path("eda_test_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Create a base dataset
    base_df = generate_test_dataset(size=500)
    
    # 1. Test with high percentage of missing values
    print("\nTesting with high percentage of missing values")
    df_missing = base_df.copy()
    
    # Add more missing values (50% missing)
    for col in df_missing.columns:
        if col != 'target':  # Keep target complete for analysis
            mask = np.random.choice([True, False], len(df_missing), p=[0.5, 0.5])
            df_missing.loc[mask, col] = np.nan
    
    analyzer = EDAAnalyzer(df_missing, target_column='target')
    analyzer.run_full_analysis(
        output_path=str(output_dir / "report_high_missing.html"),
        title="Test Report - High Missing Values",
        lazy_loading=True,
    )
    
    # 2. Test with constant columns
    print("\nTesting with constant columns")
    df_constant = base_df.copy()
    
    # Add constant columns
    df_constant['constant_numeric'] = 5
    df_constant['constant_string'] = "same_value"
    
    analyzer = EDAAnalyzer(df_constant, target_column='target')
    analyzer.run_full_analysis(
        output_path=str(output_dir / "report_constant_columns.html"),
        title="Test Report - Constant Columns",
        lazy_loading=True,
    )
    
    # 3. Test with extremely high cardinality
    print("\nTesting with extremely high cardinality")
    df_high_card = base_df.copy()
    
    # Add a unique ID column (100% cardinality)
    df_high_card['unique_id'] = [f"ID_{i}" for i in range(len(df_high_card))]
    
    analyzer = EDAAnalyzer(df_high_card, target_column='target')
    analyzer.run_full_analysis(
        output_path=str(output_dir / "report_high_cardinality.html"),
        title="Test Report - High Cardinality",
        lazy_loading=True,
    )

def main():
    """Run all test cases."""
    print("Starting EDA reporting test suite...")
    
    test_report_with_different_sizes()
    test_report_with_column_subsets()
    test_report_with_edge_cases()
    
    print("\nAll test reports generated successfully!")
    print("Check the 'eda_test_reports' directory for the HTML files.")

if __name__ == "__main__":
    main()