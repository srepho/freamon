"""
Example script demonstrating the enhanced EDA reporting with large datasets and Jupyter notebook export.

This script showcases:
1. Lazy loading for images in HTML reports
2. Smart data table rendering (showing first/last rows with count indicators)
3. Export to Jupyter notebook functionality
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import from freamon
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches

# Apply matplotlib patches to handle currency symbols properly
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)

def generate_large_dataset(size=50000):
    """Generate a synthetic large dataset for demonstration."""
    print(f"Generating synthetic dataset with {size} rows...")
    
    # Create a large DataFrame with various column types
    dates = pd.date_range(start='2020-01-01', periods=size)
    
    df = pd.DataFrame({
        'date': dates,
        'id': range(1, size + 1),
        'numeric1': np.random.normal(100, 15, size),
        'numeric2': np.random.exponential(10, size),
        'numeric3': np.random.uniform(0, 1000, size),
        'price': [f"${x:.2f}" for x in np.random.uniform(10, 1000, size)],
        'category1': np.random.choice(['A', 'B', 'C', 'D'], size),
        'category2': np.random.choice(['Low', 'Medium', 'High'], size, p=[0.6, 0.3, 0.1]),
        'binary': np.random.choice([0, 1], size, p=[0.7, 0.3]),
    })
    
    # Add some missing values
    for col in ['numeric1', 'numeric2', 'price', 'category1']:
        mask = np.random.choice([True, False], size, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    # Add some correlations
    df['correlated'] = df['numeric1'] * 0.7 + np.random.normal(0, 5, size)
    
    return df

def main():
    """Run the example."""
    # Generate a large dataset
    output_dir = Path("eda_reports")
    output_dir.mkdir(exist_ok=True)
    
    df = generate_large_dataset(size=50000)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Create an EDA analyzer
    print("Creating EDA analyzer...")
    analyzer = EDAAnalyzer(
        df,
        date_column='date',
        categorical_columns=['category1', 'category2'],
        target_column='binary'
    )
    
    # Run a full analysis with the enhanced reporting options
    print("Running analysis with standard settings...")
    analyzer.run_full_analysis(
        output_path=str(output_dir / "large_dataset_report_standard.html"),
        title="Large Dataset Analysis (Standard)",
        include_multivariate=True,
        lazy_loading=False,
        include_export_button=False,
    )
    
    # Run another analysis with all enhancements enabled
    print("Running analysis with all enhancements...")
    analyzer.run_full_analysis(
        output_path=str(output_dir / "large_dataset_report_enhanced.html"),
        title="Large Dataset Analysis (Enhanced)",
        include_multivariate=True,
        lazy_loading=True,
        include_export_button=True,
    )
    
    print("EDA reports generated! Check the eda_reports directory for the HTML files.")
    print("- Open large_dataset_report_standard.html for the standard report")
    print("- Open large_dataset_report_enhanced.html for the enhanced report with lazy loading and Jupyter export")

if __name__ == "__main__":
    main()