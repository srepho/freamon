"""
Example demonstrating large dataset handling capabilities in freamon.

This example shows how to:
1. Process a large dataset in chunks
2. Save and load large datasets to/from disk in chunks
3. Perform EDA on large datasets efficiently
"""
import os
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta

from freamon.utils.dataframe_utils import (
    process_in_chunks,
    iterate_chunks,
    save_to_chunks,
    load_from_chunks,
)
from freamon.eda import EDAAnalyzer


def generate_large_dataset(n_rows=500000, n_cols=20):
    """Generate a large sample dataset for demonstration."""
    print(f"Generating dataset with {n_rows} rows and {n_cols} columns...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate numeric columns
    data = {}
    for i in range(n_cols - 5):  # Reserve 5 columns for special types
        data[f'numeric_{i}'] = np.random.randn(n_rows)
    
    # Generate categorical column
    categories = ['A', 'B', 'C', 'D', 'E']
    data['category'] = np.random.choice(categories, size=n_rows)
    
    # Generate datetime column
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    data['date'] = dates
    
    # Generate binary target column
    data['target'] = np.random.binomial(1, 0.3, size=n_rows)
    
    # Generate ID column
    data['id'] = np.arange(1, n_rows + 1)
    
    return pd.DataFrame(data)


def example_processing_in_chunks():
    """Example of processing a large dataset in chunks."""
    print("\n===== Processing Large Dataset in Chunks =====")
    
    # Generate large dataset
    df = generate_large_dataset(n_rows=500000)
    
    # 1. Example: Calculate column means in chunks
    print("\nCalculating column means in chunks...")
    def calculate_means(chunk):
        return chunk.select_dtypes(include=['number']).mean()
    
    def combine_means(means_list):
        return pd.concat(means_list).mean()
    
    means = process_in_chunks(
        df,
        func=calculate_means,
        chunk_size=100000,
        combine_func=combine_means
    )
    print("Column means:", means.to_dict())
    
    # 2. Example: Calculate correlations in chunks
    print("\nCalculating correlations in chunks...")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('id')  # Remove ID column
    
    def calculate_correlation(chunk):
        return chunk[numeric_cols].corr()
    
    # For correlation, we'll just take the first chunk's result for demo
    # In a real scenario, you might want a more sophisticated approach
    corr = process_in_chunks(
        df, 
        func=calculate_correlation,
        chunk_size=100000,
    )[0]
    print("Correlation matrix shape:", corr.shape)
    
    # 3. Example: Using the chunk iterator directly
    print("\nUsing chunk iterator directly...")
    total_sum = 0
    for i, chunk in enumerate(iterate_chunks(df, chunk_size=100000)):
        chunk_sum = chunk.select_dtypes(include=['number']).sum().sum()
        total_sum += chunk_sum
        print(f"Chunk {i+1} - Shape: {chunk.shape}, Sum: {chunk_sum:.2f}")
    
    print(f"Total sum across all chunks: {total_sum:.2f}")


def example_save_load_chunks():
    """Example of saving and loading a large dataset in chunks."""
    print("\n===== Saving and Loading Large Dataset in Chunks =====")
    
    # Generate large dataset
    df = generate_large_dataset(n_rows=300000)
    
    # Create a temporary directory for chunk files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the dataframe in chunks
        print(f"\nSaving dataframe in chunks to {tmp_dir}...")
        chunk_files = save_to_chunks(
            df,
            output_dir=tmp_dir,
            base_filename='large_data',
            chunk_size=50000,
            file_format='parquet'
        )
        print(f"Created {len(chunk_files)} chunk files.")
        
        # Load chunks back into memory
        print("\nLoading chunks back with different options...")
        
        # 1. Load as a single pandas dataframe
        df_pandas = load_from_chunks(
            input_dir=tmp_dir,
            pattern='large_data_*.parquet',
            combine=True,
            output_type='pandas'
        )
        print(f"Loaded as pandas - Shape: {df_pandas.shape}")
        
        # 2. Load as separate chunks
        chunks = load_from_chunks(
            input_dir=tmp_dir,
            pattern='large_data_*.parquet',
            combine=False,
            output_type='pandas'
        )
        print(f"Loaded as separate chunks - Number of chunks: {len(chunks)}")
        print(f"First chunk shape: {chunks[0].shape}")
        
        try:
            # 3. Load as Dask dataframe (if dask is installed)
            import dask.dataframe as dd
            df_dask = load_from_chunks(
                input_dir=tmp_dir,
                pattern='large_data_*.parquet',
                output_type='dask'
            )
            print(f"Loaded as dask - Shape: {df_dask.shape}")
        except ImportError:
            print("Dask not installed, skipping dask example")
        
        try:
            # 4. Load as Polars dataframe (if polars is installed)
            import polars as pl
            df_polars = load_from_chunks(
                input_dir=tmp_dir,
                pattern='large_data_*.parquet',
                output_type='polars'
            )
            print(f"Loaded as polars - Shape: {df_polars.shape}")
        except ImportError:
            print("Polars not installed, skipping polars example")


def example_eda_with_chunks():
    """Example of performing EDA on a large dataset efficiently."""
    print("\n===== Performing EDA on a Large Dataset =====")
    
    # Generate a smaller dataset for this example
    df = generate_large_dataset(n_rows=100000, n_cols=10)
    
    print("\nPerforming EDA analysis...")
    
    # 1. Analyze dataset with EDAAnalyzer
    # This will work well for this size dataset
    analyzer = EDAAnalyzer(df, target_column='target', date_column='date')
    analyzer.analyze_basic_stats()
    
    # 2. Perform univariate analysis
    # This automatically uses the chunk-based processing internally for large datasets
    print("\nPerforming univariate analysis...")
    univariate_results = analyzer.analyze_univariate()
    print(f"Analyzed {len(univariate_results)} columns.")
    
    # 3. Generate summary for one numeric column
    numeric_col = 'numeric_0'
    summary = univariate_results[numeric_col]
    print(f"\nSummary for {numeric_col}:")
    print(f"Mean: {summary['mean']:.4f}")
    print(f"Std Dev: {summary['std']:.4f}")
    print(f"Min: {summary['min']:.4f}")
    print(f"Max: {summary['max']:.4f}")
    
    # 4. Generate summary for categorical column
    cat_col = 'category'
    summary = univariate_results[cat_col]
    print(f"\nSummary for {cat_col}:")
    print(f"Number of unique values: {summary['n_unique']}")
    print(f"Top values: {summary['value_counts']}")
    
    # 5. Generate an HTML report
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
        report_path = tmp_file.name
    
    print(f"\nGenerating EDA report to {report_path}...")
    analyzer.generate_report(
        output_path=report_path,
        title="Large Dataset Analysis Example"
    )
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    example_processing_in_chunks()
    example_save_load_chunks()
    example_eda_with_chunks()
    
    print("\nAll examples completed successfully!")