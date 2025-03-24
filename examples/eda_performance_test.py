"""
Performance testing script for the enhanced EDA reporting features.

This script benchmarks the following:
1. Report generation time with and without lazy loading
2. Report file size comparison
3. Browser load time estimations for different dataset sizes
4. Memory usage during report generation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import psutil
import gc
from IPython.display import display, HTML

# Import from freamon
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches

# Apply matplotlib patches
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
OUTPUT_DIR = Path("eda_performance_reports")
DATASET_SIZES = [1000, 10000, 100000, 500000]  # Number of rows to test


def generate_test_dataset(size):
    """Generate a synthetic dataset with the specified number of rows."""
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


def get_memory_usage():
    """Get current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_benchmark(df_size):
    """Run benchmarks for a specific dataset size."""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING DATASET SIZE: {df_size:,} rows")
    print(f"{'='*80}")
    
    # Generate dataset
    start_time = time.time()
    df = generate_test_dataset(df_size)
    gen_time = time.time() - start_time
    print(f"Dataset generation time: {gen_time:.2f} seconds")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Create file paths
    standard_path = OUTPUT_DIR / f"report_standard_{df_size}.html"
    enhanced_path = OUTPUT_DIR / f"report_enhanced_{df_size}.html"
    
    # Benchmark 1: Standard report (no lazy loading, no export button)
    gc.collect()  # Force garbage collection before test
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    analyzer = EDAAnalyzer(
        df,
        date_column='date',
        categorical_columns=['category1', 'category2'],
        target_column='binary'
    )
    
    analyzer.run_full_analysis(
        output_path=str(standard_path),
        title=f"Standard Report ({df_size:,} rows)",
        include_multivariate=True,
        lazy_loading=False,
        include_export_button=False,
    )
    
    standard_time = time.time() - start_time
    peak_memory_standard = get_memory_usage() - initial_memory
    standard_size = os.path.getsize(standard_path) / 1024 / 1024  # MB
    
    # Benchmark 2: Enhanced report (with lazy loading and export button)
    gc.collect()  # Force garbage collection before test
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    analyzer = EDAAnalyzer(
        df,
        date_column='date',
        categorical_columns=['category1', 'category2'],
        target_column='binary'
    )
    
    analyzer.run_full_analysis(
        output_path=str(enhanced_path),
        title=f"Enhanced Report ({df_size:,} rows)",
        include_multivariate=True,
        lazy_loading=True,
        include_export_button=True,
    )
    
    enhanced_time = time.time() - start_time
    peak_memory_enhanced = get_memory_usage() - initial_memory
    enhanced_size = os.path.getsize(enhanced_path) / 1024 / 1024  # MB
    
    # Results
    results = {
        'dataset_size': df_size,
        'standard_time': standard_time,
        'enhanced_time': enhanced_time,
        'standard_size': standard_size,
        'enhanced_size': enhanced_size,
        'peak_memory_standard': peak_memory_standard,
        'peak_memory_enhanced': peak_memory_enhanced,
        'time_diff_pct': (enhanced_time - standard_time) / standard_time * 100,
        'size_diff_pct': (enhanced_size - standard_size) / standard_size * 100,
    }
    
    print("\nResults:")
    print(f"  Standard report generation time: {standard_time:.2f} seconds")
    print(f"  Enhanced report generation time: {enhanced_time:.2f} seconds")
    print(f"  Time difference: {results['time_diff_pct']:.2f}%")
    print(f"  Standard report size: {standard_size:.2f} MB")
    print(f"  Enhanced report size: {enhanced_size:.2f} MB")
    print(f"  Size difference: {results['size_diff_pct']:.2f}%")
    print(f"  Peak memory usage (standard): {peak_memory_standard:.2f} MB")
    print(f"  Peak memory usage (enhanced): {peak_memory_enhanced:.2f} MB")
    
    # Clean up to free memory
    del df, analyzer
    gc.collect()
    
    return results


def main():
    """Run the benchmark suite."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("Starting EDA reporting performance benchmarks...")
    
    all_results = []
    
    for size in DATASET_SIZES:
        try:
            result = run_benchmark(size)
            all_results.append(result)
        except Exception as e:
            print(f"Error with dataset size {size}: {str(e)}")
    
    # Create summary report
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
    
    print("\nSummary:")
    display(results_df)
    
    # Create a simple plot of the results
    plt.figure(figsize=(12, 8))
    
    # Plot generation time comparison
    plt.subplot(2, 2, 1)
    plt.plot(results_df['dataset_size'], results_df['standard_time'], marker='o', label='Standard')
    plt.plot(results_df['dataset_size'], results_df['enhanced_time'], marker='x', label='Enhanced')
    plt.title('Report Generation Time')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    # Plot file size comparison
    plt.subplot(2, 2, 2)
    plt.plot(results_df['dataset_size'], results_df['standard_size'], marker='o', label='Standard')
    plt.plot(results_df['dataset_size'], results_df['enhanced_size'], marker='x', label='Enhanced')
    plt.title('Report File Size')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Size (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot memory usage
    plt.subplot(2, 2, 3)
    plt.plot(results_df['dataset_size'], results_df['peak_memory_standard'], marker='o', label='Standard')
    plt.plot(results_df['dataset_size'], results_df['peak_memory_enhanced'], marker='x', label='Enhanced')
    plt.title('Peak Memory Usage')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot generation time ratio
    plt.subplot(2, 2, 4)
    plt.bar(range(len(results_df)), results_df['time_diff_pct'])
    plt.xticks(range(len(results_df)), [f"{size:,}" for size in results_df['dataset_size']])
    plt.title('Time Difference (%)')
    plt.xlabel('Dataset Size (rows)')
    plt.ylabel('Enhanced vs Standard (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_results.png")
    
    print(f"Benchmark complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()