"""
Example usage of the freamon package.
"""
import numpy as np
import pandas as pd

from freamon.data_quality import DataQualityAnalyzer, handle_missing_values, detect_outliers
from freamon.utils import optimize_dtypes, estimate_memory_usage


def generate_sample_data(n_rows=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Create a dataframe with some typical issues
    df = pd.DataFrame({
        "id": range(n_rows),
        "numeric_clean": np.random.normal(0, 1, n_rows),
        "numeric_with_missing": np.random.normal(0, 1, n_rows),
        "numeric_with_outliers": np.random.normal(0, 1, n_rows),
        "categorical": np.random.choice(["A", "B", "C", "D"], n_rows),
        "binary": np.random.choice([0, 1], n_rows),
        "date": pd.date_range(start="2020-01-01", periods=n_rows),
    })
    
    # Add missing values
    missing_idx = np.random.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    df.loc[missing_idx, "numeric_with_missing"] = np.nan
    
    # Add outliers
    outlier_idx = np.random.choice(n_rows, size=10, replace=False)
    df.loc[outlier_idx, "numeric_with_outliers"] = np.random.normal(10, 1, len(outlier_idx))
    
    # Add some mixed type data
    df["mixed_types"] = df["categorical"].copy()
    mixed_idx = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    df.loc[mixed_idx, "mixed_types"] = np.random.randint(0, 100, len(mixed_idx))
    
    return df


def main():
    """Main function demonstrating freamon usage."""
    print("Generating sample data...")
    df = generate_sample_data()
    print(f"Data shape: {df.shape}")
    print(df.head())
    
    # Analyze data quality
    print("\n1. Analyzing data quality...")
    analyzer = DataQualityAnalyzer(df)
    
    # Missing values analysis
    missing_analysis = analyzer.analyze_missing_values()
    print(f"\nMissing values summary:")
    print(f"Total missing: {missing_analysis['total_missing']}")
    print(f"Columns with missing values:")
    for col, count in missing_analysis['missing_count'].items():
        if count > 0:
            pct = missing_analysis['missing_percent'][col]
            print(f"  - {col}: {count} values ({pct:.2f}%)")
    
    # Data types analysis
    type_analysis = analyzer.analyze_data_types()
    print(f"\nData types summary:")
    for col, dtype in type_analysis['dtypes'].items():
        consistency = type_analysis['type_consistency'][col]
        if 'consistent' in consistency and not consistency['consistent']:
            print(f"  - {col}: {dtype} (INCONSISTENT - {consistency['types']})")
        else:
            print(f"  - {col}: {dtype}")
    
    # Handle missing values
    print("\n2. Handling missing values...")
    df_clean = handle_missing_values(df, strategy="mean")
    print(f"Missing values before: {df.isna().sum().sum()}")
    print(f"Missing values after: {df_clean.isna().sum().sum()}")
    
    # Detect outliers
    print("\n3. Detecting outliers...")
    outlier_masks = detect_outliers(df_clean, method="iqr", return_mask=True)
    outlier_counts = {col: mask.sum() for col, mask in outlier_masks.items()}
    print("Outlier counts by column:")
    for col, count in outlier_counts.items():
        if count > 0:
            print(f"  - {col}: {count} outliers")
    
    # Optimize memory usage
    print("\n4. Optimizing memory usage...")
    memory_before = estimate_memory_usage(df_clean)
    df_optimized = optimize_dtypes(df_clean)
    memory_after = estimate_memory_usage(df_optimized)
    
    print(f"Memory usage before: {memory_before['total_mb']:.2f} MB")
    print(f"Memory usage after: {memory_after['total_mb']:.2f} MB")
    print(f"Memory savings: {memory_before['total_mb'] - memory_after['total_mb']:.2f} MB "
          f"({(1 - memory_after['total_mb'] / memory_before['total_mb']) * 100:.2f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()