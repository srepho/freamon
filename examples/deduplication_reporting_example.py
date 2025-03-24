"""
Example demonstrating integrated deduplication and EDA reporting functionality.

This script showcases:
1. Data quality analysis with duplicate detection
2. Different deduplication methods with comparison
3. Enhanced EDA reports before and after deduplication
4. Visualization of the impact of deduplication on data analysis
5. Optimized reporting of large datasets with deduplication insights
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import string
import random
from pathlib import Path
from datetime import datetime, timedelta

# Import from freamon
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.text_utils import TextProcessor
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches
from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.deduplication import (
    hash_deduplication,
    ngram_fingerprint_deduplication,
    deduplicate_texts
)

# Apply matplotlib patches to handle currency symbols properly
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_structured_dataset_with_duplicates(size=10000, duplicate_rate=0.15):
    """
    Generate a structured dataset with various column types and duplicates.
    
    Parameters
    ----------
    size : int, default=10000
        Number of unique rows to generate
    duplicate_rate : float, default=0.15
        Percentage of duplicates to add
    
    Returns
    -------
    pd.DataFrame
        DataFrame with mixed column types and duplicated rows
    """
    print(f"Generating dataset with {size} unique rows...")
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    date_range = [start_date + timedelta(days=i % 365) for i in range(size)]
    
    # Generate unique data
    unique_data = pd.DataFrame({
        'id': range(1, size + 1),
        'date': date_range,
        'numeric1': np.random.normal(100, 15, size),
        'numeric2': np.random.exponential(10, size),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
        'amount': [f"${x:.2f}" for x in np.random.uniform(10, 1000, size)],
        'text': [generate_sample_text(random.randint(50, 200)) for _ in range(size)]
    })
    
    # Create target column based on numeric1
    unique_data['target'] = np.where(unique_data['numeric1'] > 100, 1, 0)
    
    # Add duplicates
    duplicate_size = int(size * duplicate_rate)
    print(f"Adding {duplicate_size} duplicated rows...")
    
    # Select random rows to duplicate
    duplicate_indices = np.random.choice(range(size), duplicate_size, replace=False)
    duplicated_rows = unique_data.iloc[duplicate_indices].copy()
    
    # Combine unique and duplicated data
    full_data = pd.concat([unique_data, duplicated_rows], ignore_index=True)
    
    # Shuffle rows
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return full_data

def generate_sample_text(length=100):
    """Generate a random text string with specified length."""
    words = [
        "data", "analysis", "machine", "learning", "model", "algorithm", "feature",
        "training", "prediction", "dataset", "variable", "function", "parameter",
        "optimization", "accuracy", "precision", "recall", "validation", "testing",
        "preprocessing", "visualization", "correlation", "regression", "classification",
        "clustering", "dimensionality", "reduction", "outlier", "normalization",
        "transformation", "experiment", "hypothesis", "statistical", "significance",
        "distribution", "probability", "sampling", "random", "variance", "covariance",
        "matrix", "vector", "tensor", "gradient", "computation", "evaluation", "metric"
    ]
    
    # Generate random text from these words
    text_words = [random.choice(words) for _ in range(length // 5)]  # Approx 5 chars per word
    return " ".join(text_words)

def analyze_duplicates(df):
    """
    Analyze duplicates in the dataset using different column combinations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    
    Returns
    -------
    dict
        Dictionary with duplicate analysis results
    """
    print("\n===== Analyzing Duplicates =====")
    
    analyses = {
        "all_columns": detect_duplicates(df, return_counts=True),
        "numeric_only": detect_duplicates(df, subset=['numeric1', 'numeric2'], return_counts=True),
        "category_only": detect_duplicates(df, subset=['category'], return_counts=True),
        "text_only": detect_duplicates(df, subset=['text'], return_counts=True)
    }
    
    # Print summary of duplicate analyses
    for analysis_name, result in analyses.items():
        print(f"\n{analysis_name.replace('_', ' ').title()}:")
        print(f"  Duplicates found: {result['duplicate_count']}")
        print(f"  Duplicate percentage: {result['duplicate_percentage']:.2f}%")
        
        # Show top duplicated values if available and not too many
        if 'value_counts' in result and len(result['value_counts']) > 0:
            print("  Top duplicated values:")
            for i, value_info in enumerate(result['value_counts'][:3]):  # Show top 3
                print(f"    - Count: {value_info['count']}")
    
    return analyses

def compare_deduplication_methods(df):
    """
    Compare different deduplication methods on the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to deduplicate
    
    Returns
    -------
    dict
        Dictionary with deduplicated DataFrames and performance metrics
    """
    print("\n===== Comparing Deduplication Methods =====")
    
    results = {}
    
    # Method 1: Simple drop_duplicates (all columns)
    start_time = time.time()
    df_dedup_all = remove_duplicates(df)
    time_all = time.time() - start_time
    
    results["all_columns"] = {
        "df": df_dedup_all,
        "time": time_all,
        "removed": len(df) - len(df_dedup_all),
        "percentage": (len(df) - len(df_dedup_all)) / len(df) * 100
    }
    
    print(f"\nFull row deduplication:")
    print(f"  Time: {time_all:.4f} seconds")
    print(f"  Rows before: {len(df)}, after: {len(df_dedup_all)}")
    print(f"  Removed: {results['all_columns']['removed']} rows ({results['all_columns']['percentage']:.2f}%)")
    
    # Method 2: Text-based deduplication using hash
    start_time = time.time()
    unique_indices = hash_deduplication(df['text'], keep='first')
    df_dedup_text = df.iloc[unique_indices].copy()
    time_text = time.time() - start_time
    
    results["text_hash"] = {
        "df": df_dedup_text,
        "time": time_text,
        "removed": len(df) - len(df_dedup_text),
        "percentage": (len(df) - len(df_dedup_text)) / len(df) * 100
    }
    
    print(f"\nText hash deduplication:")
    print(f"  Time: {time_text:.4f} seconds")
    print(f"  Rows before: {len(df)}, after: {len(df_dedup_text)}")
    print(f"  Removed: {results['text_hash']['removed']} rows ({results['text_hash']['percentage']:.2f}%)")
    
    # Method 3: Text-based deduplication using similarity
    try:
        start_time = time.time()
        unique_indices = deduplicate_texts(df['text'], threshold=0.9, method='cosine')
        df_dedup_sim = df.iloc[unique_indices].copy()
        time_sim = time.time() - start_time
        
        results["text_similarity"] = {
            "df": df_dedup_sim,
            "time": time_sim,
            "removed": len(df) - len(df_dedup_sim),
            "percentage": (len(df) - len(df_dedup_sim)) / len(df) * 100
        }
        
        print(f"\nText similarity deduplication:")
        print(f"  Time: {time_sim:.4f} seconds")
        print(f"  Rows before: {len(df)}, after: {len(df_dedup_sim)}")
        print(f"  Removed: {results['text_similarity']['removed']} rows ({results['text_similarity']['percentage']:.2f}%)")
    except Exception as e:
        print(f"\nText similarity deduplication error: {str(e)}")
    
    # Visualize results
    plot_deduplication_comparison(results)
    
    return results

def plot_deduplication_comparison(results):
    """
    Plot comparison of deduplication methods.
    
    Parameters
    ----------
    results : dict
        Dictionary with deduplication results
    """
    # Create output directory if it doesn't exist
    output_dir = Path("dedup_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Setup the plots
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Execution time
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    times = [results[method]["time"] for method in methods]
    plt.bar(methods, times, color=['blue', 'green', 'orange'][:len(methods)])
    plt.title('Deduplication Time (seconds)')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    
    # Plot 2: Rows removed
    plt.subplot(1, 2, 2)
    percentages = [results[method]["percentage"] for method in methods]
    plt.bar(methods, percentages, color=['blue', 'green', 'orange'][:len(methods)])
    plt.title('Rows Removed (%)')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'deduplication_comparison.png')
    print("\nDeduplication comparison chart saved to 'dedup_reports/deduplication_comparison.png'")

def generate_reports(original_df, dedup_df):
    """
    Generate EDA reports for original and deduplicated datasets.
    
    Parameters
    ----------
    original_df : pd.DataFrame
        Original DataFrame with duplicates
    dedup_df : pd.DataFrame
        Deduplicated DataFrame
    """
    print("\n===== Generating EDA Reports =====")
    
    # Create output directory
    output_dir = Path("dedup_reports")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Report for original data
    print("\nGenerating report for original data...")
    analyzer_original = EDAAnalyzer(
        original_df, 
        target_column='target',
        date_column='date'
    )
    
    analyzer_original.run_full_analysis(
        output_path=str(output_dir / "original_data_report.html"),
        title="Original Data Analysis (With Duplicates)",
        include_multivariate=True,
        include_feature_importance=True,
        lazy_loading=True,
        include_export_button=True
    )
    
    # 2. Report for deduplicated data
    print("\nGenerating report for deduplicated data...")
    analyzer_dedup = EDAAnalyzer(
        dedup_df, 
        target_column='target',
        date_column='date'
    )
    
    analyzer_dedup.run_full_analysis(
        output_path=str(output_dir / "deduplicated_data_report.html"),
        title="Deduplicated Data Analysis",
        include_multivariate=True,
        include_feature_importance=True,
        lazy_loading=True,
        include_export_button=True
    )
    
    print("\nReports generated successfully!")
    print("Check the 'dedup_reports' directory for HTML files:")
    print("  - original_data_report.html")
    print("  - deduplicated_data_report.html")

def main():
    """Run the example."""
    print("Starting deduplication and reporting example...")
    
    # Generate dataset with duplicates
    df = generate_structured_dataset_with_duplicates(size=5000, duplicate_rate=0.15)
    
    # Analyze duplicates in the dataset
    duplicate_analysis = analyze_duplicates(df)
    
    # Compare different deduplication methods
    dedup_results = compare_deduplication_methods(df)
    
    # Generate EDA reports for original and deduplicated data
    generate_reports(df, dedup_results['all_columns']['df'])
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()