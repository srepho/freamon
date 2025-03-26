"""
Enhanced deduplication reporting example.

This example demonstrates the enhanced reporting capabilities for deduplication,
including HTML, Excel, PowerPoint, and Jupyter report generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime, timedelta
import random

# Import from freamon
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches
from freamon.deduplication import flag_similar_records
from freamon.deduplication.evaluation import flag_and_evaluate
from freamon.deduplication.enhanced_reporting import (
    EnhancedDeduplicationReport,
    generate_enhanced_report,
    display_jupyter_report
)

# Apply matplotlib patches for better visualization
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_dataset_with_duplicates(size=1000, duplicate_rate=0.2):
    """
    Generate a structured dataset with known duplicates for testing.
    
    Parameters
    ----------
    size : int, default=1000
        Number of unique records to generate
    duplicate_rate : float, default=0.2
        Percentage of duplicates to add
        
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates and ground truth flags
    """
    print(f"Generating dataset with {size} unique records...")
    
    # Generate unique data
    names = [f"Person {i}" for i in range(size)]
    emails = [f"person{i}@example.com" for i in range(size)]
    phones = [f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}" for _ in range(size)]
    addresses = [f"{random.randint(1, 999)} Main St, City {random.randint(1, 50)}" for _ in range(size)]
    
    # Create DataFrame with unique records
    df = pd.DataFrame({
        'name': names,
        'email': emails,
        'phone': phones,
        'address': addresses,
        'age': np.random.randint(18, 80, size),
        'income': np.random.uniform(20000, 200000, size),
        'signup_date': [datetime(2020, 1, 1) + timedelta(days=i % 365) for i in range(size)]
    })
    
    # Mark all original records as not duplicates
    df['is_duplicate'] = False
    
    # Create duplicates with variations
    duplicate_size = int(size * duplicate_rate)
    print(f"Adding {duplicate_size} duplicates with variations...")
    
    duplicates = []
    for i in range(duplicate_size):
        # Randomly select a record to duplicate
        idx = random.randint(0, size - 1)
        duplicate = df.iloc[idx].copy()
        
        # Introduce random variations to simulate real-world errors
        duplicate_type = random.randint(1, 3)
        
        if duplicate_type == 1:
            # Name variation
            name_parts = duplicate['name'].split()
            duplicate['name'] = f"{name_parts[1]}, {name_parts[0]}"
        elif duplicate_type == 2:
            # Email typo
            email_parts = duplicate['email'].split('@')
            if len(email_parts[0]) > 3:
                # Swap two adjacent characters
                pos = random.randint(0, len(email_parts[0]) - 2)
                chars = list(email_parts[0])
                chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
                duplicate['email'] = ''.join(chars) + '@' + email_parts[1]
        else:
            # Phone format variation
            phone = duplicate['phone'].replace('-', '')
            duplicate['phone'] = f"({phone[2:5]}) {phone[5:8]}-{phone[8:]}"
        
        # Mark as a duplicate
        duplicate['is_duplicate'] = True
        
        duplicates.append(duplicate)
    
    # Add duplicates to the dataset
    duplicate_df = pd.DataFrame(duplicates)
    full_df = pd.concat([df, duplicate_df], ignore_index=True)
    
    # Shuffle the dataset
    return full_df.sample(frac=1).reset_index(drop=True)


def compare_deduplicated_original(df, deduplicated_df):
    """Compare original and deduplicated datasets with visualization."""
    # Create output directory
    output_dir = Path("dedup_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Record counts
    plt.subplot(2, 2, 1)
    counts = [len(df), len(deduplicated_df)]
    plt.bar(['Original', 'Deduplicated'], counts, color=['#3498db', '#2ecc71'])
    plt.title('Record Counts')
    plt.ylabel('Number of Records')
    
    # Plot 2: Duplicate percentage
    plt.subplot(2, 2, 2)
    duplicate_pct = (len(df) - len(deduplicated_df)) / len(df) * 100
    plt.pie([100 - duplicate_pct, duplicate_pct], 
            labels=['Unique', 'Duplicates'], 
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Duplicate Percentage')
    
    # Plot 3: Age distribution comparison
    plt.subplot(2, 2, 3)
    sns.kdeplot(df['age'], label='Original', color='#3498db')
    sns.kdeplot(deduplicated_df['age'], label='Deduplicated', color='#2ecc71')
    plt.title('Age Distribution Comparison')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 4: Income distribution comparison
    plt.subplot(2, 2, 4)
    sns.kdeplot(df['income'], label='Original', color='#3498db')
    sns.kdeplot(deduplicated_df['income'], label='Deduplicated', color='#2ecc71')
    plt.title('Income Distribution Comparison')
    plt.xlabel('Income')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_comparison.png')
    plt.close()
    
    print(f"Dataset comparison visualization saved to 'dedup_reports/dataset_comparison.png'")


def run_deduplication_with_evaluation(df):
    """
    Run deduplication with evaluation against known duplicate flags.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'is_duplicate' truth column
        
    Returns
    -------
    Dict[str, Any]
        Evaluation results
    """
    print("\nRunning deduplication with evaluation...")
    
    # Define columns for comparison and weights
    columns = ['name', 'email', 'phone', 'address']
    weights = {
        'name': 0.4,
        'email': 0.3,
        'phone': 0.2,
        'address': 0.1
    }
    
    # Run deduplication with evaluation
    results = flag_and_evaluate(
        df=df,
        columns=columns,
        weights=weights,
        known_duplicate_column='is_duplicate',
        threshold=0.7,
        method='weighted',
        flag_column='predicted_duplicate',
        generate_report=True,
        report_format='text',
        include_plots=True,
        auto_mode=True,
        show_progress=True
    )
    
    # Print text report
    print("\nEvaluation Report:")
    print(results['report'])
    
    # Return detected duplicates and results
    return results


def generate_all_report_formats(results):
    """
    Generate reports in various formats using the enhanced reporting system.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from flag_and_evaluate function
        
    Returns
    -------
    Dict[str, str]
        Paths to generated reports
    """
    print("\nGenerating enhanced reports in multiple formats...")
    
    # Generate reports in all available formats
    report_paths = generate_enhanced_report(
        results=results,
        formats=['html', 'excel', 'markdown', 'pptx'],
        output_dir="dedup_reports",
        title="Advanced Deduplication Analysis",
        include_pairs=True,
        max_pairs=50,
        filename_prefix="enhanced_deduplication_report"
    )
    
    # Print paths to generated reports
    print("\nGenerated the following reports:")
    for format_name, path in report_paths.items():
        print(f"- {format_name.upper()}: {path}")
    
    return report_paths


def demonstrate_reporter_class(results):
    """
    Demonstrate using the EnhancedDeduplicationReport class directly.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from flag_and_evaluate function
    """
    print("\nDemonstrating direct use of EnhancedDeduplicationReport class...")
    
    # Create reporter instance
    reporter = EnhancedDeduplicationReport(
        results=results,
        title="Custom Deduplication Analysis",
        output_dir="dedup_reports/custom",
        create_dir=True
    )
    
    # Generate HTML report with custom settings
    html_path = reporter.generate_html_report(
        output_path="dedup_reports/custom/custom_report.html",
        include_pairs=True,
        max_pairs=20,
        theme="flatly"  # Different bootstrap theme
    )
    
    print(f"Generated custom HTML report: {html_path}")
    
    # Generate Excel report with minimal pairs
    excel_path = reporter.generate_excel_report(
        output_path="dedup_reports/custom/custom_report.xlsx",
        include_pairs=True,
        max_pairs=10
    )
    
    print(f"Generated custom Excel report: {excel_path}")


def main():
    """Run the enhanced deduplication reporting example."""
    print("Enhanced Deduplication Reporting Example")
    print("=======================================")
    
    # Generate dataset with duplicates
    df = generate_dataset_with_duplicates(size=500, duplicate_rate=0.2)
    print(f"Generated dataset with {len(df)} records")
    
    # Run deduplication with evaluation
    results = run_deduplication_with_evaluation(df)
    
    # Get deduplicated dataframe
    deduplicated_df = results['dataframe'][~results['dataframe']['predicted_duplicate']]
    
    # Compare original and deduplicated datasets
    compare_deduplicated_original(df, deduplicated_df)
    
    # Generate reports in various formats
    report_paths = generate_all_report_formats(results)
    
    # Demonstrate the reporter class directly
    demonstrate_reporter_class(results)
    
    print("\nExample completed successfully!")
    print(f"Check the 'dedup_reports' directory for generated reports.")


if __name__ == "__main__":
    main()