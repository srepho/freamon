"""
Example showing how to use the split report generation for better EDA performance.

This example demonstrates how to use the split report functionality to improve
the performance of the EDA process, particularly for large datasets.
"""
import pandas as pd
import numpy as np
import time
from sklearn.datasets import make_classification, make_regression

from freamon.eda import EDAAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

# Create a synthetic classification dataset
def create_test_classification_dataset(n_samples=10000, n_features=20):
    """Create a synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=42,
    )
    
    # Create a dataframe
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    # Add a datetime column for time series analysis
    base_date = pd.Timestamp("2023-01-01")
    dates = [base_date + pd.Timedelta(days=i) for i in range(n_samples)]
    df["date"] = dates
    
    # Add some categorical features
    df["cat_1"] = np.random.choice(["A", "B", "C"], size=n_samples)
    df["cat_2"] = np.random.choice(["X", "Y", "Z"], size=n_samples)
    
    return df

# Create a synthetic regression dataset
def create_test_regression_dataset(n_samples=10000, n_features=20):
    """Create a synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        random_state=42,
    )
    
    # Create a dataframe
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    # Add a datetime column for time series analysis
    base_date = pd.Timestamp("2023-01-01")
    dates = [base_date + pd.Timedelta(days=i) for i in range(n_samples)]
    df["date"] = dates
    
    # Add some categorical features
    df["cat_1"] = np.random.choice(["A", "B", "C"], size=n_samples)
    df["cat_2"] = np.random.choice(["X", "Y", "Z"], size=n_samples)
    
    return df

def run_single_report_test():
    """Test single report performance."""
    print("\n--- Testing Single Report Performance ---")
    
    # Create dataset
    print("Creating dataset...")
    df = create_test_classification_dataset(n_samples=50000, n_features=30)
    print(f"Dataset shape: {df.shape}")
    
    # Run EDA with single report
    print("Running EDA with single report...")
    start_time = time.time()
    
    analyzer = EDAAnalyzer(df, target_column="target", date_column="date")
    analyzer.analyze_basic_stats()
    analyzer.analyze_univariate(use_sampling=True)
    analyzer.analyze_bivariate(use_sampling=True)
    analyzer.generate_report(output_path="eda_single_report.html", title="Single Report Test")
    
    end_time = time.time()
    single_report_time = end_time - start_time
    print(f"Single report completed in {single_report_time:.2f} seconds")
    
    return single_report_time

def run_split_reports_test():
    """Test split reports performance."""
    print("\n--- Testing Split Reports Performance ---")
    
    # Create dataset
    print("Creating dataset...")
    df = create_test_classification_dataset(n_samples=50000, n_features=30)
    print(f"Dataset shape: {df.shape}")
    
    # Run EDA with split reports
    print("Running EDA with split reports...")
    start_time = time.time()
    
    analyzer = EDAAnalyzer(df, target_column="target", date_column="date")
    analyzer.analyze_basic_stats()
    analyzer.analyze_univariate(use_sampling=True)
    analyzer.analyze_bivariate(use_sampling=True)
    
    # Generate split reports
    analyzer.generate_univariate_report(output_path="eda_univariate_report.html", title="Univariate Analysis")
    analyzer.generate_bivariate_report(output_path="eda_bivariate_report.html", title="Bivariate Analysis")
    
    end_time = time.time()
    split_reports_time = end_time - start_time
    print(f"Split reports completed in {split_reports_time:.2f} seconds")
    
    return split_reports_time

def run_automatic_split_test():
    """Test automatic split reports performance."""
    print("\n--- Testing Automatic Split Reports Performance ---")
    
    # Create dataset
    print("Creating dataset...")
    df = create_test_regression_dataset(n_samples=50000, n_features=30)
    print(f"Dataset shape: {df.shape}")
    
    # Run full analysis with split reports
    print("Running full analysis with split reports...")
    start_time = time.time()
    
    analyzer = EDAAnalyzer(df, target_column="target", date_column="date")
    analyzer.run_full_analysis(
        output_path="eda_full_report.html",
        title="Full Analysis Report",
        use_sampling=True,
        show_progress=True,
        split_reports=True,
    )
    
    end_time = time.time()
    automatic_split_time = end_time - start_time
    print(f"Automatic split reports completed in {automatic_split_time:.2f} seconds")
    
    return automatic_split_time

def run_bivariate_feature_importance_demo():
    """Demo for bivariate analysis with feature importance."""
    print("\n--- Demonstrating Bivariate Analysis with Feature Importance ---")
    
    # Create a smaller dataset for quicker demo
    print("Creating dataset...")
    df = create_test_classification_dataset(n_samples=5000, n_features=15)
    print(f"Dataset shape: {df.shape}")
    
    # Run analysis
    print("Running analysis...")
    analyzer = EDAAnalyzer(df, target_column="target", date_column="date")
    analyzer.analyze_basic_stats()
    analyzer.analyze_univariate()
    analyzer.analyze_bivariate()
    
    # Generate bivariate report with feature importance
    print("Generating bivariate report with feature importance...")
    analyzer.generate_bivariate_report(
        output_path="eda_bivariate_feature_importance.html",
        title="Bivariate Analysis with Feature Importance"
    )
    
    print("Feature importance report generated. Check eda_bivariate_feature_importance.html")
    
if __name__ == "__main__":
    print("=== EDA Performance Test ===")
    print("This example demonstrates the performance improvements from splitting EDA reports.")
    
    # Run bivariate feature importance demo
    run_bivariate_feature_importance_demo()
    
    # Run performance tests
    single_time = run_single_report_test()
    split_time = run_split_reports_test()
    auto_split_time = run_automatic_split_test()
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"Single report time: {single_time:.2f} seconds")
    print(f"Split reports time: {split_time:.2f} seconds")
    print(f"Automatic split time: {auto_split_time:.2f} seconds")
    
    # Note: actual performance gains depend on dataset size and complexity
    print("\nThe split report approach is particularly beneficial for:")
    print("1. Large datasets (>50K rows)")
    print("2. Many columns (>30 columns)")
    print("3. Complex analyses with feature importance calculation")
    print("4. When you need to focus on either univariate or bivariate analysis separately")