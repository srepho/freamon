"""
Example demonstrating the use of Markdown reports for EDA.

This example shows how to generate Markdown reports with Freamon EDA,
which can be more lightweight and easier to integrate with other tools
compared to HTML reports.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from freamon.eda import EDAAnalyzer

# Create a sample dataset
def create_sample_data(n_samples=1000):
    # Create date range
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Create features with some patterns
    np.random.seed(42)
    
    # Numeric features
    age = np.random.normal(35, 10, n_samples).astype(int)
    age = np.clip(age, 18, 80)  # Clip to realistic age range
    
    income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    category = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Binary features
    has_subscription = np.random.binomial(1, 0.3, n_samples)
    
    # Create a target variable with dependency on features
    churn_prob = 0.1 + 0.3 * (age < 25) + 0.2 * (category == 'D') - 0.1 * has_subscription
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    data = {
        'date': dates,
        'age': age,
        'income': income, 
        'category': category,
        'has_subscription': has_subscription,
        'churn': churn
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    income_copy = income.copy()
    income_copy[missing_indices] = np.nan
    data['income'] = income_copy
    
    # Create few more features with time patterns
    time_idx = np.arange(n_samples)
    data['signups'] = 100 + 0.5 * time_idx + 20 * np.sin(2 * np.pi * time_idx / 30) + np.random.normal(0, 5, n_samples)
    data['visits'] = 500 + time_idx + 100 * np.sin(2 * np.pi * time_idx / 7) + np.random.normal(0, 20, n_samples)
    
    return pd.DataFrame(data)

# Generate the dataset
df = create_sample_data()
print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns")

# Initialize the EDA analyzer
analyzer = EDAAnalyzer(df, target_column='churn', date_column='date')

# Run the full analysis
print("Running full analysis...")
analysis_results = analyzer.run_full_analysis()

# Generate a Markdown report
print("Generating Markdown report...")
markdown_path = 'eda_report.md'
analyzer.generate_report(
    output_path=markdown_path,
    title="Customer Churn Analysis",
    format="markdown"
)
print(f"Markdown report saved to {markdown_path}")

# Generate a Markdown report with HTML conversion
print("Generating Markdown report with HTML conversion...")
markdown_html_path = 'eda_report_with_html.md'
analyzer.generate_report(
    output_path=markdown_html_path,
    title="Customer Churn Analysis",
    format="markdown",
    convert_to_html=True
)
print(f"Markdown report saved to {markdown_html_path}")
print(f"HTML version saved to {markdown_html_path}.html")

print("\nExample completed successfully!")