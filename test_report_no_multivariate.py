"""
Test script to verify that multivariate analysis is now off by default.
"""
import pandas as pd
import numpy as np
from freamon.eda.analyzer import EDAAnalyzer
from freamon.eda.report import generate_html_report

# Create a simple dataset
np.random.seed(42)
n_rows = 100

# Create a dataframe with minimal columns
df = pd.DataFrame({
    'id': range(1, n_rows + 1),
    'numeric': np.random.normal(0, 1, n_rows),
    'integer': np.random.randint(1, 100, n_rows),
    'categorical': np.random.choice(['A', 'B', 'C'], n_rows),
    'date': pd.date_range('2020-01-01', periods=n_rows)
})

# Run EDA analysis
print("Running EDA analysis with default settings (multivariate should be off)...")
analyzer = EDAAnalyzer(df)
results = analyzer.run_full_analysis(show_progress=True)

# Check if multivariate analysis is included
print("\nChecking if multivariate analysis was included...")
if "multivariate" in results:
    print("WARNING: Multivariate analysis was still included despite being off by default!")
else:
    print("SUCCESS: Multivariate analysis was not included, as expected!")

# Generate HTML report
print("\nGenerating HTML report...")
generate_html_report(df, results, "test_report_no_multivariate.html", title="Test Report (No Multivariate)")

print("\nNow running analysis WITH multivariate explicitly enabled...")
analyzer2 = EDAAnalyzer(df)
results2 = analyzer2.run_full_analysis(include_multivariate=True, show_progress=True)

# Check if multivariate analysis is included when explicitly enabled
print("\nChecking if multivariate analysis was included when explicitly enabled...")
if "multivariate" in results2:
    print("SUCCESS: Multivariate analysis was included when explicitly enabled!")
else:
    print("WARNING: Multivariate analysis was not included despite being explicitly enabled!")

# Generate HTML report with multivariate
print("\nGenerating HTML report with multivariate...")
generate_html_report(df, results2, "test_report_with_multivariate.html", title="Test Report (With Multivariate)")