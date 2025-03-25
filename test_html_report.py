"""
Test script to verify the HTML report generation with working accordion components.
"""
import pandas as pd
import numpy as np
from freamon.eda.analyzer import EDAAnalyzer
from freamon.eda.report import generate_html_report

# Create a simple dataset that will demonstrate the HTML features
np.random.seed(42)
n_rows = 100

# Create a dataframe with minimal columns to avoid type detection issues
df = pd.DataFrame({
    'id': range(1, n_rows + 1),
    'numeric': np.random.normal(0, 1, n_rows),
    'integer': np.random.randint(1, 100, n_rows),
    'categorical': np.random.choice(['A', 'B', 'C'], n_rows),
    'date': pd.date_range('2020-01-01', periods=n_rows)
})

# Set explicit data types
df['id'] = df['id'].astype('int32')
df['numeric'] = df['numeric'].astype('float32')
df['integer'] = df['integer'].astype('int32')
df['categorical'] = df['categorical'].astype('category')
df['date'] = pd.to_datetime(df['date'])

# Run EDA analysis
print("Running EDA analysis...")
analyzer = EDAAnalyzer(df)
results = analyzer.run_full_analysis()

# Generate HTML report
print("Generating HTML report...")
generate_html_report(df, results, "test_report.html", title="Test EDA Report")

print("Report generated successfully at test_report.html")
print("Please open the file in a browser to test if the accordion components work as expected.")