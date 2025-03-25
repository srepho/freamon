import pandas as pd
from freamon.eda import EDAAnalyzer

# Create test data with month-year formats that include the 2024.0 format
data = {
    'date_col': ['Jun-24', 'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24'],
    'numeric': [10, 20, 30, 40, 50]
}

# Create a dataframe with some missing values
df = pd.DataFrame(data)
df.loc[1, 'date_col'] = None  # Add a missing value

print(f"Original DataFrame:\n{df}")

# Run EDA analysis
eda = EDAAnalyzer(df)
eda.analyze_basic_stats()  # Should automatically detect column types
print("\nColumn type detection results:")
print(f"Datetime columns: {eda.datetime_columns}")
print(f"Numeric columns: {eda.numeric_columns}")
print(f"Categorical columns: {eda.categorical_columns}")

# Run univariate analysis (should handle the date column)
result = eda.analyze_univariate()
print("\nSuccessfully completed univariate analysis")

# Generate a small report to verify fix works end-to-end
eda.run_full_analysis(output_path="test_report.html")
print("\nSuccessfully generated report at test_report.html")