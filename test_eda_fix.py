"""Test script for EDA fix with complex currency data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path if not already there
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from freamon.utils.matplotlib_fixes import patch_freamon_eda, apply_comprehensive_matplotlib_patches

# Apply patches to handle currency symbols
apply_comprehensive_matplotlib_patches()
patch_freamon_eda()

# Create test dataframe with various currency formats and special characters
df = pd.DataFrame({
    'Price_USD': ['$1,234.56', '$2,345.67', '$3,456.78', '$4,567.89', '$5,678.90'],
    'Price_EUR': ['€1,234.56', '€2,345.67', '€3,456.78', '€4,567.89', '€5,678.90'],
    'Price_GBP': ['£1,234.56', '£2,345.67', '£3,456.78', '£4,567.89', '£5,678.90'],
    'Price_JPY': ['¥123,456', '¥234,567', '¥345,678', '¥456,789', '¥567,890'],
    'Price_Mix': ['$1,234.56', '€2,345.67', '£3,456.78', '¥456,789', '$5,678.90'],
    'Special_Chars': ['abc_123', 'def^456', 'ghi&789', 'jkl*012', 'mno%345'],
    'LaTeX_Symbols': ['$alpha$', '$beta$', '$gamma$', '$delta$', '$epsilon$'],
    'Numeric': [1234.56, 2345.67, 3456.78, 4567.89, 5678.90],
    'Category': ['A', 'B', 'A', 'C', 'B']
})

# Explicitly set column types to avoid datetime conversion attempts
for col in df.columns:
    if col == 'Numeric':
        df[col] = df[col].astype(float)
    elif col == 'Category':
        df[col] = df[col].astype('category')
    else:
        df[col] = df[col].astype(str)

# Run EDA analysis
from freamon.eda.analyzer import EDAAnalyzer
analyzer = EDAAnalyzer(df, target_column='Category')
report = analyzer.run_full_analysis(output_path="test_currency_report.html", title="Currency Test Report")

print("Test complete! Report saved to test_currency_report.html")

# Create a simple plot to verify currency display works correctly
plt.figure(figsize=(10, 6))
plt.bar(df['Category'], df['Numeric'])
plt.title('Test with Category labels and numeric values')
plt.xlabel('Category')
plt.ylabel('Value ($)')
plt.savefig('test_currency_plot.png')
plt.close()

# Test with various types of problematic strings
test_strings = [
    "$100.00",
    "Price is $100.00",
    "100%",
    "Text with _underscores_",
    "Text with ^carets^",
    "Both $price and _underscores_",
    "Complex $alpha$ and $beta$",
    "Mix of € and £ symbols"
]

plt.figure(figsize=(12, 8))
for i, s in enumerate(test_strings):
    plt.text(0.1, 0.9 - (i * 0.1), s, fontsize=12)
plt.title("Testing various problematic strings")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.savefig('test_string_rendering.png')
plt.close()

print("All tests completed successfully!")