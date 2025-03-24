"""
Test script for automatic special character fixes in matplotlib.
This script demonstrates that the special character handling is now automatic.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from freamon.eda.analyzer import EDAAnalyzer

# Create test data with underscores and other special characters
data = {
    'category_name': ['product_a', 'product_b', 'product_c', 'product_d', 'product_e'],
    'price_$': [10.99, 15.99, 5.99, 20.99, 25.99],
    'quantity_%': [5, 10, 15, 20, 25],
    'profit_{margin}': [1.5, 2.0, 0.8, 3.0, 4.5]
}

# Create DataFrame
df = pd.DataFrame(data)

print(f"Created test DataFrame with problematic column names:")
print(df.head())

# Create the EDAAnalyzer with our test data
analyzer = EDAAnalyzer(df)

# Run the analysis and generate a report
# All special character fixes should be applied automatically
analyzer.run_full_analysis(output_path="test_automatic_fixes_report.html", 
                          title="Automatic Fixes Test Report")

print("\nGenerated report: test_automatic_fixes_report.html")
print("Special characters should be properly rendered without needing explicit fixes")

# Also test direct plotting
plt.figure(figsize=(10, 6))
plt.bar(df['category_name'], df['price_$'])
plt.title("Product Prices (With Underscore Characters)")
plt.xlabel("Product Category")
plt.ylabel("Price ($)")
plt.savefig("test_automatic_fixes_plot.png")
plt.close()

print("\nGenerated plot: test_automatic_fixes_plot.png")
print("Done! Check the report and plot to verify that special characters are displayed correctly.")