"""
Example of using Freamon EDA with financial data containing currency symbols.

This example demonstrates how to use the patched version of Freamon EDA
to analyze financial data containing dollar signs and other special characters
that would normally cause rendering errors in matplotlib.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path if running the example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply the patches for matplotlib and freamon EDA
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches, patch_freamon_eda

apply_comprehensive_matplotlib_patches()
patch_freamon_eda()

# Create sample financial dataset
np.random.seed(42)
n_samples = 100

# Generate some realistic financial data
df = pd.DataFrame({
    'Product_ID': [f'PROD-{i:04d}' for i in range(1, n_samples + 1)],
    'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home & Garden'], n_samples),
    'Price': [f"${np.random.uniform(10, 1000):.2f}" for _ in range(n_samples)],
    'Discount': [f"{np.random.uniform(0, 30):.1f}%" for _ in range(n_samples)],
    'Rating': np.random.uniform(1, 5, n_samples),
    'Stock': np.random.randint(0, 1000, n_samples),
    'Profit_Margin': [f"${np.random.uniform(1, 100):.2f}" for _ in range(n_samples)],
    'Supplier_Code': [f"SUP_{np.random.choice(['A', 'B', 'C', 'D', 'E'])}{np.random.randint(100, 999)}" for _ in range(n_samples)],
    'Last_Restocked': pd.date_range(start='2023-01-01', periods=n_samples),
})

# Add a numeric price column for analysis
df['Price_Numeric'] = df['Price'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
df['Profit_Margin_Numeric'] = df['Profit_Margin'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
df['Discount_Numeric'] = df['Discount'].apply(lambda x: float(x.replace('%', '')))

# Print some sample data
print("Sample financial data:")
print(df.head())

# Run EDA analysis
from freamon.eda.analyzer import EDAAnalyzer

print("\nRunning EDA analysis on financial data...\n")
analyzer = EDAAnalyzer(df, target_column='Category', date_column='Last_Restocked')
report = analyzer.run_full_analysis(output_path="financial_data_report.html", 
                                  title="Financial Data Analysis Report",
                                  show_progress=True)

print("\nAnalysis complete. Report saved to financial_data_report.html")

# Create a simple plot to demonstrate currency handling
plt.figure(figsize=(10, 6))
categories = df['Category'].unique()
avg_prices = [df[df['Category'] == cat]['Price_Numeric'].mean() for cat in categories]

plt.bar(categories, avg_prices)
plt.title('Average Price by Category ($)')
plt.xlabel('Product Category')
plt.ylabel('Average Price ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('category_price_analysis.png')

print("\nCategory price analysis chart saved to category_price_analysis.png")

# Plot with dollar signs in the data labels
plt.figure(figsize=(12, 8))
plt.scatter(df['Price_Numeric'], df['Profit_Margin_Numeric'], alpha=0.7)

# Add product labels with dollar signs for some points
for i in range(0, 10, 2):
    plt.annotate(f"${df['Price_Numeric'].iloc[i]:.2f}", 
                 (df['Price_Numeric'].iloc[i], df['Profit_Margin_Numeric'].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')

plt.title('Price vs. Profit Margin Analysis')
plt.xlabel('Price ($)')
plt.ylabel('Profit Margin ($)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('price_profit_analysis.png')

print("Price-profit analysis chart saved to price_profit_analysis.png")