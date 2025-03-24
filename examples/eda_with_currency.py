"""
Example demonstrating how to use the patched EDA module with data containing
currency symbols and other special characters.

This example shows how to apply the robust patches and generate a report without errors.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Apply the patches to make EDA more robust
from configure_matplotlib_for_currency import patch_freamon
patch_freamon()

# Create a sample dataframe with financial data including currency values
def create_financial_dataframe(n=1000):
    """Create a sample financial dataframe with currency values."""
    np.random.seed(42)
    
    # Generate dates for a year of daily data
    dates = pd.date_range(start='2023-01-01', periods=n)
    
    # Create base data
    df = pd.DataFrame({
        'date': dates,
        'product_id': np.random.randint(1, 11, n),
        'store_id': np.random.randint(1, 6, n),
        'revenue': np.random.uniform(1000, 10000, n).round(2),
        'cost': np.random.uniform(500, 8000, n).round(2),
        'units_sold': np.random.randint(1, 100, n),
        'discount_%': np.random.uniform(0, 30, n).round(1)
    })
    
    # Calculate profit
    df['profit'] = (df['revenue'] - df['cost']).round(2)
    
    # Format currency columns with dollar signs
    df['revenue_$'] = df['revenue'].apply(lambda x: f"${x:.2f}")
    df['cost_$'] = df['cost'].apply(lambda x: f"${x:.2f}")
    df['profit_$'] = df['profit'].apply(lambda x: f"${x:.2f}")
    
    # Add profit margin with percentage
    df['profit_margin_%'] = (df['profit'] / df['revenue'] * 100).round(2)
    
    # Create product categories with underscores (which can cause LaTeX issues)
    products = {
        1: 'electronics_large', 
        2: 'electronics_small', 
        3: 'home_goods', 
        4: 'kitchen_appliances',
        5: 'office_supplies', 
        6: 'furniture_large', 
        7: 'furniture_small',
        8: 'clothing_adult', 
        9: 'clothing_child', 
        10: 'misc_items'
    }
    df['product_category'] = df['product_id'].map(products)
    
    # Add some metrics with special characters that might cause issues
    df['revenue_per_unit'] = (df['revenue'] / df['units_sold']).round(2)
    df['revenue/unit_$'] = df['revenue_per_unit'].apply(lambda x: f"${x:.2f}")
    
    return df

# Create and display the dataframe
df = create_financial_dataframe()
print("Sample financial dataframe created:")
print(df.head())

# Now run EDA with the patched version
print("\nRunning EDA analysis...")
from freamon.eda import EDAAnalyzer

# Initialize the analyzer with target column
analyzer = EDAAnalyzer(
    df, 
    target_column='profit',
    date_column='date'
)

# Run the full analysis with all features
analyzer.run_full_analysis(
    output_path='financial_eda_report.html',
    title='Financial Data Analysis Report',
    include_multivariate=True,
    include_feature_importance=True,
    use_sampling=False,
    show_progress=True
)

print("\nAnalysis complete! Report saved to 'financial_eda_report.html'")
print("Open the HTML file in your browser to view the full report.")
print("\nKey features of this example:")
print("1. Successfully handles dollar signs in column names and values")
print("2. Properly displays underscores in category names")
print("3. Correctly processes percentage signs")
print("4. Renders all visualizations without LaTeX errors")
print("5. Generates a complete analysis without crashes")