"""
Test script to verify fixes for EDA report generation when handling currency symbols
and other special characters.

This script tests the robust version of the EDA report generation with data 
containing potentially problematic values like currency symbols, underscores, etc.
"""
import pandas as pd
import numpy as np
from configure_matplotlib_for_currency import patch_freamon

# Apply the patches to make EDA more robust
patch_freamon()

# Create a test dataframe with currency values
def create_test_dataframe():
    # Create a test dataframe with currency values
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'id': range(1, n+1),
        'price_$': np.random.uniform(10, 1000, n).round(2),  # Has $ in column name
        'revenue': [f"${x:.2f}" for x in np.random.uniform(1000, 10000, n)],  # Has $ in values
        'growth_rate_%': np.random.uniform(-5, 15, n).round(2),  # Has % in column name
        'status': np.random.choice(['active', 'inactive', 'pending'], n),
        'category': np.random.choice(['A_cat', 'B_cat', 'C_cat'], n),  # Has underscores
        'date': pd.date_range(start='2020-01-01', periods=n),
        'markdown_code': [f"`code_{i}`" for i in range(n)],  # Has backticks and underscores
        'latex_math': [f"$y = {i}x + {i*2}$" for i in range(n)],  # Has LaTeX math syntax
    })
    
    # Add some missing values
    df.loc[np.random.choice(n, 20), 'price_$'] = np.nan
    df.loc[np.random.choice(n, 20), 'revenue'] = np.nan
    
    return df

# Create the test dataframe
df = create_test_dataframe()

# Print the first few rows to verify
print("Test dataframe created:")
print(df.head())

# Run EDA analysis with our patched version
try:
    from freamon.eda import EDAAnalyzer
    
    # Initialize the analyzer
    analyzer = EDAAnalyzer(df, date_column='date')
    
    # Run full analysis with robust error handling
    success = analyzer.run_full_analysis(
        output_path='test_fix_report.html',
        title='Test EDA Report with Currency Values',
        include_multivariate=True,
        show_progress=True
    )
    
    if success:
        print("\nEDA analysis completed successfully!")
        print("Output saved to test_fix_report.html")
    else:
        print("\nEDA analysis completed with some errors.")
        print("Check the warnings above and review the output file.")
        
except Exception as e:
    print(f"\nError occurred: {str(e)}")

print("\nTest completed.")