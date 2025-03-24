"""
Example demonstrating the Jupyter notebook export functionality in Freamon EDA.

This script:
1. Creates a sample dataset with financial data (including currency symbols)
2. Generates an EDA report with the export to Jupyter button enabled
3. Provides instructions on how to use the export feature
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import from freamon
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches

# Apply matplotlib patches for handling currency symbols
apply_comprehensive_matplotlib_patches()

# Set random seed for reproducibility
np.random.seed(42)

def create_financial_dataset():
    """Create a financial dataset with currency symbols for demonstration."""
    n = 1000
    
    # Generate dates for the last 3 years with business days only
    dates = pd.date_range(
        start=pd.Timestamp.now() - pd.DateOffset(years=3),
        end=pd.Timestamp.now(),
        freq='B'  # Business days
    )
    
    # Sample n dates from this range
    sampled_dates = np.random.choice(dates, n, replace=False)
    sampled_dates.sort()
    
    # Create company data
    companies = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    sectors = ['Technology', 'Technology', 'Technology', 'Retail', 'Social Media']
    
    # Create DataFrame with financial metrics
    df = pd.DataFrame({
        'date': sampled_dates,
        'company': np.random.choice(companies, n),
        'sector': np.random.choice(sectors, n),
        # Financial metrics
        'revenue': [f"${x:,.2f}M" for x in np.random.uniform(100, 1000, n)],
        'profit_margin': [f"{x:.2f}%" for x in np.random.uniform(5, 25, n)],
        'stock_price': [f"${x:.2f}" for x in np.random.uniform(50, 500, n)],
        'volume': np.random.randint(1000000, 10000000, n),
        'pe_ratio': np.random.uniform(10, 40, n),
        'market_cap': [f"${x:,.2f}B" for x in np.random.uniform(100, 2000, n)],
        'dividend_yield': [f"{x:.2f}%" for x in np.random.uniform(0, 5, n)],
        # Target metric
        'rating': np.random.choice(['Buy', 'Hold', 'Sell'], n, p=[0.5, 0.3, 0.2])
    })
    
    # Add some missing values
    for col in ['profit_margin', 'dividend_yield', 'pe_ratio']:
        mask = np.random.choice([True, False], n, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    return df

def main():
    """Run the example."""
    print("Creating financial dataset...")
    df = create_financial_dataset()
    print(f"Dataset shape: {df.shape}")
    
    # Create output directory
    output_dir = Path("jupyter_export_example")
    output_dir.mkdir(exist_ok=True)
    
    # Create an EDA analyzer
    print("Running EDA analysis...")
    analyzer = EDAAnalyzer(
        df,
        date_column='date',
        categorical_columns=['company', 'sector', 'rating'],
        target_column='rating'
    )
    
    # Run full analysis with Jupyter export enabled
    report_path = str(output_dir / "financial_analysis.html")
    analyzer.run_full_analysis(
        output_path=report_path,
        title="Financial Data Analysis",
        include_multivariate=True,
        lazy_loading=True,
        include_export_button=True,
    )
    
    print("\nReport generated successfully!")
    print(f"Open {report_path} in your web browser")
    print("\nTo export to Jupyter notebook:")
    print("1. Open the HTML report in your browser")
    print("2. Click the 'Export to Jupyter Notebook' button in the top-right corner")
    print("3. The notebook will be downloaded automatically")
    print("4. Open the downloaded notebook in Jupyter to interact with it")
    print("\nThe exported notebook will contain:")
    print("- All the visualizations from the report")
    print("- Sample data from the analyzed dataset")
    print("- Code to reproduce each visualization")
    print("- Proper setup code with the matplotlib fixes for currency symbols")

if __name__ == "__main__":
    main()