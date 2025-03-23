"""
Example script demonstrating month-year format detection and conversion.
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Import the specialized functions from our utility module
from freamon.utils.date_converters import convert_month_year_format, is_month_year_format


def main():
    """Run the month-year detection example."""
    print("Month-Year Format Detection Example")
    print("===================================")
    
    # Create a dataframe with various month-year formats
    df = pd.DataFrame({
        'transaction_date': [
            'Aug-24',       # Abbreviated month with 2-digit year
            'Sep-2024',     # Abbreviated month with 4-digit year
            'October-23',   # Full month with 2-digit year
            'November-2023',# Full month with 4-digit year
            'Dec 24',       # Month with space separator
            '01-25',        # Numeric month with hyphen
            '02/25',        # Numeric month with slash
            '03.25',        # Numeric month with dot
        ],
        'amount': [150.25, 200.50, 75.00, 300.75, 125.60, 450.00, 50.25, 90.80]
    })
    
    print("\nThis example demonstrates how to detect and convert month-year formats")
    print("like 'Aug-24', 'Sep-2024', 'Dec 24', '01-25', etc. to proper datetime objects.")
    
    print("\nOriginal DataFrame:")
    print(df)
    print(f"\nOriginal datatypes:\n{df.dtypes}")
    
    # Check if column has month-year format
    print("\nChecking if values match month-year format pattern:")
    is_month_year = is_month_year_format(df['transaction_date'])
    print(f"Is month-year format: {is_month_year}")
    
    # Convert the month-year format values to datetime
    print("\nConverting month-year values to datetime:")
    converted_dates = convert_month_year_format(df['transaction_date'])
    print(converted_dates)
    
    # Check how many were successfully converted
    success_count = converted_dates.notna().sum()
    print(f"\nSuccessfully converted {success_count} out of {len(converted_dates)} dates")
    
    # If we were able to parse them, we would display them properly
    print("\nFormatted dates (month and year):")
    for i, (orig, date) in enumerate(zip(df['transaction_date'], converted_dates)):
        if pd.notna(date):
            print(f"  {orig} -> {date.strftime('%B %Y')}")
        else:
            print(f"  {orig} -> Failed to parse")
    
    # Create a DataFrame with the converted dates
    df_with_dates = df.copy()
    df_with_dates['date'] = converted_dates
    
    # Show how this could be used in a business context
    print("\nBusiness example - Grouping sales by month:")
    monthly_data = df_with_dates.copy()
    monthly_data['month_year'] = monthly_data['date'].dt.strftime('%Y-%m')
    print(monthly_data.groupby('month_year')['amount'].sum())
    
    # Demonstrate the utility with another example - mixed formats
    print("\nAnother example - Different month-year formats in the same column:")
    mixed_formats = pd.DataFrame({
        'report_period': ['Jan/22', 'Feb-2022', 'March 22', 'Apr.22', 'May22'],
        'revenue': [10000, 12000, 9500, 11200, 13500]
    })
    
    print(mixed_formats)
    
    # Check and convert
    is_my_format = is_month_year_format(mixed_formats['report_period'], threshold=0.8)
    print(f"\nIs month-year format: {is_my_format}")
    
    if is_my_format:
        mixed_formats['report_date'] = convert_month_year_format(mixed_formats['report_period'])
        print("\nConverted dates:")
        print(mixed_formats)
        
        # Business reporting by quarter
        print("\nQuarterly revenue report:")
        mixed_formats['quarter'] = mixed_formats['report_date'].dt.to_period('Q').astype(str)
        quarterly = mixed_formats.groupby('quarter')['revenue'].sum()
        print(quarterly)


if __name__ == "__main__":
    main()