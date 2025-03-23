"""
Example showing the visual detection report in a Jupyter notebook.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from freamon.utils.datatype_detector import DataTypeDetector


def create_sample_df(rows=200):
    """Create a sample dataframe with various data types for the example."""
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Create dates
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(rows)]
    
    # Create a diverse sample DataFrame
    df = pd.DataFrame({
        # IDs
        'id': range(1, rows + 1),
        'uuid': [f"abc-{i:03d}-xyz-{i*2:03d}" for i in range(1, rows + 1)],
        
        # Numeric data
        'amount': np.random.normal(100, 25, rows),
        'price': np.random.uniform(10, 1000, rows),
        'quantity': np.random.randint(1, 50, rows),
        'category_num': np.random.choice([1, 2, 3, 4, 5], rows),
        'binary': np.random.choice([0, 1], rows),
        'scientific': np.random.uniform(0.00001, 0.1, rows),  # Will be displayed in scientific notation
        
        # Date and time data
        'date': dates,
        'date_str': [d.strftime('%Y-%m-%d') for d in dates],
        'month_year': [d.strftime('%b-%y') for d in dates],
        'timestamp': [int(d.timestamp()) for d in dates],
        'excel_date': [d.toordinal() - datetime(1899, 12, 30).toordinal() for d in dates],
        
        # Categorical data
        'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
        'status': np.random.choice(['Pending', 'Approved', 'Rejected', 'On Hold'], rows),
        
        # String data with patterns
        'email': [f"user{i}@example.com" for i in range(rows)],
        'phone': [f"+1 (555) 123-{i:04d}"[:16] for i in range(rows)],
        'zipcode': [f"{np.random.randint(10000, 99999)}" for _ in range(rows)],
        'ip': [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(rows)],
        
        # Australian data
        'au_postcode': [f"{np.random.randint(1000, 9999)}" for _ in range(rows)],
        'au_phone': [f"+61 {np.random.choice([2, 3, 7, 8])}{np.random.randint(1000, 9999)} {np.random.randint(1000, 9999)}" for _ in range(rows)],
        
        # Mixed/messy data
        'mixed': [i if i % 3 == 0 else f"Value-{i}" for i in range(rows)],
        'with_nulls': [None if i % 5 == 0 else i for i in range(rows)],
    })
    
    # Convert some numeric columns to display scientific notation
    df['scientific'] = df['scientific'].map(lambda x: f"{x:.2e}")
    
    # Add text with nulls
    df['description'] = [None if i % 7 == 0 else f"This is description #{i}" for i in range(rows)]
    
    return df


def main():
    """Run the example - this is meant to be run in a Jupyter notebook."""
    print("Data Type Detection Report Example")
    print("=================================")
    print("\nThis example demonstrates how to generate a visually enhanced")
    print("detection report that's ideal for Jupyter notebook usage.")
    
    # Create a sample DataFrame
    df = create_sample_df()
    
    print(f"\nCreated sample DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nColumns in the sample data:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Initialize the detector and run detection
    detector = DataTypeDetector(df)
    results = detector.detect_all_types()
    
    print("\nDetection completed. In a Jupyter notebook, you would run:")
    print("detector.display_detection_report()")
    print("\nThis would display a styled DataFrame with color-coding for different types")
    
    # Create a basic representation of the report for CLI display
    print("\nDetection Results Summary:")
    print("--------------------------")
    for col, info in results.items():
        semantic = info.get('semantic_type', 'none')
        print(f"{col}: {info['storage_type']} → {info['logical_type']} ({semantic})")
    
    print("\nFor a full report with color-coding in a Jupyter notebook, use:")
    print("detector.display_detection_report()")
    
    # If running in IPython/Jupyter, this would return the styled DataFrame
    report = detector.display_detection_report()
    
    # When running in a script, save an HTML version of the report
    try:
        # Get the DataFrame from the styled report
        if hasattr(report, 'data'):
            report_df = report.data
        else:
            report_df = report
            
        # Save an HTML version of the report for viewing in a browser
        html_file = 'datatype_detection_report.html'
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DataTypeDetector Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .datetime {{ background-color: #BBDEFB; }}
                .categorical {{ background-color: #C8E6C9; }}
                .numeric {{ background-color: #FFF9C4; }}
                .string {{ background-color: #F8BBD0; }}
                .id {{ background-color: #D1C4E9; }}
                .suggestion {{ background-color: #FFE0B2; }}
            </style>
        </head>
        <body>
            <h1>DataTypeDetector Analysis Report</h1>
            <p>DataFrame with {len(df.columns)} columns and {len(df)} rows</p>
            
            <h2>Detection Results</h2>
            {report_df.to_html(index=False)}
            
            <h2>Color Legend</h2>
            <ul>
                <li><span class="datetime">&nbsp;&nbsp;&nbsp;&nbsp;</span> - Date/Time data (including month-year formats)</li>
                <li><span class="categorical">&nbsp;&nbsp;&nbsp;&nbsp;</span> - Categorical data</li>
                <li><span class="numeric">&nbsp;&nbsp;&nbsp;&nbsp;</span> - Numeric data</li>
                <li><span class="string">&nbsp;&nbsp;&nbsp;&nbsp;</span> - Text data</li>
                <li><span class="id">&nbsp;&nbsp;&nbsp;&nbsp;</span> - IDs and other semantic types</li>
                <li><span class="suggestion">&nbsp;&nbsp;&nbsp;&nbsp;</span> - Suggested conversions</li>
            </ul>
            
            <p><i>Generated by freamon.utils.datatype_detector.DataTypeDetector</i></p>
        </body>
        </html>
        """
        with open(html_file, 'w') as f:
            f.write(html_content)
        print(f"\nHTML report saved to {html_file}")
    except Exception as e:
        print(f"Error saving HTML report: {e}")
    
    return report


if __name__ == "__main__":
    main()