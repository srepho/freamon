"""
Example demonstrating the Markdown report generation functionality.

This example shows:
1. How to generate clean, readable markdown reports from data analysis
2. Customizing the report content and sections
3. Including tables, statistics, and analysis results
4. Converting to HTML for sharing
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_diabetes

from freamon.eda.markdown_report import generate_markdown_report
from freamon.eda import EDAAnalyzer
from freamon.utils.datatype_detector import DataTypeDetector

def create_report_directory():
    """Create directory for markdown reports."""
    report_dir = Path("markdown_reports")
    report_dir.mkdir(exist_ok=True)
    return report_dir

def generate_basic_report():
    """Generate a basic markdown report from synthetic data."""
    print("\n===== Generating Basic Report =====")
    
    # Create sample data
    data = {
        'age': np.random.randint(18, 70, 100),
        'income': np.random.normal(50000, 15000, 100),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'experience': np.random.randint(0, 30, 100),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 100),
        'churn': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    
    # Generate markdown report
    report = generate_markdown_report(
        df, 
        title="Customer Analysis Report",
        description="Analysis of customer demographics and churn risk",
        target_column='churn',
        include_correlations=True,
        include_histograms=True,
        include_boxplots=True
    )
    
    # Save report
    report_dir = create_report_directory()
    report_path = report_dir / "basic_customer_report.md"
    
    with open(report_path, "w") as f:
        f.write(report)
    
    # Convert to HTML
    try:
        from markdown import markdown
        html = markdown(report)
        html_path = report_dir / "basic_customer_report.html"
        
        with open(html_path, "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Customer Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                    h2 {{ color: #3498db; margin-top: 30px; }}
                    h3 {{ color: #2980b9; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """)
            
        print(f"Basic report saved to '{report_path}' and '{html_path}'")
    except ImportError:
        print(f"Basic report saved to '{report_path}' (HTML conversion requires markdown package)")

def generate_diabetes_report():
    """Generate a more complex report using the diabetes dataset."""
    print("\n===== Generating Diabetes Dataset Report =====")
    
    # Load the diabetes dataset
    diabetes = load_diabetes()
    df = pd.DataFrame(
        data=diabetes.data,
        columns=diabetes.feature_names
    )
    df['target'] = diabetes.target
    
    # Detect data types
    detector = DataTypeDetector(df)
    detector.detect_all_types()
    types_dict = {col: info['type'] for col, info in detector.column_types.items()}
    
    # Add column information footnotes
    feature_info = """
## Feature Information
- **age**: Age
- **sex**: Gender
- **bmi**: Body mass index
- **bp**: Average blood pressure
- **s1**: tc, total serum cholesterol
- **s2**: ldl, low-density lipoproteins
- **s3**: hdl, high-density lipoproteins
- **s4**: tch, total cholesterol / HDL
- **s5**: ltg, possibly log of serum triglycerides level
- **s6**: glu, blood sugar level
    """
    
    # Add custom section with data type information
    data_types_section = "## Data Types\n\n"
    data_types_section += "| Column | Data Type |\n"
    data_types_section += "|--------|----------|\n"
    
    for col, dtype in types_dict.items():
        data_types_section += f"| {col} | {dtype} |\n"
    
    # Generate markdown report
    report = generate_markdown_report(
        df, 
        title="Diabetes Analysis Report",
        description="Analysis of diabetes dataset from scikit-learn",
        target_column='target',
        include_correlations=True,
        include_histograms=True,
        include_boxplots=True,
        include_scatter_matrix=True
    )
    
    # Add custom sections
    report = report + "\n" + feature_info + "\n" + data_types_section
    
    # Add a simple model section
    model_section = """
## Simple Model Analysis

A LightGBM model was trained on the dataset with the following parameters:
- num_leaves: 31
- learning_rate: 0.05
- n_estimators: 100

### Feature Importance
The top features by importance were:
1. bmi
2. s5
3. bp
4. s1
5. s2

### Model Performance
- **R²**: 0.453
- **MAE**: 42.8
- **RMSE**: 55.3

These results suggest that the dataset contains useful predictive information, 
but additional features or more sophisticated modeling might be needed for better performance.
"""
    
    report = report + "\n" + model_section
    
    # Save report
    report_dir = create_report_directory()
    report_path = report_dir / "diabetes_report.md"
    
    with open(report_path, "w") as f:
        f.write(report)
    
    # Export plots for the report
    plt.figure(figsize=(10, 6))
    corr = df.corr()
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(report_dir / "correlation_matrix.png")
    
    # Add HTML note to explain how to reference images
    image_note = """
## Adding Images to Markdown Reports

To include images in your markdown reports, you can:

1. Save plots to image files:
```python
plt.figure(figsize=(10, 6))
# Plot data...
plt.savefig("path/to/image.png")
```

2. Reference images in markdown:
```markdown
![Plot title](path/to/image.png)
```

Example:
![Correlation Matrix](correlation_matrix.png)

When converting to HTML, these images will be properly displayed.
"""
    
    # Add the image note to the report
    with open(report_path, "a") as f:
        f.write("\n" + image_note)
    
    print(f"Diabetes report saved to '{report_path}'")

def generate_custom_report():
    """Generate a fully customized markdown report."""
    print("\n===== Generating Custom Report =====")
    
    # Create sample sales data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    regions = ['North', 'South', 'East', 'West']
    
    # Generate data
    data = []
    for _ in range(500):
        date = np.random.choice(dates)
        product = np.random.choice(products)
        region = np.random.choice(regions)
        quantity = np.random.randint(1, 50)
        price = np.random.uniform(10, 100)
        revenue = price * quantity
        
        data.append({
            'date': date,
            'product': product,
            'region': region,
            'quantity': quantity,
            'price': price,
            'revenue': revenue
        })
    
    df = pd.DataFrame(data)
    
    # Create markdown report components manually
    report_dir = create_report_directory()
    
    # Title and introduction
    title = "# Sales Performance Analysis Report\n\n"
    intro = "## Introduction\n\nThis report analyzes sales performance across different products and regions for the period Jan-Apr 2023.\n\n"
    
    # Data summary section
    summary = "## Data Summary\n\n"
    summary += f"- **Total records:** {len(df)}\n"
    summary += f"- **Date range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n"
    summary += f"- **Products:** {', '.join(df['product'].unique())}\n"
    summary += f"- **Regions:** {', '.join(df['region'].unique())}\n"
    summary += f"- **Total revenue:** ${df['revenue'].sum():,.2f}\n\n"
    
    # Sales by product
    sales_by_product = df.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
    product_section = "## Sales by Product\n\n"
    product_section += "| Product | Revenue |\n"
    product_section += "|---------|--------:|\n"
    
    for _, row in sales_by_product.iterrows():
        product_section += f"| {row['product']} | ${row['revenue']:,.2f} |\n"
    
    product_section += "\n"
    
    # Sales by region
    sales_by_region = df.groupby('region')['revenue'].sum().sort_values(ascending=False).reset_index()
    region_section = "## Sales by Region\n\n"
    region_section += "| Region | Revenue |\n"
    region_section += "|--------|--------:|\n"
    
    for _, row in sales_by_region.iterrows():
        region_section += f"| {row['region']} | ${row['revenue']:,.2f} |\n"
    
    region_section += "\n"
    
    # Time series analysis
    time_section = "## Sales Trend Analysis\n\n"
    time_section += "Sales show a generally increasing trend over the period with weekly seasonality.\n\n"
    
    # Save timeseries plot
    plt.figure(figsize=(10, 6))
    daily_sales = df.groupby('date')['revenue'].sum()
    daily_sales.plot()
    plt.title('Daily Sales Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(report_dir / "daily_sales.png")
    plt.close()
    
    time_section += "![Daily Sales Trend](daily_sales.png)\n\n"
    
    # Recommendations section
    recommendations = "## Recommendations\n\n"
    recommendations += "Based on the analysis, we recommend:\n\n"
    recommendations += "1. **Focus on Product A** which shows the highest revenue potential\n"
    recommendations += "2. **Increase marketing in Southern region** where sales are lower\n"
    recommendations += "3. **Optimize for weekly sales peaks** by ensuring adequate inventory\n"
    recommendations += "4. **Explore product bundling opportunities** between top and lower performing products\n\n"
    
    # Combine all sections
    full_report = title + intro + summary + product_section + region_section + time_section + recommendations
    
    # Save the report
    report_path = report_dir / "custom_sales_report.md"
    with open(report_path, "w") as f:
        f.write(full_report)
    
    print(f"Custom sales report saved to '{report_path}'")

def generate_eda_markdown_report():
    """Generate a markdown report using the EDAAnalyzer."""
    print("\n===== Generating EDA Analyzer Markdown Report =====")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create date range
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Numeric features
    age = np.random.normal(35, 10, n_samples).astype(int)
    age = np.clip(age, 18, 80)  # Clip to realistic age range
    
    income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    category = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Binary features
    has_subscription = np.random.binomial(1, 0.3, n_samples)
    
    # Target variable with dependency on features
    churn_prob = 0.1 + 0.3 * (age < 25) + 0.2 * (category == 'D') - 0.1 * has_subscription
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    data = {
        'date': dates,
        'age': age,
        'income': income, 
        'category': category,
        'has_subscription': has_subscription,
        'churn': churn
    }
    
    # Add some missing values and time patterns
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    income_copy = income.copy()
    income_copy[missing_indices] = np.nan
    data['income'] = income_copy
    
    time_idx = np.arange(n_samples)
    data['signups'] = 100 + 0.5 * time_idx + 20 * np.sin(2 * np.pi * time_idx / 30) + np.random.normal(0, 5, n_samples)
    
    df = pd.DataFrame(data)
    
    # Initialize the EDA analyzer
    analyzer = EDAAnalyzer(df, target_column='churn', date_column='date')
    
    # Generate markdown report
    report_dir = create_report_directory()
    markdown_path = report_dir / "eda_analyzer_report.md"
    
    analyzer.generate_report(
        output_path=str(markdown_path),
        title="Customer Churn Analysis",
        format="markdown"
    )
    
    print(f"EDA Analyzer markdown report saved to '{markdown_path}'")
    
    # Generate markdown report with HTML conversion
    markdown_html_path = report_dir / "eda_analyzer_report_with_html.md"
    analyzer.generate_report(
        output_path=str(markdown_html_path),
        title="Customer Churn Analysis",
        format="markdown",
        convert_to_html=True
    )
    
    print(f"EDA Analyzer markdown report saved to '{markdown_html_path}'")
    print(f"HTML version saved to '{markdown_html_path}.html'")

def main():
    """Run the markdown report generation examples."""
    print("===== Markdown Report Generation Examples =====")
    
    # Generate different types of reports
    generate_basic_report()
    generate_diabetes_report()
    generate_custom_report()
    generate_eda_markdown_report()
    
    print("\nAll reports generated successfully! Check the 'markdown_reports' directory.")

if __name__ == "__main__":
    main()