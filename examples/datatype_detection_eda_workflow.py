"""
Data Type Detection and EDA Workflow Example

This example demonstrates an integrated workflow combining Freamon's data type detection
with exploratory data analysis (EDA) capabilities, showing how to:

1. Detect data types in a dataset, including semantic types and Excel dates
2. Apply recommended type conversions to optimize the dataframe
3. Perform EDA using detected type information
4. Visualize key insights based on data type categories

The workflow showcases how automatic type detection can enhance your EDA process.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types
from freamon.eda import EDAAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_dataset(rows=1000):
    """Create a sample dataset with diverse data types for demonstration."""
    # Date range (including some as Excel dates)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(rows)]
    
    # Excel dates (days since 1899-12-30)
    excel_date_offset = (start_date - datetime(1899, 12, 30)).days
    excel_dates = [excel_date_offset + i for i in range(rows)]
    
    # Create different data types
    df = pd.DataFrame({
        # IDs and categorical features
        'customer_id': [f"CUST-{i:06d}" for i in range(rows)],
        'product_id': [f"PROD-{np.random.randint(1, 100):04d}" for _ in range(rows)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys'], rows),
        'subcategory': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], rows),
        'store_num': np.random.randint(1, 50, rows),
        
        # Date features in various formats
        'purchase_date': dates,
        'excel_date': excel_dates,
        'date_string': [d.strftime('%Y-%m-%d') for d in dates],
        
        # Numeric features (continuous and categorical)
        'price': np.random.lognormal(3, 1, rows),
        'quantity': np.random.randint(1, 10, rows),
        'discount_pct': np.round(np.random.choice([0, 5, 10, 15, 20, 25], rows), 2),
        'customer_rating': np.random.choice([1, 2, 3, 4, 5], rows, p=[0.05, 0.1, 0.2, 0.3, 0.35]),
        
        # Contact information
        'email': [f"customer{i}@example.com" for i in range(rows)],
        'phone': [f"+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(rows)],
        
        # Location data
        'zipcode': np.random.choice(['90210', '10001', '60601', '77002', '98101'], rows),
        'au_postcode': np.random.choice([2000, 3000, 4000, 5000, 6000], rows),
        'latitude': np.random.uniform(-90, 90, rows),
        'longitude': np.random.uniform(-180, 180, rows),
        
        # Other metrics
        'page_views': np.random.negative_binomial(10, 0.5, rows),
        'time_on_site': np.random.exponential(300, rows),
        'conversion_rate': np.random.beta(2, 5, rows),
    })
    
    # Calculate some target variables
    df['total_sale'] = df['price'] * df['quantity'] * (1 - df['discount_pct']/100)
    df['is_large_order'] = (df['quantity'] > 5).astype(int)
    
    # Add some missing values
    for col in df.columns:
        if col not in ['customer_id', 'product_id']:
            mask = np.random.random(rows) < 0.05
            df.loc[mask, col] = np.nan
    
    return df

# Create the sample dataset
print("Creating sample dataset...")
df = create_sample_dataset()
print(f"Dataset shape: {df.shape}")
print("\nDataset preview:")
print(df.head())

# Step 1: Detect data types using DataTypeDetector
print("\n--- Step 1: Detecting Data Types ---")
detector = DataTypeDetector(df)
type_results = detector.detect_all_types()

# Categorize columns by their detected types
semantic_types = {}
for col, info in type_results.items():
    if 'semantic_type' in info:
        sem_type = info['semantic_type']
        if sem_type not in semantic_types:
            semantic_types[sem_type] = []
        semantic_types[sem_type].append(col)

logical_types = {}
for col, info in type_results.items():
    log_type = info['logical_type']
    if log_type not in logical_types:
        logical_types[log_type] = []
    logical_types[log_type].append(col)

# Print detection summary
print("\nDetected Semantic Types:")
for sem_type, cols in semantic_types.items():
    print(f"  {sem_type}: {', '.join(cols)}")

print("\nDetected Logical Types:")
for log_type, cols in logical_types.items():
    print(f"  {log_type}: {', '.join(cols)}")

# Step 2: Apply type conversions
print("\n--- Step 2: Applying Type Conversions ---")
print("Before conversion:")
print(df.dtypes)

optimized_df = detector.convert_types()
print("\nAfter conversion:")
print(optimized_df.dtypes)

# Group columns by data type for analysis
date_cols = [col for col, info in type_results.items() if info['logical_type'] == 'datetime']
numeric_cols = [
    col for col, info in type_results.items() 
    if 'float' in info['logical_type'] or 'integer' in info['logical_type']
    and 'categorical' not in info['logical_type']
]
categorical_cols = [
    col for col, info in type_results.items() 
    if 'categorical' in info['logical_type'] or info['logical_type'] == 'string'
]

# Step 3: Perform EDA based on detected types
print("\n--- Step 3: Performing EDA Based on Detected Types ---")
analyzer = EDAAnalyzer(optimized_df)
analyzer.run_full_analysis()

# Show missing value information
print("\nMissing Values Summary:")
for col, miss_pct in analyzer.analysis_results['basic_stats']['missing_values_percent'].items():
    if miss_pct > 0:
        print(f"  {col}: {miss_pct:.2f}%")

# Step 4: Create visualizations based on data types
print("\n--- Step 4: Creating Type-Based Visualizations ---")

# 1. Visualize numeric distributions
if numeric_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols[:9], 1):  # Limit to 9 plots for readability
        plt.subplot(3, 3, i)
        sns.histplot(optimized_df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    print("Saved numeric distributions plot to 'numeric_distributions.png'")

# 2. Visualize categorical distributions
if categorical_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols[:6], 1):
        plt.subplot(2, 3, i)
        value_counts = optimized_df[col].value_counts().head(10)
        value_counts.plot(kind='bar')
        plt.title(f'Top values in {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    plt.savefig('categorical_distributions.png')
    print("Saved categorical distributions plot to 'categorical_distributions.png'")

# 3. Visualize time series data
if date_cols and numeric_cols:
    # Pick a date column and a numeric column for time series
    date_col = date_cols[0]
    numeric_col = [col for col in numeric_cols if optimized_df[col].dtype != 'category'][0]
    
    # Create time series plot
    plt.figure(figsize=(12, 6))
    # Group by date and calculate mean of numeric column
    time_series_data = optimized_df.groupby(date_col)[numeric_col].mean()
    time_series_data.plot()
    plt.title(f'{numeric_col} over Time')
    plt.xlabel('Date')
    plt.ylabel(numeric_col)
    plt.grid(True, alpha=0.3)
    plt.savefig('time_series_plot.png')
    print("Saved time series plot to 'time_series_plot.png'")

# 4. Visualize correlations between numeric features
plt.figure(figsize=(12, 10))
corr_matrix = optimized_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Saved correlation matrix plot to 'correlation_matrix.png'")

# 5. Create a summary of data types
plt.figure(figsize=(10, 6))
type_counts = pd.Series({k: len(v) for k, v in logical_types.items()})
type_counts.plot(kind='bar', color='skyblue')
plt.title('Column Counts by Logical Type')
plt.xlabel('Logical Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('logical_type_counts.png')
print("Saved logical type counts plot to 'logical_type_counts.png'")

print("\nExample complete!")

# Optional: Save cleaned dataset
optimized_df.to_csv('optimized_dataset.csv', index=False)
print("Saved optimized dataset to 'optimized_dataset.csv'")