"""
Example demonstrating the advanced data type detection capabilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from freamon.eda import (
    DataTypeDetector,
    detect_column_types,
    optimize_dataframe_types,
    EDAAnalyzer
)

# Create a synthetic dataset with various data types
def create_sample_dataset(n_rows=1000):
    """Create a synthetic dataset with various data types for demonstration."""
    np.random.seed(42)
    
    # Basic types
    data = {
        'id': range(1, n_rows + 1),
        'uuid': [f"f47ac10b-58cc-4372-a567-0e02b2c3d{i:03d}" for i in range(n_rows)],
        'name': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Emma'], n_rows),
        'age': np.random.randint(18, 80, n_rows),
        'height': np.random.normal(170, 10, n_rows),
        'weight': np.random.normal(70, 15, n_rows),
        'income': np.random.randint(30000, 150000, n_rows),
        'email': [f"user{i}@example.com" for i in range(n_rows)],
        'phone': [f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'is_customer': np.random.choice([True, False], n_rows),
        
        # Categorical with numeric coding
        'status_code': np.random.choice([1, 2, 3, 4, 5], n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'rating': np.random.randint(1, 6, n_rows),
        
        # Date-like columns in different formats
        'date_iso': [(datetime(2020, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_rows)],
        'date_us': [(datetime(2020, 1, 1) + timedelta(days=i)).strftime('%m/%d/%Y') for i in range(n_rows)],
        'timestamp': [(datetime(2020, 1, 1) + timedelta(days=i)).timestamp() for i in range(n_rows)],
        
        # Currency and special formats
        'price': [f"${np.random.randint(10, 1000)}.{np.random.randint(0, 100):02d}" for _ in range(n_rows)],
        'zip_code': np.random.choice(['12345', '23456', '34567', '45678', '56789'], n_rows),
        'latitude': np.random.uniform(25, 48, n_rows),
        'longitude': np.random.uniform(-125, -70, n_rows),
        
        # Continuous float with many unique values
        'measurement1': np.random.normal(100, 20, n_rows),
        'measurement2': np.random.exponential(10, n_rows),
    }
    
    return pd.DataFrame(data)

# Main example code
if __name__ == "__main__":
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print("\n===== Dataset Overview =====")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\n===== Using detect_column_types function =====")
    types = detect_column_types(df)
    
    print("Detected column types:")
    for column, info in types.items():
        print(f"\n{column}:")
        print(f"  Storage type: {info['storage_type']}")
        print(f"  Logical type: {info['logical_type']}")
        
        if 'semantic_type' in info:
            print(f"  Semantic type: {info['semantic_type']}")
            
        if 'suggested_conversion' in info:
            print(f"  Suggested conversion: {info['suggested_conversion']}")
    
    print("\n===== Using DataTypeDetector class directly =====")
    detector = DataTypeDetector(df, sample_size=200, threshold=0.8)
    detector.detect_all_types()
    
    print("Getting detailed column report...")
    report = detector.get_column_report()
    
    # Display report for a few selected columns
    selected_columns = ['id', 'age', 'status_code', 'date_iso', 'timestamp', 'latitude']
    for col in selected_columns:
        print(f"\nDetailed report for '{col}':")
        for key, value in report[col].items():
            print(f"  {key}: {value}")
    
    print("\n===== Optimizing data types =====")
    print("Before optimization:")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    
    optimized_df = optimize_dataframe_types(df)
    
    print("\nAfter optimization:")
    print(f"Memory usage: {optimized_df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
    print("Optimized data types:")
    for col, dtype in zip(optimized_df.columns, optimized_df.dtypes):
        print(f"  {col}: {dtype}")
    
    print("\n===== Using with EDAAnalyzer =====")
    analyzer = EDAAnalyzer(df)
    
    # Analyze basic stats which includes detected types
    stats = analyzer.analyze_basic_stats()
    
    print("Detected semantic types in EDAAnalyzer:")
    if "semantic_types" in stats:
        for col, sem_type in stats["semantic_types"].items():
            print(f"  {col}: {sem_type}")
    
    print("\nDone!")