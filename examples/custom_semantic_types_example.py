"""
Example demonstrating how to use custom semantic type detection patterns.
"""
import pandas as pd
import numpy as np

from freamon.eda import (
    DataTypeDetector,
    detect_column_types,
    optimize_dataframe_types,
    EDAAnalyzer
)

# Create a sample dataset with domain-specific data types
def create_sample_dataset():
    """Create a synthetic dataset with domain-specific data types."""
    # Sample size
    n_rows = 100
    
    # Generate some data with custom patterns
    data = {
        # Standard columns
        'id': range(1, n_rows + 1),
        'name': [f"Customer {i}" for i in range(n_rows)],
        
        # Financial domain-specific patterns
        'account_number': [f"ACCT-{np.random.randint(10000, 99999)}" for _ in range(n_rows)],
        'tax_id': [f"{np.random.randint(10, 99)}-{np.random.randint(1000000, 9999999)}" for _ in range(n_rows)],
        'swift_code': [f"BANK{np.random.choice(['US', 'UK', 'FR', 'DE', 'JP'])}{np.random.choice(['XX', 'YY', 'ZZ'])}" for _ in range(n_rows)],
        
        # Healthcare domain-specific patterns
        'medical_record_number': [f"MRN-{np.random.randint(100000, 999999)}" for _ in range(n_rows)],
        'diagnosis_code': [f"ICD-{np.random.randint(10, 99)}.{np.random.randint(1, 9)}" for _ in range(n_rows)],
        
        # E-commerce domain-specific patterns
        'product_sku': [f"SKU-{np.random.choice(['A', 'B', 'C'])}{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'order_id': [f"ORD-{np.random.randint(10000, 99999)}" for _ in range(n_rows)],
        
        # Sensor data patterns (IoT domain)
        'device_id': [f"DEV-{np.random.choice(['TEMP', 'HUM', 'PRES'])}-{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'reading': np.random.normal(25, 5, n_rows),  # Temperature readings
    }
    
    return pd.DataFrame(data)

# Define custom patterns for domain-specific data types
def get_custom_patterns():
    """Define regex patterns for custom semantic types."""
    return {
        # Financial domain patterns
        'account_number': r'^ACCT-\d{5}$',
        'tax_id': r'^\d{2}-\d{7}$',
        'swift_code': r'^BANK[A-Z]{2}[A-Z]{2}$',
        
        # Healthcare domain patterns
        'medical_record_number': r'^MRN-\d{6}$',
        'diagnosis_code': r'^ICD-\d{2}\.\d{1}$',
        
        # E-commerce domain patterns
        'product_sku': r'^SKU-[A-C]\d{4}$',
        'order_id': r'^ORD-\d{5}$',
        
        # IoT domain patterns
        'device_id': r'^DEV-(TEMP|HUM|PRES)-\d{4}$',
    }

if __name__ == "__main__":
    # Create the sample dataset
    df = create_sample_dataset()
    
    print("=== Sample Dataset ===")
    print(df.head())
    
    # Get custom patterns
    custom_patterns = get_custom_patterns()
    
    print("\n=== Custom Pattern Definitions ===")
    for pattern_name, pattern in custom_patterns.items():
        print(f"{pattern_name}: {pattern}")
    
    print("\n=== Standard Type Detection (without custom patterns) ===")
    standard_types = detect_column_types(df)
    for col in df.columns:
        semantic_type = standard_types[col].get('semantic_type', 'None')
        print(f"{col}: {semantic_type}")
    
    print("\n=== Enhanced Type Detection (with custom patterns) ===")
    enhanced_types = detect_column_types(df, custom_patterns=custom_patterns)
    for col in df.columns:
        semantic_type = enhanced_types[col].get('semantic_type', 'None')
        print(f"{col}: {semantic_type}")
    
    print("\n=== Using DataTypeDetector Class Directly ===")
    # Create a detector with custom patterns
    detector = DataTypeDetector(
        df,
        sample_size=50,  # Use a smaller sample size for this example
        threshold=0.85,  # Slightly lower threshold
        custom_patterns=custom_patterns
    )
    
    # Detect types
    detector.detect_all_types()
    
    # Get the detailed column report
    report = detector.get_column_report()
    
    # Show detailed info for a few columns
    for col in ['medical_record_number', 'product_sku', 'device_id']:
        print(f"\nDetailed report for '{col}':")
        print(f"  Storage type: {report[col]['storage_type']}")
        print(f"  Logical type: {report[col]['logical_type']}")
        print(f"  Semantic type: {report[col].get('semantic_type', 'None')}")
        print(f"  Null count: {report[col]['null_count']}")
        print(f"  Unique count: {report[col]['unique_count']}")
    
    print("\n=== Using with EDAAnalyzer ===")
    analyzer = EDAAnalyzer(df)
    
    # Unfortunately, we can't directly pass custom patterns to the EDAAnalyzer.
    # We could enhance the EDAAnalyzer to accept custom patterns in a future update.
    analyzer.analyze_basic_stats()
    
    print("\n=== Suggestion for Enhancing EDAAnalyzer ===")
    print("To fully support custom patterns in EDAAnalyzer, we should:")
    print("1. Update the EDAAnalyzer.__init__ method to accept custom_patterns")
    print("2. Pass these patterns to the DataTypeDetector in _set_column_types()")
    print("3. Store the detected semantic types in the analysis results")
    
    print("\nExample implementation:")
    print("""
    def __init__(
        self,
        df: Any,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        \"\"\"Initialize the EDAAnalyzer.\"\"\"
        self.dataframe_type = check_dataframe_type(df)
        self.custom_patterns = custom_patterns
        
        # Convert to pandas for analysis
        if self.dataframe_type != 'pandas':
            self.df = convert_dataframe(df, 'pandas')
        else:
            self.df = df.copy()
        
        self.target_column = target_column
        self.date_column = date_column
        
        # Validate columns...
        
        # Set column types
        self._set_column_types()
        
        # Set basic stats
        self.n_rows, self.n_cols = self.df.shape
        
        # Initialize result storage
        self.analysis_results = {}
    
    def _set_column_types(self) -> None:
        \"\"\"Identify column types with advanced type detection.\"\"\"
        # Use advanced type detection with custom patterns
        detector = DataTypeDetector(
            self.df, 
            custom_patterns=self.custom_patterns
        )
        detected_types = detector.detect_all_types()
        
        # Rest of the method...
    """)
    
    print("\nDone!")