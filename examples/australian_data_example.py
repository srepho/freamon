"""
Example demonstrating Australian-specific data type detection.
"""
import pandas as pd
import numpy as np
from datetime import datetime

from freamon.eda import (
    DataTypeDetector,
    detect_column_types,
    optimize_dataframe_types,
    EDAAnalyzer
)

# Create a sample dataset with Australian data
def create_sample_au_dataset(n_rows=100):
    """Create a synthetic dataset with Australian data types."""
    np.random.seed(42)
    
    # Generate Australian postcodes - deliberately introducing common issues
    postcodes = []
    postcode_regions = [
        (200, 299),     # ACT (leading zero when stored as 4-digit string)
        (800, 899),     # NT (leading zero when stored as 4-digit string)
        (1000, 1999),   # NSW
        (2000, 2999),   # NSW
        (3000, 3999),   # VIC
        (4000, 4999),   # QLD
        (5000, 5999),   # SA
        (6000, 6999),   # WA
        (7000, 7999),   # TAS
    ]
    
    for _ in range(n_rows):
        region = np.random.choice(len(postcode_regions))
        min_val, max_val = postcode_regions[region]
        postcodes.append(np.random.randint(min_val, max_val))
    
    # Generate other Australian data
    data = {
        # Basic info
        'id': range(1, n_rows + 1),
        'name': [f"Customer {i}" for i in range(n_rows)],
        
        # Locations with both numeric and string postcodes
        'postcode_numeric': postcodes,
        'postcode_string': [f"{code:04d}" for code in postcodes],  # Properly formatted with leading zeros
        
        # Australian phone numbers
        'phone': [f"0{np.random.randint(2, 9)}{np.random.randint(1000000, 9999999)}" for _ in range(n_rows)],
        'mobile': [f"04{np.random.randint(10, 99)}{np.random.randint(100000, 999999)}" for _ in range(n_rows)],
        
        # Australian business identifiers
        'abn': [f"{np.random.randint(10, 99)} {np.random.randint(100, 999)} {np.random.randint(100, 999)} {np.random.randint(100, 999)}" for _ in range(n_rows)],
        'acn': [f"{np.random.randint(100, 999)} {np.random.randint(100, 999)} {np.random.randint(100, 999)}" for _ in range(n_rows)],
        'tfn': [f"{np.random.randint(100, 999)} {np.random.randint(100, 999)} {np.random.randint(100, 999)}" for _ in range(n_rows)],
    }
    
    return pd.DataFrame(data)

# Create state-specific data
def get_state_from_postcode(postcode):
    """Convert an Australian postcode to its state."""
    if isinstance(postcode, str):
        code = int(postcode)
    else:
        code = postcode
        
    if code >= 1000 and code <= 2599 or code >= 2619 and code <= 2899 or code >= 2921 and code <= 2999:
        return 'NSW'
    elif code >= 200 and code <= 299 or code >= 2600 and code <= 2618 or code >= 2900 and code <= 2920:
        return 'ACT'
    elif code >= 3000 and code <= 3999:
        return 'VIC'
    elif code >= 4000 and code <= 4999:
        return 'QLD'
    elif code >= 5000 and code <= 5999:
        return 'SA'
    elif code >= 6000 and code <= 6999:
        return 'WA'
    elif code >= 7000 and code <= 7999:
        return 'TAS'
    elif code >= 800 and code <= 999:
        return 'NT'
    else:
        return 'Unknown'

# Main example code
if __name__ == "__main__":
    print("Creating sample Australian dataset...")
    df = create_sample_au_dataset()
    
    # Add state column based on postcodes
    df['state'] = df['postcode_numeric'].apply(get_state_from_postcode)
    
    print("\n===== Dataset Preview =====")
    print(df.head())
    
    print("\n===== Basic Data Types =====")
    print(df.dtypes)
    
    print("\n===== Detecting Australian Data Types =====")
    detected_types = detect_column_types(df)
    
    # Show detected semantic types
    print("\nDetected Semantic Types:")
    for col, info in detected_types.items():
        if 'semantic_type' in info:
            print(f"{col}: {info['semantic_type']}")
    
    # Show conversion suggestions for postcodes
    print("\nConversion Suggestions:")
    for col, info in detected_types.items():
        if 'suggested_conversion' in info:
            print(f"{col}: {info['suggested_conversion']}")
    
    print("\n===== Optimizing Postcode Data =====")
    optimized_df = optimize_dataframe_types(df)
    
    # Show the difference in postcode columns
    print("\nBefore optimization:")
    print(f"postcode_numeric type: {df['postcode_numeric'].dtype}")
    print("First 5 postcodes:", df['postcode_numeric'].head().tolist())
    
    print("\nAfter optimization:")
    print(f"postcode_numeric type: {optimized_df['postcode_numeric'].dtype}")
    print("First 5 postcodes:", optimized_df['postcode_numeric'].head().tolist())
    
    # Count how many NT/ACT postcodes had leading zeros
    nt_act_postcodes = df[(df['state'] == 'NT') | (df['state'] == 'ACT')]['postcode_numeric']
    numeric_count = len(nt_act_postcodes)
    
    if numeric_count > 0:
        print(f"\nFound {numeric_count} NT/ACT postcodes that should have leading zeros:")
        for code in nt_act_postcodes.head(5):
            print(f"Original: {code} -> Properly formatted: {code:04d}")
    
    print("\n===== Using with EDAAnalyzer =====")
    analyzer = EDAAnalyzer(df)
    stats = analyzer.analyze_basic_stats()
    
    if "semantic_types" in stats:
        print("\nSemantic types detected by EDAAnalyzer:")
        for col, sem_type in stats["semantic_types"].items():
            print(f"  {col}: {sem_type}")
            
    print("\nDone!")