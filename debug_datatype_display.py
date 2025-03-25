"""Debug script to diagnose datatype detector issues"""
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path if not already there
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the problematic modules
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.matplotlib_fixes import (
    preprocess_text_for_matplotlib, 
    apply_comprehensive_matplotlib_patches,
    patch_freamon_eda
)

# Apply the matplotlib fixes to see if they affect the datatype detector
apply_comprehensive_matplotlib_patches()
patch_freamon_eda()

# Create simple test dataframe with underscore in column name
df = pd.DataFrame({
    'normal_column': [1, 2, 3],
    'column_with_underscore': [4, 5, 6],
    'column with space': [7, 8, 9],
    'percent_column': [10, 20, 30]
})

# Apply a null value to see percentage display
df.loc[0, 'percent_column'] = None

# Print original dataframe info
print("Original DataFrame:")
print(df)
print("\nColumn names:")
for col in df.columns:
    print(f"  - '{col}'")

# Create a datatype detector
print("\nCreating DataTypeDetector...")
detector = DataTypeDetector(df)

# Run detection
print("Running type detection...")
detector.detect_all_types()

# Show missing percentage calculation
print("\nMissing values percentages:")
for col in df.columns:
    null_count = df[col].isna().sum()
    null_pct = 100 * null_count / len(df)
    print(f"  - {col}: {null_count} nulls ({null_pct}%)")

# Print the column report and check for placeholders
print("\nColumn report content:")
try:
    column_report = detector.get_column_report()
    print(column_report)
except AttributeError:
    print("get_column_report method not found")

# Check if the column types dictionary contains placeholders
print("\nColumn types dictionary:")
for col, type_info in detector.column_types.items():
    print(f"  - '{col}': {type_info}")

# Check if matplotlib preprocessing affects column names
print("\nTesting matplotlib preprocessing on column names:")
for col in df.columns:
    processed = preprocess_text_for_matplotlib(col)
    print(f"  - '{col}' → '{processed}'")

# Check if there's any integration between datatype_detector and matplotlib_fixes
print("\nInspecting the detector object to find preprocessed strings:")
for attr_name in dir(detector):
    if not attr_name.startswith('__'):
        attr = getattr(detector, attr_name)
        if isinstance(attr, dict):
            print(f"\nChecking dictionary '{attr_name}':")
            for k, v in attr.items():
                if isinstance(k, str) and any(marker in k for marker in ['[UNDERSCORE]', '[PERCENT]']):
                    print(f"  Found preprocessed key: '{k}'")
                if isinstance(v, str) and any(marker in v for marker in ['[UNDERSCORE]', '[PERCENT]']):
                    print(f"  Found preprocessed value for key '{k}': '{v}'")
        elif isinstance(attr, list):
            print(f"\nChecking list '{attr_name}':")
            for item in attr:
                if isinstance(item, str) and any(marker in item for marker in ['[UNDERSCORE]', '[PERCENT]']):
                    print(f"  Found preprocessed item: '{item}'")