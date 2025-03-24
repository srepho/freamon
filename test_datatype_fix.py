"""
Test the datatype detector fixes to handle matplotlib placeholder issues.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path if not already there
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# First import the matplotlib_fixes to ensure patches are applied
from freamon.utils.matplotlib_fixes import (
    configure_matplotlib_for_currency,
    preprocess_text_for_matplotlib,
    fix_matplotlib_placeholders
)

# Apply the matplotlib fixes to trigger the issue
configure_matplotlib_for_currency()

# Now import our datatype_fixes to patch the DataTypeDetector
from freamon.utils.datatype_fixes import apply_datatype_detector_patches

# Import the DataTypeDetector after the patches
from freamon.utils.datatype_detector import DataTypeDetector

# Function to test a column name for placeholders
def check_for_placeholders(text):
    """Check if a string contains any matplotlib placeholder markers."""
    if not isinstance(text, str):
        return False
        
    placeholders = ['[UNDERSCORE]', '[PERCENT]', '[DOLLAR]', '[CARET]']
    return any(p in text for p in placeholders)

# Create a test dataframe with column names that would be affected
df = pd.DataFrame({
    'normal_column': [1, 2, 3],
    'column_with_underscore': [4, 5, 6],
    'column_with_percent%': [7, 8, 9],
    'column_with_$dollar': [10, 20, 30]
})

# Apply a null value to see percentage display
df.loc[0, 'column_with_percent%'] = None

print("Test 1: Column Name Processing")
print("-" * 50)

# First, manually check what matplotlib preprocessing would do
print("Original column names vs preprocessed by matplotlib:")
for col in df.columns:
    preprocessed = preprocess_text_for_matplotlib(col)
    fixed = fix_matplotlib_placeholders(preprocessed)
    print(f"Original: '{col}' → Preprocessed: '{preprocessed}' → Fixed: '{fixed}'")

print("\nTest 2: DataTypeDetector Integration")
print("-" * 50)

# Create a datatype detector with our patched methods
detector = DataTypeDetector(df)

# Run detection
print("Running type detection...")
detector.detect_all_types()

# Get the report
report = detector.get_column_report()

# Check report column names
print("\nReport column names:")
for col in report.keys():
    print(f"  - '{col}' (has placeholders: {check_for_placeholders(col) if isinstance(col, str) else False})")

# Check null percentages
print("\nMissing values percentages:")
for col, info in report.items():
    if 'null_percentage' in info:
        placeholder_in_pct = check_for_placeholders(str(info['null_percentage']))
        print(f"  - {col}: {info['null_percentage']} (has placeholders: {placeholder_in_pct})")

print("\nTest complete!")