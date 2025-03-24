#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test and apply the underscore character fix in the matplotlib formatting.

This script creates a simple test chart with various formatting challenges to 
verify that the fixes for underscores, dollars, and other special characters 
are working correctly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path if needed
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the fixes
from freamon.utils.matplotlib_fixes import (
    configure_matplotlib_for_currency, 
    apply_comprehensive_matplotlib_patches
)

# Apply the patches
print("Applying matplotlib patches...")
configure_matplotlib_for_currency()
apply_comprehensive_matplotlib_patches()

# Create test data with various special characters
labels = [
    "variable_with_underscore",
    "price_$_dollars",
    "percent_%_value", 
    "mixed_$_%_^_chars",
    "normal text"
]

values = np.random.rand(len(labels)) * 100

# Create a plot to test the rendering
plt.figure(figsize=(12, 8))
plt.barh(labels, values)
plt.title("Testing Special Characters in Labels")
plt.xlabel("Value")
plt.ylabel("Variable Name")

# Add annotations with special characters
for i, label in enumerate(labels):
    plt.text(values[i] + 2, i, f"Value: {values[i]:.2f} for {label}")

# Save the plot
plt.tight_layout()
plt.savefig("underscore_fix_test.png")
plt.close()

print("Test completed. Check underscore_fix_test.png for results.")

# Test text preprocessing directly
from freamon.utils.matplotlib_fixes import preprocess_text_for_matplotlib, fix_matplotlib_placeholders

test_strings = [
    "variable_with_underscore",
    "price_$100.00",
    "75% completion",
    "mixed_$_%_^_{}_values",
    "normal text"
]

print("\nTesting text preprocessing and restoration:")
for original in test_strings:
    processed = preprocess_text_for_matplotlib(original)
    restored = fix_matplotlib_placeholders(processed)
    print(f"Original: '{original}'")
    print(f"Processed: '{processed}'")
    print(f"Restored: '{restored}'")
    print(f"Match: {'✓' if original == restored else '✗'}")
    print()