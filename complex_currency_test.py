"""
Test script to verify the fix for matplotlib LaTeX error with the exact problematic string.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Configure matplotlib to disable LaTeX parsing
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# The exact string that was causing the error
problematic_string = "INSD quotes are $5,995.00 for FWD65A95L SONY 65\" 4K UHD QD OLED TV and $5,384.00"

# Helper function to escape dollar signs for matplotlib
def escape_dollars(text):
    return text.replace('$', r'\$')

# Escape dollar signs in the strings
fixed_problematic_string = escape_dollars(problematic_string)

# Create a DataFrame
data = pd.DataFrame({
    'Description': [fixed_problematic_string, escape_dollars("$2,984.00 for Samsung TV"), escape_dollars("$1,999.00 for LG TV")],
    'Value': [5995, 2984, 1999]
})

# Create figure and plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(data)), data['Value'])

# Add shortened labels
plt.xticks(range(len(data)), [text[:30] + '...' for text in data['Description']], rotation=30, ha='right')

# Add values as text (avoiding problematic strings in annotations)
for i, val in enumerate(data['Value']):
    plt.text(i, val + 100, f"${val:,.2f}", ha='center')

plt.title('Values with Currency Symbols (Fixed)')
plt.tight_layout()
plt.savefig('complex_currency_test.png')
print("Complex test complete! Check complex_currency_test.png")