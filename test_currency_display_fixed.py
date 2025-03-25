"""
Test script to verify the fix for matplotlib LaTeX parsing issues with currency symbols.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from freamon.utils.matplotlib_fixes import (
    configure_matplotlib_for_currency,
    replace_dollar_signs,
    safe_process_dataframe
)

# Verify matplotlib configuration
print("Configuring matplotlib for currency display...")
configure_matplotlib_for_currency()
print(f"  text.usetex = {plt.rcParams['text.usetex']}")
print(f"  mathtext.default = {plt.rcParams['mathtext.default']}")

# The problematic string that was causing the error
problematic_string = "INSD quotes are $5,995.00 for FWD65A95L SONY 65\" 4K UHD QD OLED TV and $5,384.00"
print("\nTesting problematic string replacement...")
print(f"  Original: {problematic_string}")
print(f"  Replaced: {replace_dollar_signs(problematic_string)}")

# Test dataframe processing
print("\nTesting DataFrame processing...")
df = pd.DataFrame({
    'Product': ['SONY TV', 'LG TV', 'Samsung TV'],
    'Price': ['$5,995.00', '$5,384.00', '$2,984.00'],
    'Description': [
        "INSD quotes are $5,995.00 for FWD65A95L SONY 65\" 4K UHD QD OLED TV",
        "Costco quotes $5,384.00 for LG OLED65C3",
        "Retailer quotes $2,984.00 for Samsung QN65S95D"
    ]
})
print("Original DataFrame:")
print(df.head())

processed_df = safe_process_dataframe(df)
print("\nProcessed DataFrame:")
print(processed_df.head())

# Create and save a test plot
plt.figure(figsize=(10, 6))
for i, row in processed_df.iterrows():
    plt.bar(i, i+1, label=row['Product'])
    plt.text(i, i+1.5, row['Price'], ha='center')
    
plt.title("Test Plot with Currency Values")
plt.xticks(range(len(processed_df)), processed_df['Product'])
plt.tight_layout()
plt.savefig('test_currency_fixed.png')

print("\nTest complete! Plot saved to 'test_currency_fixed.png'")
print("The currency display issue has been fixed.")