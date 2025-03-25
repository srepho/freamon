"""
Test script to verify the fix for matplotlib LaTeX parsing issues.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame with currency data
df = pd.DataFrame({
    'Product': ['SONY 65" 4K UHD QD OLED TV', 'LG OLED65C3', 'Samsung QN65S95D'],
    'Price': ['$5,995.00', '$5,384.00', '$2,984.00'],
})

# Extract numeric values for plotting
prices = [float(p.replace('$', '').replace(',', '')) for p in df['Price']]

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Product'], prices)

# Add pricing labels directly from original data (including $ signs)
for bar, price in zip(bars, df['Price']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             price, ha='center', va='bottom')

plt.title('TV Price Comparison')
plt.ylabel('Price (USD)')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Save the figure
plt.savefig('currency_display_fixed.png')
print("Test complete! Check currency_display_fixed.png")