"""
Test script to diagnose and fix matplotlib dollar sign rendering issues.
"""
import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Backend: {matplotlib.get_backend()}")

# Configure matplotlib to avoid LaTeX parsing for dollar signs
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Print all text-related rcParams for debugging
print("\nMatplotlib text configuration:")
for key, value in plt.rcParams.items():
    if 'text' in key or 'math' in key:
        print(f"{key}: {value}")

# Create a simple dataframe with dollar values
import pandas as pd
import numpy as np

# Data with currency values
data = pd.DataFrame({
    'Product': ['TV', 'Laptop', 'Phone', 'Tablet'],
    'Price': ['$5,995.00', '$2,499.99', '$1,199.99', '$899.99']
})

# Create a plot with dollar signs
plt.figure(figsize=(10, 6))

# Create a bar chart
y_pos = np.arange(len(data['Product']))
prices = [float(p.replace('$', '').replace(',', '')) for p in data['Price']]
bars = plt.bar(y_pos, prices)

# Add price labels with dollar signs (which would normally cause LaTeX issues)
for i, (bar, price) in enumerate(zip(bars, data['Price'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             price, ha='center', va='bottom')

plt.xticks(y_pos, data['Product'])
plt.title('Product Prices with Dollar Signs')
plt.ylabel('Price')
plt.tight_layout()

# Save the figure
plt.savefig('dollar_sign_test.png')
print(f"\nTest completed successfully. Plot saved to dollar_sign_test.png")