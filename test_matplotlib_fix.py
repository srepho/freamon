"""
Test script to fix matplotlib LaTeX parsing issues.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure matplotlib to avoid treating $ as LaTeX math delimiters
plt.rcParams['text.usetex'] = False  
plt.rcParams['mathtext.default'] = 'regular'

# Create sample data with $ values
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Value': [100, 200, 300, 400],
    'Label': ['$100', '$200', '$300', '$400']
})

# Create a simple plot
plt.figure(figsize=(10, 6))
plt.bar(data['Category'], data['Value'])

# Add labels with dollar signs
for i, (cat, val, label) in enumerate(zip(data['Category'], data['Value'], data['Label'])):
    plt.text(i, val + 10, label, ha='center')

plt.title('Sample Plot with Dollar Sign Values')
plt.xlabel('Category')
plt.ylabel('Value ($)')
plt.savefig('test_dollar_signs.png')
print("Test plot created successfully!")