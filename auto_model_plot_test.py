"""
Simple test script to verify that matplotlib plotting works in the conda environment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset manually
np.random.seed(42)
data = np.random.randn(50, 3)
feature_names = ['Feature A', 'Feature B', 'Feature C']

# Create feature importance values
importance_values = np.array([0.5, 0.3, 0.2])

# Create plot
plt.figure(figsize=(8, 6))
plt.bar(feature_names, importance_values)
plt.title('Feature Importance Test')
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_test.png')

print("Plot saved as feature_importance_test.png")