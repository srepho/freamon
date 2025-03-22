"""
Scientific Notation Detection Example

This example demonstrates how the Freamon library can detect scientific notation
numbers in datasets. This is particularly useful when working with scientific data
that includes very large or very small numbers expressed in scientific notation
(e.g., 1.23e-10, 4.56e+5).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from freamon.utils.datatype_detector import DataTypeDetector, optimize_dataframe_types

# Create sample data with scientific notation
def create_scientific_notation_df():
    """Create a sample dataframe with scientific notation values."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Very small numbers (e.g., molecular concentrations)
    small_values = np.array([1.23e-10, 5.67e-9, 3.45e-11, 7.89e-10, 2.34e-12])
    
    # Large numbers (e.g., astronomical distances in meters)
    large_values = np.array([1.496e+11, 5.9e+9, 1.28e+10, 7.78e+11, 4.5e+12])
    
    # Medium numbers in scientific notation
    medium_values = np.array([3.84e+2, 6.02e+3, 9.81e+1, 2.998e+2, 6.626e+2])
    
    # Regular floating point numbers for comparison
    regular_values = np.array([0.1234, 56.78, 901.23, 45.67, 89.01])
    
    # Scientific calculations based on the values (to demonstrate usage)
    calculations = small_values * large_values
    
    return pd.DataFrame({
        'small_scientific': small_values,
        'large_scientific': large_values,
        'medium_scientific': medium_values,
        'regular_float': regular_values,
        'calculation_result': calculations,
        'description': ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']
    })

# Create and display the dataframe
print("Creating sample dataframe with scientific notation values...")
df = create_scientific_notation_df()
print("\nOriginal DataFrame:")
print(df)
print("\nDataFrame Info (original):")
print(df.dtypes)

# Use the DataTypeDetector to identify scientific notation
print("\nDetecting data types...")
detector = DataTypeDetector(df)
results = detector.detect_all_types()

# Print the detection results
print("\nDetection Results:")
for col, info in results.items():
    print(f"\n{col}:")
    print(f"  Storage type: {info['storage_type']}")
    print(f"  Logical type: {info['logical_type']}")
    if 'semantic_type' in info:
        print(f"  Semantic type: {info['semantic_type']}")
    if 'suggested_conversion' in info:
        print(f"  Suggested conversion: {info['suggested_conversion']['convert_to']} using {info['suggested_conversion']['method']}")
        if 'note' in info['suggested_conversion']:
            print(f"  Note: {info['suggested_conversion']['note']}")

# Visualize the different scales with a logarithmic plot
print("\nCreating logarithmic plot to visualize scientific notation values...")
plt.figure(figsize=(12, 6))

# Prepare data for plotting (absolute values for log scale)
plot_df = df[['small_scientific', 'large_scientific', 'medium_scientific', 'regular_float']].abs()

# Log scale bar chart
ax = plot_df.plot(kind='bar', logy=True, width=0.8, figsize=(12, 6))
plt.title('Scientific Notation Values (Log Scale)')
plt.ylabel('Value (log scale)')
plt.xlabel('Sample Index')
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('scientific_notation_example.png')
print("Plot saved as 'scientific_notation_example.png'")

print("\nScientific notation advantages:")
print("1. Compact representation of very large or very small numbers")
print("2. Maintains precision with extreme values")
print("3. Makes arithmetic operations more readable with extreme values")
print("4. Standard format for scientific and engineering data")

print("\nHow Scientific Notation Detection Works:")
print("1. Converts numeric values to string representation")
print("2. Uses regex pattern matching to identify scientific notation format (e.g., 1.23e-10)")
print("3. Calculates ratio of values matching the pattern")
print("4. If threshold is met, identifies column as containing scientific notation")
print("5. Provides information for proper handling in subsequent processing")

print("\nExample complete!")