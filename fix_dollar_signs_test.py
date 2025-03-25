"""
Test script to verify the fix for matplotlib LaTeX parsing issues with dollar signs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from freamon.utils.matplotlib_fixes import replace_dollar_signs

# Configure matplotlib
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# The exact problematic string that was causing the error
problematic_string = "INSD quotes are $5,995.00 for FWD65A95L SONY 65\" 4K UHD QD OLED TV and $5,384.00"

# Fix the string by replacing dollar signs
fixed_string = replace_dollar_signs(problematic_string)
print(f"Original: {problematic_string}")
print(f"Fixed: {fixed_string}")

# Create a figure with the fixed string
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(['Item 1', 'Item 2', 'Item 3'], [5995, 5384, 2984])

# Add the fixed string as a title - this would previously cause an error
plt.title(fixed_string)

# Add other text with dollar signs
plt.text(0, 3000, fixed_string, fontsize=10)
plt.text(1, 4000, "Price: [DOLLAR]5,384.00", fontsize=10)
plt.text(2, 2000, "Price: [DOLLAR]2,984.00", fontsize=10)

# Save the figure
plt.tight_layout()
plt.savefig('fix_dollar_signs_test.png')
print("Test complete! Figure saved to fix_dollar_signs_test.png")
print("The dollar sign issue has been fixed.")