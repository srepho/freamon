"""
Helper script to configure matplotlib to properly handle currency symbols, 
especially dollar signs, in plot labels and text.

This script applies comprehensive patches to matplotlib and the freamon EDA module
to prevent crashes and errors when displaying currency values in reports and plots.

Example usage:
-------------
# Import the helper function
from configure_matplotlib_for_currency import patch_freamon

# Apply the patches
patch_freamon()

# Now use freamon as usual
from freamon.eda import EDAAnalyzer
analyzer = EDAAnalyzer(df)
analyzer.run_full_analysis(output_path='robust_EDA_report.html')
"""

from freamon.utils.matplotlib_fixes import patch_freamon_eda, configure_matplotlib_for_currency

def patch_freamon():
    """
    Apply comprehensive patches to matplotlib and freamon.
    
    This function:
    1. Configures matplotlib to properly handle currency symbols
    2. Patches the freamon EDA module to handle errors gracefully
    3. Makes EDA reports more robust against rendering issues
    
    Returns
    -------
    bool
        True if patching was successful, False otherwise
    """
    # Apply all patches
    success = patch_freamon_eda()
    if success:
        print("Successfully applied EDA and matplotlib patches.")
        print("You can now safely use freamon with data containing currency symbols.")
    return success

# Apply patches when script is run directly
if __name__ == "__main__":
    patch_freamon()