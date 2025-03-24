"""
Helper script for patching matplotlib to handle currency symbols.

This script provides a simple entry point to apply all the necessary patches
to make matplotlib and the freamon EDA module work correctly with text
containing dollar signs and other special characters.
"""

from freamon.utils.matplotlib_fixes import (
    configure_matplotlib_for_currency,
    apply_comprehensive_matplotlib_patches,
    patch_freamon_eda
)

def patch_freamon():
    """
    Apply all necessary patches to make freamon work with currency symbols.
    
    This function:
    1. Configures matplotlib to handle dollar signs and special characters
    2. Patches the text rendering functionality to avoid LaTeX interpretation
    3. Patches the freamon EDA module to handle errors gracefully
    
    Returns
    -------
    bool
        True if all patches were successfully applied, False otherwise
    """
    # First apply the matplotlib configuration
    matplotlib_config_success = configure_matplotlib_for_currency()
    
    # Apply comprehensive patches to fix text rendering
    matplotlib_patches_success = apply_comprehensive_matplotlib_patches()
    
    # Patch the freamon EDA module
    freamon_patches_success = patch_freamon_eda()
    
    return matplotlib_config_success and matplotlib_patches_success and freamon_patches_success

if __name__ == "__main__":
    success = patch_freamon()
    if success:
        print("Successfully applied all patches to handle currency symbols")
    else:
        print("Warning: Some patches could not be applied")