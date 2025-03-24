"""
Fixes for the datatype detector to handle matplotlib placeholder issues.

This module patches the DataTypeDetector class to properly handle
column names and other text that may have been processed by matplotlib_fixes.
"""

import functools
import logging
from typing import Dict, Any, Optional

# Fix the import for matplotlib_fixes
try:
    from freamon.utils.matplotlib_fixes import fix_matplotlib_placeholders
except ImportError:
    # Define a fallback if the module can't be imported
    def fix_matplotlib_placeholders(text):
        if not isinstance(text, str):
            return text
            
        replacements = {
            '[UNDERSCORE]': '_',
            '[PERCENT]': '%',
            '[DOLLAR]': '$',
            '[CARET]': '^',
            '[BACKSLASH]': '\\',
            '[LBRACE]': '{',
            '[RBRACE]': '}'
        }
        
        for placeholder, original in replacements.items():
            text = text.replace(placeholder, original)
            
        return text

def apply_datatype_detector_patches():
    """
    Apply fixes to the DataTypeDetector class to handle matplotlib placeholder issues.
    
    This function patches the DataTypeDetector methods that generate reports
    to ensure column names and other text values are displayed correctly.
    
    Returns
    -------
    bool
        True if patching was successful, False otherwise
    """
    try:
        # Import the DataTypeDetector class
        from freamon.utils.datatype_detector import DataTypeDetector
        
        # Get original methods to patch
        original_get_column_report = DataTypeDetector.get_column_report
        
        @functools.wraps(original_get_column_report)
        def patched_get_column_report(self):
            """
            Patched version of get_column_report that fixes matplotlib placeholders.
            """
            # Get the original report
            report = original_get_column_report(self)
            
            # Process the report to fix placeholders
            fixed_report = {}
            for col_name, col_info in report.items():
                # Fix the column name if it contains placeholders
                fixed_col_name = fix_matplotlib_placeholders(col_name) if isinstance(col_name, str) else col_name
                
                # Fix values in the column info dictionary
                fixed_info = {}
                for k, v in col_info.items():
                    if isinstance(v, str):
                        fixed_info[k] = fix_matplotlib_placeholders(v)
                    elif isinstance(v, dict):
                        # Handle nested dictionaries (like suggested_conversion)
                        fixed_nested = {}
                        for nk, nv in v.items():
                            if isinstance(nv, str):
                                fixed_nested[nk] = fix_matplotlib_placeholders(nv)
                            else:
                                fixed_nested[nk] = nv
                        fixed_info[k] = fixed_nested
                    else:
                        fixed_info[k] = v
                
                fixed_report[fixed_col_name] = fixed_info
            
            return fixed_report
        
        # Apply the patch
        DataTypeDetector.get_column_report = patched_get_column_report
        
        # If there's a display_detection_report method, patch it too
        if hasattr(DataTypeDetector, 'display_detection_report'):
            # The display method will automatically benefit from the patched get_column_report
            pass
        
        # Also patch the get_column_report_html method if it exists
        if hasattr(DataTypeDetector, 'get_column_report_html'):
            original_get_html = DataTypeDetector.get_column_report_html
            
            @functools.wraps(original_get_html)
            def patched_get_html(self):
                """
                Patched version of get_column_report_html that fixes matplotlib placeholders.
                """
                # Get the HTML report
                html = original_get_html(self)
                
                # Replace placeholders in the HTML
                html = fix_matplotlib_placeholders(html)
                
                return html
            
            # Apply the HTML patch
            DataTypeDetector.get_column_report_html = patched_get_html
        
        logging.info("Successfully applied fixes to DataTypeDetector")
        return True
        
    except Exception as e:
        logging.error(f"Failed to apply datatype detector patches: {str(e)}")
        return False

# Automatically apply the patches when this module is imported
apply_datatype_detector_patches()