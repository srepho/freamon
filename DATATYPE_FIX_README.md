# Freamon Formatting Fix

## Issue Description

The DataTypeDetector class was displaying modified column names and values with placeholders:

- Column names with underscores showing as `column[UNDERSCORE]name`
- Percentage values displaying as `33.3[PERCENT]`
- Dollar signs showing as `[DOLLAR]100`

This happened because the matplotlib preprocessing that protects special characters for plotting was being applied to column names and values in datatype reports.

## Solution

We implemented a fix with three key components:

1. Added a `fix_matplotlib_placeholders()` function to the `matplotlib_fixes.py` module that reverses the placeholder replacements
2. Created a `datatype_fixes.py` module that patches the DataTypeDetector's report methods to clean up placeholders
3. Modified `safe_process_dataframe()` to add an option to skip column name processing

### The `fix_matplotlib_placeholders()` Function

This function reverses the replacements made by the `preprocess_text_for_matplotlib()` function:

```python
def fix_matplotlib_placeholders(text):
    """Replace matplotlib placeholder markers with their original characters."""
    if not isinstance(text, str):
        return text
        
    # Replace common placeholder markers
    text = text.replace('[UNDERSCORE]', '_')
    text = text.replace('[PERCENT]', '%')
    text = text.replace('[DOLLAR]', '$')
    text = text.replace('[CARET]', '^')
    text = text.replace('[BACKSLASH]', '\\')
    text = text.replace('[LBRACE]', '{')
    text = text.replace('[RBRACE]', '}')
    
    return text
```

### The DataType Fixes Module

The `datatype_fixes.py` module patches the DataTypeDetector methods:

```python
def apply_datatype_detector_patches():
    """Apply fixes to handle matplotlib placeholder issues."""
    # Get original methods
    original_get_column_report = DataTypeDetector.get_column_report
    
    @functools.wraps(original_get_column_report)
    def patched_get_column_report(self):
        # Get the original report
        report = original_get_column_report(self)
        
        # Process the report to fix placeholders
        fixed_report = {}
        for col_name, col_info in report.items():
            # Fix the column name
            fixed_col_name = fix_matplotlib_placeholders(col_name)
            
            # Fix values in the column info dictionary
            fixed_info = {}
            for k, v in col_info.items():
                if isinstance(v, str):
                    fixed_info[k] = fix_matplotlib_placeholders(v)
                else:
                    fixed_info[k] = v
            
            fixed_report[fixed_col_name] = fixed_info
        
        return fixed_report
    
    # Apply the patch
    DataTypeDetector.get_column_report = patched_get_column_report
```

## Usage

To apply these fixes to your project, import the datatype_fixes module:

```python
from freamon.utils.datatype_fixes import apply_datatype_detector_patches
```

This will automatically apply the patches. If you want to manually fix placeholder text:

```python
from freamon.utils.matplotlib_fixes import fix_matplotlib_placeholders

# Fix a string with placeholders
cleaned_text = fix_matplotlib_placeholders("column[UNDERSCORE]name")
# Result: "column_name"
```

## Testing

A test file `test_datatype_fix.py` verifies the fix works correctly. It tests:

1. Column name preprocessing and fixing
2. Integration with the DataTypeDetector
3. Proper cleaning of null percentage values

## Implementation Details

The issue was caused by matplotlib's special character handling. When plotting text with characters like '$', '_', and '%', matplotlib tries to interpret them as math notation. To prevent this, the code replaces these characters with placeholder text.

However, this replacement was happening for all text, including column names and report data, not just plot labels. Our fix ensures that the original characters are restored for display in reports while still protecting them for plotting.

By applying these fixes, the DataTypeDetector now displays column names and values correctly.