# DataType Detection Fixes in Freamon

This document explains the fixes implemented for handling special characters and formatting in the DataTypeDetector.

## The Problem

When generating reports and visualizations with the DataTypeDetector, some special characters in column names and values were being displayed incorrectly as placeholder texts:

- Column names with underscores (`_`) appeared as `column[UNDERSCORE]name`
- Percentage values (%) appeared as `50[PERCENT]`
- Dollar signs ($) appeared as `[DOLLAR]100`
- And other special characters were similarly affected

This issue occurred because matplotlib uses LaTeX rendering, which requires special handling of certain characters.

## The Solution

Freamon now includes automatic fixes to handle these placeholder substitutions properly:

1. **Transparent Patching**: The DataTypeDetector is automatically patched at runtime to fix these issues
2. **Placeholder Reversal**: Special character placeholders are reversed when displaying results
3. **HTML Report Fixes**: All reports (interactive HTML, text, etc.) now display special characters correctly

## Using the Fixes

The fixes are applied automatically when importing the package. No additional code is required:

```python
import pandas as pd
from freamon.utils.datatype_detector import DataTypeDetector

# Create a DataFrame with special characters in column names
df = pd.DataFrame({
    'price_$': [100, 200, 300],
    'growth_%': [5.2, 7.8, 10.5],
    'category^type': ['A', 'B', 'C']
})

# The DataTypeDetector will handle these special characters correctly
detector = DataTypeDetector(df)
detector.detect_all_types()

# Generate report - column names will display correctly
report = detector.get_column_report()
print(report.keys())  # Will show: ['price_$', 'growth_%', 'category^type']

# HTML report will also display correctly
html_report = detector.get_column_report_html()
with open("datatype_report.html", "w") as f:
    f.write(html_report)
```

## Manual Applications

If you need to apply the fixes manually to other parts of your code, you can use the `fix_matplotlib_placeholders` function:

```python
from freamon.utils.matplotlib_fixes import fix_matplotlib_placeholders

# Fix a single string
column_name = "revenue[UNDERSCORE]2023[DOLLAR]"
fixed_name = fix_matplotlib_placeholders(column_name)
print(fixed_name)  # Output: "revenue_2023$"

# Fix all column names in a DataFrame
df.columns = [fix_matplotlib_placeholders(col) for col in df.columns]
```

## Compatibility

These fixes are compatible with all existing code that uses the DataTypeDetector. The changes are applied through function patching, so no modification to your existing code is needed.

## Affected Special Characters

The following special characters are now handled correctly:

| Character | LaTeX Special | Previous Display | Fixed Display |
|-----------|---------------|------------------|---------------|
| _ (underscore) | Subscript | [UNDERSCORE] | _ |
| % (percent) | Comment | [PERCENT] | % |
| $ (dollar) | Math mode | [DOLLAR] | $ |
| ^ (caret) | Superscript | [CARET] | ^ |
| \ (backslash) | Escape | [BACKSLASH] | \ |
| { (left brace) | Group begin | [LBRACE] | { |
| } (right brace) | Group end | [RBRACE] | } |

## Example: Currency and Percentages

```python
import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.eda.analyzer import EDAAnalyzer

# Create sample financial data
data = {
    'product': ['A', 'B', 'C', 'D', 'E'],
    'price_$': [199.99, 299.99, 149.99, 399.99, 249.99],
    'sales_volume': [150, 75, 300, 50, 125],
    'growth_%': [5.2, -1.8, 10.5, 3.7, 0.0],
    'discount_%': [0, 10, 5, 15, 7.5]
}

df = pd.DataFrame(data)

# Generate EDA report with correct column names and values
analyzer = EDAAnalyzer(df)
analyzer.run_full_analysis(
    output_path="financial_data_report.html",
    title="Financial Data Analysis"
)

# The report will show correct column names like "price_$" and "growth_%"
# Values will correctly display as "$199.99" and "5.2%"
```