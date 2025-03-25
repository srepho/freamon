# Markdown Reports in Freamon

This document provides an overview of the markdown report generation functionality in Freamon.

## Introduction

Markdown reports offer a lightweight alternative to full HTML reports, providing several advantages:

- **Simplicity**: Plain text format that's easy to read even without rendering
- **Version Control**: Easy to track changes in git repositories
- **Portability**: Can be displayed on any platform, including GitHub, GitLab, etc.
- **Convertibility**: Easy to convert to other formats (HTML, PDF, etc.)
- **Performance**: Significantly smaller file size compared to HTML with embedded images

## Report Generation Options

Freamon offers two main approaches to creating markdown reports:

### 1. Using the `generate_markdown_report` Function

The `generate_markdown_report` function provides a simple way to generate standard reports:

```python
from freamon.eda.markdown_report import generate_markdown_report

report = generate_markdown_report(
    df, 
    title="Data Analysis Report",
    description="Analysis of key metrics and trends",
    target_column='target',
    include_correlations=True,
    include_histograms=True,
    include_boxplots=True
)

# Save the report to a file
with open("analysis_report.md", "w") as f:
    f.write(report)
```

### 2. Using EDAAnalyzer with Markdown Format

The EDAAnalyzer class supports markdown output through its `generate_report` method:

```python
from freamon.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df, target_column='target')
analyzer.run_full_analysis()

# Generate a markdown report
analyzer.generate_report(
    output_path="eda_report.md",
    title="EDA Report",
    format="markdown"
)

# Generate markdown with HTML conversion
analyzer.generate_report(
    output_path="eda_report_with_html.md",
    title="EDA Report with HTML",
    format="markdown",
    convert_to_html=True
)
```

## Customization Options

### Basic Sections

The `generate_markdown_report` function supports various sections that can be included or excluded:

```python
report = generate_markdown_report(
    df, 
    title="Custom Analysis Report",
    description="Detailed analysis with specific sections",
    target_column='target',
    include_summary=True,          # Dataset summary
    include_correlations=True,     # Correlation analysis
    include_histograms=True,       # Histograms for numeric columns
    include_boxplots=True,         # Boxplots for numeric columns
    include_value_counts=True,     # Value counts for categorical columns
    include_scatter_matrix=False,  # Scatter matrix plot
    include_missing_analysis=True, # Missing value analysis
    max_categorical_values=10      # Limit for categorical value display
)
```

### Creating Custom Reports

For full customization, you can build markdown reports from scratch or extend generated ones:

```python
# Generate a base report
base_report = generate_markdown_report(df, title="Base Report")

# Add custom sections
custom_section = """
## Advanced Analysis

Based on the data, we observe the following insights:
1. Feature X shows a strong correlation with the target (r=0.75)
2. Categories A and B perform significantly better than C and D

### Recommendations
- Focus on Category A for best results
- Investigate the drop in performance for Category C
"""

# Combine base report with custom sections
full_report = base_report + "\n" + custom_section

# Save the combined report
with open("enhanced_report.md", "w") as f:
    f.write(full_report)
```

## Converting to HTML

Freamon provides utilities to convert markdown to HTML for sharing:

```python
# Method 1: Using EDAAnalyzer (automatic conversion)
analyzer.generate_report(
    output_path="report.md",
    format="markdown",
    convert_to_html=True  # Creates both report.md and report.md.html
)

# Method 2: Manual conversion
from markdown import markdown

with open("report.md", "r") as f:
    md_content = f.read()

html_content = markdown(md_content)

# Add HTML styling
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        h3 {{ color: #2980b9; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

with open("report.html", "w") as f:
    f.write(styled_html)
```

## Including Images in Reports

To add images to your markdown reports:

1. Generate and save plots:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Create your plot...
plt.savefig("my_plot.png")
```

2. Reference images in the markdown:
```markdown
![Plot Title](my_plot.png)
```

When converting to HTML, these images will be properly displayed.

## Common Use Cases

- **Executive Summaries**: Create concise reports focusing on key metrics
- **Data Quality Reports**: Document data quality issues and recommendations
- **Model Performance Reports**: Generate model evaluation metrics and visualizations
- **Integration with Documentation**: Include in software documentation or wikis
- **Version-Controlled Analysis**: Keep analysis reports in version control alongside code

## Example

### Generating a Complex Report

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from freamon.eda.markdown_report import generate_markdown_report

# Create report directory
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

# Generate base report
report = generate_markdown_report(
    df, 
    title="Sales Analysis Report",
    description="Monthly analysis of product performance",
    target_column='sales',
    include_correlations=True,
    include_histograms=True
)

# Add custom visualization
plt.figure(figsize=(10, 6))
df.groupby('month')['sales'].sum().plot(kind='bar')
plt.title('Monthly Sales')
plt.tight_layout()
plt.savefig(report_dir / "monthly_sales.png")

# Add custom section with the visualization
custom_section = """
## Monthly Performance

The chart below shows monthly sales performance:

![Monthly Sales](monthly_sales.png)

### Key Insights:
1. November and December show highest sales (holiday season)
2. February shows lowest performance across all products
3. Summer months show consistent growth trend
"""

# Combine base report with custom section
full_report = report + "\n" + custom_section

# Save the report
with open(report_dir / "sales_report.md", "w") as f:
    f.write(full_report)
```

## Benefits Over HTML Reports

- **File Size**: Typically 10-20x smaller than equivalent HTML reports
- **Version Control**: Easy to track changes with meaningful diffs
- **Editing**: Can be edited directly in text editors or IDEs
- **Platform Compatibility**: Renders correctly on GitHub, GitLab, Notion, and other platforms
- **Conversion**: Easy to convert to multiple formats (HTML, PDF, DOCX) as needed

For full documentation of all markdown report options, refer to the docstrings in the code or review the example scripts in the `examples` directory.