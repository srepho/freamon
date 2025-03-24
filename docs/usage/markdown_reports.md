# Markdown Reports in Freamon EDA

Freamon EDA supports generating reports in Markdown format, which offers several advantages over HTML reports:

1. **Readability**: Markdown files are human-readable even without rendering
2. **Compatibility**: Markdown is supported by many platforms including GitHub, GitLab, Notion, etc.
3. **Version Control**: Markdown files work well with version control systems like Git
4. **Flexibility**: Markdown can be easily converted to other formats (HTML, PDF, etc.)
5. **Lightweight**: Markdown files are typically smaller than HTML files

## Basic Usage

To generate a Markdown report from your EDA analysis:

```python
from freamon.eda import EDAAnalyzer

# Create analyzer and run analysis
analyzer = EDAAnalyzer(df, target_column='target', date_column='date')
analyzer.run_full_analysis()

# Generate Markdown report
analyzer.generate_report(
    output_path='eda_report.md',  # Output file path
    title='My EDA Report',        # Report title 
    format='markdown'             # Specify Markdown format
)
```

## Converting to HTML

You can also generate a Markdown report and convert it to HTML in a single step:

```python
analyzer.generate_report(
    output_path='eda_report.md',
    title='My EDA Report',
    format='markdown',
    convert_to_html=True          # Also create an HTML version
)
```

This will create both `.md` and `.md.html` files, allowing you to choose the format that best suits your needs.

## Example

Here's a complete example that demonstrates generating a Markdown report with analytics for a sample dataset:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from freamon.eda import EDAAnalyzer

# Create a sample dataset
np.random.seed(42)
n_samples = 500
dates = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(n_samples)]
sales = 1000 + np.random.normal(0, 200, n_samples) + np.arange(n_samples) * 0.5
categories = np.random.choice(['A', 'B', 'C'], n_samples)

df = pd.DataFrame({
    'date': dates,
    'sales': sales,
    'category': categories,
    'in_stock': np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
})

# Run the analysis
analyzer = EDAAnalyzer(df, target_column='in_stock', date_column='date')
results = analyzer.run_full_analysis()

# Generate Markdown report
analyzer.generate_report(
    output_path='sales_analysis.md',
    title='Sales Data Analysis',
    format='markdown'
)

# Generate Markdown report with HTML conversion
analyzer.generate_report(
    output_path='sales_analysis_with_html.md',
    title='Sales Data Analysis',
    format='markdown',
    convert_to_html=True
)
```

## Customization

The Markdown report includes:

- Dataset overview and basic statistics
- Univariate analysis for each column
- Bivariate relationships (correlations, feature-target relationships)
- Multivariate analysis (if performed)
- Time series analysis (if time column provided)
- Feature importance (if calculated)

All visualizations are embedded as base64-encoded images, ensuring the report is self-contained.

## Dependencies

To use the Markdown report functionality, you may need to install the optional dependency:

```bash
pip install freamon[markdown_reports]
```

Or install the markdown package directly:

```bash
pip install markdown
```

## Best Practices

- **Version Control**: Commit Markdown reports to your Git repository for future reference
- **Documentation**: Use Markdown reports in your project documentation
- **Collaboration**: Share Markdown reports with stakeholders who may not have access to your development environment
- **Conversion**: Convert to other formats using tools like Pandoc when needed
- **Integration**: Integrate with tools like Jupyter Book or MkDocs for comprehensive documentation