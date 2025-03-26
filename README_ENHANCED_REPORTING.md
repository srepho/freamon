# Enhanced Deduplication Reporting

Freamon provides a comprehensive reporting system for deduplication results, allowing you to generate detailed reports in multiple formats, including HTML, Excel, PowerPoint, and Markdown. The reporting system can also render interactive reports directly in Jupyter notebooks.

## Key Features

- **Multiple Export Formats**: Generate reports in HTML, Excel, PowerPoint, and Markdown formats from a single function call
- **Interactive Jupyter Integration**: Display interactive reports with widgets directly in Jupyter notebooks
- **Comprehensive Visualizations**: View confusion matrices, feature importances, and threshold evaluations
- **Business Impact Analysis**: Analyze the business impact of different threshold settings
- **Detailed Duplicate Pairs**: View sample duplicate pairs with similarity scores
- **Customizable Styling**: Control the appearance and content of reports

## Installation

To use the enhanced reporting functionality, install Freamon with the required dependencies:

```bash
pip install "freamon[enhanced_reporting]"
```

Or for the full set of features:

```bash
pip install "freamon[all]"  # Includes all optional dependencies
```

## Basic Usage

```python
from freamon.deduplication import flag_and_evaluate
from freamon.deduplication import generate_enhanced_report

# Run deduplication with evaluation
results = flag_and_evaluate(
    df=df,
    columns=['name', 'email', 'address'],
    known_duplicate_column='is_duplicate',
    threshold=0.7,
    method='weighted',
    flag_column='predicted_duplicate',
    generate_report=True,
    report_format='text'
)

# Generate reports in multiple formats
report_paths = generate_enhanced_report(
    results=results,
    formats=['html', 'excel', 'markdown', 'pptx'],
    output_dir="dedup_reports",
    title="Deduplication Analysis Report",
    include_pairs=True,
    max_pairs=50
)

# Print paths to generated reports
for format_name, path in report_paths.items():
    print(f"{format_name.upper()}: {path}")
```

## Jupyter Notebook Integration

For interactive reports in Jupyter notebooks, use the `display_jupyter_report` function:

```python
from freamon.deduplication import flag_and_evaluate
from freamon.deduplication import display_jupyter_report

# Run deduplication with evaluation
results = flag_and_evaluate(
    df=df,
    columns=['name', 'email', 'address'],
    known_duplicate_column='is_duplicate',
    threshold=0.7,
    method='weighted',
    flag_column='predicted_duplicate',
    generate_report=True,
    report_format='text'
)

# Display interactive report in the notebook
display_jupyter_report(
    results=results,
    title="Interactive Deduplication Analysis",
    include_pairs=True,
    max_pairs=10,
    interactive=True
)
```

## Advanced Usage with the EnhancedDeduplicationReport Class

For more control over the reporting process, use the `EnhancedDeduplicationReport` class directly:

```python
from freamon.deduplication import flag_and_evaluate
from freamon.deduplication.enhanced_reporting import EnhancedDeduplicationReport

# Run deduplication with evaluation
results = flag_and_evaluate(
    df=df,
    columns=['name', 'email', 'address'],
    known_duplicate_column='is_duplicate',
    threshold=0.7,
    method='weighted',
    flag_column='predicted_duplicate'
)

# Create reporter instance
reporter = EnhancedDeduplicationReport(
    results=results,
    title="Custom Deduplication Analysis",
    output_dir="dedup_reports/custom",
    create_dir=True
)

# Generate HTML report with custom settings
html_path = reporter.generate_html_report(
    output_path="dedup_reports/custom/custom_report.html",
    include_pairs=True,
    max_pairs=20,
    theme="flatly"  # Different bootstrap theme
)

# Generate Excel report with minimal pairs
excel_path = reporter.generate_excel_report(
    output_path="dedup_reports/custom/custom_report.xlsx",
    include_pairs=True,
    max_pairs=10
)

# Display in Jupyter notebook
reporter.display_jupyter_report(
    include_pairs=True,
    max_pairs=10,
    interactive=True
)
```

## Example Reports

### HTML Report
The HTML report includes:
- Summary statistics with key metrics
- Interactive visualizations with tabs
- Threshold evaluation with business impact analysis
- Sample duplicate pairs with record data
- Feature importance analysis (if available)

### Excel Report
The Excel report includes multiple sheets:
- Summary metrics
- Threshold evaluation data
- Feature importances (if available)
- Sample duplicate pairs with detailed information
- Model information (if available)

### PowerPoint Report
The PowerPoint presentation includes:
- Title slide with report information
- Metrics slide with key performance indicators
- Visualizations including confusion matrix and threshold analysis
- Sample duplicate pairs with record information
- Model information (if available)

### Markdown Report
The Markdown report includes:
- Summary metrics in a table format
- Threshold evaluation data
- Sample duplicate pairs
- Feature importance information

## Additional Resources

For more information and examples, see:
- [Enhanced Deduplication Reporting Example](examples/enhanced_deduplication_reporting_example.py)
- [Deduplication Evaluation Example](examples/deduplication_evaluation_example.py)
- The full API documentation for the `enhanced_reporting` module