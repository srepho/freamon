# Large Dataset Handling

Freamon provides robust mechanisms for handling large datasets efficiently, both during analysis and when displaying results. This guide explains the techniques used and options available to optimize performance with large datasets.

## Automatic Sampling and Chunking

When working with large datasets, Freamon automatically uses:

1. **Sampling**: Takes representative subsets of data when appropriate for statistical analysis
2. **Chunking**: Processes data in manageable pieces to reduce memory usage

These techniques are applied automatically and adaptively based on dataset size, ensuring that analysis remains efficient even with millions of rows.

## EDA Reporting for Large Datasets

When generating EDA reports for large datasets, several optimizations are available:

### 1. Smart Data Table Rendering

Instead of truncating large tables to just the first few rows, Freamon now uses an intelligent data display approach:

```python
from freamon.eda.analyzer import EDAAnalyzer

analyzer = EDAAnalyzer(large_df)
analyzer.generate_report(
    output_path="large_data_report.html", 
    title="Large Dataset Analysis"
)
```

The report will show:
- First N rows (sample size adjusted based on dataset size)
- Last N rows
- A row count indicator showing how many rows are between the displayed portions

This approach provides better context about the full dataset structure while keeping the report efficient.

### 2. Lazy Loading for Images

For reports with many visualizations, you can enable lazy loading of images:

```python
analyzer.generate_report(
    output_path="large_data_report.html",
    lazy_loading=True  # Enable lazy loading for images
)
```

With lazy loading enabled:
- Images are only loaded when they scroll into view
- Initial page load is faster
- Browser memory usage is reduced
- Overall report performance is improved, especially for large reports

### 3. Adaptive Column Type Analysis

For large datasets, column type detection is optimized:
- Uses sampling to efficiently determine column types
- Automatically adjusts sampling rates based on dataset size
- Handles edge cases like mixed types in large datasets

## Export to Jupyter Notebook

You can now export EDA reports to Jupyter notebooks for interactive analysis:

```python
analyzer.generate_report(
    output_path="report.html",
    include_export_button=True  # Add export to Jupyter button
)
```

Benefits of the Jupyter export:
- No server-side dependencies required (operates entirely in browser)
- Extracts all visualizations with their context
- Includes sample data from the analyzed dataset
- Creates executable code cells for each visualization
- Adds proper imports and configuration

## Example: Advanced Large Dataset Workflow

Here's an example that combines all these features:

```python
from freamon.eda.analyzer import EDAAnalyzer
from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches

# Apply fixes for special characters (like currency symbols)
apply_comprehensive_matplotlib_patches()

# Create analyzer for a large dataset
analyzer = EDAAnalyzer(
    large_df,
    date_column='transaction_date',
    target_column='status'
)

# Generate an optimized report
analyzer.run_full_analysis(
    output_path="large_dataset_report.html",
    title="Large Dataset Analysis",
    include_multivariate=True,
    lazy_loading=True,
    include_export_button=True
)
```

## Performance Benchmarks

Internal benchmarks show significant performance improvements with these optimizations:

| Dataset Size | Standard Report | With Optimizations | Improvement |
|--------------|-----------------|---------------------|-------------|
| 10,000 rows  | 8.2s            | 7.5s                | 8.5%        |
| 100,000 rows | 42.1s           | 32.8s               | 22.1%       |
| 1,000,000 rows | 285.4s        | 198.7s              | 30.4%       |

Browser rendering time also improves significantly, especially for reports with many visualizations.

## Related Examples

To see these features in action, check out these example scripts:

- `examples/large_dataset_eda_example.py`: Demonstrates all optimizations with a synthetic large dataset
- `examples/eda_performance_test.py`: Benchmarks report generation with different optimizations
- `examples/jupyter_export_example.py`: Shows how to use the Jupyter notebook export feature