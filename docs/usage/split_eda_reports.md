# Split EDA Reports for Improved Performance

The Freamon EDA module now supports generating separate reports for different types of analyses, which significantly improves performance for large datasets. This allows users to focus on specific aspects of their data analysis and makes working with large datasets more efficient.

## Benefits of Split Reports

1. **Better Performance**: Separating univariate and bivariate analyses reduces the computational load and speeds up report generation.
2. **Reduced Memory Usage**: Processing and rendering smaller reports uses less memory.
3. **Faster Browser Rendering**: Smaller HTML files load faster in web browsers.
4. **Focused Analysis**: Target-specific reports make it easier to focus on what's important.
5. **Enhanced Feature Importance**: Bivariate reports now include more detailed feature importance metrics.

## Report Types

The Freamon EDA module supports three types of reports:

1. **Full Report**: Contains all analyses (univariate, bivariate, multivariate, time series).
2. **Univariate Report**: Contains only basic statistics and univariate analysis.
3. **Bivariate Report**: Contains basic statistics, bivariate analysis, and feature importance.

## Basic Usage

### Generating Separate Reports Manually

```python
from freamon.eda import EDAAnalyzer

# Create an analyzer
analyzer = EDAAnalyzer(df, target_column="target")

# Run the analyses
analyzer.analyze_basic_stats()
analyzer.analyze_univariate()
analyzer.analyze_bivariate()

# Generate separate reports
analyzer.generate_univariate_report(output_path="univariate_report.html")
analyzer.generate_bivariate_report(output_path="bivariate_report.html")
```

### Automatically Generating Split Reports

You can also use the `run_full_analysis` method with the `split_reports` parameter:

```python
from freamon.eda import EDAAnalyzer

# Create an analyzer
analyzer = EDAAnalyzer(df, target_column="target")

# Run full analysis and generate split reports
analyzer.run_full_analysis(
    output_path="full_report.html",
    split_reports=True,
    show_progress=True
)
```

This will generate three reports:
- `full_report.html`: The complete analysis
- `full_report_univariate.html`: Univariate analysis only
- `full_report_bivariate.html`: Bivariate analysis with feature importance

## Enhanced Feature Importance in Bivariate Reports

The bivariate reports now include more detailed feature importance metrics for target-feature relationships:

- **Numeric vs. Numeric**: R-squared (coefficient of determination)
- **Numeric vs. Categorical**: F-statistic from ANOVA
- **Categorical vs. Numeric**: Eta-squared (effect size)
- **Categorical vs. Categorical**: Cramer's V (normalized chi-square)

These metrics help to quantify the strength of relationships between features and the target variable, making it easier to identify the most important features.

## Performance Considerations

For large datasets (>50K rows) or datasets with many columns (>30), using split reports can significantly improve performance:

```python
# For large datasets, use sampling and split reports
analyzer = EDAAnalyzer(large_df, target_column="target")
analyzer.run_full_analysis(
    output_path="large_dataset_report.html",
    split_reports=True,
    use_sampling=True,
    sample_size=50000,  # Adjust based on your needs
    show_progress=True
)
```

The performance gain will be most noticeable when:
1. The dataset is large (many rows)
2. There are many columns to analyze
3. You're calculating feature importance
4. You need to focus on either univariate or bivariate analysis separately

## Examples

For a complete working example, see the `eda_split_reports_example.py` in the examples directory:

```python
# Run the example to see the performance difference
python examples/eda_split_reports_example.py
```

## API Reference

### New Methods in EDAAnalyzer

- **generate_univariate_report**: Generate a report containing only univariate analysis.
- **generate_bivariate_report**: Generate a report containing only bivariate analysis and feature importance.
- **generate_report** (with new parameters): The existing method now supports a `report_type` parameter that can be set to `'full'`, `'univariate'`, or `'bivariate'`.
- **run_full_analysis** (with new parameters): The method now supports a `split_reports` parameter that generates separate reports when set to `True`.

### New Parameters in bivariate.py

- **include_importance**: Whether to include feature importance metrics in feature-target analysis (default: `True`).