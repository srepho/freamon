# Data Drift Detection

Data drift detection is a critical component of machine learning operations. It helps identify changes in data distributions over time that might affect model performance. Freamon provides robust tools for detecting and quantifying data drift across different types of features.

## Overview

The data drift detection module in Freamon allows you to:

- Detect distribution changes between reference and current datasets
- Analyze drift in numeric, categorical, and datetime features
- Calculate statistical measures of drift
- Generate comprehensive visual reports
- Monitor model input data quality over time

## Basic Usage

### Using the DataDriftDetector Class

```python
from freamon.data_quality import DataDriftDetector

# Initialize with reference and current datasets
detector = DataDriftDetector(reference_data, current_data)

# Detect drift in all features
results = detector.detect_all_drift()

# Get summary of drifting features
summary = results['summary']
print(f"Features with drift: {summary['dataset_summary']['total_drifting']}")

# Generate an HTML report
detector.generate_drift_report("drift_report.html")
```

### Quick Detection with Convenience Function

```python
from freamon.data_quality import detect_drift

# Detect drift between datasets
results = detect_drift(reference_data, current_data)
```

## Drift Detection Methods

### Numeric Features

The detector analyzes numeric features using statistical tests to identify distribution changes:

```python
# Detect drift in specific numeric features
numeric_results = detector.detect_numeric_drift(
    features=['feature1', 'feature2'],
    threshold=0.05,    # p-value threshold
    method='ks'        # Kolmogorov-Smirnov test
)
```

Available methods:
- `'ks'`: Kolmogorov-Smirnov test (default)
- `'anderson'`: Anderson-Darling test
- `'wasserstein'`: Wasserstein distance

### Categorical Features

For categorical features, the detector uses chi-square and divergence measures:

```python
# Detect drift in categorical features
categorical_results = detector.detect_categorical_drift(
    features=['category1', 'category2'],
    threshold=0.05,
    max_categories=20   # Skip features with too many categories
)
```

### Datetime Features

Datetime features are analyzed for temporal shifts:

```python
# Detect drift in datetime features
datetime_results = detector.detect_datetime_drift(
    features=['date1', 'date2'],
    threshold=0.05
)
```

## Advanced Usage

### Custom Feature Types

You can specify which features should be treated as which type:

```python
detector = DataDriftDetector(
    reference_data, 
    current_data,
    cat_features=['category1', 'category2'],
    num_features=['numeric1', 'numeric2'],
    datetime_features=['date1']
)
```

### Accessing Detailed Results

The drift detection results contain detailed information for each feature:

```python
# Get details for a specific numeric feature
feature_result = results['numeric']['feature_name']
print(f"P-value: {feature_result['p_value']}")
print(f"Reference mean: {feature_result['ref_mean']}")
print(f"Current mean: {feature_result['cur_mean']}")
print(f"Is drift detected: {feature_result['is_drift']}")
```

### PSI and Divergence Measures

The results include Population Stability Index (PSI) and other statistical measures:

```python
# PSI for numeric features
numeric_psi = results['numeric']['feature_name']['psi']

# Jensen-Shannon divergence for categorical features
js_div = results['categorical']['feature_name']['js_divergence']
```

PSI interpretation:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.2: Moderate change
- PSI ≥ 0.2: Significant change

## HTML Report Generation

The detector can generate comprehensive HTML reports for easy visualization:

```python
detector.generate_drift_report(
    "drift_report.html",
    title="Data Drift Analysis Report"
)
```

The report includes:
- Summary statistics
- List of drifting features
- Detailed analysis of each feature
- Visual distribution comparisons
- Statistical test results

## Example: Complete Drift Analysis Workflow

```python
import pandas as pd
from freamon.data_quality import DataDriftDetector

# Load reference and current datasets
reference_data = pd.read_csv("reference_data.csv")
current_data = pd.read_csv("current_data.csv")

# Initialize detector
detector = DataDriftDetector(reference_data, current_data)

# Run all drift detection methods
results = detector.detect_all_drift()

# Print summary statistics
summary = results['summary']['dataset_summary']
print(f"Total features analyzed: {summary['total_features']}")
print(f"Features with drift: {summary['total_drifting']} ({summary['drift_percentage']:.1f}%)")

# Print top drifting features
print("\nTop drifting features:")
for i, feature in enumerate(results['summary']['drifting_features'][:5]):
    print(f"{i+1}. {feature['feature']} ({feature['type']}): p-value = {feature.get('p_value', 'N/A')}")

# Generate HTML report
detector.generate_drift_report("drift_report.html", "Data Drift Analysis Report")
```

## Integration with ML Workflows

Data drift detection can be integrated into model monitoring and retraining workflows:

```python
from freamon.data_quality import DataDriftDetector
import pandas as pd

# Load training data (reference) and new data (current)
train_data = pd.read_csv("train_data.csv")
new_data = pd.read_csv("new_data.csv")

# Detect drift
detector = DataDriftDetector(train_data, new_data)
results = detector.detect_all_drift()

# Check if significant drift requires model retraining
drift_percentage = results['summary']['dataset_summary']['drift_percentage']
if drift_percentage > 20:
    print("Significant drift detected. Model retraining recommended.")
    # Trigger model retraining
else:
    print("No significant drift detected. Model is still valid.")
```

## Performance Considerations

- For large datasets, consider sampling to speed up the analysis
- Set appropriate thresholds based on your domain knowledge
- The `max_categories` parameter helps manage high-cardinality categorical features