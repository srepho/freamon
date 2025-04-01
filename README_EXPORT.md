# Freamon Export Capabilities

This feature provides comprehensive export capabilities for EDA reports, model performance evaluations, hyperparameter tuning results, and topic modeling outcomes to Excel (.xlsx) and PowerPoint (.pptx) formats.

## Overview

The export functionality enables you to:

- Export exploratory data analysis (EDA) reports to Excel workbooks with multiple sheets
- Create PowerPoint presentations with data visualizations and key insights
- Export model performance metrics and visualizations
- Document hyperparameter tuning experiments
- Capture topic modeling results in structured formats 

## Usage

### EDA Report Export

```python
from freamon.eda.simple_report import generate_simple_eda_report

# Generate and export EDA report to Excel
report = generate_simple_eda_report(
    df=my_dataframe,
    target_column='target',
    export_format='xlsx',
    output_path='eda_report.xlsx'
)

# Export to PowerPoint
report = generate_simple_eda_report(
    df=my_dataframe,
    target_column='target',
    export_format='pptx',
    output_path='eda_report.pptx'
)
```

### Model Performance Report

```python
from freamon.modeling.model_report import generate_model_performance_report

# Create model results dictionary
model_results = {
    'model_type': 'classification',  # or 'regression'
    'model_library': 'lightgbm',     # or 'sklearn', etc.
    'metrics': {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1': 0.85
    },
    'predictions': y_pred,
    'actuals': y_test,
    'confusion_matrix': confusion_matrix(y_test, y_pred)
}

# Export to Excel
report = generate_model_performance_report(
    model_results=model_results,
    feature_importance=feature_importance,
    export_format='xlsx',
    output_path='model_performance.xlsx'
)
```

### Hyperparameter Tuning Report

```python
from freamon.modeling.model_report import generate_hyperparameter_tuning_report

# Export tuning results
report = generate_hyperparameter_tuning_report(
    tuning_results=tuning_results,
    parameter_importance=parameter_importance,
    trial_history=trial_history,
    export_format='pptx',
    output_path='hyperparameter_tuning.pptx'
)
```

### Topic Modeling Report

```python
from freamon.utils.topic_report import generate_topic_modeling_report

# Export topic modeling results
report = generate_topic_modeling_report(
    topic_model_results=topic_model_results,
    text_data=text_df,
    category_column='category',
    export_format='xlsx',
    output_path='topic_modeling.xlsx'
)
```

## Export Content Details

### Excel Export Format

#### EDA Report
- **Overview**: Dataset statistics, row/column counts, data type distribution
- **Sample Data**: Preview of the first rows of data
- **Missing Values**: Analysis of missing data
- **Numeric Columns**: Statistical summaries of numeric features
- **Correlations**: Correlation matrix between numeric features
- **Categorical Columns**: Value counts and distributions
- **Text Columns**: Text statistics and word counts
- **Datetime Columns**: Temporal patterns and statistics

#### Model Report
- **Overview**: Model type, library, performance metrics
- **CV Results**: Cross-validation metrics by fold
- **Feature Importance**: Ranked feature importance
- **Predictions**: Sample of predictions vs actuals
- **Confusion Matrix**: For classification models

#### Hyperparameter Report
- **Overview**: Optimization algorithm, trials, best score
- **Best Parameters**: Optimal hyperparameter values
- **Parameter Importance**: Impact of each parameter
- **Trial History**: All trials and their scores
- **Parameter Ranges**: Search space configuration

#### Topic Modeling Report
- **Overview**: Model type, topics, coherence metrics
- **Topics**: Top terms for each topic with weights
- **Document Topics**: Document-topic distribution
- **Topic Coherence**: Coherence scores by topic
- **Category Topics**: Topic distribution by category
- **Topic Similarity**: Similarity matrix between topics

### PowerPoint Export Format

Each report type creates a PowerPoint presentation with slides containing:

- Title slide with report type and timestamp
- Overview slide with key metrics
- Data visualizations as high-quality images
- Summary tables with key statistics
- Topic visualization slides for topic models
- Model performance charts
- Feature importance visualizations
- Hyperparameter tuning convergence plots

## Example

See the full example in [`examples/export_example.py`](examples/export_example.py) which demonstrates exporting all report types to both Excel and PowerPoint formats.

## Requirements

- Python 3.6+
- pandas
- matplotlib
- seaborn
- numpy
- For Excel export: `openpyxl`
- For PowerPoint export: `python-pptx`

To install the required dependencies:

```bash
pip install openpyxl python-pptx
```