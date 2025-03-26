# Freamon Quick Start Guide

This guide provides quick examples of the most commonly used features in Freamon.

## Installation

### Basic Installation

```bash
pip install freamon
```

### Full Installation (All Features)

```bash
pip install "freamon[all]"
```

## 1. Exploratory Data Analysis

```python
import pandas as pd
from freamon.eda import EDAAnalyzer

# Load data
df = pd.read_csv('your_data.csv')

# Create analyzer
analyzer = EDAAnalyzer(df, target_column='target')

# Run basic analysis
analyzer.run_full_analysis()

# Generate report
analyzer.generate_report('eda_report.html')
```

## 2. Automated Modeling

```python
import pandas as pd
from freamon import auto_model

# Load data
df = pd.read_csv('your_data.csv')

# Run automated modeling
results = auto_model(
    df=df,
    target_column='target',
    problem_type='classification',  # or 'regression'
    auto_split=True,  # Automatically split into train/test
    test_size=0.2
)

# Access results
model = results['model']
test_metrics = results['test_metrics']
feature_importance = results['feature_importance']

# Make predictions
new_data = pd.read_csv('new_data.csv')
predictions = model.predict(new_data)
```

## 3. Deduplication

### Basic Deduplication

```python
import pandas as pd
from freamon.deduplication import hash_deduplication

# Load data
df = pd.read_csv('data_with_duplicates.csv')

# Get indices of unique records
unique_indices = hash_deduplication(df)

# Create a deduplicated dataframe
deduplicated_df = df.iloc[unique_indices]
```

### Duplicate Flagging (Without Removal)

```python
import pandas as pd
from freamon.deduplication import flag_text_duplicates, flag_similar_records

# Flag text duplicates
result_df = flag_text_duplicates(
    df,
    text_column='description',
    method='lsh',
    threshold=0.8,
    flag_column='is_text_duplicate',
    group_column='duplicate_group_id'
)

# Flag similar records based on multiple fields
result_df = flag_similar_records(
    result_df,
    columns=['name', 'address', 'phone', 'email'],
    weights={'name': 0.4, 'address': 0.3, 'phone': 0.2, 'email': 0.1},
    threshold=0.7,
    flag_column='is_similar',
    similarity_column='similarity_score'
)
```

## 4. Advanced Features

### Feature Selection

```python
from freamon.features.selector import select_features
from freamon.features.categorical_selection import chi2_selection

# Statistical feature selection
selected_features = select_features(
    X, y, 
    method='mutual_info', 
    k=10
)

# Categorical feature selection
categorical_features = chi2_selection(
    df, 
    target='target', 
    k=5
)
```

### Time Series Features

```python
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer

# Create engineer
ts_engineer = TimeSeriesFeatureEngineer(
    date_column='date',
    target_column='target'
)

# Create time features
enhanced_df = ts_engineer.fit_transform(df)
```

### Text Processing and Topic Modeling

```python
from freamon.utils.text_utils import create_topic_model_optimized

# Create topic model
result = create_topic_model_optimized(
    df=text_df,
    text_column='text',
    n_topics='auto',  # Automatically determine optimal number
    method='nmf',
    max_sample_size=10000
)

# Access components
topic_model = result['topic_model']
topics = result['topics']
doc_topics = result['document_topics']
```

## 5. Report Export

```python
from freamon.eda.export import export_to_powerpoint, export_to_excel

# Export EDA results to PowerPoint
export_to_powerpoint(
    analyzer.get_report_data(),
    'eda_presentation.pptx',
    report_type='eda'
)

# Export model results to Excel
export_to_excel(
    model_data,
    'model_results.xlsx',
    report_type='model'
)
```

## More Resources

For complete examples and detailed documentation, see:

- [Advanced EDA Features](README_ADVANCED_EDA.md)
- [Auto-Split Modeling](README_AUTO_SPLIT.md)
- [Export Capabilities](README_EXPORT.md)
- [Deduplication Tracking](README_DEDUPLICATION_TRACKING.md)
- [Markdown Report Generation](README_MARKDOWN_REPORTS.md)
- [LSH Deduplication](README_LSH_DEDUPLICATION.md)

Example scripts in the `examples/` directory show complete workflows for various use cases.