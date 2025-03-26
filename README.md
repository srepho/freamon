# Freamon: Feature-Rich EDA, Analytics, and Modeling Toolkit

<p align="center">
  <img src="package_logo.webp" alt="Freamon Logo" width="250"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/freamon.svg)](https://pypi.org/project/freamon/)
[![GitHub release](https://img.shields.io/github/v/release/srepho/freamon)](https://github.com/srepho/freamon/releases)

Freamon is a comprehensive Python toolkit for exploratory data analysis, feature engineering, and model development with a focus on practical data science workflows.

## Features

- **Exploratory Data Analysis**: Automatic EDA with comprehensive reporting in HTML, Markdown, Excel, PowerPoint, and Jupyter notebooks
- **Advanced Multivariate Analysis**: PCA visualization, correlation networks, and target-oriented analysis
- **Feature Engineering**: Advanced feature engineering for numeric, categorical, and text data
- **Feature Selection**: Statistical feature selection including Chi-square, ANOVA F-test, and effect size analysis
- **Deduplication**: High-performance deduplication with Polars optimization (2-5x faster, 60-70% less memory), LSH, supervised ML, and active learning
- **Topic Modeling**: Optimized text analysis with NMF and LDA, supporting large datasets up to 100K documents
- **Automated Modeling**: Intelligent end-to-end modeling workflow for text, tabular, and time series data
- **Modeling**: Custom model implementations with feature importance and model interpretation
- **Pipeline**: Scikit-learn compatible pipeline with additional features
- **Drift Analysis**: Tools for detecting and analyzing data drift
- **Word Embeddings**: Integration with various word embedding techniques
- **Visualization**: Publication-quality visualizations with proper handling of all special characters
- **Performance Optimization**: Multiprocessing support and intelligent sampling for large dataset analysis

## Installation

```bash
pip install freamon
```

## Quick Start

```python
from freamon.eda import EDAAnalyzer

# Create an analyzer instance
analyzer = EDAAnalyzer(df, target_column='target')

# Run the analysis
analyzer.run_full_analysis()

# Generate a report
analyzer.generate_report('eda_report.html')

# Or a markdown report for version control
analyzer.generate_report('eda_report.md', format='markdown')
```

## Key Components

### Automated Modeling with Data Type Detection

Easily build an end-to-end model with automatic data type detection and processing:

```python
import pandas as pd
from freamon import auto_model

# Load sample data with mixed types (text, categorical, numeric)
df = pd.read_csv('customer_data.csv')

# Run automated modeling with just a target column - all others are auto-detected
results = auto_model(
    df=df,
    target_column='churn',
    problem_type='classification'
)

# See the detected data types
print(f"Detected text columns: {results['dataset_info']['text_columns']}")
print(f"Detected categorical columns: {results['dataset_info']['categorical_columns']}")
print(f"Detected numeric columns: {results['dataset_info']['numeric_columns']}")

# Review model performance
print(f"Cross-validation metrics: {results['metrics']}")
if 'test_metrics' in results:
    print(f"Test set metrics: {results['test_metrics']}")

# Make predictions on new data
new_data = pd.read_csv('new_customers.csv')
predictions = results['autoflow'].predict(new_data)
```

### Deduplication Workflow with Model Training

Perform efficient deduplication and train a model on deduplicated data:

```python
import pandas as pd
from freamon.deduplication.exact_deduplication import hash_deduplication 
from freamon.deduplication.lsh_deduplication import lsh_deduplication
from examples.deduplication_tracking_example import IndexTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create sample data with text and duplicates
df = pd.read_csv('data_with_duplicates.csv')

# Initialize index tracker
tracker = IndexTracker().initialize_from_df(df)

# Find duplicates using LSH (locality-sensitive hashing) for text similarity
kept_indices = lsh_deduplication(
    df['description'],
    threshold=0.8,
    num_bands=20,
    preprocess=True
)

# Create deduplicated dataframe
deduped_df = df.iloc[kept_indices].copy()

# Update tracker with kept indices
tracker.update_from_kept_indices(kept_indices, deduped_df)

# Train model on deduplicated data
X = deduped_df.drop('target', axis=1)
y = deduped_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and generate results dataframe
y_pred = pd.Series(model.predict(X_test), index=X_test.index)
results_df = pd.DataFrame({'prediction': y_pred})

# Map results back to original dataset with all records
full_results = tracker.create_full_result_df(
    results_df, df, fill_value={'prediction': None}
)

print(f"Original dataset size: {len(df)}")
print(f"Deduplicated dataset size: {len(deduped_df)}")
print(f"Number of records with predictions: {full_results['prediction'].notna().sum()}")
```

### Deduplication on Unlabeled Data

Identify potential duplicates without removing them:

```python
import pandas as pd
import numpy as np
from freamon.deduplication.flag_duplicates import flag_similar_records, flag_text_duplicates

# Load unlabeled dataset
unlabeled_df = pd.read_csv('unlabeled_customer_data.csv')

# Flag potential text duplicates
text_df = flag_text_duplicates(
    unlabeled_df,
    text_column='description',
    threshold=0.8,
    method='lsh',
    add_group_id=True
)

# Check how many potential duplicate groups were found
duplicate_groups = text_df['duplicate_group_id'].dropna().nunique()
print(f"Found {duplicate_groups} potential duplicate text groups")

# Flag similar records across multiple fields
similar_df = flag_similar_records(
    unlabeled_df,
    columns=['name', 'address', 'phone', 'email'],
    weights={'name': 0.4, 'address': 0.3, 'phone': 0.2, 'email': 0.1},
    threshold=0.7,
    add_similarity_score=True
)

# Examine records with high similarity
high_similarity = similar_df[similar_df['similarity_score'] > 0.9]
print(f"Found {len(high_similarity)} records with >90% similarity")

# Export flagged duplicates for review
high_similarity.to_csv('potential_duplicates.csv', index=False)
```

### Advanced EDA and Feature Selection

Perform advanced multivariate analysis and feature selection:

```python
from freamon.eda.advanced_multivariate import visualize_pca, analyze_target_relationships
from freamon.features.categorical_selection import chi2_selection, anova_f_selection

# PCA visualization with target coloring
fig, pca_results = visualize_pca(df, target_column='target')

# Target-oriented feature analysis
figures, target_results = analyze_target_relationships(df, target_column='target')

# Select important categorical features
selected_features, scores = chi2_selection(df, target='target', k=5, return_scores=True)
```

See [Advanced EDA documentation](README_ADVANCED_EDA.md) for more details.

### EDA Module

The EDA module provides comprehensive data analysis:

```python
from freamon.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df, target_column='target')
analyzer.run_full_analysis()

# Generate different types of reports
analyzer.generate_report('report.html')  # HTML report
analyzer.generate_report('report.md', format='markdown')  # Markdown report
analyzer.generate_report('report.md', format='markdown', convert_to_html=True)  # Both formats
```

## Documentation

For more detailed information, refer to the examples directory and the following resources:

- [Deduplication Tracking](README_DEDUPLICATION_TRACKING.md)
- [Markdown Report Generation](README_MARKDOWN_REPORTS.md)
- [LSH Deduplication](README_LSH_DEDUPLICATION.md)

## License

[MIT License](LICENSE)