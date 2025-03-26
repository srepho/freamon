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

### Basic Installation

For basic functionality (EDA, visualization, core deduplication):

```bash
pip install freamon
```

### Installation with All Features

For full functionality including advanced modeling, text processing, and performance optimizations:

```bash
pip install "freamon[all]"
```

### Feature-Specific Installation

For specific feature sets:

```bash
# For high-performance with Polars acceleration
pip install "freamon[performance]"

# For text analysis and topic modeling
pip install "freamon[topic_modeling]"

# For word embeddings support
pip install "freamon[word_embeddings]"

# For extended features (modeling, Polars, LightGBM, SHAP, etc.)
pip install "freamon[extended]"

# For Markdown report generation
pip install "freamon[markdown_reports]"
```

### Dependencies by Feature

Here's what each optional dependency provides:

- **Core** (always installed):
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`

- **Performance** [`freamon[performance]`]:
  - `pyarrow` - For faster data processing

- **Extended** [`freamon[extended]`]:
  - `polars` - High-performance DataFrame library (2-5x faster than pandas)
  - `lightgbm` - Gradient boosting framework
  - `optuna` - Hyperparameter optimization
  - `shap` - Model explanation
  - `spacy` - NLP processing
  - `statsmodels` - Statistical modeling
  - `dask` - Parallel computing

- **Topic Modeling** [`freamon[topic_modeling]`]:
  - `gensim` - Topic modeling
  - `pyldavis` - Topic visualization
  - `wordcloud` - Word cloud generation

- **Word Embeddings** [`freamon[word_embeddings]`]:
  - `gensim` - Word vectors
  - `nltk` - Natural language toolkit
  - `spacy` - Linguistic features

- **Markdown Reports** [`freamon[markdown_reports]`]:
  - `markdown` - Report generation

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

### Step-by-Step Workflow Example

Below is a complete workflow showing data type detection, EDA analysis, modeling, and reporting:

```python
import pandas as pd
import numpy as np
from freamon.eda import EDAAnalyzer
from freamon.utils.datatype_detector import detect_datatypes
from freamon import auto_model

# Required for PowerPoint/Excel reports
# pip install "freamon[extended]"
from freamon.eda.export import export_to_powerpoint, export_to_excel

# 1. Load sample data
df = pd.read_csv('customer_data.csv')
print(f"Dataset shape: {df.shape}")

# 2. Run data type detection
datatype_results = detect_datatypes(df)
print("\nDetected data types:")
print(f"Text columns: {datatype_results['text_columns']}")
print(f"Categorical columns: {datatype_results['categorical_columns']}")
print(f"Numeric columns: {datatype_results['numeric_columns']}")
print(f"Date columns: {datatype_results['date_columns']}")

# 3. Generate data type detection report
from freamon.utils.datatype_fixes import save_detection_report
save_detection_report(
    datatype_results,
    'datatype_detection_report.html',
    title='Customer Data Type Detection'
)
print("\nData type detection report saved to 'datatype_detection_report.html'")

# 4. Run EDA analysis
analyzer = EDAAnalyzer(
    df,
    target_column='churn',  # For supervised analysis
    text_columns=datatype_results['text_columns'],
    categorical_columns=datatype_results['categorical_columns'],
    numeric_columns=datatype_results['numeric_columns'],
    datetime_columns=datatype_results['date_columns']
)
analyzer.run_full_analysis()

# 5. Generate EDA reports in different formats
analyzer.generate_report('eda_report.html')  # HTML report
analyzer.generate_report('eda_report.md', format='markdown')  # Markdown report
print("\nEDA reports generated in HTML and Markdown formats")

# 6. Export EDA results to PowerPoint for presentations
export_to_powerpoint(
    analyzer.get_report_data(),
    'eda_presentation.pptx',
    report_type='eda'
)
print("\nEDA results exported to PowerPoint")

# 7. Run automated modeling with data type detection
# Note: Install required dependencies for advanced modeling:
# pip install "freamon[extended,topic_modeling]"
results = auto_model(
    df=df,
    target_column='churn',
    problem_type='classification',
    # Use our detected data types
    text_columns=datatype_results['text_columns'],
    categorical_columns=datatype_results['categorical_columns'],
    date_column=datatype_results['date_columns'][0] if datatype_results['date_columns'] else None
)

# 8. Examine model results
print("\nModel Performance:")
for metric, value in results['metrics'].items():
    if 'mean' in metric:
        print(f"{metric}: {value:.4f}")

# 9. Plot model visualizations
fig1 = results['autoflow'].plot_metrics()
fig1.savefig('cv_metrics.png')

fig2 = results['autoflow'].plot_importance(top_n=15)
fig2.savefig('feature_importance.png')

# 10. Export model results to Excel
model_data = {
    'model_type': results['autoflow'].model_type,
    'metrics': results['metrics'],
    'feature_importance': results['feature_importance'],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
}
export_to_excel(model_data, 'model_performance.xlsx', report_type='model')

# 11. Export model results to PowerPoint
export_to_powerpoint(model_data, 'model_presentation.pptx', report_type='model')
print("\nModel results exported to Excel and PowerPoint")

# 12. Make predictions on new data
new_data = pd.read_csv('new_customers.csv')
predictions = results['autoflow'].predict(new_data)
new_data['predicted_churn'] = predictions
new_data.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")
```

### Comprehensive Deduplication Workflow

Complete step-by-step process for deduplication, including analysis, visualization, and modeling:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from freamon.deduplication.exact_deduplication import hash_deduplication 
from freamon.deduplication.lsh_deduplication import lsh_deduplication
from freamon.data_quality.duplicates import detect_duplicates, get_duplicate_groups
from examples.deduplication_tracking_example import IndexTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# For network visualization (optional)
# pip install networkx
import networkx as nx

# 1. Load sample data with text and duplicates
df = pd.read_csv('data_with_duplicates.csv')
print(f"Original dataset shape: {df.shape}")

# 2. Analyze duplicates using built-in detection
duplicate_stats = detect_duplicates(df)
print(f"\nDuplicate analysis:")
print(f"Exact duplicates: {duplicate_stats['duplicate_count']} records")
print(f"Duplicate percentage: {duplicate_stats['duplicate_percent']:.2f}%")

# 3. Get duplicate groups for examination
duplicate_groups = get_duplicate_groups(df)
print(f"\nFound {len(duplicate_groups)} duplicate groups")
for i, group in enumerate(duplicate_groups[:3]):  # Show first 3 groups
    print(f"\nDuplicate group {i+1}:")
    print(df.iloc[group].head(1))  # Show one example from each group

# 4. Initialize index tracker to maintain mapping
tracker = IndexTracker().initialize_from_df(df)

# 5. Find duplicates using LSH (locality-sensitive hashing) for text similarity
print("\nRunning LSH deduplication...")
kept_indices, similarity_dict = lsh_deduplication(
    df['description'],
    threshold=0.8,
    num_bands=20,
    preprocess=True,
    return_similarity_dict=True
)

# 6. Analyze LSH results
print(f"LSH kept {len(kept_indices)} out of {len(df)} records ({len(kept_indices)/len(df)*100:.1f}%)")

# 7. Visualize similarity network (for smaller datasets)
if len(df) < 1000:
    import networkx as nx
    G = nx.Graph()
    
    # Add all nodes (documents)
    for i in range(len(df)):
        G.add_node(i)
    
    # Add edges (similarities)
    for doc_id, similar_docs in similarity_dict.items():
        for similar_id in similar_docs:
            G.add_edge(doc_id, similar_id)
    
    # Plot network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=50, node_color='blue', alpha=0.6)
    plt.title('Document Similarity Network')
    plt.savefig('similarity_network.png')
    plt.close()
    print("\nSaved similarity network visualization to 'similarity_network.png'")

# 8. Create deduplicated dataframe
deduped_df = df.iloc[kept_indices].copy()

# 9. Update tracker with kept indices
tracker.update_from_kept_indices(kept_indices, deduped_df)

# 10. Train model on deduplicated data
print("\nTraining model on deduplicated data...")
X = deduped_df.drop(['target', 'description'], axis=1)  # Exclude text column
y = deduped_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Evaluate model
y_pred = model.predict(X_test)
print("\nModel performance on deduplicated test data:")
print(classification_report(y_test, y_pred))

# 12. Make predictions and generate results dataframe
y_pred_series = pd.Series(y_pred, index=X_test.index)
results_df = pd.DataFrame({'prediction': y_pred_series, 'actual': y_test})

# 13. Map results back to original dataset with all records
full_results = tracker.create_full_result_df(
    results_df, df, fill_value={'prediction': None, 'actual': None}
)

print(f"\nMapping results:")
print(f"Original dataset size: {len(df)}")
print(f"Deduplicated dataset size: {len(deduped_df)}")
print(f"Number of records with predictions: {full_results['prediction'].notna().sum()}")

# 14. Save full dataset with deduplication information
df['is_duplicate'] = ~df.index.isin(kept_indices)
df['has_prediction'] = full_results['prediction'].notna()
df['predicted'] = full_results['prediction']
df.to_csv('deduplication_results.csv', index=False)
print("\nSaved full dataset with deduplication and prediction information to 'deduplication_results.csv'")
```

### Duplicate Flagging for Unlabeled Data

Comprehensive workflow to identify potential duplicates without removing them, with analysis and visualization:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Required for duplicate flagging functionality 
# pip install "freamon[extended]"
from freamon.deduplication.flag_duplicates import flag_similar_records, flag_text_duplicates

# Required for PowerPoint/Excel export
# pip install "freamon[extended]"
from freamon.eda.export import export_to_excel, export_to_powerpoint

# 1. Load unlabeled dataset
unlabeled_df = pd.read_csv('unlabeled_customer_data.csv')
print(f"Dataset shape: {unlabeled_df.shape}")

# 2. Flag potential text duplicates using LSH
print("\nProcessing text duplicates...")
text_df = flag_text_duplicates(
    unlabeled_df,
    text_column='description',
    threshold=0.8,
    method='lsh',
    add_group_id=True,
    add_similarity_score=True,
    add_duplicate_flag=True
)

# 3. Analyze text duplicate results
duplicate_text_groups = text_df['duplicate_group_id'].dropna().nunique()
duplicate_text_records = text_df['is_text_duplicate'].sum()
print(f"Text duplicate analysis:")
print(f"Found {duplicate_text_groups} potential duplicate text groups")
print(f"Found {duplicate_text_records} records ({duplicate_text_records/len(text_df)*100:.1f}%) with similar text")

# 4. Flag similar records across multiple fields using weighted similarity
print("\nProcessing multi-field similarity...")
similar_df = flag_similar_records(
    text_df,  # Use the dataframe that already has text duplicate info
    columns=['name', 'address', 'phone', 'email'],
    weights={'name': 0.4, 'address': 0.3, 'phone': 0.2, 'email': 0.1},
    threshold=0.7,
    add_similarity_score=True,
    add_group_id=True,
    group_id_column="multifield_group_id",
    duplicate_flag_column="is_multifield_duplicate"
)

# 5. Analyze multi-field similarity results
multifield_groups = similar_df['multifield_group_id'].dropna().nunique()
multifield_duplicates = similar_df['is_multifield_duplicate'].sum()
print(f"Multi-field duplicate analysis:")
print(f"Found {multifield_groups} potential duplicate groups based on multiple fields")
print(f"Found {multifield_duplicates} records ({multifield_duplicates/len(similar_df)*100:.1f}%) with similar fields")

# 6. Create a combined duplicate flag
similar_df['is_potential_duplicate'] = similar_df['is_text_duplicate'] | similar_df['is_multifield_duplicate']
total_duplicates = similar_df['is_potential_duplicate'].sum()
print(f"\nCombined results: {total_duplicates} potential duplicates ({total_duplicates/len(similar_df)*100:.1f}%)")

# 7. Visualize similarity score distribution
plt.figure(figsize=(10, 6))
sns.histplot(similar_df['similarity_score'].dropna(), bins=20)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Count')
plt.axvline(x=0.7, color='r', linestyle='--', label='Threshold (0.7)')
plt.axvline(x=0.9, color='g', linestyle='--', label='High Similarity (0.9)')
plt.legend()
plt.savefig('similarity_distribution.png')
plt.close()
print("\nSaved similarity distribution chart to 'similarity_distribution.png'")

# 8. Create a group size analysis
group_sizes = similar_df[similar_df['multifield_group_id'].notna()].groupby('multifield_group_id').size()
plt.figure(figsize=(10, 6))
sns.histplot(group_sizes, bins=10)
plt.title('Duplicate Group Size Distribution')
plt.xlabel('Group Size')
plt.ylabel('Count')
plt.savefig('group_size_distribution.png')
plt.close()
print(f"Largest duplicate group has {group_sizes.max()} records")

# 9. Add confidence level based on combined evidence
similar_df['duplicate_confidence'] = 'None'
# Both text and multifield similarity = high confidence
similar_df.loc[(similar_df['is_text_duplicate']) & 
               (similar_df['is_multifield_duplicate']), 'duplicate_confidence'] = 'High'
# Only one method but high score = medium confidence
similar_df.loc[(similar_df['is_potential_duplicate']) & 
               (similar_df['similarity_score'] > 0.9) &
               (similar_df['duplicate_confidence'] == 'None'), 'duplicate_confidence'] = 'Medium'
# Flagged but lower score = low confidence
similar_df.loc[(similar_df['is_potential_duplicate']) & 
               (similar_df['duplicate_confidence'] == 'None'), 'duplicate_confidence'] = 'Low'

confidence_counts = similar_df['duplicate_confidence'].value_counts()
print("\nDuplicate confidence levels:")
for level, count in confidence_counts.items():
    print(f"{level} confidence: {count} records")

# 10. Export high confidence duplicates for review
high_confidence = similar_df[similar_df['duplicate_confidence'] == 'High']
medium_confidence = similar_df[similar_df['duplicate_confidence'] == 'Medium']

# 11. Create summary report with examples from each confidence level
report_data = []
for group_id in high_confidence['multifield_group_id'].dropna().unique()[:5]:  # Top 5 high confidence groups
    group_records = similar_df[similar_df['multifield_group_id'] == group_id]
    report_data.append({
        'confidence': 'High',
        'group_id': group_id,
        'group_size': len(group_records),
        'similarity_score': group_records['similarity_score'].mean(),
        'sample_records': group_records.head(2).to_dict('records')
    })

# 12. Export results in different formats
# Excel report
similar_df.to_csv('duplicate_analysis_complete.csv', index=False)
high_confidence.to_csv('high_confidence_duplicates.csv', index=False)
medium_confidence.to_csv('medium_confidence_duplicates.csv', index=False)

# 13. Export summary data for PowerPoint
summary_data = {
    'dataframe_size': len(similar_df),
    'duplicate_count': total_duplicates,
    'duplicate_percent': total_duplicates/len(similar_df)*100,
    'confidence_distribution': confidence_counts.to_dict(),
    'group_count': multifield_groups,
    'largest_group_size': group_sizes.max(),
    'similarity_scores': similar_df['similarity_score'].dropna().tolist(),
    'threshold': 0.7
}

# Create presentation-ready dictionary
presentation_data = {
    'metrics': {
        'dataset_size': len(similar_df),
        'duplicate_count': total_duplicates,
        'duplicate_percent': total_duplicates/len(similar_df)*100,
        'high_confidence': confidence_counts.get('High', 0),
        'medium_confidence': confidence_counts.get('Medium', 0),
        'low_confidence': confidence_counts.get('Low', 0),
    }
}

# 14. Export to PowerPoint (use model_type report since it has charts)
export_to_powerpoint(
    presentation_data, 
    'duplicate_analysis.pptx', 
    report_type='model'
)
print("\nExported reports to CSV files and PowerPoint")

print("\nDuplicate analysis complete.")
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