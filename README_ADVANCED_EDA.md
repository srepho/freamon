# Advanced EDA and Feature Selection

This module provides advanced exploratory data analysis (EDA) capabilities for high-dimensional data and sophisticated feature selection methods for categorical variables.

## Key Features

### 1. PCA Visualization

Principal Component Analysis (PCA) is a powerful technique for visualizing high-dimensional data by reducing it to a lower dimension while preserving as much variance as possible.

```python
from freamon.eda.advanced_multivariate import visualize_pca

# Visualize data using PCA
fig, pca_results = visualize_pca(
    df,  # Your dataframe 
    target_column='target',  # Optional: color points by target
    n_components=2,  # 2D or 3D visualization
    scale=True,  # Standardize features
    plot_loading=True  # Show feature loadings
)

# Access PCA results
print(f"Explained variance: {pca_results['explained_variance_ratio']}")
print(f"Feature loadings: {pca_results['loadings']}")
```

### 2. Correlation Network Visualization

This feature creates a network visualization of feature correlations, helping to identify clusters of related features and multicollinearity.

```python
from freamon.eda.advanced_multivariate import create_correlation_network

# Create correlation network
fig, network_results = create_correlation_network(
    df,
    threshold=0.7,  # Correlation threshold for edges
    method='pearson'  # Correlation method
)

# Access network properties
print(f"Number of nodes: {len(network_results['nodes'])}")
print(f"Number of edges: {len(network_results['edges'])}")
```

### 3. Target-Oriented Analysis

This feature performs comprehensive analysis of relationships between features and a target variable, including both statistical significance and effect size measures.

```python
from freamon.eda.advanced_multivariate import analyze_target_relationships

# Analyze relationships with target
figures, results = analyze_target_relationships(
    df,
    target_column='target',
    max_features=10  # Max features to show in visualizations
)

# Access top features by importance
top_features = list(results['feature_ranking'].keys())[:5]
for feature in top_features:
    score = results['feature_ranking'][feature]['avg_score']
    print(f"{feature}: {score:.4f}")
```

### 4. Categorical Feature Selection

Freamon provides specialized methods for selecting the most informative categorical features:

#### Chi-Square Test for Classification

```python
from freamon.features.categorical_selection import chi2_selection

# Select top categorical features using Chi-square
selected_features, scores = chi2_selection(
    df,
    target='target',  # Target column name or Series
    k=5,  # Top k features to select
    return_scores=True  # Return importance scores
)

print("Selected features:", selected_features)
print("Feature scores:", scores)
```

#### ANOVA F-Test for Regression or Classification

```python
from freamon.features.categorical_selection import anova_f_selection

# Select top categorical features using ANOVA F-test
selected_features, scores = anova_f_selection(
    df,
    target='target',  # Target column name or Series
    k=5,  # Top k features to select
    return_scores=True  # Return importance scores
)

print("Selected features:", selected_features)
print("Feature scores:", scores)
```

#### Visualizing Feature Importance

```python
from freamon.features.categorical_selection import plot_categorical_importance

# Visualize feature importance
fig, fig_effect = plot_categorical_importance(
    scores,  # Feature scores from selection function
    method='chi2',  # Method used ('chi2' or 'anova_f')
    top_n=10,  # Number of top features to show
    show_effect_size=True  # Also show effect size plot
)
```

### 5. Comprehensive Feature Selector Class

For more control, use the `CategoricalFeatureSelector` class:

```python
from freamon.features.categorical_selection import CategoricalFeatureSelector

# Create selector
selector = CategoricalFeatureSelector(
    method='chi2',  # 'chi2', 'anova_f', or 'mutual_info'
    k=5,  # Top k features
    target_type='classification'  # 'classification' or 'regression'
)

# Fit selector
selector.fit(X, y)

# Get selected features
selected_features = selector.selected_features_

# Get feature importance
importance = selector.feature_importance_

# Plot feature importance
fig = selector.plot_feature_importance()

# Plot effect sizes
fig_effect = selector.plot_effect_sizes()
```

## Integration with EDA Reports

These advanced analyses can be integrated with the EDA reporting system:

```python
from freamon.eda.simple_report import generate_simple_eda_report

# Generate report with advanced analyses
report = generate_simple_eda_report(
    df,
    target_column='target',
    # Other parameters...
    export_format='pptx',  # Export to PowerPoint
    output_path='advanced_eda_report.pptx'
)
```

## Example

See [`examples/advanced_eda_example.py`](examples/advanced_eda_example.py) for a comprehensive demonstration of these features.