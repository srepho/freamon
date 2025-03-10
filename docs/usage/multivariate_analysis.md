# Multivariate Analysis

Freamon provides powerful tools for multivariate analysis through the `EDAAnalyzer` class and the `multivariate` module. These tools help you understand complex relationships between multiple variables in your dataset.

## Available Multivariate Techniques

The multivariate analysis module includes several techniques:

1. **Principal Component Analysis (PCA)** - Reduce dimensionality while preserving variance
2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** - Non-linear dimensionality reduction for visualization
3. **Correlation Networks** - Graph-based visualization of feature relationships 
4. **Interaction Heatmaps** - Hierarchically clustered correlation matrices

## Basic Usage

The simplest way to perform multivariate analysis is through the `EDAAnalyzer` class:

```python
from freamon.eda import EDAAnalyzer
import pandas as pd

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Create analyzer
analyzer = EDAAnalyzer(df)

# Perform multivariate analysis with all methods
results = analyzer.analyze_multivariate()

# Generate an HTML report with the results
analyzer.generate_report("eda_report.html")
```

## Customizing Multivariate Analysis

You can customize the multivariate analysis process with various parameters:

```python
# Perform specific multivariate analyses
results = analyzer.analyze_multivariate(
    # Select specific columns to analyze
    columns=['feature1', 'feature2', 'feature3'],
    
    # Choose which methods to use:
    # 'pca', 'tsne', 'correlation_network', 'interaction_heatmap', or 'all'
    method='all',
    
    # PCA and t-SNE configuration
    n_components=3,  # Extract 3 components
    scale=True,      # Standardize data before analysis
    
    # t-SNE specific parameters
    tsne_perplexity=30.0,
    tsne_learning_rate='auto',
    tsne_n_iter=1000,
    
    # Correlation network parameters
    correlation_threshold=0.5,  # Minimum correlation to show an edge
    correlation_method='pearson',  # 'pearson', 'spearman', or 'kendall'
    correlation_layout='spring',  # Graph layout algorithm
    
    # Interaction heatmap parameters
    max_heatmap_features=20,  # Max features to include in heatmap
)
```

## Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that identifies the directions (principal components) along which your data varies the most.

```python
from freamon.eda.multivariate import perform_pca

# Perform PCA directly
pca_results = perform_pca(
    df,
    columns=['col1', 'col2', 'col3', 'col4'],
    n_components=2,
    scale=True
)

# Access PCA results
explained_variance = pca_results['explained_variance']
feature_loadings = pca_results['loadings']
```

The PCA results include:
- Transformed data points
- Explained variance for each component
- Feature loadings (contributions of each feature to the components)
- Visualizations of the components and explained variance

## t-SNE Visualization

t-SNE is particularly effective for visualizing high-dimensional data in 2D or 3D space, preserving local structures.

```python
from freamon.eda.multivariate import perform_tsne

# Perform t-SNE
tsne_results = perform_tsne(
    df,
    columns=['col1', 'col2', 'col3', 'col4'],
    perplexity=30.0,
    learning_rate='auto',
    n_iter=1000
)
```

## Correlation Networks

Correlation networks visualize relationships between features as a graph, where:
- Nodes represent features
- Edges represent correlations above a threshold
- Node size indicates connectivity (degree centrality)
- Edge colors represent correlation strength

```python
from freamon.eda.multivariate import create_correlation_network

# Create correlation network
network_results = create_correlation_network(
    df,
    columns=['col1', 'col2', 'col3', 'col4'],
    threshold=0.5,  # Minimum correlation to show
    method='pearson',
    layout='spring'
)

# Access community detection results
communities = network_results['communities']
```

The correlation network can detect feature communities (groups of related features) using the Louvain algorithm, which can help identify feature clusters.

## Interaction Heatmaps

Interaction heatmaps provide a hierarchically clustered view of feature correlations:

```python
from freamon.eda.multivariate import create_interaction_heatmap

# Create interaction heatmap
heatmap_results = create_interaction_heatmap(
    df,
    columns=['col1', 'col2', 'col3', 'col4'],
    max_features=20,
    method='pearson'
)

# Access feature clusters identified by hierarchical clustering
clusters = heatmap_results['feature_clusters']
```

The heatmap automatically reorders features based on hierarchical clustering to highlight patterns and groups of related features.

## Visualizations

All multivariate analysis functions return visualizations encoded as base64 strings that can be embedded in HTML reports:

```python
# Access visualizations
pca_viz = results['pca']['visualization']
tsne_viz = results['tsne']['visualization']
network_viz = results['correlation_network']['visualization']
heatmap_viz = results['interaction_heatmap']['visualization']

# You can also get the community detection visualization
community_viz = results['correlation_network']['community_visualization']
```

These visualizations are automatically included when using `analyzer.generate_report()`.

## Working with Large Datasets

For large datasets, the multivariate analysis module automatically uses optimized processing. You can also manually control memory usage:

```python
# For large datasets, limit features by variance
import pandas as pd

# Get top 50 features by variance
variances = df.var().sort_values(ascending=False)
top_features = variances.index[:50].tolist()

# Analyze only top features
results = analyzer.analyze_multivariate(columns=top_features)
```

See the large dataset handling documentation for more advanced techniques for processing very large datasets.