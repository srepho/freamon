"""
Example demonstrating multivariate analysis capabilities in freamon.

This example shows how to:
1. Perform PCA (Principal Component Analysis)
2. Perform t-SNE (t-Distributed Stochastic Neighbor Embedding)
3. Create correlation network visualizations
4. Generate interactive heatmaps for feature interactions
5. Integrate multivariate analysis in the EDA workflow
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, fetch_california_housing

from freamon.eda import EDAAnalyzer
from freamon.eda.multivariate import (
    perform_pca,
    perform_tsne,
    create_correlation_network,
    create_interaction_heatmap,
    analyze_multivariate
)


def load_datasets():
    """Load example datasets for multivariate analysis."""
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    iris_df['target'] = iris.target
    
    # Load Wine dataset
    wine = load_wine()
    wine_df = pd.DataFrame(
        wine.data,
        columns=wine.feature_names
    )
    wine_df['target'] = wine.target
    
    # Load California Housing dataset
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    housing_df['target'] = housing.target
    
    return {
        "iris": iris_df,
        "wine": wine_df,
        "housing": housing_df
    }


def pca_example(datasets):
    """Demonstrate Principal Component Analysis (PCA)."""
    print("\n===== PCA (Principal Component Analysis) =====")
    
    # Get the Wine dataset
    wine_df = datasets["wine"]
    
    print(f"Wine dataset shape: {wine_df.shape}")
    print(f"Features: {wine_df.columns[:-1].tolist()}")
    
    # Perform PCA directly
    pca_results = perform_pca(
        wine_df,
        columns=wine_df.columns[:-1],  # Exclude target column
        n_components=2,
        scale=True
    )
    
    # Print results
    print("\nPCA Results:")
    print(f"Explained variance ratios: {[f'{v:.2%}' for v in pca_results['explained_variance']]}")
    print(f"Cumulative explained variance: {[f'{v:.2%}' for v in pca_results['cumulative_variance']]}")
    
    # Show top feature contributions to PC1
    loadings_df = pd.DataFrame(pca_results['loadings'])
    top_features_pc1 = loadings_df['PC1'].abs().sort_values(ascending=False)
    
    print("\nTop features contributing to PC1:")
    for feature, loading in top_features_pc1.head(5).items():
        print(f"  - {feature}: {loading:.4f}")


def tsne_example(datasets):
    """Demonstrate t-SNE visualization."""
    print("\n===== t-SNE Visualization =====")
    
    # Get the Iris dataset
    iris_df = datasets["iris"]
    
    print(f"Iris dataset shape: {iris_df.shape}")
    print(f"Features: {iris_df.columns[:-1].tolist()}")
    
    # Perform t-SNE
    tsne_results = perform_tsne(
        iris_df,
        columns=iris_df.columns[:-1],  # Exclude target column
        perplexity=30.0,
        learning_rate='auto',
        n_iter=1000,
        random_state=42
    )
    
    # Print results
    print("\nt-SNE Results:")
    print(f"Configuration: perplexity={tsne_results['perplexity']}, "
          f"learning_rate={tsne_results['learning_rate']}, "
          f"n_iter={tsne_results['n_iter']}")
    
    # Create a scatter plot colored by target (done automatically in visualization)
    print("\nt-SNE visualization has been generated and encoded as base64 string.")
    print("This visualization would be included in the HTML report.")


def correlation_network_example(datasets):
    """Demonstrate correlation network visualization."""
    print("\n===== Correlation Network Visualization =====")
    
    # Get the Housing dataset
    housing_df = datasets["housing"]
    
    print(f"Housing dataset shape: {housing_df.shape}")
    print(f"Features: {housing_df.columns[:-1].tolist()}")
    
    # Create correlation network
    network_results = create_correlation_network(
        housing_df,
        columns=housing_df.columns[:-1],  # Exclude target column
        threshold=0.3,  # Lower threshold to include more connections
        method='pearson',
        layout='spring',
        node_size_factor=500
    )
    
    # Print results
    print("\nCorrelation Network Results:")
    print(f"Number of nodes (features): {len(network_results['graph_nodes'])}")
    print(f"Number of edges (connections): {len(network_results['graph_edges'])}")
    
    # Print highest correlations
    sorted_edges = sorted(network_results['graph_edges'], 
                          key=lambda x: abs(x[2]), 
                          reverse=True)
    
    print("\nTop 5 strongest correlations:")
    for u, v, weight in sorted_edges[:5]:
        print(f"  - {u} ↔ {v}: {weight:.4f}")
    
    # Check if communities were detected
    if network_results['communities']:
        print("\nFeature communities detected:")
        for community_id, members in network_results['communities'].items():
            print(f"  - Group {community_id + 1}: {', '.join(members)}")


def interaction_heatmap_example(datasets):
    """Demonstrate interaction heatmap visualization."""
    print("\n===== Feature Interaction Heatmap =====")
    
    # Get the Wine dataset
    wine_df = datasets["wine"]
    
    print(f"Wine dataset shape: {wine_df.shape}")
    
    # Create interaction heatmap
    heatmap_results = create_interaction_heatmap(
        wine_df,
        columns=wine_df.columns[:-1],  # Exclude target column
        max_features=20,
        method='pearson'
    )
    
    # Print results
    print("\nInteraction Heatmap Results:")
    print(f"Features ordered by clustering: {heatmap_results['feature_order']}")
    
    # Print feature clusters
    print("\nFeature clusters identified by hierarchical clustering:")
    for n_clusters, clusters in heatmap_results['feature_clusters'].items():
        print(f"\nWith {n_clusters} clusters:")
        for cluster_id, members in clusters.items():
            print(f"  - Cluster {cluster_id}: {', '.join(members)}")


def combined_multivariate_example(datasets):
    """Demonstrate combined multivariate analysis."""
    print("\n===== Combined Multivariate Analysis =====")
    
    # Get the Wine dataset
    wine_df = datasets["wine"]
    
    # Perform all multivariate analyses
    all_results = analyze_multivariate(
        wine_df,
        columns=wine_df.columns[:-1],  # Exclude target column
        method='all',  # Perform all methods
        n_components=2,
        correlation_threshold=0.4,
        correlation_method='pearson',
        max_heatmap_features=13  # Include all wine features
    )
    
    # Print what's available in the results
    print("\nMultivariate analysis complete. Results include:")
    for method, results in all_results.items():
        print(f"  - {method}: {', '.join(results.keys())}")


def eda_analyzer_example(datasets):
    """Demonstrate multivariate analysis through EDAAnalyzer."""
    print("\n===== EDA Analyzer with Multivariate Analysis =====")
    
    # Use all three datasets
    for dataset_name, df in datasets.items():
        print(f"\nAnalyzing {dataset_name} dataset...")
        
        # Create EDA analyzer
        analyzer = EDAAnalyzer(df, target_column='target')
        
        # Run basic analysis
        analyzer.analyze_basic_stats()
        analyzer.analyze_univariate()
        analyzer.analyze_bivariate()
        
        # Run multivariate analysis
        multivariate_results = analyzer.analyze_multivariate(
            method='all',
            correlation_threshold=0.3,
            max_heatmap_features=min(15, len(df.columns) - 1)
        )
        
        # Generate report
        output_path = f"{dataset_name}_eda_report.html"
        analyzer.generate_report(
            output_path=output_path,
            title=f"{dataset_name.capitalize()} Dataset Analysis"
        )
        
        print(f"Generated EDA report with multivariate analysis: {output_path}")


if __name__ == "__main__":
    print("Multivariate Analysis Example")
    print("============================")
    
    # Load example datasets
    datasets = load_datasets()
    
    # Run examples
    pca_example(datasets)
    tsne_example(datasets)
    correlation_network_example(datasets)
    interaction_heatmap_example(datasets)
    combined_multivariate_example(datasets)
    eda_analyzer_example(datasets)
    
    print("\nAll examples completed successfully!")