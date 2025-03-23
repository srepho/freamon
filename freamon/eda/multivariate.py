"""
Module for multivariate analysis functions.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import io
import base64
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _encode_plot(fig: plt.Figure) -> str:
    """
    Encode a matplotlib figure as a base64 string.
    
    Parameters
    ----------
    fig : plt.Figure
        The figure to encode.
        
    Returns
    -------
    str
        The encoded figure as a base64 string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def perform_pca(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_components: int = 2,
    scale: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform PCA on the specified columns and return the results with visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : Optional[List[str]], default=None
        The columns to use for PCA. If None, all numeric columns are used.
    n_components : int, default=2
        The number of components to extract.
    scale : bool, default=True
        Whether to standardize the data before PCA.
    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional arguments to pass to the plot function.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with PCA results.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Select numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter only numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required for PCA")
    
    # Extract the features
    X = df[columns].values
    
    # Handle missing values
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    
    # Standardize the data if requested
    if scale:
        X = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, len(columns)))
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame with the results
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot of first two components
    if X_pca.shape[1] >= 2:
        scatter_kwargs = {
            'alpha': 0.6, 
            's': 50,
            'edgecolor': 'k',
            'linewidth': 0.5
        }
        scatter_kwargs.update(plot_kwargs.get('scatter', {}))
        
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], **scatter_kwargs)
        axes[0].set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
        axes[0].set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
        axes[0].set_title('PCA: First Two Principal Components')
        axes[0].grid(alpha=0.3)
    
    # Explained variance plot
    bar_kwargs = {'alpha': 0.7, 'color': 'steelblue'}
    bar_kwargs.update(plot_kwargs.get('bar', {}))
    
    axes[1].bar(range(1, len(explained_variance) + 1), 
                explained_variance, **bar_kwargs)
    
    line_kwargs = {'color': 'red', 'marker': 'o', 'linestyle': '--'}
    line_kwargs.update(plot_kwargs.get('line', {}))
    
    ax2 = axes[1].twinx()
    ax2.plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance, **line_kwargs)
    
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    ax2.set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Explained Variance by Component')
    axes[1].set_xticks(range(1, len(explained_variance) + 1))
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Get the loadings (feature importance for each component)
    loadings = pca.components_
    loadings_df = pd.DataFrame(
        loadings.T,
        index=columns,
        columns=[f"PC{i+1}" for i in range(loadings.shape[0])]
    )
    
    # Calculate feature contributions to principal components
    abs_loadings = np.abs(loadings)
    contribution = abs_loadings / np.sum(abs_loadings, axis=1, keepdims=True)
    contribution_df = pd.DataFrame(
        contribution.T,
        index=columns,
        columns=[f"PC{i+1}" for i in range(contribution.shape[0])]
    )
    
    # Create loadings visualization
    if len(columns) <= 15:  # Only create this plot if there aren't too many features
        fig2, ax = plt.subplots(figsize=(10, 8))
        
        heatmap_kwargs = {'cmap': 'coolwarm', 'center': 0, 'annot': True}
        heatmap_kwargs.update(plot_kwargs.get('heatmap', {}))
        
        sns.heatmap(loadings_df.iloc[:, :min(5, loadings_df.shape[1])], 
                   **heatmap_kwargs)
        ax.set_title('Feature Loadings for Top Principal Components')
        
        loadings_plot = _encode_plot(fig2)
    else:
        loadings_plot = None
    
    return {
        "pca_results": pca_df.to_dict(orient='list'),
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "loadings": loadings_df.to_dict(),
        "feature_contributions": contribution_df.to_dict(),
        "n_components": n_components,
        "visualization": _encode_plot(fig),
        "loadings_visualization": loadings_plot,
    }


def perform_tsne(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: Union[float, str] = 'auto',
    n_iter: int = 1000,
    scale: bool = True,
    random_state: int = 42,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform t-SNE on the specified columns and return the results with visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : Optional[List[str]], default=None
        The columns to use for t-SNE. If None, all numeric columns are used.
    n_components : int, default=2
        The number of components to extract.
    perplexity : float, default=30.0
        The perplexity parameter for t-SNE.
    learning_rate : Union[float, str], default='auto'
        The learning rate for t-SNE.
    n_iter : int, default=1000
        The number of iterations for t-SNE.
    scale : bool, default=True
        Whether to standardize the data before t-SNE.
    random_state : int, default=42
        Random seed for reproducibility.
    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional arguments to pass to the plot function.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with t-SNE results.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Select numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter only numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required for t-SNE")
    
    # Extract the features
    X = df[columns].values
    
    # Handle missing values
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    
    # Standardize the data if requested
    if scale:
        X = StandardScaler().fit_transform(X)
    
    # Apply t-SNE (can be computationally expensive)
    import time
    start_time = time.time()
    
    print(f"Running t-SNE on {X.shape[0]} samples with {X.shape[1]} features...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(df) - 1),  # Perplexity must be less than sample size
        learning_rate=learning_rate,
        n_iter=n_iter,
        verbose=1,  # Show progress during computation
        random_state=random_state
    )
    X_tsne = tsne.fit_transform(X)
    
    end_time = time.time()
    print(f"t-SNE completed in {end_time - start_time:.2f} seconds")
    
    # Create a DataFrame with the results
    tsne_df = pd.DataFrame(
        X_tsne,
        columns=[f"TSNE{i+1}" for i in range(X_tsne.shape[1])]
    )
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter_kwargs = {
        'alpha': 0.6, 
        's': 50,
        'edgecolor': 'k',
        'linewidth': 0.5,
        'c': 'steelblue'
    }
    scatter_kwargs.update(plot_kwargs.get('scatter', {}))
    
    if X_tsne.shape[1] >= 2:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], **scatter_kwargs)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Visualization')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return {
        "tsne_results": tsne_df.to_dict(orient='list'),
        "perplexity": perplexity,
        "learning_rate": learning_rate if isinstance(learning_rate, float) else str(learning_rate),
        "n_iter": n_iter,
        "n_components": n_components,
        "visualization": _encode_plot(fig),
    }


def analyze_multivariate(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: Literal['pca', 'tsne', 'both', 'correlation_network', 'interaction_heatmap', 'all'] = 'all',
    n_components: int = 2,
    scale: bool = True,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: Union[float, str] = 'auto',
    tsne_n_iter: int = 1000,
    correlation_threshold: float = 0.5,
    correlation_method: str = 'pearson',
    correlation_layout: str = 'spring',
    max_heatmap_features: int = 20,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform multivariate analysis on the specified columns using various techniques.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : Optional[List[str]], default=None
        The columns to use for analysis. If None, all numeric columns are used.
    method : Literal['pca', 'tsne', 'correlation_network', 'interaction_heatmap', 'all'], default='all'
        The analysis method(s) to use.
    n_components : int, default=2
        The number of components to extract for PCA and t-SNE.
    scale : bool, default=True
        Whether to standardize the data before PCA and t-SNE.
    tsne_perplexity : float, default=30.0
        The perplexity parameter for t-SNE.
    tsne_learning_rate : Union[float, str], default='auto'
        The learning rate for t-SNE.
    tsne_n_iter : int, default=1000
        The number of iterations for t-SNE.
    correlation_threshold : float, default=0.5
        Minimum absolute correlation value for the correlation network.
    correlation_method : str, default='pearson'
        The correlation method to use ('pearson', 'spearman', or 'kendall').
    correlation_layout : str, default='spring'
        The graph layout algorithm for the correlation network.
    max_heatmap_features : int, default=20
        Maximum number of features to include in the interaction heatmap.
    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional arguments to pass to the plot functions.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary with multivariate analysis results.
    """
    result = {}
    
    # Select numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter only numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required for multivariate analysis")
    
    # Perform PCA if requested
    if method in ['pca', 'both', 'all']:
        result['pca'] = perform_pca(
            df=df,
            columns=columns,
            n_components=n_components,
            scale=scale,
            plot_kwargs=plot_kwargs
        )
    
    # Perform t-SNE if requested
    if method in ['tsne', 'both', 'all']:
        result['tsne'] = perform_tsne(
            df=df,
            columns=columns,
            n_components=n_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            n_iter=tsne_n_iter,
            scale=scale,
            plot_kwargs=plot_kwargs
        )
    
    # Perform correlation network analysis if requested
    if method in ['correlation_network', 'all']:
        try:
            result['correlation_network'] = create_correlation_network(
                df=df,
                columns=columns,
                threshold=correlation_threshold,
                method=correlation_method,
                layout=correlation_layout,
                plot_kwargs=plot_kwargs
            )
        except ValueError as e:
            # Handle the case where there are no correlations above threshold
            result['correlation_network'] = {
                "error": str(e),
                "recommendation": "Try lowering the correlation threshold or using a different correlation method."
            }
    
    # Perform interaction heatmap if requested
    if method in ['interaction_heatmap', 'all']:
        result['interaction_heatmap'] = create_interaction_heatmap(
            df=df,
            columns=columns,
            max_features=max_heatmap_features,
            method=correlation_method,
            plot_kwargs=plot_kwargs
        )
    
    return result


def create_correlation_network(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.5,
    method: str = 'pearson',
    node_size_factor: float = 500,
    layout: str = 'spring',
    colormap: str = 'coolwarm',
    figsize: Tuple[int, int] = (12, 10),
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a network visualization of feature correlations.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : Optional[List[str]], default=None
        The columns to use for network creation. If None, all numeric columns are used.
    threshold : float, default=0.5
        Minimum absolute correlation value to include an edge in the network.
    method : str, default='pearson'
        The correlation method to use ('pearson', 'spearman', or 'kendall').
    node_size_factor : float, default=500
        Factor to scale node sizes based on their connectivity.
    layout : str, default='spring'
        The graph layout algorithm to use ('spring', 'circular', 'kamada_kawai', 'spectral').
    colormap : str, default='coolwarm'
        Colormap to use for edge colors based on correlation values.
    figsize : Tuple[int, int], default=(12, 10)
        Figure size for the network plot.
    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional arguments to pass to the plot functions.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with correlation network results.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Select numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter only numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required for correlation network")
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (features)
    for col in columns:
        G.add_node(col)
    
    # Add edges (correlations above threshold)
    edge_weights = []
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) >= threshold:
                G.add_edge(col1, col2, weight=corr_value)
                edge_weights.append(corr_value)
    
    if len(edge_weights) == 0:
        raise ValueError(
            f"No correlations above threshold {threshold} found. "
            "Try lowering the threshold or using a different correlation method."
        )
    
    # Calculate node connectivity (degree centrality)
    centrality = nx.degree_centrality(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine layout
    pos_funcs = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout
    }
    
    pos_func = pos_funcs.get(layout, nx.spring_layout)
    pos = pos_func(G, seed=42)
    
    # Draw nodes
    node_kwargs = {
        'alpha': 0.8,
        'linewidths': 1,
        'edgecolors': 'black'
    }
    node_kwargs.update(plot_kwargs.get('node', {}))
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[centrality[node] * node_size_factor * len(G.nodes) for node in G.nodes],
        node_color='skyblue',
        **node_kwargs
    )
    
    # Draw node labels
    label_kwargs = {
        'font_size': 10,
        'font_weight': 'bold'
    }
    label_kwargs.update(plot_kwargs.get('label', {}))
    
    nx.draw_networkx_labels(G, pos, **label_kwargs)
    
    # Draw edges with color based on correlation
    edge_kwargs = {
        'width': 2,
        'alpha': 0.7
    }
    edge_kwargs.update(plot_kwargs.get('edge', {}))
    
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=[G[u][v]['weight'] for u, v in G.edges()],
        edge_cmap=plt.colormaps.get_cmap(colormap),
        edge_vmin=-1, edge_vmax=1,
        **edge_kwargs
    )
    
    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps.get_cmap(colormap), norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Correlation Strength')
    
    # Add legend for node sizes
    sizes = sorted(set([centrality[node] for node in G.nodes]))
    if len(sizes) > 3:
        sizes = [min(sizes), np.median(sizes), max(sizes)]
    
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', alpha=0.8,
              label=f'Degree: {size:.2f}')
        for size in sizes
    ]
    plt.legend(handles=legend_elements, title="Node Connectivity", loc="upper right")
    
    plt.title(f'Feature Correlation Network (threshold={threshold})')
    plt.axis('off')
    plt.tight_layout()
    
    # Calculate community structure using Louvain algorithm
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
    except ImportError:
        communities = None
    
    # Create community visualization if communities were detected
    if communities and len(communities) > 1:
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Draw nodes colored by community
        cmap = plt.colormaps.get_cmap('tab10', max(communities.keys()) + 1)
        
        nx.draw_networkx_nodes(
            G, pos,
            node_size=[centrality[node] * node_size_factor * len(G.nodes) for node in G.nodes],
            node_color=[partition[node] for node in G.nodes],
            cmap=cmap,
            **node_kwargs
        )
        
        nx.draw_networkx_labels(G, pos, **label_kwargs)
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=[G[u][v]['weight'] for u, v in G.edges()],
            edge_cmap=plt.colormaps.get_cmap(colormap),
            edge_vmin=-1, edge_vmax=1,
            **edge_kwargs
        )
        
        # Create legend for communities
        legend_elements = [
            Patch(facecolor=cmap(comm_id), edgecolor='black', alpha=0.8,
                  label=f'Group {comm_id + 1}: {len(members)} features')
            for comm_id, members in communities.items()
        ]
        plt.legend(handles=legend_elements, title="Feature Groups", loc="upper right")
        
        plt.title('Feature Correlation Network with Community Detection')
        plt.axis('off')
        plt.tight_layout()
        
        community_viz = _encode_plot(fig2)
    else:
        community_viz = None
        communities = None
    
    return {
        "graph_nodes": list(G.nodes()),
        "graph_edges": [(u, v, G[u][v]['weight']) for u, v in G.edges()],
        "centrality": centrality,
        "communities": communities,
        "method": method,
        "threshold": threshold,
        "visualization": _encode_plot(fig),
        "community_visualization": community_viz,
    }


def create_interaction_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_features: int = 20,
    method: str = 'pearson',
    cmap: str = 'coolwarm',
    figsize: Tuple[int, int] = (12, 10),
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an interactive heatmap for feature interactions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : Optional[List[str]], default=None
        The columns to use for the heatmap. If None, all numeric columns are used.
    max_features : int, default=20
        Maximum number of features to include in the heatmap.
    method : str, default='pearson'
        The correlation method to use ('pearson', 'spearman', or 'kendall').
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap.
    figsize : Tuple[int, int], default=(12, 10)
        Figure size for the heatmap.
    plot_kwargs : Optional[Dict[str, Any]], default=None
        Additional arguments to pass to the plot functions.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with interaction heatmap results.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Select numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter only numeric columns
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required for interaction heatmap")
    
    # Limit the number of features if needed
    if len(columns) > max_features:
        # Compute variance of each feature
        variances = df[columns].var().sort_values(ascending=False)
        columns = variances.index[:max_features].tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    # Create hierarchical clusters to determine feature order
    link = sns.clustermap(
        corr_matrix,
        cmap=cmap,
        col_cluster=True,
        row_cluster=True,
        figsize=figsize,
        cbar_pos=None  # No colorbar
    )
    plt.close()  # Close the clustermap figure
    
    # Get the reordered index based on clustering
    row_order = link.dendrogram_row.reordered_ind
    reordered_columns = [corr_matrix.index[i] for i in row_order]
    
    # Create ordered correlation matrix
    ordered_corr = corr_matrix.loc[reordered_columns, reordered_columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap_kwargs = {
        'cmap': cmap,
        'center': 0,
        'annot': True,
        'fmt': '.2f',
        'linewidths': 0.5,
        'cbar_kws': {'shrink': 0.8, 'label': f'{method.capitalize()} Correlation'}
    }
    heatmap_kwargs.update(plot_kwargs.get('heatmap', {}))
    
    # Create heatmap
    sns.heatmap(ordered_corr, ax=ax, **heatmap_kwargs)
    
    ax.set_title(f'Feature Interaction Heatmap ({method.capitalize()} Correlation)')
    plt.tight_layout()
    
    # Calculate feature clusters
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(ordered_corr, method='ward')
    
    # Get clusters at various levels
    max_clusters = min(5, len(columns) // 2)  # Reasonable max number of clusters
    cluster_results = {}
    
    for n_clusters in range(2, max_clusters + 1):
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        grouped_features = {}
        
        for i, cluster_id in enumerate(clusters):
            feature = reordered_columns[i]
            if cluster_id not in grouped_features:
                grouped_features[cluster_id] = []
            grouped_features[cluster_id].append(feature)
        
        cluster_results[n_clusters] = grouped_features
    
    return {
        "correlation_matrix": ordered_corr.to_dict(),
        "feature_order": reordered_columns,
        "method": method,
        "feature_clusters": cluster_results,
        "visualization": _encode_plot(fig),
    }