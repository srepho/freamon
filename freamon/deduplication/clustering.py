"""Clustering-based deduplication methods."""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import re

import pandas as pd
import numpy as np
import networkx as nx

from freamon.utils.text_utils import TextProcessor
from freamon.deduplication.fuzzy_deduplication import find_similar_texts


def cluster_texts_hierarchical(
    texts: Union[List[str], pd.Series],
    distance_threshold: float = 0.3,  # Lower threshold = more clusters
    method: str = 'cosine',
    preprocess: bool = True,
    text_processor: Optional[TextProcessor] = None
) -> Dict[int, List[int]]:
    """
    Cluster texts using hierarchical clustering.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to cluster.
    distance_threshold : float, default=0.3
        Distance threshold for clustering. Lower values create more clusters.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    text_processor : Optional[TextProcessor], default=None
        Instance of TextProcessor. If None, a new one will be created.
    
    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping cluster IDs to lists of text indices in each cluster.
    """
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
    except ImportError:
        raise ImportError("scipy is not installed. Install it with 'pip install scipy'.")
    
    # Create TextProcessor if not provided
    if text_processor is None:
        text_processor = TextProcessor()
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Get similarity matrix using find_similar_texts
    similarity_matrix = np.zeros((len(texts), len(texts)))
    
    similarity_threshold = 1.0 - distance_threshold  # Convert distance to similarity
    
    # Find similar texts
    similar_texts = find_similar_texts(
        texts=texts,
        threshold=0.0,  # Get all pairs
        method=method,
        preprocess=preprocess,
        return_scores=True,
        text_processor=text_processor
    )
    
    # Fill the similarity matrix
    for i in range(len(texts)):
        similarity_matrix[i, i] = 1.0  # Self-similarity
        
        for j_tuple in similar_texts.get(i, []):
            j, sim = j_tuple if isinstance(j_tuple, tuple) else (j_tuple, 1.0)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric
    
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    
    # Create condensed distance matrix (upper triangular)
    condensed_distance = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_distance, method='average')
    
    # Form flat clusters
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')
    
    # Create cluster dictionary
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters


def cluster_texts_dbscan(
    texts: Union[List[str], pd.Series],
    eps: float = 0.3,  # Maximum distance between samples in a cluster
    min_samples: int = 2,  # Minimum samples in a core point neighborhood
    method: str = 'cosine',
    preprocess: bool = True,
    text_processor: Optional[TextProcessor] = None
) -> Dict[int, List[int]]:
    """
    Cluster texts using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to cluster.
    eps : float, default=0.3
        Maximum distance between samples for them to be in the same neighborhood.
    min_samples : int, default=2
        Minimum number of samples in a neighborhood for a point to be a core point.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    text_processor : Optional[TextProcessor], default=None
        Instance of TextProcessor. If None, a new one will be created.
    
    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping cluster IDs to lists of text indices in each cluster.
        Cluster ID -1 represents noise points (outliers).
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("scikit-learn is not installed. Install it with 'pip install scikit-learn'.")
    
    # Create TextProcessor if not provided
    if text_processor is None:
        text_processor = TextProcessor()
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Get similarity matrix using find_similar_texts
    similarity_matrix = np.zeros((len(texts), len(texts)))
    
    # Find similar texts with all pairs
    similar_texts = find_similar_texts(
        texts=texts,
        threshold=0.0,  # Get all pairs
        method=method,
        preprocess=preprocess,
        return_scores=True,
        text_processor=text_processor
    )
    
    # Fill the similarity matrix
    for i in range(len(texts)):
        similarity_matrix[i, i] = 1.0  # Self-similarity
        
        for j_tuple in similar_texts.get(i, []):
            j, sim = j_tuple if isinstance(j_tuple, tuple) else (j_tuple, 1.0)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric
    
    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    dbscan.fit(distance_matrix)
    
    # Get cluster labels
    labels = dbscan.labels_
    
    # Create cluster dictionary
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters
