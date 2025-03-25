"""Fuzzy text deduplication methods using similarity metrics."""

from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import re

import pandas as pd
import numpy as np
import networkx as nx

from freamon.utils.text_utils import TextProcessor


def calculate_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein similarity between two strings.
    
    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.
        
    Returns
    -------
    float
        Similarity score between 0 and 1, where 1 indicates identical strings.
    """
    if s1 == s2:
        return 1.0
        
    if not s1 or not s2:
        return 0.0
        
    # Implement Levenshtein distance
    m, n = len(s1), len(s2)
    
    # Initialize the distance matrix
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Source prefixes can be transformed into empty string by
    # dropping all characters
    for i in range(m + 1):
        d[i][0] = i
        
    # Target prefixes can be reached from empty source prefix
    # by inserting every character
    for j in range(n + 1):
        d[0][j] = j
        
    # Fill in the distance matrix
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                d[i][j] = d[i-1][j-1]  # No operation required
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # Deletion
                    d[i][j-1] + 1,      # Insertion
                    d[i-1][j-1] + 1     # Substitution
                )
                
    # Calculate similarity from distance
    distance = d[m][n]
    max_len = max(m, n)
    
    if max_len == 0:
        return 1.0  # Both strings are empty
        
    return 1.0 - (distance / max_len)


def calculate_jaccard_similarity(s1: str, s2: str, n: int = 1) -> float:
    """
    Calculate Jaccard similarity between two strings using character n-grams.
    
    Parameters
    ----------
    s1 : str
        First string.
    s2 : str
        Second string.
    n : int, default=1
        Size of n-grams.
        
    Returns
    -------
    float
        Similarity score between 0 and 1, where 1 indicates identical strings.
    """
    if s1 == s2:
        return 1.0
        
    if not s1 or not s2:
        return 0.0
        
    # Create character n-grams
    ngrams1 = [s1[i:i+n] for i in range(max(0, len(s1) - n + 1))]
    ngrams2 = [s2[i:i+n] for i in range(max(0, len(s2) - n + 1))]
    
    # Handle empty n-grams
    if not ngrams1 or not ngrams2:
        return 0.0
        
    # Calculate Jaccard similarity
    set1, set2 = set(ngrams1), set(ngrams2)
    return len(set1 & set2) / len(set1 | set2)


def find_similar_texts(
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    return_scores: bool = False,
    text_processor: Optional[TextProcessor] = None
) -> Dict[int, List[Union[int, Tuple[int, float]]]]:
    """
    Find duplicate texts based on similarity.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to check for duplicates.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    return_scores : bool, default=False
        Whether to include similarity scores in results.
    text_processor : Optional[TextProcessor], default=None
        Instance of TextProcessor. If None, a new one will be created.
    
    Returns
    -------
    Dict[int, List[Union[int, Tuple[int, float]]]]
        Dictionary mapping each text index to a list of duplicate indices 
        (with similarity scores if return_scores=True).
    """
    # Create TextProcessor if not provided
    if text_processor is None:
        text_processor = TextProcessor()
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Handle empty texts list
    if not texts:
        return {}
    
    # Preprocess texts if requested
    processed_texts = texts
    if preprocess:
        processed_texts = [
            text_processor.preprocess_text(
                text,
                lowercase=True,
                remove_punctuation=True
            ) if text is not None and not pd.isna(text) else ''
            for text in texts
        ]
    
    # For embedding-based similarity
    if method == 'embedding':
        # Create document embeddings
        doc_embeddings = []
        
        # Check if word2vec model exists, otherwise create one
        if not hasattr(text_processor, '_word_vectors'):
            embeddings = text_processor.create_word2vec_embeddings(
                texts=processed_texts,
                vector_size=100
            )
            text_processor._word_vectors = embeddings['wv']
        
        # Create document embeddings
        doc_embeddings = text_processor.create_document_embeddings(
            texts=processed_texts,
            word_vectors=text_processor._word_vectors,
            method='mean'
        )
        
        # Find duplicates using embeddings
        duplicates = {}
        for i in range(len(processed_texts)):
            duplicates[i] = []
            for j in range(len(processed_texts)):
                if i == j:
                    continue
                    
                similarity = text_processor.calculate_embedding_similarity(
                    doc_embeddings[i], 
                    doc_embeddings[j],
                    method='cosine'
                )
                
                if similarity >= threshold:
                    if return_scores:
                        duplicates[i].append((j, similarity))
                    else:
                        duplicates[i].append(j)
        
        return duplicates
    
    # For text-based similarity
    duplicates = {}
    
    # Initialize with empty lists
    for i in range(len(processed_texts)):
        duplicates[i] = []
    
    # Compare all pairs of texts
    for i in range(len(processed_texts)):
        for j in range(i+1, len(processed_texts)):  # Only compare each pair once
            # Skip if both texts are empty
            if not processed_texts[i] and not processed_texts[j]:
                continue
                
            # Calculate similarity based on method
            if method == 'cosine':
                similarity = text_processor.calculate_document_similarity(
                    processed_texts[i],
                    processed_texts[j],
                    method='cosine'
                )
            elif method == 'jaccard':
                if processed_texts[i] and processed_texts[j]:  # Non-empty check
                    similarity = calculate_jaccard_similarity(
                        processed_texts[i],
                        processed_texts[j],
                        n=3  # Use character trigrams
                    )
                else:
                    similarity = 0.0
            elif method == 'levenshtein':
                if processed_texts[i] and processed_texts[j]:  # Non-empty check
                    similarity = calculate_levenshtein_similarity(
                        processed_texts[i],
                        processed_texts[j]
                    )
                else:
                    similarity = 0.0
            else:
                raise ValueError(f"Unknown similarity method: {method}. Use 'cosine', 'jaccard', 'levenshtein', or 'embedding'.")
            
            # Add to duplicates if similarity exceeds threshold
            if similarity >= threshold:
                if return_scores:
                    duplicates[i].append((j, similarity))
                    duplicates[j].append((i, similarity))
                else:
                    duplicates[i].append(j)
                    duplicates[j].append(i)
    
    return duplicates


def deduplicate_texts(
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    method: str = 'cosine',
    preprocess: bool = True,
    keep: str = 'first',
    text_processor: Optional[TextProcessor] = None,
    return_indices: bool = True,
    return_similarity_dict: bool = False
) -> Union[List[int], Tuple[List[int], Dict[int, List[Tuple[int, float]]]]]:
    """
    Deduplicate texts based on similarity.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    method : str, default='cosine'
        Similarity method: 'cosine', 'jaccard', 'levenshtein', or 'embedding'.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    text_processor : Optional[TextProcessor], default=None
        Instance of TextProcessor. If None, a new one will be created.
    return_indices : bool, default=True
        This parameter is kept for API compatibility and has no effect as
        the function always returns indices.
    return_similarity_dict : bool, default=False
        If True, also return the similarity dictionary with scores.
    
    Returns
    -------
    Union[List[int], Tuple[List[int], Dict[int, List[Tuple[int, float]]]]]
        If return_similarity_dict=False: List of indices of unique texts after deduplication.
        If return_similarity_dict=True: Tuple of (kept_indices, similarity_dict)
    """
    # Create TextProcessor if not provided
    if text_processor is None:
        text_processor = TextProcessor()
    
    # Find all duplicates
    duplicates_dict = find_similar_texts(
        texts=texts,
        threshold=threshold,
        method=method,
        preprocess=preprocess,
        return_scores=True,
        text_processor=text_processor
    )
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Create a graph of similar texts
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(len(texts)):
        G.add_node(i)
    
    # Add edges for duplicates
    for i, duplicates in duplicates_dict.items():
        for dup in duplicates:
            idx, score = dup if isinstance(dup, tuple) else (dup, 1.0)
            G.add_edge(i, idx, weight=score)
    
    # Find connected components (clusters of duplicates)
    clusters = list(nx.connected_components(G))
    
    # Choose which texts to keep from each cluster
    keep_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single text, keep it
            keep_indices.append(list(cluster)[0])
        else:
            # Multiple similar texts, choose based on keep strategy
            cluster_list = list(cluster)
            
            if keep == 'first':
                keep_idx = min(cluster_list)
            elif keep == 'last':
                keep_idx = max(cluster_list)
            elif keep == 'longest':
                keep_idx = max(cluster_list, key=lambda i: len(texts[i]) if texts[i] is not None and not pd.isna(texts[i]) else 0)
            else:
                raise ValueError(f"Unknown keep strategy: {keep}. Use 'first', 'last', or 'longest'.")
            
            keep_indices.append(keep_idx)
    
    result = sorted(keep_indices)
    
    # Return result based on return_similarity_dict parameter
    if return_similarity_dict:
        return result, duplicates_dict
    else:
        return result
