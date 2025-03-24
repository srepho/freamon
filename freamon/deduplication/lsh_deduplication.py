"""Locality-Sensitive Hashing (LSH) implementation for efficient text deduplication."""

from typing import Dict, List, Tuple, Union, Optional, Any, Set
import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
from collections import defaultdict
import random

from freamon.utils.text_utils import TextProcessor
from freamon.deduplication.fingerprinting import create_minhash_signature


def create_lsh_bands(
    signature: List[int], 
    num_bands: int, 
    rows_per_band: Optional[int] = None
) -> List[Tuple[int, Tuple]]:
    """
    Divide a minhash signature into bands for LSH.
    
    Parameters
    ----------
    signature : List[int]
        The minhash signature to divide into bands.
    num_bands : int
        Number of bands to divide the signature into.
    rows_per_band : Optional[int], default=None
        Number of rows per band. If None, will be calculated as len(signature) // num_bands.
    
    Returns
    -------
    List[Tuple[int, Tuple]]
        List of (band_idx, band_hash) tuples.
    """
    if rows_per_band is None:
        rows_per_band = len(signature) // num_bands
        
    # Ensure we have enough rows in the signature
    if len(signature) < num_bands * rows_per_band:
        raise ValueError(f"Signature length {len(signature)} is less than num_bands * rows_per_band ({num_bands * rows_per_band})")
    
    bands = []
    for i in range(num_bands):
        # Extract the current band
        start_idx = i * rows_per_band
        end_idx = start_idx + rows_per_band
        band = tuple(signature[start_idx:end_idx])
        
        # Add (band_idx, band_tuple) to list
        bands.append((i, band))
    
    return bands


def lsh_deduplication(
    texts: Union[List[str], pd.Series],
    threshold: float = 0.7,
    num_minhash_permutations: int = 100,
    num_bands: int = 20,
    preprocess: bool = True,
    text_processor: Optional[TextProcessor] = None,
    shingle_size: int = 3,
    return_indices: bool = True,
    keep: str = 'first',
    return_similarity_dict: bool = False,
) -> Union[List[int], Tuple[List[int], Dict[int, List[int]]]]:
    """
    Find duplicate texts using Locality-Sensitive Hashing (LSH).
    
    This implementation uses MinHash signatures and banding technique to efficiently
    find similar documents without comparing all possible pairs.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    threshold : float, default=0.7
        Similarity threshold (0-1). Higher values require more similarity.
    num_minhash_permutations : int, default=100
        Number of permutations to use for MinHash signature generation.
    num_bands : int, default=20
        Number of bands to divide the MinHash signatures into for LSH.
        More bands increase recall but reduce precision.
    preprocess : bool, default=True
        Whether to preprocess texts before comparison.
    text_processor : Optional[TextProcessor], default=None
        Instance of TextProcessor. If None, a new one will be created.
    shingle_size : int, default=3
        Size of the shingles (n-grams) to create from documents.
    return_indices : bool, default=True
        Whether to return indices of unique texts after deduplication.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    return_similarity_dict : bool, default=False
        If True, also return the candidate pairs dictionary.
    
    Returns
    -------
    Union[List[int], Tuple[List[int], Dict[int, List[int]]]]
        If return_similarity_dict=False: List of indices of unique texts.
        If return_similarity_dict=True: Tuple of (kept_indices, candidate_pairs_dict)
    """
    # Create TextProcessor if not provided
    if text_processor is None:
        text_processor = TextProcessor()
    
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Handle empty texts list
    if not texts:
        return [] if not return_similarity_dict else ([], {})
    
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
    
    # Calculate rows per band
    rows_per_band = num_minhash_permutations // num_bands
    if rows_per_band < 1:
        rows_per_band = 1
        num_bands = num_minhash_permutations
    
    # Create LSH hash tables (one per band)
    hash_tables = [defaultdict(list) for _ in range(num_bands)]
    
    # Generate MinHash signatures and hash them into bands
    signatures = []
    for idx, text in enumerate(processed_texts):
        if not text:  # Skip empty texts
            signatures.append([])
            continue
            
        # Create MinHash signature for the text
        signature = create_minhash_signature(
            text=text,
            shingle_size=shingle_size,
            num_permutations=num_minhash_permutations
        )
        
        signatures.append(signature)
        
        # Divide signature into bands and hash each band
        bands = create_lsh_bands(signature, num_bands, rows_per_band)
        
        # Add document index to appropriate hash buckets
        for band_idx, band_hash in bands:
            hash_tables[band_idx][band_hash].append(idx)
    
    # Find candidate pairs from hash tables
    candidate_pairs = defaultdict(list)
    
    for band_idx in range(num_bands):
        for bucket in hash_tables[band_idx].values():
            if len(bucket) > 1:  # Potential similar documents
                for i in bucket:
                    for j in bucket:
                        if i != j and j not in candidate_pairs[i]:
                            candidate_pairs[i].append(j)
    
    # Verify candidate pairs by computing actual similarities
    similar_pairs = defaultdict(list)
    
    for i, candidates in candidate_pairs.items():
        for j in candidates:
            # Skip if either signature is empty (indicating empty text)
            if not signatures[i] or not signatures[j]:
                continue
                
            # Compute Jaccard similarity using signatures
            intersect = sum(1 for h1, h2 in zip(signatures[i], signatures[j]) if h1 == h2)
            jaccard = intersect / num_minhash_permutations
            
            if jaccard >= threshold:
                similar_pairs[i].append(j)
    
    # If no return is needed, return empty list
    if not return_indices:
        if return_similarity_dict:
            return [], dict(similar_pairs)
        else:
            return []
    
    # Find connected components (clusters of duplicates)
    visited = set()
    clusters = []
    
    for i in range(len(texts)):
        if i in visited:
            continue
            
        # Find all connected documents using breadth-first search
        cluster = {i}
        queue = [i]
        
        while queue:
            node = queue.pop(0)
            for neighbor in similar_pairs.get(node, []):
                if neighbor not in cluster:
                    cluster.add(neighbor)
                    queue.append(neighbor)
        
        visited.update(cluster)
        clusters.append(cluster)
    
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
        return result, dict(similar_pairs)
    else:
        return result