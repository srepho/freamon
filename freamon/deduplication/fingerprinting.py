"""Document fingerprinting methods for efficient text similarity detection."""

from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
import re
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np


def create_shingled_document(
    text: str,
    k: int = 3,
    is_character_shingles: bool = True
) -> List[str]:
    """
    Generate k-shingles from text (character or word n-grams).
    
    Parameters
    ----------
    text : str
        Input text.
    k : int, default=3
        Shingle size.
    is_character_shingles : bool, default=True
        If True, create character shingles. If False, create word shingles.
    
    Returns
    -------
    List[str]
        List of k-shingles.
    """
    if not text or not isinstance(text, str):
        return []
        
    if is_character_shingles:
        # Character shingles
        return [text[i:i+k] for i in range(max(0, len(text) - k + 1))]
    else:
        # Word shingles
        words = text.split()
        if len(words) < k:
            return []
        return [' '.join(words[i:i+k]) for i in range(len(words) - k + 1)]


def create_minhash_signature(
    text: str,
    num_perm: int = 100,
    k_shingles: int = 3
) -> List[int]:
    """
    Create a MinHash signature for a document.
    
    Parameters
    ----------
    text : str
        Input text.
    num_perm : int, default=100
        Number of permutations for the signature.
    k_shingles : int, default=3
        Size of shingles to use for the document.
    
    Returns
    -------
    List[int]
        MinHash signature as a list of integers.
    """
    # Create shingles (character trigrams by default)
    shingles = create_shingled_document(text, k=k_shingles)
    
    if not shingles:
        return [0] * num_perm
    
    # Create signature using multiple hash functions
    signature = [float('inf')] * num_perm
    
    # Since we can't use actual permutations efficiently,
    # we'll use multiple hash functions with different seeds
    for i in range(num_perm):
        for shingle in shingles:
            # Create a hash for the shingle using a seed based on permutation index
            h = hashlib.md5(f"{shingle}_{i}".encode('utf-8')).hexdigest()
            # Convert to an integer and mod by a large number to reduce to a reasonable range
            hash_val = int(h, 16) % (2**31 - 1)
            # Update the signature with the minimum hash value
            signature[i] = min(signature[i], hash_val)
    
    # Replace any infinity values that might remain
    signature = [0 if x == float('inf') else x for x in signature]
    
    return signature


def calculate_minhash_similarity(
    sig1: List[int],
    sig2: List[int]
) -> float:
    """
    Calculate similarity between two MinHash signatures.
    
    Parameters
    ----------
    sig1 : List[int]
        First MinHash signature.
    sig2 : List[int]
        Second MinHash signature.
    
    Returns
    -------
    float
        Similarity between the two signatures (estimate of Jaccard similarity).
    """
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have the same length.")
        
    # Count how many hash values match
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    
    # Similarity is the fraction of matching values
    return matches / len(sig1)


def create_simhash_signature(
    text: str,
    hash_bits: int = 64,
    k_shingles: int = 3
) -> int:
    """
    Create a SimHash signature for a document.
    
    Parameters
    ----------
    text : str
        Input text.
    hash_bits : int, default=64
        Number of bits in the resulting hash.
    k_shingles : int, default=3
        Size of shingles to use for the document.
    
    Returns
    -------
    int
        SimHash signature as an integer.
    """
    # Create shingles (character trigrams by default)
    shingles = create_shingled_document(text, k=k_shingles)
    
    if not shingles:
        return 0
    
    # Initialize vector of hash_bits dimensions
    vector = [0] * hash_bits
    
    # Process each shingle
    for shingle in shingles:
        # Hash the shingle to a 64-bit integer
        h = hashlib.md5(shingle.encode('utf-8')).hexdigest()
        hash_val = int(h, 16)
        
        # Update the vector based on the bits in hash_val
        for i in range(hash_bits):
            bit = (hash_val >> i) & 1  # Extract the i-th bit
            if bit == 1:
                vector[i] += 1
            else:
                vector[i] -= 1
    
    # Create final hash from the vector
    simhash = 0
    for i in range(hash_bits):
        if vector[i] > 0:
            simhash |= (1 << i)  # Set the i-th bit to 1
    
    return simhash


def calculate_simhash_similarity(
    hash1: int,
    hash2: int,
    hash_bits: int = 64
) -> float:
    """
    Calculate similarity between two SimHash signatures based on Hamming distance.
    
    Parameters
    ----------
    hash1 : int
        First SimHash signature.
    hash2 : int
        Second SimHash signature.
    hash_bits : int, default=64
        Number of bits in the hash.
    
    Returns
    -------
    float
        Similarity between the two signatures, based on Hamming distance.
    """
    # Calculate Hamming distance (number of differing bits)
    xor = hash1 ^ hash2  # Bits that differ will be 1
    # Count the number of 1 bits
    hamming_distance = bin(xor).count('1')
    
    # Similarity is inverse of normalized Hamming distance
    return 1.0 - (hamming_distance / hash_bits)


def lsh_deduplication(
    texts: Union[List[str], pd.Series],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 16,
    keep: str = 'first',
    return_indices: bool = True
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Deduplicate texts using Locality-Sensitive Hashing (LSH) for scalability.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    threshold : float, default=0.8
        Similarity threshold (0-1). Higher values require more similarity.
    num_perm : int, default=128
        Number of permutations for MinHash.
    bands : int, default=16
        Number of bands for LSH. More bands = higher recall, lower precision.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    return_indices : bool, default=True
        If True, return indices of unique texts. If False, return a dictionary
        mapping LSH bucket keys to lists of indices.
    
    Returns
    -------
    Union[List[int], Dict[str, List[int]]]
        If return_indices is True, indices of unique texts after deduplication.
        If return_indices is False, dictionary mapping LSH bucket keys to document indices.
    """
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
        
    # Calculate number of rows per band
    rows = int(num_perm / bands)
    
    # Precalculate all MinHash signatures
    minhash_signatures = []
    for text in texts:
        if text is None or pd.isna(text) or not isinstance(text, str) or not text.strip():
            # Handle empty or None values
            minhash_signatures.append([0] * num_perm)
        else:
            minhash_signatures.append(create_minhash_signature(text, num_perm=num_perm))
    
    # Create LSH buckets
    buckets = defaultdict(list)
    
    for doc_idx, signature in enumerate(minhash_signatures):
        # Split signature into bands
        for band_idx in range(bands):
            # Take r values from the signature for this band
            start = band_idx * rows
            end = start + rows
            band = tuple(signature[start:end])
            
            # Create a bucket key from the band
            bucket_key = f"{band_idx}_{hash(band)}"
            
            # Add document to the bucket
            buckets[bucket_key].append(doc_idx)
    
    if not return_indices:
        return dict(buckets)
    
    # Find candidate pairs from the buckets
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            # Add all pairs in this bucket as candidates
            for i in range(len(bucket)):
                for j in range(i+1, len(bucket)):
                    candidate_pairs.add((min(bucket[i], bucket[j]), max(bucket[i], bucket[j])))
    
    # Verify candidate pairs
    similar_pairs = []
    for i, j in candidate_pairs:
        # Calculate actual similarity for the candidate pair
        similarity = calculate_minhash_similarity(minhash_signatures[i], minhash_signatures[j])
        if similarity >= threshold:
            similar_pairs.append((i, j))
    
    # Create clusters using a graph
    import networkx as nx
    G = nx.Graph()
    
    # Add all document indices as nodes
    for i in range(len(texts)):
        G.add_node(i)
    
    # Add edges for similar pairs
    for i, j in similar_pairs:
        G.add_edge(i, j)
    
    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))
    
    # Choose which documents to keep from each cluster
    keep_indices = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single document, keep it
            keep_indices.append(list(cluster)[0])
        else:
            # Multiple similar documents, choose based on keep strategy
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
    
    return sorted(keep_indices)
