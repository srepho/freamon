"""Exact text deduplication methods using hashing and fingerprinting."""

from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
import hashlib
import re

import pandas as pd
import numpy as np


def hash_deduplication(
    texts: Union[List[str], pd.Series],
    hash_func: str = 'md5',
    case_sensitive: bool = False,
    keep: str = 'first',
    preprocess: bool = True,
    return_indices: bool = True
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Deduplicate texts using exact hashing.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    hash_func : str, default='md5'
        Hash function to use: 'md5', 'sha1', 'sha256'
    case_sensitive : bool, default=False
        Whether comparison should be case-sensitive.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    preprocess : bool, default=True
        Whether to preprocess texts (remove whitespace, punctuation) before hashing.
    return_indices : bool, default=True
        If True, return indices of unique texts. If False, return a dictionary
        mapping hash values to lists of indices.
        
    Returns
    -------
    Union[List[int], Dict[str, List[int]]]
        If return_indices is True, returns indices of unique texts.
        If return_indices is False, returns a dictionary mapping hash values to lists of indices.
    """
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Preprocess texts if requested
    processed_texts = []
    for text in texts:
        if text is None or pd.isna(text):
            processed_texts.append('')
            continue
            
        if not isinstance(text, str):
            text = str(text)
            
        if preprocess:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            
        if not case_sensitive:
            text = text.lower()
            
        processed_texts.append(text)
    
    # Choose hash function
    if hash_func == 'md5':
        hash_method = hashlib.md5
    elif hash_func == 'sha1':
        hash_method = hashlib.sha1
    elif hash_func == 'sha256':
        hash_method = hashlib.sha256
    else:
        raise ValueError(f"Unsupported hash function: {hash_func}. Use 'md5', 'sha1', or 'sha256'.")
    
    # Calculate hashes for all texts
    hashes = {}
    for i, text in enumerate(processed_texts):
        text_hash = hash_method(text.encode('utf-8')).hexdigest()
        
        if text_hash not in hashes:
            hashes[text_hash] = []
        hashes[text_hash].append(i)
    
    if not return_indices:
        return hashes
    
    # Determine which indices to keep based on the 'keep' strategy
    keep_indices = []
    
    for hash_val, indices in hashes.items():
        if len(indices) == 1:
            # Only one instance, keep it
            keep_indices.append(indices[0])
        else:
            # Multiple instances with same hash, choose based on keep strategy
            if keep == 'first':
                keep_indices.append(min(indices))
            elif keep == 'last':
                keep_indices.append(max(indices))
            elif keep == 'longest':
                keep_idx = max(indices, key=lambda i: len(texts[i]) if texts[i] is not None and not pd.isna(texts[i]) else 0)
                keep_indices.append(keep_idx)
            else:
                raise ValueError(f"Unknown keep strategy: {keep}. Use 'first', 'last', or 'longest'.")
    
    return sorted(keep_indices)


def ngram_fingerprint_deduplication(
    texts: Union[List[str], pd.Series],
    n: int = 3,
    threshold: float = 1.0,  # 1.0 means exact match of fingerprints
    case_sensitive: bool = False,
    keep: str = 'first',
    num_ngrams: int = 100,
    return_indices: bool = True
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Deduplicate texts using n-gram fingerprinting.
    
    Parameters
    ----------
    texts : Union[List[str], pd.Series]
        Collection of texts to deduplicate.
    n : int, default=3
        Size of n-grams.
    threshold : float, default=1.0
        Similarity threshold for fingerprints (1.0 means exact match).
    case_sensitive : bool, default=False
        Whether comparison should be case-sensitive.
    keep : str, default='first'
        Which instance to keep: 'first', 'last', or 'longest'.
    num_ngrams : int, default=100
        Number of n-grams to include in the fingerprint.
    return_indices : bool, default=True
        If True, return indices of unique texts. If False, return a dictionary
        mapping fingerprint values to lists of indices.
        
    Returns
    -------
    Union[List[int], Dict[str, List[int]]]
        If return_indices is True, returns indices of unique texts.
        If return_indices is False, returns a dictionary mapping fingerprints to lists of indices.
    """
    # Convert to list if pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Process texts and create fingerprints
    fingerprints = {}
    
    for i, text in enumerate(texts):
        if text is None or pd.isna(text):
            # Handle None/NaN values
            fingerprint = None
        else:
            if not isinstance(text, str):
                text = str(text)
                
            if not case_sensitive:
                text = text.lower()
                
            # Create character n-grams
            ngrams = [text[j:j+n] for j in range(len(text) - n + 1)]
            
            if not ngrams:
                fingerprint = None
            else:
                # Count n-gram frequencies
                ngram_counts = {}
                for ngram in ngrams:
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                
                # Select top n-grams by frequency for the fingerprint
                top_ngrams = sorted(ngram_counts.items(), key=lambda x: (-x[1], x[0]))[:num_ngrams]
                fingerprint = ' '.join(ngram for ngram, _ in top_ngrams)
        
        # Add to fingerprints dictionary
        if fingerprint not in fingerprints:
            fingerprints[fingerprint] = []
        fingerprints[fingerprint].append(i)
    
    if threshold < 1.0:
        # For fuzzy matching of fingerprints (not exact), we need to compare them
        # This is a more complex case and would require pairwise similarity comparison
        # We'll use a simple approach here, but this could be optimized further
        merged_fingerprints = {}
        processed_fp = set()
        
        # Compare each fingerprint with others
        for fp1, indices1 in fingerprints.items():
            if fp1 in processed_fp or fp1 is None:
                continue
                
            processed_fp.add(fp1)
            similar_indices = indices1.copy()
            
            for fp2, indices2 in fingerprints.items():
                if fp2 in processed_fp or fp1 == fp2 or fp2 is None:
                    continue
                    
                # Calculate similarity (simple Jaccard for now)
                if fp1 and fp2:  # Ensure non-None
                    set1 = set(fp1.split())
                    set2 = set(fp2.split())
                    
                    if set1 and set2:  # Ensure non-empty
                        similarity = len(set1 & set2) / len(set1 | set2)
                        
                        if similarity >= threshold:
                            similar_indices.extend(indices2)
                            processed_fp.add(fp2)
            
            # Create a merged group
            merged_key = f"group_{len(merged_fingerprints)}"
            merged_fingerprints[merged_key] = similar_indices
            
        fingerprints = merged_fingerprints
    
    if not return_indices:
        return fingerprints
    
    # Determine which indices to keep based on the 'keep' strategy
    keep_indices = []
    
    for _, indices in fingerprints.items():
        if len(indices) == 1:
            # Only one instance, keep it
            keep_indices.append(indices[0])
        else:
            # Multiple instances with similar fingerprints, choose based on keep strategy
            if keep == 'first':
                keep_indices.append(min(indices))
            elif keep == 'last':
                keep_indices.append(max(indices))
            elif keep == 'longest':
                keep_idx = max(indices, key=lambda i: len(texts[i]) if texts[i] is not None and not pd.isna(texts[i]) else 0)
                keep_indices.append(keep_idx)
            else:
                raise ValueError(f"Unknown keep strategy: {keep}. Use 'first', 'last', or 'longest'.")
    
    return sorted(keep_indices)
