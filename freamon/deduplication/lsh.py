"""
Locality-Sensitive Hashing (LSH) implementations for efficient similarity search.

LSH approximates similarity search by hashing similar items into the same buckets
with high probability, allowing for fast approximate nearest neighbor search.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Set, Union
from collections import defaultdict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_minhash_signatures(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    num_perm: int = 128,
    tokenizer: Callable = None
) -> Dict[int, Any]:
    """
    Create MinHash signatures for text data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    columns : List[str]
        Columns to use for creating signatures
    weights : Dict[str, float], default=None
        Dictionary mapping column names to their weights
    num_perm : int, default=128
        Number of permutations for MinHash
    tokenizer : Callable, default=None
        Function that splits text into tokens
        
    Returns
    -------
    Dict[int, Any]
        Dictionary mapping record indices to MinHash signatures
    """
    try:
        from datasketch import MinHash
    except ImportError:
        raise ImportError("The datasketch library is required for MinHash LSH. "
                         "Install it using 'pip install datasketch'.")
    
    if tokenizer is None:
        tokenizer = lambda x: x.lower().split()
    
    minhashes = {}
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        m = MinHash(num_perm=num_perm)
        
        # Process each column
        for col in columns:
            if col not in row:
                continue
                
            value = row[col]
            if pd.isna(value):
                continue
                
            if not isinstance(value, str):
                value = str(value)
            
            # Apply weight by repeating the column content
            weight = weights.get(col, 1.0) if weights else 1.0
            repeat_factor = max(1, int(10 * weight))
            
            # Update MinHash with tokens
            try:
                tokens = tokenizer(value)
                for _ in range(repeat_factor):
                    for token in tokens:
                        m.update(token.encode('utf8'))
            except Exception as e:
                logger.debug(f"Error tokenizing value '{value}' in column {col}: {e}")
        
        minhashes[idx] = m
    
    return minhashes


def find_similar_pairs_minhash_lsh(
    minhashes: Dict[int, Any],
    threshold: float = 0.7,
    num_perm: int = 128
) -> Set[Tuple[int, int]]:
    """
    Find similar pairs using MinHash LSH.
    
    Parameters
    ----------
    minhashes : Dict[int, Any]
        Dictionary mapping record indices to MinHash signatures
    threshold : float, default=0.7
        Similarity threshold for considering records similar
    num_perm : int, default=128
        Number of permutations used in MinHash
        
    Returns
    -------
    Set[Tuple[int, int]]
        Set of similar record index pairs
    """
    try:
        from datasketch import MinHashLSH
    except ImportError:
        raise ImportError("The datasketch library is required for MinHash LSH. "
                         "Install it using 'pip install datasketch'.")
    
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # Insert all minhashes
    for idx, minhash in minhashes.items():
        lsh.insert(str(idx), minhash)
    
    # Query for similar pairs
    similar_pairs = set()
    for idx, minhash in minhashes.items():
        similar_indices = lsh.query(minhash)
        for similar_idx in similar_indices:
            if similar_idx != str(idx):
                # Ensure consistent ordering of pairs
                pair = tuple(sorted([idx, int(similar_idx)]))
                similar_pairs.add(pair)
    
    return similar_pairs


def create_feature_vectors(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    normalize: bool = True
) -> Dict[int, np.ndarray]:
    """
    Create numerical feature vectors for random projection LSH.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    columns : List[str]
        Numerical columns to use for creating vectors
    weights : Dict[str, float], default=None
        Dictionary mapping column names to their weights
    normalize : bool, default=True
        Whether to normalize the vectors
        
    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping record indices to feature vectors
    """
    # Extract and normalize numerical columns
    vectors = {}
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        # Create feature vector
        vector = []
        for col in columns:
            if col not in row:
                vector.append(0.0)
                continue
                
            value = row[col]
            if pd.isna(value):
                vector.append(0.0)
                continue
                
            # Convert to numerical value
            if isinstance(value, (int, float)):
                num_value = float(value)
            elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                num_value = float(value)
            else:
                # For non-numerical values, use a hash
                num_value = hash(str(value)) % 1000 / 1000.0
            
            # Apply weight
            weight = weights.get(col, 1.0) if weights else 1.0
            vector.append(num_value * weight)
        
        # Convert to numpy array
        vectors[idx] = np.array(vector)
    
    # Normalize vectors if requested
    if normalize:
        for idx, vector in vectors.items():
            norm = np.linalg.norm(vector)
            if norm > 0:
                vectors[idx] = vector / norm
    
    return vectors


def find_similar_pairs_random_projection(
    vectors: Dict[int, np.ndarray],
    threshold: float = 0.7,
    num_projections: int = 50,
    bands: int = 10,
    rows: int = 5
) -> Set[Tuple[int, int]]:
    """
    Find similar pairs using random projection LSH.
    
    Parameters
    ----------
    vectors : Dict[int, np.ndarray]
        Dictionary mapping record indices to feature vectors
    threshold : float, default=0.7
        Similarity threshold for considering records similar
    num_projections : int, default=50
        Number of random projections to use
    bands : int, default=10
        Number of bands for LSH
    rows : int, default=5
        Number of rows per band
        
    Returns
    -------
    Set[Tuple[int, int]]
        Set of similar record index pairs
    """
    if not vectors:
        return set()
    
    # Get dimension of vectors
    dim = next(iter(vectors.values())).shape[0]
    
    # Generate random projection vectors
    projection_vectors = np.random.randn(num_projections, dim)
    # Normalize the projection vectors
    projection_vectors /= np.linalg.norm(projection_vectors, axis=1)[:, np.newaxis]
    
    # Calculate hash signatures
    signatures = {}
    for idx, vector in vectors.items():
        # Project vector onto random directions
        projections = projection_vectors @ vector
        # Convert to binary signature based on sign
        signature = (projections >= 0).astype(int)
        signatures[idx] = signature
    
    # Find candidate pairs using bucket hashing
    buckets = defaultdict(list)
    for idx, signature in signatures.items():
        # Use bands technique for LSH
        for band in range(bands):
            # Create a hash for this band
            band_signature = tuple(signature[band * rows: (band + 1) * rows])
            buckets[(band, band_signature)].append(idx)
    
    # Generate candidate pairs from buckets
    candidate_pairs = set()
    for indices in buckets.values():
        if len(indices) > 1:
            for i, idx1 in enumerate(indices):
                for idx2 in indices[i+1:]:
                    candidate_pairs.add(tuple(sorted([idx1, idx2])))
    
    # Verify candidates meet similarity threshold
    similar_pairs = set()
    for idx1, idx2 in candidate_pairs:
        # Calculate cosine similarity
        similarity = np.dot(vectors[idx1], vectors[idx2])
        if similarity >= threshold:
            similar_pairs.add((idx1, idx2))
    
    return similar_pairs


def analyze_column_types(
    df: Any,
    columns: List[str],
    sample_size: int = 100
) -> Dict[str, str]:
    """
    Analyze column types to determine appropriate LSH method.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    columns : List[str]
        Columns to analyze
    sample_size : int, default=100
        Number of records to sample for analysis
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping column names to their inferred types ('numeric' or 'text')
    """
    column_types = {}
    
    # Sample records for analysis
    if hasattr(df, 'sample') and callable(df.sample):
        try:
            sample_df = df.sample(min(sample_size, len(df)))
            sample_iter = sample_df.iterrows()
        except:
            # Fallback for non-pandas dataframes
            sample_iter = list(df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df))
            if len(sample_iter) > sample_size:
                import random
                sample_iter = random.sample(sample_iter, sample_size)
    else:
        sample_iter = list(zip(range(len(df)), df))
        if len(sample_iter) > sample_size:
            import random
            sample_iter = random.sample(sample_iter, sample_size)
    
    # Analyze each column
    for col in columns:
        numeric_count = 0
        text_count = 0
        total_count = 0
        
        for _, row in sample_iter:
            if col not in row:
                continue
                
            value = row[col]
            if pd.isna(value):
                continue
                
            total_count += 1
            
            if isinstance(value, (int, float)):
                numeric_count += 1
            elif isinstance(value, str):
                if value.replace('.', '', 1).isdigit():
                    numeric_count += 1
                else:
                    text_count += 1
            else:
                # Other types are treated as text
                text_count += 1
        
        # Determine column type based on majority
        if total_count > 0:
            if numeric_count >= text_count:
                column_types[col] = 'numeric'
            else:
                column_types[col] = 'text'
        else:
            # Default to text if no data
            column_types[col] = 'text'
    
    return column_types


def calculate_optimal_bands_rows(
    threshold: float,
    num_perm: int = 128
) -> Tuple[int, int]:
    """
    Calculate optimal number of bands and rows for LSH.
    
    Parameters
    ----------
    threshold : float
        Similarity threshold
    num_perm : int, default=128
        Number of permutations
        
    Returns
    -------
    Tuple[int, int]
        (bands, rows) optimized for the threshold
    """
    best_bands = 0
    best_rows = 0
    min_error = float('inf')
    
    # Try different combinations of bands and rows
    for rows in range(1, num_perm + 1):
        if num_perm % rows == 0:  # Ensure rows divides num_perm
            bands = num_perm // rows
            # Calculate threshold where a pair has 50% chance of being a candidate
            s = (1/bands) ** (1/rows)
            # Calculate error from target threshold
            error = abs(s - threshold)
            
            if error < min_error:
                min_error = error
                best_bands = bands
                best_rows = rows
    
    return best_bands, best_rows


def apply_lsh_strategy(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    lsh_method: str = 'auto',
    threshold: float = 0.7,
    num_perm: int = 128,
    num_bands: int = None,
    rows_per_band: int = None
) -> Set[Tuple[int, int]]:
    """
    Apply LSH strategy to find similar pairs.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    columns : List[str]
        Columns to use for LSH
    weights : Dict[str, float], default=None
        Dictionary mapping column names to their weights
    lsh_method : str, default='auto'
        LSH method to use: 'auto', 'minhash', 'random_projection', or 'hybrid'
    threshold : float, default=0.7
        Similarity threshold for considering records similar
    num_perm : int, default=128
        Number of permutations for MinHash LSH
    num_bands : int, default=None
        Number of bands for LSH. If None, calculated from threshold.
    rows_per_band : int, default=None
        Number of rows per band for LSH. If None, calculated from threshold.
        
    Returns
    -------
    Set[Tuple[int, int]]
        Set of similar record index pairs
    """
    # Calculate optimal bands and rows if not provided
    if num_bands is None or rows_per_band is None:
        bands, rows = calculate_optimal_bands_rows(threshold, num_perm)
        num_bands = num_bands or bands
        rows_per_band = rows_per_band or rows
    
    # Ensure the product equals num_perm
    if num_bands * rows_per_band != num_perm:
        # Adjust num_perm to be divisible by bands
        num_perm = num_bands * rows_per_band
        logger.info(f"Adjusted num_perm to {num_perm} to be divisible by {num_bands} bands with {rows_per_band} rows each")
    
    # Log the approximate threshold
    approx_threshold = (1 - (1 - (1/num_bands) ** rows_per_band) ** (1/rows_per_band))
    logger.info(f"LSH configuration: {num_bands} bands with {rows_per_band} rows each")
    logger.info(f"This approximates a similarity threshold of ~{approx_threshold:.2f}")
    
    # Determine which LSH method to use
    if lsh_method == 'auto':
        # Analyze column types to determine appropriate LSH method
        column_types = analyze_column_types(df, columns)
        
        # Count numeric and text columns
        numeric_cols = [col for col, type_ in column_types.items() if type_ == 'numeric']
        text_cols = [col for col, type_ in column_types.items() if type_ == 'text']
        
        if len(numeric_cols) > len(text_cols):
            lsh_method = 'random_projection'
            logger.info(f"Auto-selected 'random_projection' LSH (found {len(numeric_cols)} numeric columns, {len(text_cols)} text columns)")
        else:
            lsh_method = 'minhash'
            logger.info(f"Auto-selected 'minhash' LSH (found {len(text_cols)} text columns, {len(numeric_cols)} numeric columns)")
    
    # Apply the selected LSH method
    if lsh_method == 'minhash':
        logger.info(f"Using MinHash LSH with {num_perm} permutations")
        minhashes = create_minhash_signatures(
            df, columns, weights, num_perm
        )
        
        similar_pairs = find_similar_pairs_minhash_lsh(
            minhashes, threshold, num_perm
        )
    
    elif lsh_method == 'random_projection':
        logger.info(f"Using Random Projection LSH with {num_perm} projections")
        vectors = create_feature_vectors(
            df, columns, weights
        )
        
        similar_pairs = find_similar_pairs_random_projection(
            vectors, threshold, num_perm, num_bands, rows_per_band
        )
    
    elif lsh_method == 'hybrid':
        # Split columns by type and apply appropriate LSH method to each
        logger.info(f"Using Hybrid LSH approach (MinHash for text, Random Projection for numeric)")
        column_types = analyze_column_types(df, columns)
        
        # Split columns by type
        text_columns = [col for col, type_ in column_types.items() if type_ == 'text']
        numeric_columns = [col for col, type_ in column_types.items() if type_ == 'numeric']
        
        logger.info(f"  Text columns ({len(text_columns)}): {text_columns}")
        logger.info(f"  Numeric columns ({len(numeric_columns)}): {numeric_columns}")
        
        # Apply appropriate LSH to each column type
        text_pairs = set()
        if text_columns:
            text_weights = {col: weights.get(col, 1.0) for col in text_columns} if weights else None
            minhashes = create_minhash_signatures(df, text_columns, text_weights, num_perm)
            text_pairs = find_similar_pairs_minhash_lsh(minhashes, threshold, num_perm)
            logger.info(f"  MinHash LSH found {len(text_pairs)} similar pairs based on text columns")
        
        numeric_pairs = set()
        if numeric_columns:
            numeric_weights = {col: weights.get(col, 1.0) for col in numeric_columns} if weights else None
            vectors = create_feature_vectors(df, numeric_columns, numeric_weights)
            numeric_pairs = find_similar_pairs_random_projection(vectors, threshold, num_perm, num_bands, rows_per_band)
            logger.info(f"  Random Projection LSH found {len(numeric_pairs)} similar pairs based on numeric columns")
        
        # Combine results
        similar_pairs = text_pairs.union(numeric_pairs)
    
    else:
        raise ValueError(f"Unknown LSH method: {lsh_method}")
    
    logger.info(f"LSH identified {len(similar_pairs)} potential similar pairs")
    return similar_pairs