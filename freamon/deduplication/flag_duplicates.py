"""
Functions for flagging duplicate records in a dataframe.

This module provides functionality to identify and flag duplicates in a dataframe
rather than removing them, useful for:
1. Maintaining the original dataset size
2. Making informed decisions about which duplicates to remove
3. Downstream filtering based on duplicate status
4. Tracking duplicates for reporting and analysis
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
import pandas as pd
import numpy as np
import networkx as nx
import time
import psutil

from freamon.utils.dataframe_utils import check_dataframe_type, convert_dataframe
import logging

# Optional imports - will be imported dynamically as needed
# from freamon.deduplication.blocking import apply_blocking_strategy
# from freamon.deduplication.lsh import apply_lsh_strategy
from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import find_similar_texts
from freamon.deduplication.lsh_deduplication import lsh_deduplication
from freamon.deduplication.polars_supervised_deduplication import PolarsSupervisedDeduplicationModel

# Import automatic parameter selection
try:
    from freamon.deduplication.auto_params import apply_auto_params
    AUTO_PARAMS_AVAILABLE = True
except ImportError:
    AUTO_PARAMS_AVAILABLE = False

# Import evaluation utilities
try:
    from freamon.deduplication.evaluation import (
        calculate_deduplication_metrics,
        plot_confusion_matrix,
        evaluate_threshold_sensitivity,
        generate_evaluation_report,
        flag_and_evaluate
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

# Import the progress tracking utilities
try:
    from freamon.utils.progress_tracker import ProgressTracker, BlockProgressTracker, create_deduplication_tracker
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False


def flag_exact_duplicates(
    df: Any,
    subset: Optional[Union[str, List[str]]] = None,
    keep: str = 'first',
    flag_column: str = 'is_duplicate',
    inplace: bool = False,
    indicator_column: Optional[str] = None,
) -> Any:
    """
    Flag exact duplicate rows in a dataframe without removing them.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    subset : Optional[Union[str, List[str]]], default=None
        Column(s) to consider for identifying duplicates. If None, uses all columns.
    keep : str, default='first'
        Which occurrence to mark as non-duplicate. Options: 'first', 'last', False.
        If 'first', only the first occurrence of each duplicate will be marked as False.
        If 'last', only the last occurrence of each duplicate will be marked as False.
        If False, all duplicate rows will be marked as True.
    flag_column : str, default='is_duplicate'
        Name of the column to add for flagging duplicates.
    inplace : bool, default=False
        If True, modify the dataframe in-place.
    indicator_column : Optional[str], default=None
        If provided, also add a column with this name that indicates the duplicate group ID.
        
    Returns
    -------
    Any
        DataFrame with duplicates flagged.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 1, 3, 1],
    ...     'B': ['x', 'y', 'x', 'z', 'x'],
    ... })
    >>> result = flag_exact_duplicates(df, subset=['A', 'B'])
    >>> result
       A  B  is_duplicate
    0  1  x         False
    1  2  y         False
    2  1  x          True
    3  3  z         False
    4  1  x          True
    """
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    # Set subset to list if it's a string
    if isinstance(subset, str):
        subset = [subset]
    
    # Create duplicate flag
    duplicated = pandas_df.duplicated(subset=subset, keep=keep)
    
    # Add flag column
    if inplace:
        pandas_df[flag_column] = duplicated
    else:
        pandas_df = pandas_df.assign(**{flag_column: duplicated})
    
    # Add indicator column if requested
    if indicator_column:
        if subset is None:
            # Use all columns to identify duplicate groups
            group_cols = pandas_df.columns.tolist()
            if flag_column in group_cols:
                group_cols.remove(flag_column)
        else:
            group_cols = subset
            
        # Create a group ID for each unique combination of values
        # Only assign group IDs to rows that have duplicates
        group_data = pandas_df[group_cols].copy()
        group_data['__temp_row_idx'] = range(len(group_data))
        
        # Find duplicate groups
        dup_mask = pandas_df.duplicated(subset=subset, keep=False)
        groups = group_data[dup_mask].groupby(group_cols)
        
        # Create dictionary to store group IDs
        group_ids = {}
        for i, (_, group) in enumerate(groups):
            for idx in group['__temp_row_idx']:
                group_ids[idx] = i + 1  # Start from 1 for better readability
        
        # Create the indicator column
        indicator_values = pd.Series([group_ids.get(i, 0) for i in range(len(pandas_df))])
        
        if inplace:
            pandas_df[indicator_column] = indicator_values
        else:
            pandas_df = pandas_df.assign(**{indicator_column: indicator_values})
    
    # Convert back to original type if needed
    if df_type != 'pandas' and not inplace:
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def flag_text_duplicates(
    df: Any,
    text_column: str,
    method: str = 'hash',
    threshold: float = 0.9,
    ngram_size: int = 3,
    hash_func: str = 'md5',
    flag_column: str = 'is_text_duplicate',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    preprocess: bool = True,
    chunk_size: Optional[int] = None,
    use_polars: bool = False,
) -> Any:
    """
    Flag duplicate text content in a dataframe without removing the rows.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    text_column : str
        Column containing text data to check for duplicates.
    method : str, default='hash'
        Method to use for finding duplicates:
        - 'hash': Exact hash-based matching (fastest)
        - 'ngram': N-gram fingerprint matching (faster than fuzzy, less exact than hash)
        - 'fuzzy': Fuzzy text matching with similarity threshold (slowest but most flexible)
        - 'lsh': Locality-sensitive hashing for approximate matching (good for large datasets)
    threshold : float, default=0.9
        Similarity threshold for fuzzy and lsh methods. Ignored for hash and ngram.
    ngram_size : int, default=3
        Size of n-grams for ngram and lsh methods.
    hash_func : str, default='md5'
        Hash function to use for hash method. Options: 'md5', 'sha1', 'sha256'.
    flag_column : str, default='is_text_duplicate'
        Name of the column to add for flagging duplicates.
    inplace : bool, default=False
        If True, modify the dataframe in-place.
    group_column : Optional[str], default=None
        If provided, add a column with this name containing the duplicate group ID.
    similarity_column : Optional[str], default=None
        If provided, add a column with this name containing the similarity score.
        Only used for fuzzy and lsh methods.
    preprocess : bool, default=True
        Whether to preprocess text data before comparison.
    chunk_size : Optional[int], default=None
        Size of chunks to process for streaming large text collections.
        If provided, uses streaming processing to handle very large datasets.
    use_polars : bool, default=False
        Whether to use polars for faster processing. Requires polars to be installed.
        Particularly effective for large text datasets.
        
    Returns
    -------
    Any
        DataFrame with text duplicates flagged.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'text': [
    ...         "This is a sample text.",
    ...         "This is another example.",
    ...         "This is a sample text!",
    ...         "Something completely different.",
    ...         "This is a sample text."
    ...     ]
    ... })
    >>> result = flag_text_duplicates(df, text_column='text', method='fuzzy', 
    ...                               threshold=0.9, group_column='dup_group')
    
    # For large datasets, use chunking with LSH
    >>> large_df = pd.concat([df] * 1000)  # Create a larger dataset
    >>> result = flag_text_duplicates(large_df, text_column='text', method='lsh',
    ...                               threshold=0.8, chunk_size=1000, use_polars=True)
    """
    # If using polars optimization and polars is available
    if use_polars:
        try:
            import polars as pl
            if method == 'lsh' and chunk_size is not None:
                # For large datasets with LSH, use the streaming implementation with polars
                return _flag_text_duplicates_streaming_polars(
                    df=df,
                    text_column=text_column,
                    threshold=threshold,
                    ngram_size=ngram_size,
                    flag_column=flag_column,
                    inplace=inplace,
                    group_column=group_column,
                    similarity_column=similarity_column,
                    preprocess=preprocess,
                    chunk_size=chunk_size,
                )
            elif method in ['hash', 'ngram', 'lsh']:
                # For other methods with polars, use the optimized implementation
                return _flag_text_duplicates_polars(
                    df=df,
                    text_column=text_column,
                    method=method,
                    threshold=threshold,
                    ngram_size=ngram_size,
                    hash_func=hash_func,
                    flag_column=flag_column,
                    inplace=inplace,
                    group_column=group_column,
                    similarity_column=similarity_column,
                    preprocess=preprocess,
                )
        except ImportError:
            # Fall back to standard or streaming implementation if polars is not available
            pass
    
    # If chunking is requested for large datasets
    if chunk_size is not None and method == 'lsh':
        # For large datasets with LSH, use the streaming implementation
        return _flag_text_duplicates_streaming(
            df=df,
            text_column=text_column,
            threshold=threshold,
            ngram_size=ngram_size,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            preprocess=preprocess,
            chunk_size=chunk_size,
        )
    
    # For standard (non-streaming) cases
    return _flag_text_duplicates_standard(
        df=df,
        text_column=text_column,
        method=method,
        threshold=threshold,
        ngram_size=ngram_size,
        hash_func=hash_func,
        flag_column=flag_column,
        inplace=inplace,
        group_column=group_column,
        similarity_column=similarity_column,
        preprocess=preprocess,
    )


def _flag_text_duplicates_standard(
    df: Any,
    text_column: str,
    method: str = 'hash',
    threshold: float = 0.9,
    ngram_size: int = 3,
    hash_func: str = 'md5',
    flag_column: str = 'is_text_duplicate',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    preprocess: bool = True,
) -> Any:
    """
    Standard implementation of text duplicate flagging.
    
    See flag_text_duplicates for parameter descriptions.
    """
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    # Extract text series
    text_series = pandas_df[text_column]
    
    # Initialize columns
    duplicate_flags = pd.Series(False, index=pandas_df.index)
    group_ids = pd.Series(0, index=pandas_df.index)
    similarity_scores = pd.Series(0.0, index=pandas_df.index)
    
    # Use the appropriate method to find duplicates
    if method == 'hash':
        # Use exact hash-based deduplication
        unique_indices = hash_deduplication(
            text_series, 
            hash_func=hash_func,
            preprocess=preprocess,
            return_indices=True
        )
        
        # All rows not in unique_indices are duplicates
        duplicate_flags = ~pandas_df.index.isin(unique_indices)
        
        # Create duplicate groups
        if group_column:
            # Create a graph to connect duplicate rows
            G = nx.Graph()
            
            # Create a dictionary mapping hashes to row indices
            hash_to_indices = {}
            for i, text in enumerate(text_series):
                if pd.isna(text):
                    continue
                    
                # Preprocess if required
                processed_text = text
                if preprocess:
                    from freamon.utils.text_utils import TextProcessor
                    processor = TextProcessor()
                    processed_text = processor.preprocess_text(
                        text, 
                        lowercase=True,
                        remove_punctuation=True
                    )
                
                # Calculate hash
                import hashlib
                if hash_func == 'md5':
                    hash_obj = hashlib.md5(processed_text.encode())
                elif hash_func == 'sha1':
                    hash_obj = hashlib.sha1(processed_text.encode())
                elif hash_func == 'sha256':
                    hash_obj = hashlib.sha256(processed_text.encode())
                else:
                    hash_obj = hashlib.md5(processed_text.encode())
                
                text_hash = hash_obj.hexdigest()
                
                if text_hash in hash_to_indices:
                    hash_to_indices[text_hash].append(i)
                else:
                    hash_to_indices[text_hash] = [i]
            
            # Add edges for duplicate rows
            for indices in hash_to_indices.values():
                if len(indices) > 1:
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            G.add_edge(indices[i], indices[j])
            
            # Find connected components (clusters of duplicates)
            clusters = list(nx.connected_components(G))
            
            # Assign group IDs
            for group_id, cluster in enumerate(clusters, 1):
                if len(cluster) > 1:  # Only assign IDs to actual duplicate groups
                    for idx in cluster:
                        group_ids.iloc[idx] = group_id
        
    elif method == 'ngram':
        # Use n-gram fingerprint deduplication
        from freamon.deduplication.exact_deduplication import ngram_fingerprint_deduplication
        
        unique_indices = ngram_fingerprint_deduplication(
            text_series,
            n=ngram_size,
            return_indices=True
        )
        
        # All rows not in unique_indices are duplicates
        duplicate_flags = ~pandas_df.index.isin(unique_indices)
        
        # Create duplicate groups
        if group_column:
            # Create a graph to connect duplicate rows
            G = nx.Graph()
            
            # Create a dictionary mapping fingerprints to row indices
            fingerprint_to_indices = {}
            for i, text in enumerate(text_series):
                if pd.isna(text):
                    continue
                    
                # Preprocess if required
                processed_text = text
                if preprocess:
                    from freamon.utils.text_utils import TextProcessor
                    processor = TextProcessor()
                    processed_text = processor.preprocess_text(
                        text, 
                        lowercase=True,
                        remove_punctuation=True
                    )
                
                # Create n-gram fingerprint
                from freamon.deduplication.fingerprinting import create_fingerprint
                fingerprint = create_fingerprint(processed_text, n=ngram_size)
                
                if fingerprint in fingerprint_to_indices:
                    fingerprint_to_indices[fingerprint].append(i)
                else:
                    fingerprint_to_indices[fingerprint] = [i]
            
            # Add edges for duplicate rows
            for indices in fingerprint_to_indices.values():
                if len(indices) > 1:
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            G.add_edge(indices[i], indices[j])
            
            # Find connected components (clusters of duplicates)
            clusters = list(nx.connected_components(G))
            
            # Assign group IDs
            for group_id, cluster in enumerate(clusters, 1):
                if len(cluster) > 1:  # Only assign IDs to actual duplicate groups
                    for idx in cluster:
                        group_ids.iloc[idx] = group_id
        
    elif method == 'fuzzy':
        # Use fuzzy text matching
        similarity_dict = find_similar_texts(
            text_series,
            threshold=threshold,
            method='cosine',  # Use cosine similarity
            preprocess=preprocess,
            return_scores=True
        )
        
        # Use a graph to find connected components (duplicate groups)
        G = nx.Graph()
        
        # Add all indices as nodes
        for i in range(len(text_series)):
            G.add_node(i)
        
        # Add edges for similar texts with similarity as edge weight
        for i, similar_texts in similarity_dict.items():
            for j_info in similar_texts:
                j, similarity = j_info
                G.add_edge(i, j, weight=similarity)
                
                # Store similarity scores if requested
                if similarity_column:
                    # Store the highest similarity score for each row
                    if similarity > similarity_scores.iloc[i]:
                        similarity_scores.iloc[i] = similarity
                    if similarity > similarity_scores.iloc[j]:
                        similarity_scores.iloc[j] = similarity
        
        # Find connected components (clusters of duplicates)
        clusters = list(nx.connected_components(G))
        
        # Mark duplicates and assign group IDs
        for group_id, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:  # Only process actual duplicate groups
                for idx in cluster:
                    duplicate_flags.iloc[idx] = True
                    if group_column:
                        group_ids.iloc[idx] = group_id
        
        # For first occurrence in each group, set duplicate flag to False
        for cluster in clusters:
            if len(cluster) > 1:
                min_idx = min(cluster)
                duplicate_flags.iloc[min_idx] = False
    
    elif method == 'lsh':
        # Use LSH for approximate duplicate detection
        # Run LSH with return_similarity_dict
        result = lsh_deduplication(
            text_series,
            threshold=threshold,
            num_minhash_permutations=128,
            num_bands=16,
            preprocess=preprocess,
            return_indices=True,
            return_similarity_dict=True
        )
        
        # Extract unique indices and similarity dict
        unique_indices, similarity_dict = result
        
        # Create clusters from similarity dict
        G = nx.Graph()
        
        # Add all nodes
        for i in range(len(text_series)):
            G.add_node(i)
            
        # Add edges from similarity dict
        for i, similar_indices in similarity_dict.items():
            for j in similar_indices:
                G.add_edge(i, j)
        
        # Find connected components as clusters
        clusters = list(nx.connected_components(G))
        
        # Mark duplicates and assign group IDs
        for group_id, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:  # Only process actual duplicate groups
                for idx in cluster:
                    duplicate_flags.iloc[idx] = True
                    if group_column:
                        group_ids.iloc[idx] = group_id
        
        # For first occurrence in each group, set duplicate flag to False
        for cluster in clusters:
            if len(cluster) > 1:
                min_idx = min(cluster)
                duplicate_flags.iloc[min_idx] = False
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hash', 'ngram', 'fuzzy', or 'lsh'.")
    
    # Add flag column
    if inplace:
        pandas_df[flag_column] = duplicate_flags
        if group_column:
            pandas_df[group_column] = group_ids
        if similarity_column and method in ['fuzzy', 'lsh']:
            pandas_df[similarity_column] = similarity_scores
    else:
        pandas_df = pandas_df.assign(**{flag_column: duplicate_flags})
        if group_column:
            pandas_df = pandas_df.assign(**{group_column: group_ids})
        if similarity_column and method in ['fuzzy', 'lsh']:
            pandas_df = pandas_df.assign(**{similarity_column: similarity_scores})
    
    # Convert back to original type if needed
    if df_type != 'pandas' and not inplace:
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def _flag_text_duplicates_streaming(
    df: Any,
    text_column: str,
    threshold: float = 0.9,
    ngram_size: int = 3,
    flag_column: str = 'is_text_duplicate',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    preprocess: bool = True,
    chunk_size: int = 1000,
) -> Any:
    """
    Streaming implementation of text duplicate flagging for large datasets.
    
    This implementation uses LSH to process the dataframe in chunks, allowing for
    efficient memory usage with large text collections.
    
    See flag_text_duplicates for parameter descriptions.
    """
    import math
    
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    n_rows = len(pandas_df)
    
    # If the dataset is smaller than chunk_size, use the standard approach
    if n_rows <= chunk_size:
        return _flag_text_duplicates_standard(
            df=pandas_df,
            text_column=text_column,
            method='lsh',
            threshold=threshold,
            ngram_size=ngram_size,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            preprocess=preprocess,
        )
    
    # Extract text series
    text_series = pandas_df[text_column]
    
    # Initialize columns
    duplicate_flags = pd.Series(False, index=pandas_df.index)
    group_ids = pd.Series(0, index=pandas_df.index)
    similarity_scores = pd.Series(0.0, index=pandas_df.index)
    
    # Create a graph to track similar texts across chunks
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(n_rows):
        G.add_node(i)
    
    # Calculate number of chunks
    n_chunks = math.ceil(n_rows / chunk_size)
    
    # Process each chunk separately first
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_rows)
        
        # Get text for this chunk
        chunk_texts = text_series.iloc[start_idx:end_idx]
        
        # Create index mapping for this chunk
        chunk_indices = {i: start_idx + i for i in range(len(chunk_texts))}
        
        # Process this chunk with LSH
        result = lsh_deduplication(
            chunk_texts,
            threshold=threshold,
            num_minhash_permutations=128,
            num_bands=16,
            preprocess=preprocess,
            return_indices=True,
            return_similarity_dict=True
        )
        
        # Extract similarity dict from the result
        _, similarity_dict = result
        
        # Add edges from within-chunk similarity dict
        for i, similar_indices in similarity_dict.items():
            for j in similar_indices:
                # Map to original dataframe indices
                global_i = chunk_indices[i]
                global_j = chunk_indices[j]
                
                # Add edge to the global graph
                G.add_edge(global_i, global_j)
                
                # Store similarity scores if requested
                if similarity_column:
                    # For simplicity, use 1.0 as similarity score for identified pairs
                    # Could be enhanced to store actual scores if needed
                    similarity_scores.iloc[global_i] = 1.0
                    similarity_scores.iloc[global_j] = 1.0
    
    # Now compare texts between chunks using signature matrices
    from freamon.deduplication.fingerprinting import create_minhash_signature
    from datasketch import MinHash
    
    # Create minhash signatures for all texts
    signatures = []
    for i, text in enumerate(text_series):
        if pd.isna(text):
            signatures.append(None)
            continue
            
        # Preprocess if required
        processed_text = text
        if preprocess:
            from freamon.utils.text_utils import TextProcessor
            processor = TextProcessor()
            processed_text = processor.preprocess_text(
                text, 
                lowercase=True,
                remove_punctuation=True
            )
        
        # Create minhash signature
        signature = create_minhash_signature(processed_text, num_perm=128, k_shingles=ngram_size)
        signatures.append(signature)
    
    # Compare signatures between chunks
    for i in range(n_chunks):
        for j in range(i+1, n_chunks):
            # Get indices for the chunks
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, n_rows)
            
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n_rows)
            
            # Compare each text in chunk i with each text in chunk j
            for idx_i in range(start_i, end_i):
                if signatures[idx_i] is None:
                    continue
                    
                for idx_j in range(start_j, end_j):
                    if signatures[idx_j] is None:
                        continue
                        
                    # Calculate Jaccard similarity between signatures
                    minhash_i = signatures[idx_i]
                    minhash_j = signatures[idx_j]
                    
                    jaccard = minhash_i.jaccard(minhash_j)
                    
                    # Add edge if similarity is above threshold
                    if jaccard >= threshold:
                        G.add_edge(idx_i, idx_j)
                        
                        # Store similarity scores if requested
                        if similarity_column:
                            if jaccard > similarity_scores.iloc[idx_i]:
                                similarity_scores.iloc[idx_i] = jaccard
                            if jaccard > similarity_scores.iloc[idx_j]:
                                similarity_scores.iloc[idx_j] = jaccard
    
    # Find connected components (clusters of similar texts)
    clusters = list(nx.connected_components(G))
    
    # Mark similar texts and assign group IDs
    for group_id, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:  # Only process actual similar groups
            for idx in cluster:
                duplicate_flags.iloc[idx] = True
                if group_column:
                    group_ids.iloc[idx] = group_id
    
    # For first occurrence in each group, set duplicate flag to False
    for cluster in clusters:
        if len(cluster) > 1:
            min_idx = min(cluster)
            duplicate_flags.iloc[min_idx] = False
    
    # Add columns to dataframe
    if inplace:
        pandas_df[flag_column] = duplicate_flags
        if group_column:
            pandas_df[group_column] = group_ids
        if similarity_column:
            pandas_df[similarity_column] = similarity_scores
    else:
        pandas_df = pandas_df.assign(**{flag_column: duplicate_flags})
        if group_column:
            pandas_df = pandas_df.assign(**{group_column: group_ids})
        if similarity_column:
            pandas_df = pandas_df.assign(**{similarity_column: similarity_scores})
    
    # Convert back to original type if needed
    if df_type != 'pandas' and not inplace:
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def _flag_text_duplicates_polars(
    df: Any,
    text_column: str,
    method: str = 'hash',
    threshold: float = 0.9,
    ngram_size: int = 3,
    hash_func: str = 'md5',
    flag_column: str = 'is_text_duplicate',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    preprocess: bool = True,
) -> Any:
    """
    Polars-optimized implementation of text duplicate flagging.
    
    This implementation uses Polars for faster processing, especially for large
    text collections with many string operations.
    
    See flag_text_duplicates for parameter descriptions.
    """
    import polars as pl
    import hashlib
    
    # Check dataframe type and convert if needed
    df_type = check_dataframe_type(df)
    
    if df_type == 'polars':
        polars_df = df.clone() if not inplace else df
    else:
        # Convert to polars
        if df_type == 'pandas':
            polars_df = pl.from_pandas(df)
        else:
            # Convert to pandas first, then to polars
            pandas_df = convert_dataframe(df, 'pandas')
            polars_df = pl.from_pandas(pandas_df)
    
    # Extract text as a Python list
    text_list = polars_df[text_column].to_list()
    
    # Initialize columns
    n_rows = len(text_list)
    duplicate_flags = [False] * n_rows
    group_ids = [0] * n_rows
    similarity_scores = [0.0] * n_rows
    
    # Use the appropriate method for deduplication
    if method == 'hash':
        # Use Polars expressions for faster hashing
        if preprocess:
            from freamon.utils.text_utils import TextProcessor
            processor = TextProcessor()
            
            # Create a preprocessed column
            processed_texts = []
            for text in text_list:
                if text is None:
                    processed_texts.append(None)
                else:
                    processed_texts.append(processor.preprocess_text(
                        text, 
                        lowercase=True,
                        remove_punctuation=True
                    ))
        else:
            processed_texts = text_list
        
        # Create a dictionary to track hash-based duplicates
        hash_to_indices = {}
        
        # Calculate hash for each text
        for i, text in enumerate(processed_texts):
            if text is None:
                continue
                
            # Calculate hash
            if hash_func == 'md5':
                hash_obj = hashlib.md5(text.encode())
            elif hash_func == 'sha1':
                hash_obj = hashlib.sha1(text.encode())
            elif hash_func == 'sha256':
                hash_obj = hashlib.sha256(text.encode())
            else:
                hash_obj = hashlib.md5(text.encode())
            
            text_hash = hash_obj.hexdigest()
            
            if text_hash in hash_to_indices:
                hash_to_indices[text_hash].append(i)
            else:
                hash_to_indices[text_hash] = [i]
        
        # Create a graph to connect duplicate rows
        G = nx.Graph()
        
        # Add all indices as nodes
        for i in range(n_rows):
            G.add_node(i)
        
        # Add edges for duplicate texts
        for indices in hash_to_indices.values():
            if len(indices) > 1:
                # First occurrence is not a duplicate
                duplicate_flags[indices[0]] = False
                
                # Other occurrences are duplicates
                for idx in indices[1:]:
                    duplicate_flags[idx] = True
                
                # Add edges to the graph for group tracking
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        G.add_edge(indices[i], indices[j])
        
        # If group tracking is requested, assign group IDs
        if group_column:
            # Find connected components (clusters of duplicates)
            clusters = list(nx.connected_components(G))
            
            # Assign group IDs
            for group_id, cluster in enumerate(clusters, 1):
                if len(cluster) > 1:  # Only assign IDs to actual duplicate groups
                    for idx in cluster:
                        group_ids[idx] = group_id
    
    elif method == 'ngram':
        from freamon.deduplication.fingerprinting import create_fingerprint
        
        # Create a dictionary to track fingerprint-based duplicates
        fingerprint_to_indices = {}
        
        # Calculate fingerprint for each text
        for i, text in enumerate(text_list):
            if text is None:
                continue
                
            # Preprocess if required
            if preprocess:
                from freamon.utils.text_utils import TextProcessor
                processor = TextProcessor()
                processed_text = processor.preprocess_text(
                    text, 
                    lowercase=True,
                    remove_punctuation=True
                )
            else:
                processed_text = text
            
            # Create n-gram fingerprint
            fingerprint = create_fingerprint(processed_text, n=ngram_size)
            
            if fingerprint in fingerprint_to_indices:
                fingerprint_to_indices[fingerprint].append(i)
            else:
                fingerprint_to_indices[fingerprint] = [i]
        
        # Create a graph to connect duplicate rows
        G = nx.Graph()
        
        # Add all indices as nodes
        for i in range(n_rows):
            G.add_node(i)
        
        # Add edges for duplicate texts
        for indices in fingerprint_to_indices.values():
            if len(indices) > 1:
                # First occurrence is not a duplicate
                duplicate_flags[indices[0]] = False
                
                # Other occurrences are duplicates
                for idx in indices[1:]:
                    duplicate_flags[idx] = True
                
                # Add edges to the graph for group tracking
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        G.add_edge(indices[i], indices[j])
        
        # If group tracking is requested, assign group IDs
        if group_column:
            # Find connected components (clusters of duplicates)
            clusters = list(nx.connected_components(G))
            
            # Assign group IDs
            for group_id, cluster in enumerate(clusters, 1):
                if len(cluster) > 1:  # Only assign IDs to actual duplicate groups
                    for idx in cluster:
                        group_ids[idx] = group_id
    
    elif method == 'lsh':
        # Convert to pandas series for lsh_deduplication
        import pandas as pd
        text_series = pd.Series(text_list)
        
        # Use LSH for approximate duplicate detection
        result = lsh_deduplication(
            text_series,
            threshold=threshold,
            num_minhash_permutations=128,
            num_bands=16,
            preprocess=preprocess,
            return_indices=True,
            return_similarity_dict=True
        )
        
        # Extract unique indices and similarity dict
        unique_indices, similarity_dict = result
        
        # Create a graph to connect similar texts
        G = nx.Graph()
        
        # Add all indices as nodes
        for i in range(n_rows):
            G.add_node(i)
        
        # Add edges from similarity dict
        for i, similar_indices in similarity_dict.items():
            for j in similar_indices:
                G.add_edge(i, j)
                
                # Store similarity scores if requested
                if similarity_column:
                    # For simplicity, use 1.0 as similarity score
                    # Could be enhanced to store actual scores
                    similarity_scores[i] = 1.0
                    similarity_scores[j] = 1.0
        
        # Find connected components (clusters of similar texts)
        clusters = list(nx.connected_components(G))
        
        # Mark similar texts and assign group IDs
        for group_id, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:  # Only process actual similar groups
                for idx in cluster:
                    duplicate_flags[idx] = True
                    if group_column:
                        group_ids[idx] = group_id
        
        # For first occurrence in each group, set duplicate flag to False
        for cluster in clusters:
            if len(cluster) > 1:
                min_idx = min(cluster)
                duplicate_flags[min_idx] = False
    
    else:
        # Fall back to standard implementation for fuzzy method
        pandas_df = polars_df.to_pandas()
        result_df = _flag_text_duplicates_standard(
            df=pandas_df,
            text_column=text_column,
            method=method,
            threshold=threshold,
            ngram_size=ngram_size,
            hash_func=hash_func,
            flag_column=flag_column,
            inplace=False,
            group_column=group_column,
            similarity_column=similarity_column,
            preprocess=preprocess,
        )
        
        # Convert back to polars and return
        result_polars = pl.from_pandas(result_df)
        if df_type != 'polars' and not inplace:
            return convert_dataframe(result_polars, df_type)
        elif inplace:
            # Copy columns back to the original dataframe
            polars_df = polars_df.with_columns([
                pl.Series(name=flag_column, values=result_df[flag_column].values)
            ])
            
            if group_column:
                polars_df = polars_df.with_columns([
                    pl.Series(name=group_column, values=result_df[group_column].values)
                ])
                
            if similarity_column and method in ['fuzzy', 'lsh']:
                polars_df = polars_df.with_columns([
                    pl.Series(name=similarity_column, values=result_df[similarity_column].values)
                ])
            
            return polars_df
        else:
            return result_polars
    
    # Add columns to the polars dataframe
    if inplace:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=duplicate_flags)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids)
            ])
            
        if similarity_column and method in ['fuzzy', 'lsh']:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores)
            ])
    else:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=duplicate_flags)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids)
            ])
            
        if similarity_column and method in ['fuzzy', 'lsh']:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores)
            ])
    
    # Convert back to original type if needed
    if df_type != 'polars' and not inplace:
        return convert_dataframe(polars_df, df_type)
    else:
        return polars_df


def _flag_text_duplicates_streaming_polars(
    df: Any,
    text_column: str,
    threshold: float = 0.9,
    ngram_size: int = 3,
    flag_column: str = 'is_text_duplicate',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    preprocess: bool = True,
    chunk_size: int = 1000,
) -> Any:
    """
    Polars-optimized streaming implementation of text duplicate flagging.
    
    This implementation combines the efficiency of Polars with chunked processing
    for handling extremely large text collections.
    
    See flag_text_duplicates for parameter descriptions.
    """
    import math
    import polars as pl
    
    # Check dataframe type and convert if needed
    df_type = check_dataframe_type(df)
    
    if df_type == 'polars':
        polars_df = df.clone() if not inplace else df
    else:
        # Convert to polars
        if df_type == 'pandas':
            polars_df = pl.from_pandas(df)
        else:
            # Convert to pandas first, then to polars
            pandas_df = convert_dataframe(df, 'pandas')
            polars_df = pl.from_pandas(pandas_df)
    
    n_rows = len(polars_df)
    
    # If the dataset is smaller than chunk_size, use the standard approach
    if n_rows <= chunk_size:
        return _flag_text_duplicates_polars(
            df=polars_df,
            text_column=text_column,
            method='lsh',
            threshold=threshold,
            ngram_size=ngram_size,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            preprocess=preprocess,
        )
    
    # Initialize columns
    duplicate_flags = [False] * n_rows
    group_ids = [0] * n_rows
    similarity_scores = [0.0] * n_rows
    
    # Create a graph to track similar texts across chunks
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(n_rows):
        G.add_node(i)
    
    # Calculate number of chunks
    n_chunks = math.ceil(n_rows / chunk_size)
    
    # Process each chunk separately first
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_rows)
        
        # Create a slice for this chunk
        chunk_df = polars_df.slice(start_idx, end_idx - start_idx)
        
        # Use Polars-optimized LSH for this chunk
        chunk_result = _flag_text_duplicates_polars(
            df=chunk_df,
            text_column=text_column,
            method='lsh',
            threshold=threshold,
            ngram_size=ngram_size,
            flag_column=flag_column,
            group_column=group_column,
            similarity_column=similarity_column,
            preprocess=preprocess,
            inplace=False,
        )
        
        # Extract duplicate flags and group IDs
        chunk_flags = chunk_result[flag_column].to_list()
        
        # Add edges for duplicate pairs within this chunk
        for i in range(len(chunk_flags)):
            if chunk_flags[i]:
                # Find which group this belongs to
                group = 0
                if group_column:
                    group = chunk_result[group_column].to_list()[i]
                
                # Find other members of this group
                if group > 0:
                    group_members = []
                    for j, g in enumerate(chunk_result[group_column].to_list()):
                        if g == group:
                            group_members.append(j)
                    
                    # Add edges between all group members
                    for idx1 in group_members:
                        for idx2 in group_members:
                            if idx1 != idx2:
                                # Map to original dataframe indices
                                global_idx1 = start_idx + idx1
                                global_idx2 = start_idx + idx2
                                G.add_edge(global_idx1, global_idx2)
    
    # Now compare texts between chunks using LSH signatures
    from freamon.deduplication.fingerprinting import create_minhash_signature
    from datasketch import MinHash
    
    # Extract all texts
    text_list = polars_df[text_column].to_list()
    
    # Create minhash signatures for all texts
    signatures = []
    for i, text in enumerate(text_list):
        if text is None:
            signatures.append(None)
            continue
            
        # Preprocess if required
        processed_text = text
        if preprocess:
            from freamon.utils.text_utils import TextProcessor
            processor = TextProcessor()
            processed_text = processor.preprocess_text(
                text, 
                lowercase=True,
                remove_punctuation=True
            )
        
        # Create minhash signature
        signature = create_minhash_signature(processed_text, num_perm=128, k_shingles=ngram_size)
        signatures.append(signature)
    
    # Compare signatures between chunks
    for i in range(n_chunks):
        for j in range(i+1, n_chunks):
            # Get indices for the chunks
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, n_rows)
            
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n_rows)
            
            # Compare signatures efficiently using batches
            batch_size = 100  # Process in smaller batches for memory efficiency
            
            for batch_i_start in range(start_i, end_i, batch_size):
                batch_i_end = min(batch_i_start + batch_size, end_i)
                
                for batch_j_start in range(start_j, end_j, batch_size):
                    batch_j_end = min(batch_j_start + batch_size, end_j)
                    
                    # Compare each text in batch i with each text in batch j
                    for idx_i in range(batch_i_start, batch_i_end):
                        if signatures[idx_i] is None:
                            continue
                            
                        for idx_j in range(batch_j_start, batch_j_end):
                            if signatures[idx_j] is None:
                                continue
                                
                            # Calculate Jaccard similarity between signatures
                            minhash_i = signatures[idx_i]
                            minhash_j = signatures[idx_j]
                            
                            jaccard = minhash_i.jaccard(minhash_j)
                            
                            # Add edge if similarity is above threshold
                            if jaccard >= threshold:
                                G.add_edge(idx_i, idx_j)
                                
                                # Store similarity scores if requested
                                if similarity_column:
                                    if jaccard > similarity_scores[idx_i]:
                                        similarity_scores[idx_i] = jaccard
                                    if jaccard > similarity_scores[idx_j]:
                                        similarity_scores[idx_j] = jaccard
    
    # Find connected components (clusters of similar texts)
    clusters = list(nx.connected_components(G))
    
    # Mark similar texts and assign group IDs
    for group_id, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:  # Only process actual similar groups
            for idx in cluster:
                duplicate_flags[idx] = True
                if group_column:
                    group_ids[idx] = group_id
    
    # For first occurrence in each group, set duplicate flag to False
    for cluster in clusters:
        if len(cluster) > 1:
            min_idx = min(cluster)
            duplicate_flags[min_idx] = False
    
    # Add columns to the polars dataframe
    if inplace:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=duplicate_flags)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids)
            ])
            
        if similarity_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores)
            ])
    else:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=duplicate_flags)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids)
            ])
            
        if similarity_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores)
            ])
    
    # Convert back to original type if needed
    if df_type != 'polars' and not inplace:
        return convert_dataframe(polars_df, df_type)
    else:
        return polars_df


def flag_similar_records(
    df: Any,
    columns: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_similar',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    chunk_size: Optional[int] = None,
    n_jobs: int = 1,
    use_polars: bool = False,
    
    # New parameters for text handling
    auto_detect_columns: bool = False,
    text_columns: Optional[List[str]] = None,
    text_method: str = 'fuzzy',
    text_threshold: Optional[float] = None,
    min_text_length: int = 20,
    text_weight_boost: float = 1.5,
    
    # New parameters for blocking
    blocking_columns: Optional[List[str]] = None,
    blocking_method: str = 'exact',
    blocking_rules: Optional[Dict[str, Callable]] = None,
    max_block_size: Optional[int] = None,
    
    # New parameters for LSH
    use_lsh: bool = False,
    lsh_method: str = 'auto',
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    num_bands: Optional[int] = None,
    rows_per_band: Optional[int] = None,
    
    # Progress tracking parameters
    show_progress: bool = False,
    jupyter_mode: Optional[bool] = None,
    
    # Auto-mode parameters
    auto_mode: bool = False,
    memory_limit_gb: Optional[float] = None,
    
    # Evaluation parameters
    known_duplicate_column: Optional[str] = None,
    evaluate: bool = False,
    generate_report: bool = False,
    report_format: str = 'text',
    include_plots: bool = True,
    
    # Legacy parameters
    add_similarity_score: bool = False,
    add_group_id: bool = False,
    group_id_column: Optional[str] = None,
    duplicate_flag_column: Optional[str] = None,
) -> Any:
    """
    Flag similar records based on multiple columns with customizable weights.
    Automatically detects and handles text columns with specialized processing.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    columns : Optional[List[str]], default=None
        Columns to consider when calculating similarity. If None and auto_detect_columns=True,
        all columns will be analyzed and appropriate ones selected automatically.
    weights : Optional[Dict[str, float]], default=None
        Dictionary mapping column names to their weights in similarity calculation.
        If None, all columns are weighted equally.
    threshold : float, default=0.8
        Similarity threshold above which records are considered similar.
    method : str, default='composite'
        Method to use for calculating similarity:
        - 'composite': Weighted combination of column-by-column similarities
        - 'exact_subset': Match if a subset of columns match exactly
        - 'fuzzy_subset': Match if a subset of columns match with high similarity
    flag_column : str, default='is_similar'
        Name of the column to add for flagging similar records.
    inplace : bool, default=False
        If True, modify the dataframe in-place.
    group_column : Optional[str], default=None
        If provided, add a column with this name containing the similarity group ID.
    similarity_column : Optional[str], default=None
        If provided, add a column with this name containing the similarity score.
    max_comparisons : Optional[int], default=None
        Maximum number of pairwise comparisons to perform. If None, compare all pairs.
        Can speed up processing for large datasets.
    chunk_size : Optional[int], default=None
        Size of chunks to process at a time for large datasets. If None, process all at once.
        Enables processing of datasets too large for all-pairs comparison.
    n_jobs : int, default=1
        Number of parallel jobs to run for similarity comparison.
        Only used when chunk_size is specified.
    use_polars : bool, default=False
        Whether to use polars for faster processing. Requires polars to be installed.
    auto_detect_columns : bool, default=False
        Whether to automatically detect which columns to use for similarity calculation.
        If True and columns=None, will select appropriate columns from the dataframe.
    text_columns : Optional[List[str]], default=None
        Columns that should be processed as text fields with specialized text similarity methods.
        If None, text columns are auto-detected when auto_detect_columns=True.
    text_method : str, default='fuzzy'
        Method to use for text similarity calculation:
        - 'fuzzy': Levenshtein-based fuzzy matching (good for short-medium text)
        - 'tfidf': TF-IDF vectorization with cosine similarity (good for longer text)
        - 'ngram': N-gram Jaccard similarity (faster than fuzzy for longer text)
        - 'lsh': Locality-sensitive hashing for text (fast for very large text fields)
    text_threshold : Optional[float], default=None
        Similarity threshold for text columns. If None, uses the main threshold value.
    min_text_length : int, default=20
        Minimum average length of text content to be considered a long-form text field
        requiring specialized text similarity methods.
    text_weight_boost : float, default=1.5
        Factor to boost weights for long text fields when auto-weighting is used.
    blocking_columns : Optional[List[str]], default=None
        Columns to use for blocking. Records in different blocks won't be compared.
        Significantly speeds up processing for large datasets.
    blocking_method : str, default='exact'
        Method to use for blocking:
        - 'exact': Exact match on blocking columns
        - 'phonetic': Phonetic matching on blocking columns
        - 'ngram': N-gram based blocking
        - 'rule': Custom blocking rules
    blocking_rules : Optional[Dict[str, Callable]], default=None
        Dictionary mapping rule names to functions that generate blocking keys.
        Required when blocking_method='rule'.
    max_block_size : Optional[int], default=None
        Maximum number of records in a block before sampling or splitting.
    use_lsh : bool, default=False
        Whether to use Locality Sensitive Hashing for faster similarity search.
        Greatly speeds up processing for large datasets with a small accuracy tradeoff.
    lsh_method : str, default='auto'
        LSH method to use:
        - 'auto': Automatically select based on column types
        - 'minhash': Use MinHash LSH for text data
        - 'random_projection': Use Random Projection LSH for numerical data
        - 'hybrid': Use combination of methods for mixed data types
    lsh_threshold : Optional[float], default=None
        LSH similarity threshold. If None, uses threshold * 0.9 as a pre-filter.
    num_perm : int, default=128
        Number of permutations for MinHash LSH.
    num_bands : Optional[int], default=None
        Number of bands for LSH. If None, calculated automatically from threshold.
    rows_per_band : Optional[int], default=None
        Number of rows per band for LSH. If None, calculated automatically from threshold.
    show_progress : bool, default=False
        Whether to show progress tracking during processing. Creates interactive
        progress bars with ETA and memory usage information.
    jupyter_mode : Optional[bool], default=None
        Whether to use Jupyter notebook widgets for progress display.
        If None, auto-detects the environment.
    auto_mode : bool, default=False
        Whether to automatically select optimal parameters based on dataset characteristics.
        If True, intelligently determines the best deduplication approach.
    memory_limit_gb : Optional[float], default=None
        Maximum memory to use for deduplication in gigabytes.
        If None, uses 75% of available system memory.
    known_duplicate_column : Optional[str], default=None
        Column name containing ground truth duplicate flags (0/1 or True/False).
        Used for evaluating deduplication performance.
    evaluate : bool, default=False
        Whether to evaluate deduplication performance against known_duplicate_column.
        If True, returns metrics along with the results.
    generate_report : bool, default=False
        Whether to generate an evaluation report.
    report_format : str, default='text'
        Format for evaluation report: 'text', 'html', or 'markdown'.
    include_plots : bool, default=True
        Whether to include visualizations in the evaluation report.
    add_similarity_score : bool, default=False
        Legacy parameter, use similarity_column instead.
        If True, add a column with similarity scores.
    add_group_id : bool, default=False
        Legacy parameter, use group_column instead.
        If True, add a column with group IDs.
    group_id_column : Optional[str], default=None
        Legacy parameter, use group_column instead.
        Name of the column to add for group IDs.
    duplicate_flag_column : Optional[str], default=None
        Legacy parameter, use flag_column instead.
        Name of the column to add for flagging duplicates.
        
    Returns
    -------
    Any
        DataFrame with similar records flagged.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'name': ['John Smith', 'Jane Doe', 'Jon Smith', 'Mary Jones', 'John Smith'],
    ...     'email': ['john@example.com', 'jane@example.com', 'jon@example.com', 
    ...               'mary@example.com', 'johnsmith@example.com'],
    ...     'phone': ['555-1234', '555-5678', '555-9012', '555-3456', '555-1234'],
    ...     'state': ['CA', 'NY', 'CA', 'TX', 'CA'],
    ...     'zipcode': ['90210', '10001', '90210', '75001', '90210']
    ... })
    >>> # Weight name and email higher than phone
    >>> weights = {'name': 0.4, 'email': 0.4, 'phone': 0.2}
    >>> result = flag_similar_records(df, columns=['name', 'email', 'phone'], 
    ...                               weights=weights, threshold=0.7)
    
    # For large datasets, use chunking
    >>> large_df = pd.concat([df] * 1000)  # Create a larger dataset
    >>> result = flag_similar_records(large_df, columns=['name', 'email', 'phone'], 
    ...                               weights=weights, chunk_size=1000, n_jobs=4)
    
    # For very large datasets (50k+ rows), use aggressive chunking and sampling
    >>> very_large_df = pd.concat([df] * 10000)  # Create a 50k row dataset
    >>> result = flag_similar_records(very_large_df, columns=['name', 'email', 'phone'], 
    ...                               weights=weights, chunk_size=500, max_comparisons=1000000)
    
    # Using blocking for more efficient processing
    >>> result = flag_similar_records(df, columns=['name', 'email', 'phone'],
    ...                              blocking_columns=['state', 'zipcode'],
    ...                              threshold=0.8)
    
    # Using LSH for fast approximate matching
    >>> result = flag_similar_records(df, columns=['name', 'email', 'phone'],
    ...                              use_lsh=True, lsh_method='minhash',
    ...                              threshold=0.8)
    
    # Combined approach for very large datasets
    >>> result = flag_similar_records(very_large_df, columns=['name', 'email', 'phone'],
    ...                              blocking_columns=['state'],
    ...                              use_lsh=True, 
    ...                              threshold=0.8)
    """
    # Handle legacy parameter names
    if duplicate_flag_column is not None and flag_column == 'is_similar':
        flag_column = duplicate_flag_column
    
    if group_id_column is not None and group_column is None:
        group_column = group_id_column
        
    if add_group_id and group_column is None:
        group_column = 'group_id'
        
    if add_similarity_score and similarity_column is None:
        similarity_column = 'similarity_score'
    
    # Check if we need to auto-detect columns
    if auto_detect_columns and columns is None:
        # Convert to pandas for detection
        df_type = check_dataframe_type(df)
        if df_type != 'pandas':
            pandas_df = convert_dataframe(df, 'pandas')
        else:
            pandas_df = df.copy() if not inplace else df
            
        # Auto-detect appropriate columns
        detected_columns = []
        detected_text_columns = []
        
        # Exclude certain column types
        exclude_types = ['datetime', 'bool', 'binary']
        
        try:
            # Try to use DataTypeDetector for more advanced detection
            from freamon.utils.datatype_detector import DataTypeDetector
            
            # Create detector and detect types
            detector = DataTypeDetector(pandas_df)
            type_results = detector.detect_all_types()
            
            # Process detected types
            for col, info in type_results.items():
                logical_type = info.get('logical_type', '')
                
                # Skip excluded types
                if logical_type in exclude_types:
                    continue
                    
                # Skip primary key-like columns (high cardinality, unique identifiers)
                if logical_type == 'id' or (logical_type == 'categorical' and 
                                          pandas_df[col].nunique() > 0.95 * len(pandas_df)):
                    continue
                
                # Identify potential text fields vs. normal columns
                if logical_type == 'string':
                    # Sample the column to check text length
                    sample = pandas_df[col].dropna().sample(min(100, len(pandas_df[col].dropna())), 
                                                          random_state=42)
                    
                    # Calculate average string length of non-empty strings
                    non_empty = [s for s in sample if isinstance(s, str) and len(s) > 0]
                    if non_empty:
                        avg_length = sum(len(s) for s in non_empty) / len(non_empty)
                        
                        # Classify based on length
                        if avg_length >= min_text_length:
                            detected_text_columns.append(col)
                        else:
                            detected_columns.append(col)
                else:
                    # Add other types (numeric, categorical) to regular columns
                    detected_columns.append(col)
                    
        except (ImportError, Exception) as e:
            print(f"Warning: Could not use DataTypeDetector: {str(e)}. Using simple detection.")
            
            # Fallback to simple detection
            for col in pandas_df.columns:
                # Skip columns with majority missing values
                if pandas_df[col].isna().mean() > 0.7:
                    continue
                
                # Analyze column type
                if pd.api.types.is_numeric_dtype(pandas_df[col].dtype):
                    detected_columns.append(col)
                elif pd.api.types.is_string_dtype(pandas_df[col].dtype) or pandas_df[col].dtype == 'object':
                    # Check if it's a text field
                    sample = pandas_df[col].dropna().sample(min(100, len(pandas_df[col].dropna())), 
                                                          random_state=42)
                    
                    # Calculate average string length for strings
                    string_vals = [s for s in sample if isinstance(s, str)]
                    if string_vals:
                        avg_length = sum(len(s) for s in string_vals) / len(string_vals)
                        
                        # Classify based on length
                        if avg_length >= min_text_length:
                            detected_text_columns.append(col)
                        else:
                            detected_columns.append(col)
        
        # Use detected columns
        if detected_columns or detected_text_columns:
            columns = detected_columns + detected_text_columns
            
            # Set text_columns if auto-detected
            if detected_text_columns:
                text_columns = detected_text_columns
                
            print(f"Auto-detected {len(detected_columns)} regular columns and {len(detected_text_columns)} text columns for similarity comparison")
        else:
            # If no columns detected, use all columns except datetime and bool
            columns = [col for col in pandas_df.columns 
                      if not pd.api.types.is_datetime64_dtype(pandas_df[col].dtype) 
                      and not pd.api.types.is_bool_dtype(pandas_df[col].dtype)]
            print(f"Using {len(columns)} columns for similarity comparison")
            
    # Set text threshold to main threshold if not specified
    if text_threshold is None:
        text_threshold = threshold
    
    # Generate weights if not provided
    if weights is None and columns:
        weights = {}
        
        # Use equal weights as a base
        equal_weight = 1.0 / len(columns)
        
        for col in columns:
            if text_columns and col in text_columns:
                # Apply text weight boost for text columns
                weights[col] = equal_weight * text_weight_boost
            else:
                weights[col] = equal_weight
                
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {col: w / weight_sum for col, w in weights.items()}
    
    # Apply auto parameter selection if requested
    if auto_mode:
        if not AUTO_PARAMS_AVAILABLE:
            print("Warning: Auto mode requested but auto_params module not available. Using manual parameters.")
        else:
            # Gather current parameters
            params = {
                'weights': weights,
                'threshold': threshold,
                'method': method,
                'flag_column': flag_column,
                'inplace': inplace,
                'group_column': group_column,
                'similarity_column': similarity_column,
                'max_comparisons': max_comparisons,
                'chunk_size': chunk_size,
                'n_jobs': n_jobs,
                'use_polars': use_polars,
                'blocking_columns': blocking_columns,
                'blocking_method': blocking_method,
                'blocking_rules': blocking_rules,
                'max_block_size': max_block_size,
                'use_lsh': use_lsh,
                'lsh_method': lsh_method,
                'lsh_threshold': lsh_threshold,
                'num_perm': num_perm,
                'num_bands': num_bands,
                'rows_per_band': rows_per_band,
                'memory_limit_gb': memory_limit_gb,
                'show_progress': show_progress,
                'jupyter_mode': jupyter_mode,
                'text_columns': text_columns,
                'text_method': text_method,
                'text_threshold': text_threshold
            }
            
            # Apply auto parameter selection
            auto_params = apply_auto_params(df, columns, **params)
            
            # Update parameters with auto-selected values
            if 'chunk_size' in auto_params and auto_params['chunk_size'] is not None:
                chunk_size = auto_params['chunk_size']
            if 'max_comparisons' in auto_params and auto_params['max_comparisons'] is not None:
                max_comparisons = auto_params['max_comparisons']
            if 'n_jobs' in auto_params:
                n_jobs = auto_params['n_jobs']
            if 'use_lsh' in auto_params:
                use_lsh = auto_params['use_lsh']
            if use_lsh and 'lsh_method' in auto_params:
                lsh_method = auto_params['lsh_method']
            if use_lsh and 'num_perm' in auto_params:
                num_perm = auto_params['num_perm']
            if use_lsh and 'lsh_threshold' in auto_params:
                lsh_threshold = auto_params['lsh_threshold']
            if use_lsh and 'num_bands' in auto_params:
                num_bands = auto_params['num_bands']
            if use_lsh and 'rows_per_band' in auto_params:
                rows_per_band = auto_params['rows_per_band']
            if 'use_blocking' in auto_params and auto_params['use_blocking']:
                if 'blocking_columns' in auto_params and auto_params['blocking_columns']:
                    blocking_columns = auto_params['blocking_columns']
                if 'blocking_method' in auto_params:
                    blocking_method = auto_params['blocking_method']
                if 'max_block_size' in auto_params:
                    max_block_size = auto_params['max_block_size']
    
    # For large datasets, set a reasonable maximum number of comparisons if not provided
    if max_comparisons is None and len(df) > 10000 and not auto_mode:
        # Use quadratic scaling but with a cap
        max_comparisons = min(10000000, len(df) * 100)
        print(f"Auto-set max_comparisons to {max_comparisons} for large dataset")
        
    # Check if progress tracking is available and requested
    use_progress_tracking = show_progress and PROGRESS_TRACKING_AVAILABLE
    
    # If progress tracking is requested but not available, warn the user
    if show_progress and not PROGRESS_TRACKING_AVAILABLE:
        print("Warning: Progress tracking requested but not available. Install required dependencies.")
    
    # If evaluation is requested, make sure we have the necessary column and dependency
    if evaluate or generate_report:
        if not known_duplicate_column:
            raise ValueError("known_duplicate_column must be provided for evaluation")
        if not EVALUATION_AVAILABLE:
            raise ImportError("Evaluation module not available. Make sure freamon.deduplication.evaluation is installed.")
        if known_duplicate_column not in df.columns:
            raise ValueError(f"Known duplicate column '{known_duplicate_column}' not found in dataframe")
    
    # If we're requesting evaluation, use the flag_and_evaluate function
    if evaluate or generate_report:
        return flag_and_evaluate(
            df=df,
            columns=columns,
            known_duplicate_column=known_duplicate_column,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column=flag_column,
            generate_report=generate_report,
            report_format=report_format,
            include_plots=include_plots,
            auto_mode=auto_mode,
            show_progress=show_progress,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            max_comparisons=max_comparisons,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            use_polars=use_polars,
            blocking_columns=blocking_columns,
            blocking_method=blocking_method,
            blocking_rules=blocking_rules,
            max_block_size=max_block_size,
            use_lsh=use_lsh,
            lsh_method=lsh_method,
            lsh_threshold=lsh_threshold,
            num_perm=num_perm,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            jupyter_mode=jupyter_mode,
            memory_limit_gb=memory_limit_gb
        )
    
    # Standard deduplication path (no evaluation)
    # If the dataset is small or chunking is not required, use the standard approach
    if chunk_size is None or len(df) <= chunk_size:
        return _flag_similar_records_standard(
            df=df,
            columns=columns,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            max_comparisons=max_comparisons,
            use_polars=use_polars,
            text_columns=text_columns,
            text_method=text_method,
            text_threshold=text_threshold,
            blocking_columns=blocking_columns,
            blocking_method=blocking_method,
            blocking_rules=blocking_rules,
            max_block_size=max_block_size,
            use_lsh=use_lsh,
            lsh_method=lsh_method,
            lsh_threshold=lsh_threshold,
            num_perm=num_perm,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            show_progress=use_progress_tracking,
            jupyter_mode=jupyter_mode,
        )
    else:
        # For larger datasets, use the chunked approach
        return _flag_similar_records_chunked(
            df=df,
            columns=columns,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            chunk_size=chunk_size,
            n_jobs=n_jobs,
            use_polars=use_polars,
            show_progress=use_progress_tracking,
            jupyter_mode=jupyter_mode,
            text_columns=text_columns,
            text_method=text_method,
            text_threshold=text_threshold,
            max_comparisons=max_comparisons,
        )


def _flag_similar_records_standard(
    df: Any,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_similar',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    use_polars: bool = False,
    text_columns: Optional[List[str]] = None,
    text_method: str = 'fuzzy',
    text_threshold: Optional[float] = None,
    blocking_columns: Optional[List[str]] = None,
    blocking_method: str = 'exact',
    blocking_rules: Optional[Dict[str, Callable]] = None,
    max_block_size: Optional[int] = None,
    use_lsh: bool = False,
    lsh_method: str = 'auto',
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    num_bands: Optional[int] = None,
    rows_per_band: Optional[int] = None,
    show_progress: bool = False,
    jupyter_mode: Optional[bool] = None,
) -> Any:
    """
    Standard implementation of similar records flagging (non-chunked).
    
    See flag_similar_records for parameter descriptions.
    """
    # Check if we should use Polars for faster processing
    if use_polars:
        try:
            import polars as pl
            return _flag_similar_records_polars(
                df=df,
                columns=columns,
                weights=weights,
                threshold=threshold,
                method=method,
                flag_column=flag_column,
                inplace=inplace,
                group_column=group_column,
                similarity_column=similarity_column,
                max_comparisons=max_comparisons,
            )
        except ImportError:
            # Fall back to pandas if polars is not available
            pass
    
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    # Validate columns
    for col in columns:
        if col not in pandas_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Initialize weights if not provided
    if weights is None or not weights:
        weights = {col: 1.0 / len(columns) for col in columns}
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {col: weight / total_weight for col, weight in weights.items()}
        else:
            # If all weights are zero or dictionary is empty, use equal weights
            weights = {col: 1.0 / len(columns) for col in columns}
        
        # Check if all columns have weights
        for col in columns:
            if col not in weights:
                weights[col] = 0.0
    
    # Initialize columns
    n_rows = len(pandas_df)
    
    # For large datasets, automatically switch to chunked processing
    # Skip auto-switching if we're using blocking or LSH, which are more efficient
    if n_rows > 20000 and max_comparisons is None and not (blocking_columns or use_lsh):
        print(f"Auto-switching to chunked processing for large dataset ({n_rows} rows)")
        return _flag_similar_records_chunked(
            df=pandas_df,
            columns=columns,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            chunk_size=2000,  # Reasonable default chunk size
            n_jobs=1,
            use_polars=use_polars,
            text_columns=text_columns,
            text_method=text_method,
            text_threshold=text_threshold,
            max_comparisons=max_comparisons,
        )
        
    # Use a sparse graph representation to save memory
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(n_rows):
        G.add_node(i)
    
    # Calculate total possible comparisons
    total_comparisons = (n_rows * (n_rows - 1)) // 2
    print(f"Total possible comparisons: {total_comparisons:,}")
    
    # Apply optimization strategies for pair selection
    comparison_pairs = None
    
    # Step 1: Apply blocking if specified
    if blocking_columns or (blocking_method == 'rule' and blocking_rules):
        try:
            from freamon.deduplication.blocking import apply_blocking_strategy
            
            print(f"Applying {blocking_method} blocking with columns: {blocking_columns}")
            
            blocking_pairs = apply_blocking_strategy(
                df=pandas_df,
                strategy=blocking_method,
                blocking_columns=blocking_columns,
                blocking_rules=blocking_rules,
                max_block_size=max_block_size,
                max_comparisons=max_comparisons
            )
            
            comparison_pairs = blocking_pairs
            percentage = (len(comparison_pairs) / total_comparisons) * 100
            print(f"Blocking created {len(comparison_pairs):,} pairs to compare ({percentage:.2f}% of all possible pairs)")
        except ImportError as e:
            print(f"Warning: Blocking module not available. Error: {e}")
            print("Proceeding without blocking optimization.")
    
    # Step 2: Apply LSH if specified
    if use_lsh:
        try:
            from freamon.deduplication.lsh import apply_lsh_strategy
            
            # Set LSH threshold if not specified
            if lsh_threshold is None:
                lsh_threshold = threshold * 0.9  # Slightly lower to catch more candidates
            
            print(f"Applying {lsh_method} LSH with threshold: {lsh_threshold}")
            
            lsh_pairs = apply_lsh_strategy(
                df=pandas_df,
                columns=columns,
                weights=weights,
                lsh_method=lsh_method,
                threshold=lsh_threshold,
                num_perm=num_perm,
                num_bands=num_bands,
                rows_per_band=rows_per_band
            )
            
            # If we also have blocking pairs, take the intersection
            if comparison_pairs is not None:
                comparison_pairs_set = set(tuple(sorted(pair)) for pair in comparison_pairs)
                lsh_pairs_set = set(lsh_pairs)  # Already sorted in LSH
                combined_pairs = lsh_pairs_set.intersection(comparison_pairs_set)
                comparison_pairs = list(combined_pairs)
                percentage = (len(comparison_pairs) / total_comparisons) * 100
                print(f"Combined blocking and LSH: {len(comparison_pairs):,} pairs to compare ({percentage:.2f}% of all possible pairs)")
            else:
                comparison_pairs = list(lsh_pairs)
                percentage = (len(comparison_pairs) / total_comparisons) * 100
                print(f"LSH identified {len(comparison_pairs):,} potential similar pairs ({percentage:.2f}% of all possible pairs)")
        except ImportError as e:
            print(f"Warning: LSH module not available. Error: {e}")
            print("Proceeding without LSH optimization.")
    
    # Step 3: Apply max_comparisons limit if needed
    comparisons_to_do = total_comparisons
    if max_comparisons is not None:
        comparisons_to_do = min(total_comparisons, max_comparisons)
        
        if comparison_pairs is not None and len(comparison_pairs) > max_comparisons:
            # Sample from optimized pairs
            import random
            random.shuffle(comparison_pairs)
            comparison_pairs = comparison_pairs[:max_comparisons]
            print(f"Limited to {max_comparisons:,} pairs due to max_comparisons setting")
    
    # Final fallback to default generator if no specialized pairs were created
    def generate_pairs():
        # If we've already determined pairs through blocking or LSH, use those
        if comparison_pairs is not None:
            for pair in comparison_pairs:
                yield pair
        # Otherwise use random sampling or all pairs
        elif comparisons_to_do < total_comparisons:
            # Randomly sample pairs for efficiency
            import random
            indices = list(range(n_rows))
            count = 0
            while count < comparisons_to_do:
                i = random.randint(0, n_rows - 2)
                j = random.randint(i + 1, n_rows - 1)
                yield i, j
                count += 1
        else:
            # Compare all pairs
            for i in range(n_rows):
                for j in range(i + 1, n_rows):
                    yield i, j
                    
    # Print info about comparisons for large datasets
    if comparison_pairs is None and n_rows > 5000:
        percentage = (comparisons_to_do / total_comparisons) * 100
        print(f"Making {comparisons_to_do:,} of {total_comparisons:,} possible comparisons ({percentage:.1f}%)")
    
    # Process pairs in batches to reduce memory pressure
    batch_size = min(10000, comparisons_to_do)
    edges_to_add = []
    similarity_dict = {}  # Maps (i, j) tuples to similarity scores
    
    # Initialize progress tracking if requested
    progress_tracker = None
    if show_progress and PROGRESS_TRACKING_AVAILABLE:
        progress_tracker = create_deduplication_tracker(
            total_comparisons=comparisons_to_do,
            description="Deduplication progress",
            jupyter_mode=jupyter_mode
        )
        progress_tracker.start()
    
    # Process batches
    pairs_processed = 0
    next_report = min(100000, comparisons_to_do // 10)  # Report progress at 10% intervals
    
    # Get subset of df with only required columns for efficiency
    col_subset = pandas_df[columns]
    
    print(f"Processing record similarities in batches of {batch_size}...")
    
    for i, j in generate_pairs():
        # Get rows by index without loading full dataframe
        row1 = col_subset.iloc[i]
        row2 = col_subset.iloc[j]
        
        # Calculate similarity with text handling
        similarity = _calculate_similarity(
            row1, row2, columns, weights, method,
            text_columns=text_columns, 
            text_method=text_method,
            text_threshold=text_threshold
        )
        
        # Add edge if similarity is above threshold
        if similarity >= threshold:
            edges_to_add.append((i, j))
            similarity_dict[(i, j)] = similarity
        
        pairs_processed += 1
        
        # Add edges in batches to conserve memory
        if len(edges_to_add) >= batch_size:
            G.add_edges_from(edges_to_add)
            edges_to_add = []
        
        # Update progress tracker if active
        if progress_tracker:
            # Update progress less frequently to reduce overhead
            if pairs_processed % 100 == 0:
                progress_tracker.update(100)
        # Legacy progress reporting for non-interactive mode
        elif pairs_processed >= next_report:
            progress = (pairs_processed / comparisons_to_do) * 100
            print(f"Processed {pairs_processed:,} pairs ({progress:.1f}%)...")
            next_report += min(100000, comparisons_to_do // 10)
    
    # Add any remaining edges
    if edges_to_add:
        G.add_edges_from(edges_to_add)
    
    # Handle any remaining progress updates
    if progress_tracker:
        # Update remaining progress to 100%
        remaining = comparisons_to_do - pairs_processed
        if remaining > 0:
            progress_tracker.update(remaining)
        progress_tracker.finish()
    
    print(f"Finding connected components among {G.number_of_edges()} similar pairs...")
    
    # Find connected components (clusters of similar records)
    clusters = list(nx.connected_components(G))
    
    print(f"Found {len(clusters)} clusters of similar records")
    
    # Initialize arrays at the last moment to save memory
    similarity_flags = np.zeros(n_rows, dtype=bool)
    if group_column:
        group_ids = np.zeros(n_rows, dtype=int)
    if similarity_column:
        similarity_scores = np.zeros(n_rows, dtype=float)
    
    # Mark similar records and assign group IDs
    for group_id, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:  # Only process actual similar groups
            for idx in cluster:
                similarity_flags[idx] = True
                if group_column:
                    group_ids[idx] = group_id
    
    # For first occurrence in each group, set similarity flag to False
    for cluster in clusters:
        if len(cluster) > 1:
            min_idx = min(cluster)
            similarity_flags[min_idx] = False
    
    # If similarity column is requested, fill it with scores
    if similarity_column:
        for (i, j), score in similarity_dict.items():
            if score > similarity_scores[i]:
                similarity_scores[i] = score
            if score > similarity_scores[j]:
                similarity_scores[j] = score
    
    # Convert arrays to pandas Series
    similarity_flags = pd.Series(similarity_flags, index=pandas_df.index)
    if group_column:
        group_ids = pd.Series(group_ids, index=pandas_df.index)
    if similarity_column:
        similarity_scores = pd.Series(similarity_scores, index=pandas_df.index)
    
    # Add columns to dataframe
    if inplace:
        pandas_df[flag_column] = similarity_flags
        if group_column:
            pandas_df[group_column] = group_ids
        if similarity_column:
            pandas_df[similarity_column] = similarity_scores
    else:
        pandas_df = pandas_df.assign(**{flag_column: similarity_flags})
        if group_column:
            pandas_df = pandas_df.assign(**{group_column: group_ids})
        if similarity_column:
            pandas_df = pandas_df.assign(**{similarity_column: similarity_scores})
    
    # Convert back to original type if needed
    if df_type != 'pandas' and not inplace:
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def _flag_similar_records_chunked(
    df: Any,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_similar',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    chunk_size: int = 1000,
    n_jobs: int = 1,
    use_polars: bool = False,
    max_comparisons: Optional[int] = None,
    text_columns: Optional[List[str]] = None,
    text_method: str = 'fuzzy',
    text_threshold: Optional[float] = None,
    show_progress: bool = False,
    jupyter_mode: Optional[bool] = None,
) -> Any:
    """
    Chunked implementation of similar records flagging for large datasets.
    
    This implementation processes the dataframe in chunks to avoid memory issues
    with large datasets, where an all-pairs comparison would be too expensive.
    
    See flag_similar_records for parameter descriptions.
    """
    import math
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import gc  # For garbage collection
    
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    n_rows = len(pandas_df)
    
    # If the dataset is smaller than chunk_size, use the standard approach
    if n_rows <= chunk_size:
        return _flag_similar_records_standard(
            df=pandas_df,
            columns=columns,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column=flag_column,
            inplace=inplace,
            group_column=group_column,
            similarity_column=similarity_column,
            max_comparisons=max_comparisons,
            use_polars=use_polars,
            blocking_columns=blocking_columns,
            blocking_method=blocking_method,
            blocking_rules=blocking_rules,
            max_block_size=max_block_size,
            use_lsh=use_lsh,
            lsh_method=lsh_method,
            lsh_threshold=lsh_threshold,
            num_perm=num_perm,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
        )
    
    # For very large datasets, make the chunk size even smaller
    if n_rows > 100000:
        original_chunk_size = chunk_size
        chunk_size = min(chunk_size, 500)
        if chunk_size != original_chunk_size:
            print(f"Reducing chunk size to {chunk_size} for very large dataset ({n_rows:,} rows)")
    
    # Calculate number of chunks
    n_chunks = math.ceil(n_rows / chunk_size)
    print(f"Processing dataset in {n_chunks} chunks of size {chunk_size}")
    
    # Use sparse data structures to save memory
    # Instead of storing all pairs immediately, process them in smaller batches
    G = nx.Graph()
    
    # Add all indices as nodes (necessary for correct connected components later)
    for i in range(n_rows):
        G.add_node(i)
    
    # Create a wrapper for local (non-parallel) use
    def _process_chunk_pair(chunk1_idx, chunk2_idx):
        return _process_chunk_pair_for_parallel((
            chunk1_idx, chunk2_idx, chunk_size, n_rows, n_chunks, 
            max_comparisons, threshold, columns, weights, method,
            text_columns, text_method, text_threshold, pandas_df[columns]
        ))
    
    # Calculate total number of chunk pairs
    total_chunk_pairs = (n_chunks * (n_chunks + 1)) // 2
    print(f"Processing {total_chunk_pairs} chunk pairs...")
    
    # Initialize progress tracking if requested
    block_progress_tracker = None
    if show_progress and PROGRESS_TRACKING_AVAILABLE:
        block_progress_tracker = BlockProgressTracker(
            total_blocks=total_chunk_pairs,
            description="Deduplication block progress",
            jupyter_mode=jupyter_mode
        )
        block_progress_tracker.start()
    
    # Generate chunk pairs strategically
    # Start with within-chunk pairs (diagonal), which are more likely to have similarities
    # Then do between-chunk pairs
    diagonal_pairs = [(i, i) for i in range(n_chunks)]
    
    off_diagonal_pairs = []
    for i in range(n_chunks):
        for j in range(i+1, n_chunks):
            off_diagonal_pairs.append((i, j))
    
    # Now put diagonal pairs first, then off-diagonal
    chunk_pairs = diagonal_pairs + off_diagonal_pairs
    
    # Define process_chunk_pair function at module level to avoid pickling issues
    def _global_process_chunk_pair(chunk1_idx, chunk2_idx, chunk_size, n_rows, pandas_df, columns, weights, method, threshold):
        """
        Global version of process_chunk_pair function that works with multiprocessing.
        Extracted from the nested function to avoid pickling issues.
        """
        start1 = chunk1_idx * chunk_size
        end1 = min((chunk1_idx + 1) * chunk_size, n_rows)
        
        start2 = chunk2_idx * chunk_size
        end2 = min((chunk2_idx + 1) * chunk_size, n_rows)
        
        # Get just the columns we need
        cols_subset = pandas_df[columns]
        
        chunk1 = cols_subset.iloc[start1:end1]
        
        # Initialize local results
        edges = []
        scores = {}
        
        # For within-chunk comparison
        if chunk1_idx == chunk2_idx:
            # Calculate max pairs for sampling
            total_chunk_pairs = (len(chunk1) * (len(chunk1) - 1)) // 2
            
            # If max_comparisons is set, limit the pairs to compare within this chunk
            chunk_max_pairs = None
            if max_comparisons is not None:
                # Proportionally allocate max_comparisons to this chunk
                chunk_size_proportion = (end1 - start1) / n_rows
                chunk_max_pairs = int(max_comparisons * chunk_size_proportion * chunk_size_proportion * n_chunks)
                
                # Make sure we have at least some pairs to compare
                chunk_max_pairs = max(100, chunk_max_pairs)
                chunk_max_pairs = min(chunk_max_pairs, total_chunk_pairs)
            
            # Generate pairs to compare - potentially with sampling
            if chunk_max_pairs is not None and chunk_max_pairs < total_chunk_pairs:
                import random
                pairs = []
                for i in range(len(chunk1)):
                    for j in range(i+1, len(chunk1)):
                        pairs.append((i, j))
                random.shuffle(pairs)
                chunk_pairs = pairs[:chunk_max_pairs]
            else:
                chunk_pairs = [(i, j) for i in range(len(chunk1)) for j in range(i+1, len(chunk1))]
            
            # Compare selected pairs
            for i, j in chunk_pairs:
                abs_i = start1 + i
                abs_j = start1 + j
                
                row1 = chunk1.iloc[i]
                row2 = chunk1.iloc[j]
                
                # Calculate similarity
                similarity = _calculate_similarity(
                    row1, row2, columns, weights, method,
                    text_columns=text_columns,
                    text_method=text_method,
                    text_threshold=text_threshold
                )
                
                # Add edge if similarity is above threshold
                if similarity >= threshold:
                    edges.append((abs_i, abs_j))
                    scores[(abs_i, abs_j)] = similarity
        else:
            # For between-chunk comparison
            chunk2 = cols_subset.iloc[start2:end2]
            
            # Calculate max pairs for sampling
            total_between_pairs = len(chunk1) * len(chunk2)
            
            # If max_comparisons is set, limit the pairs to compare between chunks
            between_max_pairs = None
            if max_comparisons is not None:
                # Proportionally allocate max_comparisons to this chunk pair
                chunk1_prop = (end1 - start1) / n_rows
                chunk2_prop = (end2 - start2) / n_rows
                between_max_pairs = int(max_comparisons * chunk1_prop * chunk2_prop * n_chunks * (n_chunks+1) / 2)
                
                # Make sure we have at least some pairs to compare
                between_max_pairs = max(100, between_max_pairs)
                between_max_pairs = min(between_max_pairs, total_between_pairs)
            
            # Generate pairs to compare - potentially with sampling
            if between_max_pairs is not None and between_max_pairs < total_between_pairs:
                import random
                # Sample pairs to compare - more efficient for large chunks
                chunk_pairs = []
                for _ in range(between_max_pairs):
                    i = random.randint(0, len(chunk1) - 1)
                    j = random.randint(0, len(chunk2) - 1)
                    chunk_pairs.append((i, j))
            else:
                chunk_pairs = [(i, j) for i in range(len(chunk1)) for j in range(len(chunk2))]
            
            # Compare selected pairs
            for i, j in chunk_pairs:
                abs_i = start1 + i
                abs_j = start2 + j
                
                row1 = chunk1.iloc[i]
                row2 = chunk2.iloc[j]
                
                # Calculate similarity
                similarity = _calculate_similarity(
                    row1, row2, columns, weights, method,
                    text_columns=text_columns,
                    text_method=text_method,
                    text_threshold=text_threshold
                )
                
                # Add edge if similarity is above threshold
                if similarity >= threshold:
                    edges.append((abs_i, abs_j))
                    scores[(abs_i, abs_j)] = similarity
        
        # Return results for this chunk pair
        return edges, scores
    
    # Process chunks
    if n_jobs > 1 and n_chunks > 1:
        print(f"Processing chunk pairs in parallel with {n_jobs} workers...")
        # Process in parallel
        all_edges = []
        all_scores = {}
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all jobs with all necessary parameters
            future_to_pair = {
                executor.submit(
                    _process_chunk_pair_for_parallel, 
                    (i, j, chunk_size, n_rows, n_chunks, 
                     max_comparisons, threshold, columns, weights, method,
                     text_columns, text_method, text_threshold, pandas_df[columns])
                ): (i, j) for i, j in chunk_pairs
            }
            
            # Process results as they complete
            for idx, future in enumerate(as_completed(future_to_pair)):
                i, j = future_to_pair[future]
                try:
                    edges, scores = future.result()
                    
                    # Only keep edges with similarity above threshold
                    if edges:
                        all_edges.extend(edges)
                        all_scores.update(scores)
                    
                    # Update progress tracking
                    if block_progress_tracker:
                        block_progress_tracker.update(1, force_display=False)
                    # Legacy progress reporting for non-interactive mode
                    elif (idx + 1) % max(1, total_chunk_pairs // 10) == 0:
                        progress = ((idx + 1) / total_chunk_pairs) * 100
                        print(f"Processed {idx + 1}/{total_chunk_pairs} chunk pairs ({progress:.1f}%)...")
                        print(f"Found {len(all_edges)} similar pairs so far")
                    
                    # Add edges to graph in batches to avoid memory spikes
                    if len(all_edges) > 100000:
                        G.add_edges_from(all_edges)
                        all_edges = []
                        # Force garbage collection
                        gc.collect()
                            
                except Exception as e:
                    print(f"Error processing chunk pair ({i}, {j}): {str(e)}")
    else:
        # Process sequentially
        print("Processing chunk pairs sequentially...")
        all_edges = []
        all_scores = {}
        
        for idx, (i, j) in enumerate(chunk_pairs):
            edges, scores = _process_chunk_pair(i, j)
            
            # Only keep edges with similarity above threshold
            if edges:
                all_edges.extend(edges)
                all_scores.update(scores)
            
            # Update progress tracking
            if block_progress_tracker:
                # Create a chunk progress tracker for detailed tracking within each chunk
                chunk_size_i = chunks[i][1] - chunks[i][0]
                chunk_size_j = chunks[j][1] - chunks[j][0]
                chunk_size = chunk_size_i * chunk_size_j if i != j else (chunk_size_i * (chunk_size_i - 1)) // 2
                chunk_tracker = block_progress_tracker.start_block(idx+1, chunk_size, f"Chunk ({i+1},{j+1})")
                chunk_tracker.finish()  # Since we're processing this synchronously
                
                # Update the main tracker
                block_progress_tracker.update(1, force_display=True)  # Force display to update
            # Legacy progress reporting for non-interactive mode
            elif (idx + 1) % max(1, total_chunk_pairs // 10) == 0:
                progress = ((idx + 1) / total_chunk_pairs) * 100
                print(f"Processed {idx + 1}/{total_chunk_pairs} chunk pairs ({progress:.1f}%)...")
                print(f"Found {len(all_edges)} similar pairs so far")
            
            # Add edges to graph in batches to avoid memory spikes
            if len(all_edges) > 100000:
                G.add_edges_from(all_edges)
                all_edges = []
                # Force garbage collection
                gc.collect()
    
    # Add any remaining edges to the graph
    if all_edges:
        G.add_edges_from(all_edges)
    
    # Clear memory we no longer need
    del all_edges
    gc.collect()
    
    # Finalize progress tracking
    if block_progress_tracker:
        block_progress_tracker.finish()
    
    print(f"Finding connected components among {G.number_of_edges()} similar pairs...")
    
    # Find connected components (clusters of similar records)
    clusters = list(nx.connected_components(G))
    
    print(f"Found {len(clusters)} clusters of similar records")
    del G  # Free up graph memory
    gc.collect()
    
    # Initialize numpy arrays for results (more memory efficient)
    similarity_flags = np.zeros(n_rows, dtype=bool)
    if group_column:
        group_ids = np.zeros(n_rows, dtype=int)
    if similarity_column:
        similarity_scores = np.zeros(n_rows, dtype=float)
    
    # Mark similar records and assign group IDs
    for group_id, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:  # Only process actual similar groups
            for idx in cluster:
                similarity_flags[idx] = True
                if group_column:
                    group_ids[idx] = group_id
    
    # For first occurrence in each group, set similarity flag to False
    for cluster in clusters:
        if len(cluster) > 1:
            min_idx = min(cluster)
            similarity_flags[min_idx] = False
    
    # If similarity column is requested, fill it with scores
    if similarity_column:
        for (i, j), score in all_scores.items():
            # Store the highest similarity score for each row
            if score > similarity_scores[i]:
                similarity_scores[i] = score
            if score > similarity_scores[j]:
                similarity_scores[j] = score
    
    # Convert to pandas Series with the right index
    similarity_flags_series = pd.Series(similarity_flags, index=pandas_df.index)
    
    if group_column:
        group_ids_series = pd.Series(group_ids, index=pandas_df.index)
    
    if similarity_column:
        similarity_scores_series = pd.Series(similarity_scores, index=pandas_df.index)
    
    # Add columns to dataframe
    if inplace:
        pandas_df[flag_column] = similarity_flags_series
        if group_column:
            pandas_df[group_column] = group_ids_series
        if similarity_column:
            pandas_df[similarity_column] = similarity_scores_series
    else:
        pandas_df = pandas_df.assign(**{flag_column: similarity_flags_series})
        if group_column:
            pandas_df = pandas_df.assign(**{group_column: group_ids_series})
        if similarity_column:
            pandas_df = pandas_df.assign(**{similarity_column: similarity_scores_series})
    
    # Force garbage collection before returning
    gc.collect()
    
    # Convert back to original type if needed
    if df_type != 'pandas' and not inplace:
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def _flag_similar_records_polars(
    df: Any,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_similar',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    blocking_columns: Optional[List[str]] = None,
    blocking_method: str = 'exact',
    blocking_rules: Optional[Dict[str, Callable]] = None,
    max_block_size: Optional[int] = None,
    use_lsh: bool = False,
    lsh_method: str = 'auto',
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    num_bands: Optional[int] = None,
    rows_per_band: Optional[int] = None,
) -> Any:
    """
    Polars-optimized implementation of similar records flagging.
    
    This implementation uses Polars for faster processing, especially for
    large datasets with many string comparisons.
    
    See flag_similar_records for parameter descriptions.
    """
    import polars as pl
    
    # Check dataframe type and convert if needed
    df_type = check_dataframe_type(df)
    
    if df_type == 'polars':
        polars_df = df.clone() if not inplace else df
    else:
        # Convert to polars
        if df_type == 'pandas':
            polars_df = pl.from_pandas(df)
        else:
            # Convert to pandas first, then to polars
            pandas_df = convert_dataframe(df, 'pandas')
            polars_df = pl.from_pandas(pandas_df)
    
    # Validate columns
    for col in columns:
        if col not in polars_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Initialize weights if not provided
    if weights is None or not weights:
        weights = {col: 1.0 / len(columns) for col in columns}
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {col: weight / total_weight for col, weight in weights.items()}
        else:
            # If all weights are zero or dictionary is empty, use equal weights
            weights = {col: 1.0 / len(columns) for col in columns}
        
        # Check if all columns have weights
        for col in columns:
            if col not in weights:
                weights[col] = 0.0
    
    # Get the dataframe as pandas for NetworkX operations
    pandas_df = polars_df.to_pandas()
    
    # Initialize columns
    n_rows = len(pandas_df)
    similarity_flags = pd.Series(False, index=pandas_df.index)
    group_ids = pd.Series(0, index=pandas_df.index)
    similarity_scores = pd.Series(0.0, index=pandas_df.index)
    
    # Create a graph to track similar records
    G = nx.Graph()
    
    # Add all indices as nodes
    for i in range(n_rows):
        G.add_node(i)
    
    # Calculate total possible comparisons
    total_comparisons = (n_rows * (n_rows - 1)) // 2
    print(f"Total possible comparisons: {total_comparisons:,}")
    
    # Apply optimization strategies for pair selection
    comparison_pairs = None
    
    # Step 1: Apply blocking if specified
    if blocking_columns or (blocking_method == 'rule' and blocking_rules):
        try:
            from freamon.deduplication.blocking import apply_blocking_strategy
            
            print(f"Applying {blocking_method} blocking with columns: {blocking_columns}")
            
            blocking_pairs = apply_blocking_strategy(
                df=pandas_df,
                strategy=blocking_method,
                blocking_columns=blocking_columns,
                blocking_rules=blocking_rules,
                max_block_size=max_block_size,
                max_comparisons=max_comparisons
            )
            
            comparison_pairs = blocking_pairs
            percentage = (len(comparison_pairs) / total_comparisons) * 100
            print(f"Blocking created {len(comparison_pairs):,} pairs to compare ({percentage:.2f}% of all possible pairs)")
        except ImportError as e:
            print(f"Warning: Blocking module not available. Error: {e}")
            print("Proceeding without blocking optimization.")
    
    # Step 2: Apply LSH if specified
    if use_lsh:
        try:
            from freamon.deduplication.lsh import apply_lsh_strategy
            
            # Set LSH threshold if not specified
            if lsh_threshold is None:
                lsh_threshold = threshold * 0.9  # Slightly lower to catch more candidates
            
            print(f"Applying {lsh_method} LSH with threshold: {lsh_threshold}")
            
            lsh_pairs = apply_lsh_strategy(
                df=pandas_df,
                columns=columns,
                weights=weights,
                lsh_method=lsh_method,
                threshold=lsh_threshold,
                num_perm=num_perm,
                num_bands=num_bands,
                rows_per_band=rows_per_band
            )
            
            # If we also have blocking pairs, take the intersection
            if comparison_pairs is not None:
                comparison_pairs_set = set(tuple(sorted(pair)) for pair in comparison_pairs)
                lsh_pairs_set = set(lsh_pairs)  # Already sorted in LSH
                combined_pairs = lsh_pairs_set.intersection(comparison_pairs_set)
                comparison_pairs = list(combined_pairs)
                percentage = (len(comparison_pairs) / total_comparisons) * 100
                print(f"Combined blocking and LSH: {len(comparison_pairs):,} pairs to compare ({percentage:.2f}% of all possible pairs)")
            else:
                comparison_pairs = list(lsh_pairs)
                percentage = (len(comparison_pairs) / total_comparisons) * 100
                print(f"LSH identified {len(comparison_pairs):,} potential similar pairs ({percentage:.2f}% of all possible pairs)")
        except ImportError as e:
            print(f"Warning: LSH module not available. Error: {e}")
            print("Proceeding without LSH optimization.")
    
    # Step 3: Apply max_comparisons limit if needed
    comparisons_to_do = total_comparisons
    if max_comparisons is not None:
        comparisons_to_do = min(total_comparisons, max_comparisons)
        
        if comparison_pairs is not None and len(comparison_pairs) > max_comparisons:
            # Sample from our optimized pairs
            import random
            random.shuffle(comparison_pairs)
            comparison_pairs = comparison_pairs[:max_comparisons]
            print(f"Limited to {max_comparisons:,} pairs due to max_comparisons setting")
    
    # Generate final pairs to compare
    if comparison_pairs is not None:
        # Use pre-filtered pairs
        pairs = comparison_pairs
    elif comparisons_to_do < total_comparisons:
        # Randomly sample pairs for efficiency
        import random
        all_pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
        random.shuffle(all_pairs)
        pairs = all_pairs[:comparisons_to_do]
    else:
        # Compare all pairs
        pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
        
    # Print info about comparisons for large datasets
    if n_rows > 5000:
        percentage = (len(pairs) / total_comparisons) * 100
        print(f"Processing {len(pairs):,} of {total_comparisons:,} possible comparisons ({percentage:.1f}%)")
    
    # Process in batches for better performance
    batch_size = 10000
    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]
        
        # Extract indices
        idx1_list = [p[0] for p in batch_pairs]
        idx2_list = [p[1] for p in batch_pairs]
        
        # Compare records in batch using Polars expressions
        similarities = []
        
        for idx1, idx2 in zip(idx1_list, idx2_list):
            row1 = pandas_df.iloc[idx1]
            row2 = pandas_df.iloc[idx2]
            
            # Calculate similarity
            similarity = _calculate_similarity(
                row1, row2, columns, weights, method,
                text_columns=text_columns,
                text_method=text_method,
                text_threshold=text_threshold
            )
            similarities.append(similarity)
        
        # Add edges for similar pairs
        for (idx1, idx2), similarity in zip(batch_pairs, similarities):
            if similarity >= threshold:
                G.add_edge(idx1, idx2, weight=similarity)
                
                # Store similarity scores if requested
                if similarity_column:
                    if similarity > similarity_scores.iloc[idx1]:
                        similarity_scores.iloc[idx1] = similarity
                    if similarity > similarity_scores.iloc[idx2]:
                        similarity_scores.iloc[idx2] = similarity
    
    # Find connected components (clusters of similar records)
    clusters = list(nx.connected_components(G))
    
    # Mark similar records and assign group IDs
    for group_id, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:  # Only process actual similar groups
            for idx in cluster:
                similarity_flags.iloc[idx] = True
                if group_column:
                    group_ids.iloc[idx] = group_id
    
    # For first occurrence in each group, set similarity flag to False
    for cluster in clusters:
        if len(cluster) > 1:
            min_idx = min(cluster)
            similarity_flags.iloc[min_idx] = False
    
    # Add columns to the polars dataframe
    if inplace:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=similarity_flags.values)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids.values)
            ])
            
        if similarity_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores.values)
            ])
    else:
        polars_df = polars_df.with_columns([
            pl.Series(name=flag_column, values=similarity_flags.values)
        ])
        
        if group_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=group_column, values=group_ids.values)
            ])
            
        if similarity_column:
            polars_df = polars_df.with_columns([
                pl.Series(name=similarity_column, values=similarity_scores.values)
            ])
    
    # Convert back to original type if needed
    if df_type != 'polars' and not inplace:
        return convert_dataframe(polars_df, df_type)
    else:
        return polars_df


# Define a function to process a pair of chunks for parallel processing
# This needs to be at module level to avoid pickling issues with multiprocessing
def _process_chunk_pair_for_parallel(args):
    chunk1_idx, chunk2_idx, chunk_size, n_rows, n_chunks, max_comparisons, threshold, columns, weights, method, text_columns, text_method, text_threshold, df_columns = args
    
    import pandas as pd
    import random
    
    start1 = chunk1_idx * chunk_size
    end1 = min((chunk1_idx + 1) * chunk_size, n_rows)
    
    start2 = chunk2_idx * chunk_size
    end2 = min((chunk2_idx + 1) * chunk_size, n_rows)
    
    # Initialize local results
    edges = []
    scores = {}
    
    # For within-chunk comparison
    if chunk1_idx == chunk2_idx:
        # Get the chunk
        chunk1 = df_columns.iloc[start1:end1]
        
        # Calculate max pairs for sampling
        total_chunk_pairs = (len(chunk1) * (len(chunk1) - 1)) // 2
        
        # If max_comparisons is set, limit the pairs to compare within this chunk
        chunk_max_pairs = None
        if max_comparisons is not None:
            # Proportionally allocate max_comparisons to this chunk
            chunk_size_proportion = (end1 - start1) / n_rows
            chunk_max_pairs = int(max_comparisons * chunk_size_proportion * chunk_size_proportion * n_chunks)
            
            # Make sure we have at least some pairs to compare
            chunk_max_pairs = max(100, chunk_max_pairs)
            chunk_max_pairs = min(chunk_max_pairs, total_chunk_pairs)
        
        # Generate pairs to compare - potentially with sampling
        if chunk_max_pairs is not None and chunk_max_pairs < total_chunk_pairs:
            pairs = []
            for i in range(len(chunk1)):
                for j in range(i+1, len(chunk1)):
                    pairs.append((i, j))
            random.shuffle(pairs)
            chunk_pairs = pairs[:chunk_max_pairs]
        else:
            chunk_pairs = [(i, j) for i in range(len(chunk1)) for j in range(i+1, len(chunk1))]
        
        # Compare selected pairs
        for i, j in chunk_pairs:
            abs_i = start1 + i
            abs_j = start1 + j
            
            row1 = chunk1.iloc[i]
            row2 = chunk1.iloc[j]
            
            # Calculate similarity using our _calculate_similarity function
            # Import here to avoid circular reference
            from freamon.deduplication.flag_duplicates import _calculate_similarity
            
            similarity = _calculate_similarity(
                row1, row2, columns, weights, method, 
                text_columns=text_columns, 
                text_method=text_method, 
                text_threshold=text_threshold
            )
            
            # Add edge if similarity is above threshold
            if similarity >= threshold:
                edges.append((abs_i, abs_j))
                scores[(abs_i, abs_j)] = similarity
    else:
        # For between-chunk comparison
        chunk1 = df_columns.iloc[start1:end1]
        chunk2 = df_columns.iloc[start2:end2]
        
        # Calculate max pairs for sampling
        total_between_pairs = len(chunk1) * len(chunk2)
        
        # If max_comparisons is set, limit the pairs to compare between chunks
        between_max_pairs = None
        if max_comparisons is not None:
            # Proportionally allocate max_comparisons to this chunk pair
            chunk1_prop = (end1 - start1) / n_rows
            chunk2_prop = (end2 - start2) / n_rows
            between_max_pairs = int(max_comparisons * chunk1_prop * chunk2_prop * n_chunks * (n_chunks+1) / 2)
            
            # Make sure we have at least some pairs to compare
            between_max_pairs = max(100, between_max_pairs)
            between_max_pairs = min(between_max_pairs, total_between_pairs)
        
        # Generate pairs to compare - potentially with sampling
        if between_max_pairs is not None and between_max_pairs < total_between_pairs:
            # Sample pairs to compare - more efficient for large chunks
            chunk_pairs = []
            for _ in range(between_max_pairs):
                i = random.randint(0, len(chunk1) - 1)
                j = random.randint(0, len(chunk2) - 1)
                chunk_pairs.append((i, j))
        else:
            chunk_pairs = [(i, j) for i in range(len(chunk1)) for j in range(len(chunk2))]
        
        # Compare selected pairs
        for i, j in chunk_pairs:
            abs_i = start1 + i
            abs_j = start2 + j
            
            row1 = chunk1.iloc[i]
            row2 = chunk2.iloc[j]
            
            # Calculate similarity using the calculate_similarity function
            # Import here to avoid circular reference
            from freamon.deduplication.flag_duplicates import _calculate_similarity
            
            similarity = _calculate_similarity(
                row1, row2, columns, weights, method, 
                text_columns=text_columns, 
                text_method=text_method, 
                text_threshold=text_threshold
            )
            
            # Add edge if similarity is above threshold
            if similarity >= threshold:
                edges.append((abs_i, abs_j))
                scores[(abs_i, abs_j)] = similarity
    
    # Return results for this chunk pair
    return edges, scores


def _calculate_similarity(row1, row2, columns, weights, method, 
                      text_columns=None, text_method='fuzzy', text_threshold=0.8):
    """
    Helper function to calculate similarity between two rows.
    
    Used by both standard and chunked implementations.
    
    Parameters
    ----------
    row1 : pd.Series
        First row to compare
    row2 : pd.Series
        Second row to compare
    columns : List[str]
        Columns to consider for similarity
    weights : Dict[str, float]
        Dictionary mapping column names to weights
    method : str
        Similarity calculation method ('composite', 'exact_subset', or 'fuzzy_subset')
    text_columns : Optional[List[str]]
        Columns to process as text with specialized methods
    text_method : str
        Method to use for text similarity calculation
    text_threshold : float
        Threshold for text similarity
    """
    # Initialize text columns if not provided
    if text_columns is None:
        text_columns = []
    
    if method == 'composite':
        # Calculate weighted similarity across all columns
        total_sim = 0.0
        for col in columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Skip if either value is missing
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Calculate column similarity based on type and text classification
            if col in text_columns and isinstance(val1, str) and isinstance(val2, str):
                # Use specialized text similarity methods
                if text_method == 'fuzzy':
                    # Levenshtein for moderate-length text
                    from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                    col_sim = calculate_levenshtein_similarity(val1, val2)
                
                elif text_method == 'tfidf':
                    # TF-IDF cosine similarity for longer text
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # Create a small corpus with just these two texts
                        corpus = [val1, val2]
                        
                        # Vectorize
                        vectorizer = TfidfVectorizer(
                            stop_words='english', 
                            min_df=1,
                            lowercase=True,
                            strip_accents='unicode'
                        )
                        
                        # Calculate similarity
                        tfidf_matrix = vectorizer.fit_transform(corpus)
                        col_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    except ImportError:
                        # Fall back to fuzzy if sklearn not available
                        from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                        col_sim = calculate_levenshtein_similarity(val1, val2)
                
                elif text_method == 'ngram':
                    # N-gram Jaccard similarity
                    try:
                        # Create character n-grams
                        def get_ngrams(text, n=3):
                            return set(' ' + text.lower() + ' ') | set(
                                text.lower()[i:i+n] for i in range(len(text) - n + 1) if len(text) >= n
                            )
                        
                        # Get n-grams for both texts
                        ngrams1 = get_ngrams(val1, 3)
                        ngrams2 = get_ngrams(val2, 3)
                        
                        # Calculate Jaccard similarity
                        if not ngrams1 or not ngrams2:
                            col_sim = 0.0
                        else:
                            col_sim = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
                    except:
                        # Fall back to fuzzy
                        from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                        col_sim = calculate_levenshtein_similarity(val1, val2)
                
                elif text_method == 'lsh':
                    # Use minhash for LSH-based similarity
                    try:
                        from datasketch import MinHash
                        
                        # Function to create shingles
                        def get_shingles(text, k=3):
                            return set(text.lower()[i:i+k] for i in range(len(text) - k + 1) if len(text) >= k)
                        
                        # Create MinHash objects
                        m1, m2 = MinHash(), MinHash()
                        for s in get_shingles(val1):
                            m1.update(s.encode('utf-8'))
                        for s in get_shingles(val2):
                            m2.update(s.encode('utf-8'))
                        
                        # Calculate Jaccard similarity estimate
                        col_sim = m1.jaccard(m2)
                    except ImportError:
                        # Fall back to fuzzy
                        from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                        col_sim = calculate_levenshtein_similarity(val1, val2)
                
                else:
                    # Default to Levenshtein
                    from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                    col_sim = calculate_levenshtein_similarity(val1, val2)
            
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    col_sim = 1.0 if val1 == val2 else 0.0
                else:
                    col_sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
            
            elif isinstance(val1, str) and isinstance(val2, str):
                # Regular text/string similarity
                from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                col_sim = calculate_levenshtein_similarity(val1, val2)
            
            else:
                # Other types - exact match only
                col_sim = 1.0 if val1 == val2 else 0.0
            
            # Add weighted similarity
            total_sim += weights.get(col, 0.0) * col_sim
        
        return total_sim
        
    elif method == 'exact_subset':
        # Match if a subset of columns match exactly
        matching_weight = 0.0
        for col in columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Skip if either value is missing
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Check exact match
            if val1 == val2:
                matching_weight += weights.get(col, 0.0)
        
        return matching_weight
        
    elif method == 'fuzzy_subset':
        # Match if a subset of columns match with high similarity
        matching_weight = 0.0
        for col in columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Skip if either value is missing
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Calculate column similarity based on type and text classification
            if col in text_columns and isinstance(val1, str) and isinstance(val2, str):
                # Use specialized text similarity methods with appropriate threshold
                if text_method == 'fuzzy':
                    # Levenshtein for moderate-length text
                    from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                    col_sim = calculate_levenshtein_similarity(val1, val2)
                    # Use text-specific threshold
                    if col_sim >= text_threshold:
                        matching_weight += weights.get(col, 0.0)
                
                elif text_method == 'tfidf':
                    # TF-IDF cosine similarity for longer text
                    try:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        # Create a small corpus with just these two texts
                        corpus = [val1, val2]
                        
                        # Vectorize
                        vectorizer = TfidfVectorizer(
                            stop_words='english', 
                            min_df=1,
                            lowercase=True,
                            strip_accents='unicode'
                        )
                        
                        # Calculate similarity
                        tfidf_matrix = vectorizer.fit_transform(corpus)
                        col_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        
                        # Use text-specific threshold
                        if col_sim >= text_threshold:
                            matching_weight += weights.get(col, 0.0)
                    except ImportError:
                        # Fall back to fuzzy if sklearn not available
                        from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                        col_sim = calculate_levenshtein_similarity(val1, val2)
                        if col_sim >= text_threshold:
                            matching_weight += weights.get(col, 0.0)
                
                elif text_method in ['ngram', 'lsh']:
                    # These methods use Jaccard similarity
                    try:
                        # Create character n-grams
                        def get_ngrams(text, n=3):
                            return set(' ' + text.lower() + ' ') | set(
                                text.lower()[i:i+n] for i in range(len(text) - n + 1) if len(text) >= n
                            )
                        
                        # Get n-grams for both texts
                        ngrams1 = get_ngrams(val1, 3)
                        ngrams2 = get_ngrams(val2, 3)
                        
                        # Calculate Jaccard similarity
                        if not ngrams1 or not ngrams2:
                            col_sim = 0.0
                        else:
                            col_sim = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
                        
                        # Use text-specific threshold
                        if col_sim >= text_threshold:
                            matching_weight += weights.get(col, 0.0)
                    except:
                        # Fall back to fuzzy
                        from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                        col_sim = calculate_levenshtein_similarity(val1, val2)
                        if col_sim >= text_threshold:
                            matching_weight += weights.get(col, 0.0)
                
                else:
                    # Default to Levenshtein
                    from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                    col_sim = calculate_levenshtein_similarity(val1, val2)
                    if col_sim >= text_threshold:
                        matching_weight += weights.get(col, 0.0)
            
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    col_sim = 1.0 if val1 == val2 else 0.0
                else:
                    col_sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                
                # Consider as matching if similarity is high enough
                if col_sim >= 0.9:  # High threshold for numerical columns
                    matching_weight += weights.get(col, 0.0)
            
            elif isinstance(val1, str) and isinstance(val2, str):
                # Regular text/string similarity
                from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                col_sim = calculate_levenshtein_similarity(val1, val2)
                
                # Consider as matching if similarity is high enough
                if col_sim >= 0.9:  # High threshold for short string columns
                    matching_weight += weights.get(col, 0.0)
            
            else:
                # Other types - exact match only
                if val1 == val2:
                    matching_weight += weights.get(col, 0.0)
        
        return matching_weight
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'composite', 'exact_subset', or 'fuzzy_subset'.")


def flag_supervised_duplicates(
    df: Any,
    model: Optional[PolarsSupervisedDeduplicationModel] = None,
    duplicate_pairs: Optional[List[Tuple[int, int]]] = None,
    key_features: Optional[List[str]] = None,
    date_features: Optional[List[str]] = None,
    threshold: float = 0.5,
    flag_column: str = 'is_duplicate',
    duplicate_of_column: Optional[str] = 'duplicate_of',
    probability_column: Optional[str] = 'duplicate_probability',
    model_type: str = 'lightgbm',
    inplace: bool = False,
    **kwargs
) -> Any:
    """
    Flag potential duplicates using a supervised machine learning model.
    
    This function either uses a provided model or trains a new one on the specified
    duplicate pairs, then predicts potential duplicates in the dataset.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    model : Optional[PolarsSupervisedDeduplicationModel], default=None
        Pre-trained supervised deduplication model. If None, a new model will be
        trained using the provided duplicate_pairs.
    duplicate_pairs : Optional[List[Tuple[int, int]]], default=None
        List of tuples containing index pairs of known duplicates for training.
        Required if model is None.
    key_features : Optional[List[str]], default=None
        List of column names for features important for duplicate detection.
        Required if model is None.
    date_features : Optional[List[str]], default=None
        List of column names containing date/time features.
    threshold : float, default=0.5
        Probability threshold above which pairs are considered duplicates.
    flag_column : str, default='is_duplicate'
        Name of the column to add for flagging duplicates.
    duplicate_of_column : Optional[str], default='duplicate_of'
        If provided, add a column with this name containing the ID of the record
        this is a duplicate of. Uses the first column of the dataframe as the ID column.
    probability_column : Optional[str], default='duplicate_probability'
        If provided, add a column with this name containing the duplicate probability.
    model_type : str, default='lightgbm'
        Type of model to use if training a new model: 'lightgbm', 'random_forest', or 'gradient_boosting'.
    inplace : bool, default=False
        If True, modify the dataframe in-place.
    **kwargs
        Additional keyword arguments to pass to the model's fit method or find_duplicates method.
        
    Returns
    -------
    Any
        DataFrame with duplicates flagged.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'name': ['John Smith', 'Jane Doe', 'Jon Smith', 'Mary Jones', 'John Smith'],
    ...     'email': ['john@example.com', 'jane@example.com', 'jon@example.com', 
    ...               'mary@example.com', 'johnsmith@example.com'],
    ...     'amount': [100, 200, 150, 300, 100],
    ...     'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-03', 
    ...                            '2023-02-15', '2023-01-01'])
    ... })
    >>> # Provide known duplicate pairs for training
    >>> pairs = [(0, 4), (0, 2)]  # Indices of known duplicates
    >>> result = flag_supervised_duplicates(
    ...     df, 
    ...     duplicate_pairs=pairs,
    ...     key_features=['name', 'email', 'amount'],
    ...     date_features=['date'],
    ...     threshold=0.6
    ... )
    """
    # Check dataframe type
    df_type = check_dataframe_type(df)
    
    # Convert to pandas if needed
    if df_type != 'pandas':
        pandas_df = convert_dataframe(df, 'pandas')
    else:
        pandas_df = df.copy() if not inplace else df
    
    # Validate inputs
    if model is None and duplicate_pairs is None:
        raise ValueError("Either model or duplicate_pairs must be provided")
    
    if model is None and key_features is None:
        raise ValueError("key_features must be provided when training a new model")
    
    # Train or use the provided model
    if model is None:
        # Create and train a new model
        model = PolarsSupervisedDeduplicationModel(
            model_type=model_type,
            key_features=key_features,
            date_features=date_features or [],
            use_polars=True
        )
        
        # Get training kwargs
        train_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['validation_pairs', 'validation_fraction']}
        
        model.fit(pandas_df, duplicate_pairs, **train_kwargs)
    
    # Get predictions
    predict_kwargs = {k: v for k, v in kwargs.items() 
                     if k in ['max_comparisons', 'chunk_size']}
    
    # Predict duplicates with probabilities
    duplicate_df = model.predict_duplicate_probability(
        df=pandas_df,
        **predict_kwargs
    )
    
    # Filter by threshold
    duplicate_df = duplicate_df[duplicate_df['duplicate_probability'] >= threshold]
    
    # Initialize columns
    pandas_df[flag_column] = False
    
    if duplicate_of_column:
        # Use the first column as the ID column by default
        id_column = pandas_df.columns[0]
        pandas_df[duplicate_of_column] = None
    
    if probability_column:
        pandas_df[probability_column] = 0.0
    
    # Create a dictionary to store the best match for each record
    best_matches = {}
    
    # Find the best match for each record (highest probability)
    for _, row in duplicate_df.iterrows():
        idx1, idx2 = int(row['idx1']), int(row['idx2'])
        prob = row['duplicate_probability']
        
        # For idx1
        if idx1 not in best_matches or prob > best_matches[idx1]['probability']:
            best_matches[idx1] = {
                'duplicate_of': idx2,
                'probability': prob
            }
        
        # For idx2
        if idx2 not in best_matches or prob > best_matches[idx2]['probability']:
            best_matches[idx2] = {
                'duplicate_of': idx1,
                'probability': prob
            }
    
    # Update the dataframe with the best matches
    for idx, match_info in best_matches.items():
        # Don't flag the record with the lowest index in each pair
        if match_info['duplicate_of'] > idx:
            continue
            
        # Flag this record as a duplicate
        pandas_df.loc[idx, flag_column] = True
        
        # Add duplicate_of reference if requested
        if duplicate_of_column:
            duplicate_of_idx = match_info['duplicate_of']
            pandas_df.loc[idx, duplicate_of_column] = pandas_df.iloc[duplicate_of_idx][id_column]
        
        # Add probability if requested
        if probability_column:
            pandas_df.loc[idx, probability_column] = match_info['probability']
    
    # Return the result
    if not inplace and df_type != 'pandas':
        return convert_dataframe(pandas_df, df_type)
    else:
        return pandas_df


def add_duplicate_detection_columns(
    df: Any,
    method: str = 'exact',
    text_column: Optional[str] = None,
    columns: Optional[List[str]] = None,
    threshold: float = 0.8,
    flag_column: str = 'is_duplicate',
    group_column: str = 'duplicate_group',
    similarity_column: Optional[str] = None,
    duplicate_of_column: Optional[str] = None,
    inplace: bool = False,
    **kwargs
) -> Any:
    """
    Comprehensive function to add duplicate detection columns to a dataframe.
    
    This is a high-level wrapper around the specific flagging functions that
    makes it easier to add duplicate detection to a dataframe.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    method : str, default='exact'
        Method to use for finding duplicates:
        - 'exact': Exact matching on columns
        - 'hash': Hash-based text matching
        - 'ngram': N-gram fingerprint text matching
        - 'fuzzy': Fuzzy text matching
        - 'lsh': Locality-sensitive hashing for text
        - 'similar': Record similarity across multiple columns
        - 'supervised': Supervised machine learning model
    text_column : Optional[str], default=None
        Column containing text data to check for duplicates.
        Required for 'hash', 'ngram', 'fuzzy', and 'lsh' methods.
    columns : Optional[List[str]], default=None
        Columns to consider when calculating similarity or for exact matching.
        Required for 'exact' and 'similar' methods.
    threshold : float, default=0.8
        Similarity threshold for fuzzy methods.
    flag_column : str, default='is_duplicate'
        Name of the column to add for flagging duplicates.
    group_column : str, default='duplicate_group'
        Name of the column to add for duplicate group IDs.
    similarity_column : Optional[str], default=None
        If provided, add a column with this name containing similarity scores.
    duplicate_of_column : Optional[str], default=None
        If provided, add a column with this name containing the ID of the record
        this is a duplicate of. Only used for 'supervised' method.
    inplace : bool, default=False
        If True, modify the dataframe in-place.
    **kwargs
        Additional keyword arguments to pass to the specific flagging function.
        
    Returns
    -------
    Any
        DataFrame with duplicate detection columns added.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4, 5],
    ...     'name': ['John Smith', 'Jane Doe', 'Jon Smith', 'Mary Jones', 'John Smith'],
    ...     'text': ['This is a sample.', 'Different text.', 'Similar sample.', 
    ...              'Unique content.', 'This is a sample.']
    ... })
    >>> # Exact matching on certain columns
    >>> result1 = add_duplicate_detection_columns(df, method='exact', 
    ...                                           columns=['name'])
    >>> # Fuzzy text matching
    >>> result2 = add_duplicate_detection_columns(df, method='fuzzy', 
    ...                                           text_column='text',
    ...                                           threshold=0.7)
    """
    # Validate required parameters based on method
    if method in ['hash', 'ngram', 'fuzzy', 'lsh'] and text_column is None:
        raise ValueError(f"Text column must be specified for method '{method}'")
        
    if method in ['exact', 'similar'] and columns is None:
        raise ValueError(f"Columns must be specified for method '{method}'")
    
    if method == 'supervised' and 'model' not in kwargs and 'duplicate_pairs' not in kwargs:
        raise ValueError("Either model or duplicate_pairs must be provided for supervised method")
    
    # Call the appropriate function based on method
    if method == 'exact':
        return flag_exact_duplicates(
            df=df,
            subset=columns,
            flag_column=flag_column,
            indicator_column=group_column,
            inplace=inplace,
            **kwargs
        )
        
    elif method in ['hash', 'ngram', 'fuzzy', 'lsh']:
        return flag_text_duplicates(
            df=df,
            text_column=text_column,
            method=method,
            threshold=threshold,
            flag_column=flag_column,
            group_column=group_column,
            similarity_column=similarity_column,
            inplace=inplace,
            **kwargs
        )
        
    elif method == 'similar':
        return flag_similar_records(
            df=df,
            columns=columns,
            threshold=threshold,
            flag_column=flag_column,
            group_column=group_column,
            similarity_column=similarity_column,
            inplace=inplace,
            **kwargs
        )
        
    elif method == 'supervised':
        return flag_supervised_duplicates(
            df=df,
            threshold=threshold,
            flag_column=flag_column,
            duplicate_of_column=duplicate_of_column,
            probability_column=similarity_column,
            inplace=inplace,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'exact', 'hash', 'ngram', 'fuzzy', 'lsh', 'similar', or 'supervised'.")