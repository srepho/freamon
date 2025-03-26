"""
Functions for flagging duplicate records in a dataframe.

This module provides functionality to identify and flag duplicates in a dataframe
rather than removing them, useful for:
1. Maintaining the original dataset size
2. Making informed decisions about which duplicates to remove
3. Downstream filtering based on duplicate status
4. Tracking duplicates for reporting and analysis
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Set
import pandas as pd
import numpy as np
import networkx as nx

from freamon.utils.dataframe_utils import check_dataframe_type, convert_dataframe
from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import find_similar_texts
from freamon.deduplication.lsh_deduplication import lsh_deduplication
from freamon.deduplication.polars_supervised_deduplication import PolarsSupervisedDeduplicationModel


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


def flag_similar_records(
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
) -> Any:
    """
    Flag similar records based on multiple columns with customizable weights.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    columns : List[str]
        Columns to consider when calculating similarity.
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
    ...     'phone': ['555-1234', '555-5678', '555-9012', '555-3456', '555-1234']
    ... })
    >>> # Weight name and email higher than phone
    >>> weights = {'name': 0.4, 'email': 0.4, 'phone': 0.2}
    >>> result = flag_similar_records(df, columns=['name', 'email', 'phone'], 
    ...                               weights=weights, threshold=0.7)
    """
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
    if weights is None:
        weights = {col: 1.0 / len(columns) for col in columns}
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {col: weight / total_weight for col, weight in weights.items()}
        
        # Check if all columns have weights
        for col in columns:
            if col not in weights:
                weights[col] = 0.0
    
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
    
    # Define a function to calculate similarity between two rows
    def calculate_similarity(row1, row2):
        if method == 'composite':
            # Calculate weighted similarity across all columns
            total_sim = 0.0
            for col in columns:
                val1 = row1[col]
                val2 = row2[col]
                
                # Skip if either value is missing
                if pd.isna(val1) or pd.isna(val2):
                    continue
                
                # Calculate column similarity based on type
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        col_sim = 1.0 if val1 == val2 else 0.0
                    else:
                        col_sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Text similarity
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
                
                # Calculate column similarity based on type
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    max_val = max(abs(val1), abs(val2))
                    if max_val == 0:
                        col_sim = 1.0 if val1 == val2 else 0.0
                    else:
                        col_sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
                
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Text similarity
                    from freamon.deduplication.fuzzy_deduplication import calculate_levenshtein_similarity
                    col_sim = calculate_levenshtein_similarity(val1, val2)
                
                else:
                    # Other types - exact match only
                    col_sim = 1.0 if val1 == val2 else 0.0
                
                # Consider as matching if similarity is high enough
                if col_sim >= 0.9:  # High threshold for individual columns
                    matching_weight += weights.get(col, 0.0)
            
            return matching_weight
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'composite', 'exact_subset', or 'fuzzy_subset'.")
    
    # Compare rows and add edges for similar records
    total_comparisons = (n_rows * (n_rows - 1)) // 2
    comparisons_to_do = min(total_comparisons, max_comparisons or total_comparisons)
    
    if comparisons_to_do < total_comparisons:
        # Randomly sample pairs for efficiency
        import random
        pairs = []
        for i in range(n_rows):
            for j in range(i+1, n_rows):
                pairs.append((i, j))
        random.shuffle(pairs)
        pairs = pairs[:comparisons_to_do]
    else:
        # Compare all pairs
        pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
    
    for i, j in pairs:
        row1 = pandas_df.iloc[i]
        row2 = pandas_df.iloc[j]
        
        # Calculate similarity
        similarity = calculate_similarity(row1, row2)
        
        # Add edge if similarity is above threshold
        if similarity >= threshold:
            G.add_edge(i, j, weight=similarity)
            
            # Store similarity scores if requested
            if similarity_column:
                # Store the highest similarity score for each row
                if similarity > similarity_scores.iloc[i]:
                    similarity_scores.iloc[i] = similarity
                if similarity > similarity_scores.iloc[j]:
                    similarity_scores.iloc[j] = similarity
    
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
    
    # Add flag column
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