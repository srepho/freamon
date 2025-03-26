"""
Blocking strategies for efficient record deduplication.

Blocking is a technique to partition data into smaller groups (blocks) where
similar records are likely to be in the same block. Only records within the same
block are compared, drastically reducing the number of comparisons needed.
"""

from typing import List, Dict, Any, Callable, Tuple, Set, Union
from collections import defaultdict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_exact_blocks(
    df: Any, 
    blocking_columns: List[str]
) -> Dict[Tuple, List[int]]:
    """
    Create blocks based on exact matches in blocking columns.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    blocking_columns : List[str]
        Columns to use for blocking
        
    Returns
    -------
    Dict[Tuple, List[int]]
        Dictionary mapping block keys to lists of record indices
    """
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        # Create blocking key based on blocking columns
        # Handle missing values by using a sentinel
        block_key = tuple(str(row.get(col, "MISSING") if hasattr(row, 'get') else 
                             row[col] if col in row else "MISSING") 
                          for col in blocking_columns)
        blocks[block_key].append(idx)
    
    return blocks


def create_phonetic_blocks(
    df: Any, 
    column: str,
    phonetic_func: Callable = None
) -> Dict[str, List[int]]:
    """
    Create blocks based on phonetic encoding of a column.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    column : str
        Column to use for phonetic blocking
    phonetic_func : Callable, default=None
        Function that converts a string to a phonetic code.
        If None, uses the first character of the string.
        
    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping phonetic codes to lists of record indices
    """
    # If no phonetic function is provided, use the first character
    if phonetic_func is None:
        phonetic_func = lambda x: x[0].upper() if x and isinstance(x, str) and len(x) > 0 else "_"
        
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        value = row[column] if column in row else None
        if value is not None and not pd.isna(value):
            if isinstance(value, str):
                try:
                    encoded = phonetic_func(value)
                    blocks[encoded].append(idx)
                except Exception as e:
                    # If encoding fails, use a fallback key
                    logger.debug(f"Phonetic encoding failed for '{value}': {e}")
                    blocks["_error"].append(idx)
            else:
                # For non-string values, use string representation
                blocks[str(value)].append(idx)
        else:
            # Handle missing values
            blocks["_missing"].append(idx)
    
    return blocks


def create_ngram_blocks(
    df: Any,
    column: str,
    n: int = 2,
    num_grams: int = 1
) -> Dict[Tuple[str, ...], List[int]]:
    """
    Create blocks based on n-grams of a column.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    column : str
        Column to use for n-gram blocking
    n : int, default=2
        Size of each n-gram
    num_grams : int, default=1
        Number of most frequent n-grams to use for blocking
        
    Returns
    -------
    Dict[Tuple[str, ...], List[int]]
        Dictionary mapping n-gram combinations to lists of record indices
    """
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        value = row[column] if column in row else None
        if value is not None and not pd.isna(value) and isinstance(value, str):
            # Generate n-grams
            value = value.lower()
            ngrams = [value[i:i+n] for i in range(len(value) - n + 1)]
            
            if ngrams:
                # Use the most frequent n-grams for blocking
                if num_grams == 1:
                    # Use each n-gram as a separate block
                    for gram in ngrams:
                        blocks[gram].append(idx)
                else:
                    # Use combinations of the most frequent n-grams
                    blocks[tuple(sorted(ngrams[:num_grams]))].append(idx)
            else:
                # String too short for n-grams
                blocks["_short"].append(idx)
        else:
            # Handle missing or non-string values
            blocks["_missing"].append(idx)
    
    return blocks


def create_custom_blocks(
    df: Any, 
    blocking_func: Callable
) -> Dict[Any, List[int]]:
    """
    Create blocks using a custom blocking function.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    blocking_func : Callable
        Function that takes a row and returns a block key or list of block keys
        
    Returns
    -------
    Dict[Any, List[int]]
        Dictionary mapping block keys to lists of record indices
    """
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        try:
            block_key = blocking_func(row)
            if isinstance(block_key, list):
                # Allow a row to be in multiple blocks
                for key in block_key:
                    blocks[key].append(idx)
            else:
                blocks[block_key].append(idx)
        except Exception as e:
            # Handle errors in custom function
            logger.warning(f"Custom blocking function failed for record {idx}: {e}")
            blocks["_error"].append(idx)
    
    return blocks


def get_comparison_pairs_from_blocks(
    blocks: Dict[Any, List[int]],
    max_block_size: int = None,
    max_comparisons: int = None
) -> List[Tuple[int, int]]:
    """
    Extract pairs to compare from blocks.
    
    Parameters
    ----------
    blocks : Dict[Any, List[int]]
        Dictionary mapping block keys to lists of record indices
    max_block_size : int, default=None
        Maximum number of records in a block before sampling
    max_comparisons : int, default=None
        Maximum total number of pairs to return
        
    Returns
    -------
    List[Tuple[int, int]]
        List of index pairs to compare
    """
    pairs = []
    total_pairs = 0
    
    # Calculate total potential pairs
    for indices in blocks.values():
        block_size = len(indices)
        block_pairs = (block_size * (block_size - 1)) // 2
        total_pairs += block_pairs
    
    # Determine sampling strategy if needed
    sampling_ratio = 1.0
    if max_comparisons and total_pairs > max_comparisons:
        sampling_ratio = max_comparisons / total_pairs
        logger.info(f"Will sample approximately {sampling_ratio:.1%} of pairs to limit to {max_comparisons:,} comparisons")
    
    # Process each block
    for block_key, indices in blocks.items():
        block_indices = indices
        
        # If max_block_size is specified, sample or split large blocks
        if max_block_size and len(indices) > max_block_size:
            logger.info(f"Block {block_key} has {len(indices)} records, limiting to {max_block_size}")
            block_indices = np.random.choice(indices, max_block_size, replace=False)
        
        # Generate pairs within the block, with possible sampling
        if sampling_ratio < 1.0:
            # Probabilistic sampling
            for i, idx1 in enumerate(block_indices):
                for idx2 in block_indices[i+1:]:
                    if np.random.random() <= sampling_ratio:
                        pairs.append((idx1, idx2))
        else:
            # No sampling needed
            for i, idx1 in enumerate(block_indices):
                for idx2 in block_indices[i+1:]:
                    pairs.append((idx1, idx2))
    
    # Final limit if we still have too many pairs
    if max_comparisons and len(pairs) > max_comparisons:
        np.random.shuffle(pairs)
        pairs = pairs[:max_comparisons]
    
    return pairs


def apply_blocking_strategy(
    df: Any,
    strategy: str = 'exact',
    blocking_columns: List[str] = None,
    blocking_rules: Dict[str, Callable] = None,
    phonetic_algorithm: str = None,
    ngram_size: int = 2,
    ngram_count: int = 1,
    max_block_size: int = 1000,
    max_comparisons: int = None
) -> List[Tuple[int, int]]:
    """
    Apply the specified blocking strategy and return pairs to compare.
    
    Parameters
    ----------
    df : Any
        The dataframe to process
    strategy : str, default='exact'
        Blocking strategy: 'exact', 'phonetic', 'ngram', or 'rule'
    blocking_columns : List[str], default=None
        Columns to use for blocking
    blocking_rules : Dict[str, Callable], default=None
        Dictionary mapping rule names to functions that generate blocking keys
    phonetic_algorithm : str, default=None
        Phonetic algorithm to use: 'soundex', 'metaphone', or 'nysiis'
        If None, uses first character blocking
    ngram_size : int, default=2
        Size of n-grams for 'ngram' strategy
    ngram_count : int, default=1
        Number of n-grams to use per record for 'ngram' strategy
    max_block_size : int, default=1000
        Maximum number of records in a block before sampling
    max_comparisons : int, default=None
        Maximum total number of pairs to return
        
    Returns
    -------
    List[Tuple[int, int]]
        List of index pairs to compare
    """
    if strategy == 'exact' and blocking_columns:
        logger.info(f"Using exact blocking on columns: {blocking_columns}")
        blocks = create_exact_blocks(df, blocking_columns)
    
    elif strategy == 'phonetic' and blocking_columns:
        # Get the phonetic function based on algorithm name
        phonetic_func = None
        if phonetic_algorithm:
            try:
                import jellyfish
                if phonetic_algorithm.lower() == 'soundex':
                    phonetic_func = jellyfish.soundex
                elif phonetic_algorithm.lower() == 'metaphone':
                    phonetic_func = jellyfish.metaphone
                elif phonetic_algorithm.lower() == 'nysiis':
                    phonetic_func = jellyfish.nysiis
                else:
                    logger.warning(f"Unknown phonetic algorithm: {phonetic_algorithm}, using first character")
            except ImportError:
                logger.warning("Jellyfish library not found, using first character for phonetic blocking")
        
        # Apply phonetic blocking to each column and combine
        logger.info(f"Using phonetic blocking on columns: {blocking_columns}")
        all_blocks = {}
        for col in blocking_columns:
            col_blocks = create_phonetic_blocks(df, col, phonetic_func)
            # Combine with existing blocks
            for key, indices in col_blocks.items():
                block_key = (col, key)
                all_blocks[block_key] = indices
        
        blocks = all_blocks
    
    elif strategy == 'ngram' and blocking_columns:
        logger.info(f"Using {ngram_size}-gram blocking on columns: {blocking_columns}")
        all_blocks = {}
        for col in blocking_columns:
            col_blocks = create_ngram_blocks(df, col, ngram_size, ngram_count)
            # Combine with existing blocks
            for key, indices in col_blocks.items():
                block_key = (col, key) if not isinstance(key, tuple) else (col,) + key
                all_blocks[block_key] = indices
        
        blocks = all_blocks
    
    elif strategy == 'rule' and blocking_rules:
        # Apply each custom rule and combine blocks
        logger.info(f"Using rule-based blocking with rules: {list(blocking_rules.keys())}")
        all_blocks = {}
        for rule_name, rule_func in blocking_rules.items():
            rule_blocks = create_custom_blocks(df, rule_func)
            # Add rule name to block key to avoid collisions
            for key, indices in rule_blocks.items():
                block_key = (rule_name, key)
                all_blocks[block_key] = indices
        
        blocks = all_blocks
    
    else:
        raise ValueError(f"Invalid blocking configuration. Strategy '{strategy}' with provided parameters is not supported.")
    
    # Log blocking statistics
    n_blocks = len(blocks)
    n_blocks_with_multiple = sum(1 for indices in blocks.values() if len(indices) > 1)
    total_records_in_blocks = sum(len(indices) for indices in blocks.values())
    largest_block = max((len(indices), key) for key, indices in blocks.items())
    
    logger.info(f"Created {n_blocks} blocks, {n_blocks_with_multiple} with multiple records")
    logger.info(f"Total records in blocks: {total_records_in_blocks}")
    logger.info(f"Largest block has {largest_block[0]} records with key {largest_block[1]}")
    
    # Generate comparison pairs from blocks
    pairs = get_comparison_pairs_from_blocks(blocks, max_block_size, max_comparisons)
    logger.info(f"Generated {len(pairs):,} comparison pairs from blocks")
    
    return pairs