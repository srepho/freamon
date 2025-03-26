# Implementation Plan for LSH and Blocking in flag_similar_records

## Overview

This document outlines the steps to integrate blocking and locality-sensitive hashing (LSH) features into the existing `flag_similar_records` function in the Freamon package.

## Phase 1: Code Organization and Dependencies

### Step 1: Add Required Dependencies
- Add `datasketch` to package dependencies in `pyproject.toml` and `setup.py`
- Consider adding other useful libraries like `python-Levenshtein` for string distance

### Step 2: Create New Modules
- Create `freamon/deduplication/blocking.py` for blocking strategies
- Create `freamon/deduplication/lsh.py` for LSH implementations
- Update `freamon/deduplication/__init__.py` to expose new modules

## Phase 2: Blocking Implementation

### Step 1: Implement Core Blocking Functions
```python
# In blocking.py
from typing import List, Dict, Any, Callable, Tuple, Set
from collections import defaultdict

def create_exact_blocks(
    df: Any, 
    blocking_columns: List[str]
) -> Dict[Tuple, List[int]]:
    """Create blocks based on exact matches in blocking columns."""
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        # Create blocking key based on blocking columns
        block_key = tuple(str(row[col]) for col in blocking_columns)
        blocks[block_key].append(idx)
    
    return blocks

def create_phonetic_blocks(
    df: Any, 
    column: str,
    phonetic_func: Callable = None
) -> Dict[str, List[int]]:
    """Create blocks based on phonetic encoding of a column."""
    import jellyfish  # For phonetic encoding
    
    if phonetic_func is None:
        phonetic_func = jellyfish.soundex
        
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        value = row[column]
        if isinstance(value, str):
            try:
                encoded = phonetic_func(value)
                blocks[encoded].append(idx)
            except:
                # If encoding fails, use a fallback key
                blocks["_error"].append(idx)
        else:
            # For non-string values, use string representation
            blocks[str(value)].append(idx)
    
    return blocks

def create_custom_blocks(
    df: Any, 
    blocking_func: Callable
) -> Dict[Any, List[int]]:
    """Create blocks using a custom blocking function."""
    blocks = defaultdict(list)
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        block_key = blocking_func(row)
        if isinstance(block_key, list):
            # Allow a row to be in multiple blocks
            for key in block_key:
                blocks[key].append(idx)
        else:
            blocks[block_key].append(idx)
    
    return blocks

def get_comparison_pairs_from_blocks(
    blocks: Dict[Any, List[int]],
    max_block_size: int = None
) -> List[Tuple[int, int]]:
    """Extract pairs to compare from blocks."""
    pairs = []
    
    for indices in blocks.values():
        # If max_block_size is specified, sample or split large blocks
        if max_block_size and len(indices) > max_block_size:
            import numpy as np
            # Option 1: Random sampling
            indices = np.random.choice(indices, max_block_size, replace=False)
        
        # Generate all pairs within the block
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                pairs.append((idx1, idx2))
    
    return pairs
```

### Step 2: Implement Higher-Level Blocking Interface
```python
# In blocking.py

def apply_blocking_strategy(
    df: Any,
    strategy: str = 'exact',
    blocking_columns: List[str] = None,
    blocking_rules: Dict[str, Callable] = None,
    phonetic_algorithm: str = 'soundex',
    max_block_size: int = 1000
) -> List[Tuple[int, int]]:
    """Apply the specified blocking strategy and return pairs to compare."""
    if strategy == 'exact' and blocking_columns:
        blocks = create_exact_blocks(df, blocking_columns)
    
    elif strategy == 'phonetic' and blocking_columns:
        # Get the phonetic function based on algorithm name
        import jellyfish
        phonetic_func = getattr(jellyfish, phonetic_algorithm.lower(), jellyfish.soundex)
        
        # Apply phonetic blocking to each column and combine
        all_blocks = {}
        for col in blocking_columns:
            col_blocks = create_phonetic_blocks(df, col, phonetic_func)
            # Combine with existing blocks
            for key, indices in col_blocks.items():
                block_key = (col, key)
                all_blocks[block_key] = indices
        
        blocks = all_blocks
    
    elif strategy == 'rule' and blocking_rules:
        # Apply each custom rule and combine blocks
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
    
    # Generate comparison pairs from blocks
    return get_comparison_pairs_from_blocks(blocks, max_block_size)
```

## Phase 3: LSH Implementation

### Step 1: Implement Text-Based LSH
```python
# In lsh.py
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Set
from datasketch import MinHash, MinHashLSH

def create_minhash_signatures(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    num_perm: int = 128,
    tokenizer: Callable = None
) -> Dict[int, MinHash]:
    """Create MinHash signatures for text data."""
    if tokenizer is None:
        tokenizer = lambda x: x.lower().split()
    
    minhashes = {}
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        m = MinHash(num_perm=num_perm)
        
        # Process each column
        for col in columns:
            value = row[col]
            if not isinstance(value, str):
                value = str(value)
            
            # Apply weight by repeating the column content
            weight = weights.get(col, 1.0) if weights else 1.0
            repeat_factor = max(1, int(10 * weight))
            
            # Update MinHash with tokens
            tokens = tokenizer(value)
            for _ in range(repeat_factor):
                for token in tokens:
                    m.update(token.encode('utf8'))
        
        minhashes[idx] = m
    
    return minhashes

def find_similar_pairs_minhash_lsh(
    minhashes: Dict[int, MinHash],
    threshold: float = 0.7,
    num_perm: int = 128
) -> Set[Tuple[int, int]]:
    """Find similar pairs using MinHash LSH."""
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
```

### Step 2: Implement Numerical-Based LSH
```python
# In lsh.py
from typing import List, Dict, Any, Callable, Tuple, Set
import numpy as np

def create_feature_vectors(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    normalize: bool = True
) -> Dict[int, np.ndarray]:
    """Create numerical feature vectors for random projection LSH."""
    # Extract and normalize numerical columns
    vectors = {}
    df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
    
    for idx, row in df_iter:
        # Create feature vector
        vector = []
        for col in columns:
            value = row[col]
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
    num_projections: int = 50
) -> Set[Tuple[int, int]]:
    """Find similar pairs using random projection LSH."""
    if not vectors:
        return set()
    
    # Get dimension of vectors
    dim = len(next(iter(vectors.values())))
    
    # Generate random projection vectors
    projection_vectors = np.random.randn(num_projections, dim)
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
        bands = 10
        rows = num_projections // bands
        
        for band in range(bands):
            # Create a hash for this band
            band_signature = tuple(signature[band * rows: (band + 1) * rows])
            buckets[band, band_signature].append(idx)
    
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
```

### Step 3: Implement Hybrid LSH Interface
```python
# In lsh.py

def apply_lsh_strategy(
    df: Any,
    columns: List[str],
    weights: Dict[str, float] = None,
    lsh_method: str = 'auto',
    threshold: float = 0.7,
    num_perm: int = 128,
    num_bands: int = 20,
    rows_per_band: int = 4
) -> Set[Tuple[int, int]]:
    """Apply LSH strategy to find similar pairs."""
    # Analyze column types to determine appropriate LSH method
    if lsh_method == 'auto':
        # Simple heuristic: if most columns are numeric, use random projection
        numeric_cols = 0
        df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
        
        # Sample a few rows to determine column types
        for _, row in list(df_iter)[:100]:
            for col in columns:
                if isinstance(row[col], (int, float)):
                    numeric_cols += 1
        
        if numeric_cols > len(columns) * 50:  # If >50% numeric values
            lsh_method = 'random_projection'
        else:
            lsh_method = 'minhash'
    
    # Apply the selected LSH method
    if lsh_method == 'minhash':
        minhashes = create_minhash_signatures(
            df, columns, weights, num_perm
        )
        
        similar_pairs = find_similar_pairs_minhash_lsh(
            minhashes, threshold, num_perm
        )
    
    elif lsh_method == 'random_projection':
        vectors = create_feature_vectors(
            df, columns, weights
        )
        
        similar_pairs = find_similar_pairs_random_projection(
            vectors, threshold, num_perm
        )
    
    elif lsh_method == 'hybrid':
        # Split columns by type and apply appropriate LSH method to each
        text_columns = []
        numeric_columns = []
        
        # Determine column types
        df_iter = df.iterrows() if hasattr(df, 'iterrows') else zip(range(len(df)), df)
        for _, row in list(df_iter)[:10]:
            for col in columns:
                if isinstance(row[col], (int, float)):
                    if col not in numeric_columns and col not in text_columns:
                        numeric_columns.append(col)
                else:
                    if col not in text_columns and col not in numeric_columns:
                        text_columns.append(col)
        
        # Apply appropriate LSH to each column type
        text_pairs = set()
        if text_columns:
            text_weights = {col: weights.get(col, 1.0) for col in text_columns} if weights else None
            minhashes = create_minhash_signatures(df, text_columns, text_weights, num_perm)
            text_pairs = find_similar_pairs_minhash_lsh(minhashes, threshold, num_perm)
        
        numeric_pairs = set()
        if numeric_columns:
            numeric_weights = {col: weights.get(col, 1.0) for col in numeric_columns} if weights else None
            vectors = create_feature_vectors(df, numeric_columns, numeric_weights)
            numeric_pairs = find_similar_pairs_random_projection(vectors, threshold, num_perm)
        
        # Combine results
        similar_pairs = text_pairs.union(numeric_pairs)
    
    else:
        raise ValueError(f"Unknown LSH method: {lsh_method}")
    
    return similar_pairs
```

## Phase 4: Integration with flag_similar_records

### Step 1: Modify Function Signature
```python
# In flag_duplicates.py

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
    chunk_size: Optional[int] = None,
    n_jobs: int = 1,
    use_polars: bool = False,
    
    # New parameters
    blocking_columns: Optional[List[str]] = None,
    blocking_method: str = 'exact',
    blocking_rules: Optional[Dict[str, Callable]] = None,
    max_block_size: Optional[int] = None,
    use_lsh: bool = False,
    lsh_method: str = 'auto',
    lsh_threshold: Optional[float] = None,
    num_perm: int = 128,
    num_bands: int = 20,
    rows_per_band: int = 4,
    
    # Legacy parameters
    add_similarity_score: bool = False,
    add_group_id: bool = False,
    group_id_column: Optional[str] = None,
    duplicate_flag_column: Optional[str] = None,
) -> Any:
    """
    Flag similar records based on multiple columns with customizable weights.
    
    Parameters
    ----------
    # ... existing parameters documentation ...
    
    blocking_columns : Optional[List[str]], default=None
        Columns to use for blocking. Records in different blocks won't be compared.
    blocking_method : str, default='exact'
        Method to use for blocking:
        - 'exact': Exact match on blocking columns
        - 'phonetic': Phonetic matching on blocking columns
        - 'rule': Custom blocking rules
    blocking_rules : Optional[Dict[str, Callable]], default=None
        Dictionary mapping rule names to functions that generate blocking keys.
        Required when blocking_method='rule'.
    max_block_size : Optional[int], default=None
        Maximum number of records in a block before sampling or splitting.
    use_lsh : bool, default=False
        Whether to use Locality Sensitive Hashing for faster similarity search.
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
    num_bands : int, default=20
        Number of bands for MinHash LSH.
    rows_per_band : int, default=4
        Number of rows per band for MinHash LSH.
    
    # ... legacy parameters documentation ...
    
    Returns
    -------
    Any
        DataFrame with similar records flagged.
    """
```

### Step 2: Modify Core Implementation Logic
```python
# In flag_duplicates.py

def _flag_similar_records_standard(
    # ... existing parameters ...
    # New parameters for blocking and LSH
    blocking_columns=None,
    blocking_method='exact',
    blocking_rules=None,
    max_block_size=None,
    use_lsh=False,
    lsh_method='auto',
    lsh_threshold=None,
    num_perm=128,
    num_bands=20,
    rows_per_band=4,
):
    # ... existing initialization ...
    
    # Apply optimizations if specified
    comparison_pairs = None
    
    # Step 1: Apply blocking if specified
    if blocking_columns or (blocking_method == 'rule' and blocking_rules):
        from freamon.deduplication.blocking import apply_blocking_strategy
        
        blocking_pairs = apply_blocking_strategy(
            df=pandas_df,
            strategy=blocking_method,
            blocking_columns=blocking_columns,
            blocking_rules=blocking_rules,
            max_block_size=max_block_size
        )
        
        comparison_pairs = blocking_pairs
        print(f"Blocking created {len(comparison_pairs):,} pairs to compare")
    
    # Step 2: Apply LSH if specified
    if use_lsh:
        from freamon.deduplication.lsh import apply_lsh_strategy
        
        # Set LSH threshold if not specified
        if lsh_threshold is None:
            lsh_threshold = threshold * 0.9  # Slightly lower to catch more candidates
        
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
            combined_pairs = lsh_pairs.intersection(comparison_pairs_set)
            comparison_pairs = list(combined_pairs)
            print(f"Combined blocking and LSH: {len(comparison_pairs):,} pairs to compare")
        else:
            comparison_pairs = list(lsh_pairs)
            print(f"LSH identified {len(comparison_pairs):,} potential similar pairs")
    
    # Step 3: Apply max_comparisons limit if specified
    if max_comparisons is not None:
        if comparison_pairs is not None:
            if len(comparison_pairs) > max_comparisons:
                import random
                random.shuffle(comparison_pairs)
                comparison_pairs = comparison_pairs[:max_comparisons]
                print(f"Limited to {max_comparisons:,} pairs due to max_comparisons setting")
        else:
            # If no optimizations were applied, we'll sample from all pairs
            total_pairs = (n_rows * (n_rows - 1)) // 2
            if total_pairs > max_comparisons:
                # Generate all pairs and sample
                all_pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
                import random
                random.shuffle(all_pairs)
                comparison_pairs = all_pairs[:max_comparisons]
                print(f"Sampling {max_comparisons:,} pairs from {total_pairs:,} total pairs")
    
    # If we still don't have pairs, use all pairs or generate them efficiently
    if comparison_pairs is None:
        # Use a generator for memory efficiency
        def generate_pairs():
            for i in range(n_rows):
                for j in range(i+1, n_rows):
                    yield (i, j)
        
        comparison_iterator = generate_pairs()
    else:
        comparison_iterator = iter(comparison_pairs)
    
    # ... continue with existing logic for similarity calculation ...
```

## Phase 5: Testing and Documentation

### Step 1: Create Test Cases
- Create unit tests for each blocking method
- Create unit tests for each LSH method
- Create integration tests for combined approaches
- Test with various dataset sizes and characteristics

### Step 2: Update Documentation
- Update function docstring with new parameters
- Add examples to README showing blocking and LSH usage
- Create example scripts demonstrating performance improvements
- Document parameter selection guidelines

### Step 3: Create Benchmarks
- Create benchmarks comparing different approaches
- Measure accuracy vs. speed tradeoffs
- Document results in README

## Phase 6: Performance Optimization

### Step 1: Profile and Optimize
- Identify and optimize bottlenecks
- Implement more efficient data structures
- Add caching for repeated operations

### Step 2: Add Advanced Features
- Implement adaptive parameter selection
- Add progress reporting for long-running operations
- Implement auto-tuning based on dataset characteristics

## Timeline

1. Basic implementation of blocking and LSH (2 weeks)
2. Integration with existing code (1 week)
3. Testing and validation (1 week)
4. Documentation and examples (1 week)
5. Performance optimization (2 weeks)

Total estimated time: 7 weeks