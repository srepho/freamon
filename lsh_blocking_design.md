# LSH and Blocking Enhancement Design for flag_similar_records

## Overview

This document outlines the design for enhancing the `flag_similar_records` function with blocking and locality-sensitive hashing (LSH) capabilities to significantly improve performance on large datasets.

## Proposed Features

### 1. Blocking Strategy

Blocking is a technique to partition data into smaller groups (blocks) where similar records are likely to be in the same block. Only records within the same block are compared, drastically reducing comparison pairs.

#### Proposed Implementation:

- **Simple Blocking**: Allow users to specify one or more blocking columns where exact matches are required
- **Rule-based Blocking**: Allow users to define custom rules for blocking (e.g., first letter of name + zip code)
- **Fuzzy Blocking**: Create blocks based on similar values (e.g., soundex encoding of names)

### 2. Locality-Sensitive Hashing Integration

LSH approximates similarity search by hashing similar items into the same buckets with high probability.

#### Proposed Implementation:

- **MinHash LSH**: For text and categorical data
- **Random Projection LSH**: For numerical data
- **Hybrid Approach**: Combine different LSH methods for mixed data types

### 3. Function Parameter Extensions

Extend `flag_similar_records` with new parameters:

```python
def flag_similar_records(
    df: Any,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    # ... existing parameters ...
    
    # New parameters:
    blocking_columns: Optional[List[str]] = None,
    blocking_method: str = 'exact',  # 'exact', 'rule', 'fuzzy'
    blocking_rules: Optional[Dict[str, Callable]] = None,
    use_lsh: bool = False,
    lsh_method: str = 'auto',  # 'minhash', 'random_projection', 'auto'
    lsh_bands: int = 20,
    lsh_rows: int = 5,
    lsh_threshold: Optional[float] = None,  # If None, derived from bands/rows
) -> Any:
    # ...
```

## Implementation Plan

### Phase 1: Blocking Implementation

1. Implement exact match blocking
   - Add preprocessing step to partition data by blocking column values
   - Modify comparison logic to only compare within blocks

2. Implement rule-based blocking
   - Allow custom functions to generate blocking keys
   - Support common blocking functions (e.g., first N characters)

3. Implement fuzzy blocking
   - Integrate phonetic algorithms (Soundex, Metaphone)
   - Support n-gram based blocking

### Phase 2: LSH Implementation

1. Integrate MinHash LSH for text data
   - Implement MinHash signature generation
   - Implement banding technique for similarity thresholds

2. Implement Random Projection LSH for numerical data
   - Generate random projection vectors
   - Implement bit sampling

3. Create automatic LSH selection based on column types
   - Analyze column data types and select appropriate LSH method
   - Combine results from different LSH methods

### Phase 3: Integration and Optimization

1. Integrate blocking and LSH with existing chunking mechanism
   - Apply blocking before chunking
   - Optimize memory usage with streaming approach

2. Implement hybrid approach for mixed data types
   - Combine different LSH methods for different columns
   - Weight similarity results based on column weights

3. Optimize parameters based on dataset characteristics
   - Provide automatic parameter tuning based on data size and memory constraints
   - Add progress reporting for long-running operations

## Performance Expectations

| Dataset Size | Method | Expected Speedup | Memory Reduction |
|--------------|--------|------------------|------------------|
| 50k-100k     | Blocking | 5-10x          | 80-90%           |
| 50k-100k     | LSH      | 10-20x         | 60-80%           |
| 50k-100k     | Combined | 20-50x         | 90-95%           |
| 100k-500k    | Blocking | 10-20x         | 90-95%           |
| 100k-500k    | LSH      | 20-50x         | 80-90%           |
| 100k-500k    | Combined | 50-100x        | 95-98%           |
| 500k+        | Blocking | 20-50x         | 95-98%           |
| 500k+        | LSH      | 50-100x        | 90-95%           |
| 500k+        | Combined | 100-1000x      | 98-99%           |

## Example Usage

```python
# Using blocking with exact match
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    blocking_columns=["state", "zipcode_prefix"],
    threshold=0.8
)

# Using LSH for text columns
result = flag_similar_records(
    df,
    columns=["name", "email", "description"],
    use_lsh=True,
    lsh_method='minhash',
    threshold=0.7
)

# Combined approach with custom blocking rules
def year_block(date_str):
    # Extract year from date string
    return date_str.split('-')[0] if '-' in date_str else '0000'

result = flag_similar_records(
    df,
    columns=["name", "address", "phone", "email"],
    blocking_method='rule',
    blocking_rules={
        "registration_date": year_block,
        "name": lambda x: x[0].upper() if x else ''  # First letter
    },
    use_lsh=True,
    threshold=0.8
)
```

## Integration with Existing Features

The new blocking and LSH features will:
- Maintain backward compatibility with all existing parameters
- Work alongside chunk_size and max_comparisons features
- Support both pandas and polars DataFrames
- Maintain the same output format and options