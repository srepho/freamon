# Using Locality-Sensitive Hashing (LSH) with flag_similar_records

This guide explains how to use Locality-Sensitive Hashing (LSH) optimization with the `flag_similar_records` function to efficiently identify similar records in large datasets.

## What is LSH?

Locality-Sensitive Hashing (LSH) is a technique for finding similar items efficiently, without needing to compare all possible pairs. It works by:

1. **Hashing similar items to the same buckets** - Using special hash functions that preserve similarity
2. **Comparing only potential matches** - Only comparing records that share the same hash buckets
3. **Dramatically reducing total comparisons** - Making the similarity search scalable for large datasets

For duplicate detection, LSH can speed up the process by 10-1000x, depending on dataset size and characteristics.

## When to Use LSH

You should consider using LSH optimization when:

- Your dataset has **more than 20,000 records**
- You're facing performance or memory issues with standard pairwise comparison
- You need to find similar text records efficiently
- You're working with high-dimensional numerical data

## Basic Usage with LSH

```python
import pandas as pd
from freamon.deduplication.flag_duplicates import flag_similar_records

# Create or load your dataset
df = pd.read_csv("large_dataset.csv")

# Use flag_similar_records with LSH optimization
result = flag_similar_records(
    df,
    columns=["name", "address", "description"],
    threshold=0.8,
    use_lsh=True,  # Enable LSH optimization
    lsh_method="auto",  # Automatically choose the best LSH method
    lsh_threshold=0.7  # Pre-filtering threshold (can be lower than main threshold)
)

# View results
print(f"Found {result['is_similar'].sum()} similar records")
```

## LSH Parameters

### Key LSH Parameters

| Parameter | Description | Default | Recommended Settings |
|-----------|-------------|---------|---------------------|
| `use_lsh` | Enable LSH optimization | `False` | `True` for large datasets |
| `lsh_method` | LSH implementation to use | `"auto"` | `"auto"`, `"minhash"`, `"simhash"` |
| `lsh_threshold` | Similarity threshold for LSH pre-filtering | `None` (90% of main threshold) | `0.6`-`0.7` - lower than main threshold |

### Additional LSH-Related Parameters

These parameters can be used for advanced tuning:

```python
result = flag_similar_records(
    df,
    columns=["name", "address", "description"],
    threshold=0.8,
    use_lsh=True,
    lsh_method="minhash",
    lsh_threshold=0.7,
    
    # Advanced LSH parameters:
    lsh_num_permutations=128,  # Number of permutations for MinHash
    lsh_bands=20,              # Number of bands for LSH banding
    lsh_rows=5                 # Number of rows per band
)
```

## LSH Methods

### Automatic Method Selection (`"auto"`)

When `lsh_method="auto"` (the default when using LSH), the function automatically selects the best LSH method based on your data:

- For text-heavy columns: Uses MinHash or SimHash
- For numerical data: Uses Random Projection LSH
- For mixed data types: Uses a hybrid approach

### MinHash (`"minhash"`)

MinHash is excellent for text data and uses a technique called shingling to capture text similarity:

```python
# Optimized for text data
result = flag_similar_records(
    df,
    columns=["product_description", "customer_review"],
    threshold=0.8,
    use_lsh=True,
    lsh_method="minhash",
    lsh_threshold=0.7
)
```

### SimHash (`"simhash"`)

SimHash is well-suited for text data where term frequency is important:

```python
# Optimized for text documents
result = flag_similar_records(
    df,
    columns=["document_text"],
    threshold=0.8,
    use_lsh=True,
    lsh_method="simhash",
    lsh_threshold=0.7
)
```

## Combining LSH with Blocking

For the best performance on very large datasets, you can combine LSH with blocking:

```python
# Using both LSH and blocking for maximum performance
result = flag_similar_records(
    df,
    columns=["name", "address", "description"],
    threshold=0.8,
    
    # LSH parameters
    use_lsh=True,
    lsh_method="minhash",
    lsh_threshold=0.7,
    
    # Blocking parameters
    blocking_columns=["state", "zip_code_prefix"],
    blocking_method="exact"
)
```

This combination can provide incredible performance gains, reducing comparisons by 90-99.9% compared to the standard approach.

## Performance Comparison

Real-world performance comparison for different methods (on a dataset with 100,000 records):

| Method | Comparisons | Runtime | Memory Usage |
|--------|-------------|---------|-------------|
| Standard (All pairs) | 4.99 billion | Hours | High |
| Blocking only | 50 million | Minutes | Medium |
| LSH only | 5 million | Minutes | Medium |
| LSH + Blocking | 500,000 | Seconds | Low |

## Advanced LSH Examples

### Example 1: Mixed Data Types

```python
# Dataset with mixed data types
result = flag_similar_records(
    df,
    columns=["name", "age", "description"],
    weights={"name": 0.5, "age": 0.1, "description": 0.4},
    threshold=0.8,
    use_lsh=True,
    lsh_method="auto"  # Will use hybrid approach for mixed types
)
```

### Example 2: Fine-tuning LSH Parameters

```python
# Fine-tuning LSH parameters for better recall
result = flag_similar_records(
    df,
    columns=["product_name", "description"],
    threshold=0.8,
    use_lsh=True,
    lsh_method="minhash",
    lsh_threshold=0.65,  # Lower threshold captures more potential matches
    lsh_num_permutations=256,  # More permutations = higher accuracy but slower
    lsh_bands=32,  # More bands = higher recall
    lsh_rows=4  # Fewer rows per band = higher recall
)
```

### Example 3: Very Large Dataset

```python
# For very large datasets (millions of records)
result = flag_similar_records(
    df,
    columns=["name", "address", "phone"],
    threshold=0.8,
    
    # LSH parameters
    use_lsh=True,
    lsh_method="minhash",
    lsh_threshold=0.7,
    
    # Chunking for memory efficiency
    chunk_size=10000,
    
    # Limit total comparisons
    max_comparisons=5000000,
    
    # Parallelization
    n_jobs=4,
    
    # Progress tracking
    show_progress=True
)
```

## LSH Parameter Tuning Tips

For optimal results with LSH:

1. **Start with defaults** - The default settings work well in most cases
2. **Adjust LSH threshold** - Lower values increase recall but may reduce precision
3. **Tune banding parameters** - More bands with fewer rows increases recall
4. **Consider dataset characteristics** - Text-heavy data works best with MinHash or SimHash

## Handling LSH Failure Cases

If LSH optimization isn't finding enough similar records:

1. **Lower the LSH threshold** - Try `lsh_threshold=0.6` or even `0.5`
2. **Increase permutations** - More permutations (128-256) improve accuracy
3. **Adjust banding** - More bands with fewer rows per band increases recall
4. **Fallback strategy** - The function automatically falls back to standard comparison if LSH finds no candidates

## Conclusion

LSH optimization allows `flag_similar_records` to efficiently handle large datasets that would be impractical with standard pairwise comparison. By using LSH, you can identify similar records in large datasets with dramatically better performance and memory efficiency.

For more details, refer to:
- `FLAG_SIMILAR_RECORDS_GUIDE.md` - Complete guide to all function features
- `FLAG_SIMILAR_RECORDS_PARAMETERS.md` - Detailed parameter documentation
- Example files in the `examples/` directory