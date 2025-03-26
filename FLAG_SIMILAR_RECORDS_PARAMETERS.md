# Comprehensive Guide to `flag_similar_records` Function Parameters

The `flag_similar_records` function is a powerful tool in the Freamon library for identifying similar records across multiple columns in a dataset. Here's a detailed explanation of its parameters and how to use them effectively:

## Core Parameters

### `df` (DataFrame, required)
- The input dataframe containing records to check for similarity
- Supports pandas, polars, or dask dataframes
- Example: `df = pd.read_csv('data.csv')`

### `columns` (List[str], required)
- List of column names to consider for similarity calculations
- These are the fields that will be compared between records
- Example: `columns=['name', 'address', 'phone']`

### `weights` (Dict[str, float], optional)
- Dictionary mapping column names to their importance weights
- Higher weights give columns more influence in similarity calculations
- Weights are normalized to sum to 1.0
- Default: Equal weights for all columns
- Example: `weights={'name': 0.6, 'address': 0.3, 'phone': 0.1}`

### `threshold` (float, default=0.8)
- Similarity threshold above which records are considered duplicates
- Range: 0.0 to 1.0 (higher is more strict)
- Example: `threshold=0.75` (more lenient) or `threshold=0.95` (more strict)

### `method` (str, default='composite')
- Algorithm to calculate similarity between records:
  - `'composite'`: Weighted average similarity across all columns
  - `'exact_subset'`: Sum of weights for columns that match exactly
  - `'fuzzy_subset'`: Sum of weights for columns with high similarity (≥0.9)
- Example: `method='fuzzy_subset'`

### `flag_column` (str, default='is_similar')
- Name of the output column that will contain the duplicate flags
- Contains `True` for duplicates, `False` for non-duplicates
- First occurrence in each group is marked as `False`
- Example: `flag_column='is_duplicate'`

### `group_column` (str, optional)
- If provided, adds a column with unique IDs for each duplicate group
- Records in the same group will have the same ID
- Example: `group_column='duplicate_group'`

### `similarity_column` (str, optional)
- If provided, adds a column with the highest similarity score for each record
- Useful for understanding why records were flagged
- Example: `similarity_column='similarity_score'`

### `inplace` (bool, default=False)
- If True, modifies the input dataframe directly
- If False, returns a new dataframe with the added columns
- Example: `inplace=True`

## Performance Optimization Parameters

### `n_jobs` (int, default=1)
- Number of parallel jobs for processing large datasets
- Use -1 to use all available processor cores
- Example: `n_jobs=4`

### `max_comparisons` (int, optional)
- Maximum number of record pairs to compare
- Useful for limiting computation on large datasets
- If set, pairs are randomly sampled
- Example: `max_comparisons=1000000`

### `chunk_size` (int, optional)
- Size of chunks for processing very large datasets
- Enables memory-efficient processing
- Example: `chunk_size=5000`

### `use_polars` (bool, default=False)
- Whether to use polars for faster string operations
- Particularly effective for large text datasets
- Example: `use_polars=True`

### `show_progress` (bool, default=False)
- Whether to display a progress bar during processing
- Example: `show_progress=True`

### `jupyter_mode` (bool, default=False)
- Whether to use Jupyter-friendly progress bars
- Example: `jupyter_mode=True`

## Advanced Optimization Parameters

### `blocking_columns` (List[str], optional)
- Columns used for blocking optimization
- Only compares records within the same block
- Dramatically reduces comparison count for large datasets
- Example: `blocking_columns=['zip_code', 'state']`

### `blocking_method` (str, default='exact')
- Method for blocking:
  - `'exact'`: Exact matching on blocking columns
  - `'phonetic'`: Phonetic matching (soundex, metaphone)
  - `'ngram'`: N-gram-based blocking
  - `'rule'`: Custom blocking rules
- Example: `blocking_method='phonetic'`

### `use_lsh` (bool, default=False)
- Whether to use Locality-Sensitive Hashing for optimization
- Approximates similarity to reduce comparisons
- Example: `use_lsh=True`

### `lsh_method` (str, default='auto')
- LSH implementation to use:
  - `'auto'`: Automatically choose best method
  - `'minhash'`: MinHash LSH for sets/text
  - `'simhash'`: SimHash for text
- Example: `lsh_method='minhash'`

### `lsh_threshold` (float, optional)
- Similarity threshold for LSH optimization
- If not specified, uses 90% of the main threshold
- Example: `lsh_threshold=0.7`

## Examples

### Basic Usage
```python
similar_df = flag_similar_records(
    df,
    columns=['name', 'address', 'phone'],
    threshold=0.8,
    group_column='duplicate_group'
)
```

### With Column Weights
```python
similar_df = flag_similar_records(
    df,
    columns=['name', 'address', 'phone'],
    weights={'name': 0.6, 'address': 0.3, 'phone': 0.1},
    threshold=0.75,
    method='fuzzy_subset'
)
```

### Large Dataset Optimization
```python
similar_df = flag_similar_records(
    df,
    columns=['name', 'address', 'phone'],
    threshold=0.8,
    use_lsh=True,
    blocking_columns=['zip_code'],
    max_comparisons=1000000,
    n_jobs=4,
    show_progress=True
)
```

### With Polars for Performance
```python
similar_df = flag_similar_records(
    df,
    columns=['name', 'address', 'phone'],
    threshold=0.8,
    use_polars=True,
    chunk_size=10000,
    similarity_column='similarity_score'
)
```

This function offers a flexible approach to finding similar records while providing numerous options for performance optimization on large datasets.