# LSH Text Deduplication in Freamon

This document provides an overview of the Locality-Sensitive Hashing (LSH) deduplication functionality in Freamon.

## What is LSH Deduplication?

Locality-Sensitive Hashing (LSH) is a technique for finding similar items efficiently without needing to compare all possible pairs. For text deduplication, this means:

1. **Scale Efficiently**: Process millions of documents without comparing every possible pair
2. **Find Near-Duplicates**: Identify texts that are similar but not exact duplicates
3. **Customize Sensitivity**: Adjust the similarity threshold to your specific needs

## How LSH Works

1. **MinHash Signatures**: Each document is converted to a compact signature that preserves similarity
2. **Banding Technique**: Signatures are divided into bands and hashed to buckets
3. **Candidate Pairs**: Only documents that share bucket(s) are compared
4. **Similarity Verification**: Actual similarity is computed only for candidate pairs

## Using LSH Deduplication in Freamon

```python
from freamon.deduplication import lsh_deduplication

# Basic usage
unique_indices = lsh_deduplication(
    df['text'],
    threshold=0.7,
    num_minhash_permutations=100,
    num_bands=20
)

# Get deduplicated dataframe
df_unique = df.iloc[unique_indices].copy()
```

### Key Parameters

- **threshold**: Similarity threshold (0-1). Higher values require more similarity (default=0.7)
- **num_minhash_permutations**: Number of permutations for MinHash (default=100)
- **num_bands**: Number of bands for LSH (default=20)
- **keep**: Which duplicate to keep ('first', 'last', or 'longest')
- **return_similarity_dict**: Whether to return similarity information

### Advanced Usage

```python
# Get both unique indices and similarity information
unique_indices, similarity_dict = lsh_deduplication(
    df['text'],
    threshold=0.8,
    num_minhash_permutations=200,
    num_bands=25,
    keep='longest',
    return_similarity_dict=True
)

# See which documents were found similar to document 10
similar_to_ten = similarity_dict.get(10, [])
print(f"Documents similar to #10: {similar_to_ten}")

# Process with text preprocessing
from freamon.utils.text_utils import TextProcessor

processor = TextProcessor(use_spacy=False)
unique_indices = lsh_deduplication(
    df['text'],
    preprocess=True,
    text_processor=processor,
    shingle_size=4  # N-gram size for shingling
)
```

## Performance Considerations

- **Memory Usage**: Scales with number of documents but much better than all-pairs comparison
- **Processing Time**: 
  - Increases with number of documents
  - Increases with num_minhash_permutations
  - Generally much faster than all-pairs comparisons for large datasets
- **Accuracy Tradeoffs**:
  - More MinHash permutations = more accurate but slower
  - More bands = higher recall but lower precision

## Comparison with Other Methods

| Method | Best For | Advantages | Limitations |
|--------|----------|------------|------------|
| Hash deduplication | Exact duplicates | Fastest, lowest memory | Misses near-duplicates |
| N-gram fingerprinting | Slight variations | Fast, robust to small changes | Less sensitive than LSH |
| LSH deduplication | Near-duplicates at scale | Scalable similarity search | Parameter tuning needed |
| All-pairs similarity | Highest precision | Most accurate | Does not scale to large datasets |

## Example: Benchmarking LSH Performance

```python
import pandas as pd
import time
from freamon.deduplication import lsh_deduplication, hash_deduplication, deduplicate_texts

# Load your data
df = pd.read_csv("your_texts.csv")

# Test LSH with different parameters
start_time = time.time()
lsh_results = lsh_deduplication(
    df['text'],
    threshold=0.7,
    num_minhash_permutations=100,
    num_bands=20
)
lsh_time = time.time() - start_time
print(f"LSH found {len(lsh_results)} unique texts in {lsh_time:.2f} seconds")

# Compare with hash-based deduplication (exact matches only)
start_time = time.time()
hash_results = hash_deduplication(df['text'])
hash_time = time.time() - start_time
print(f"Hash found {len(hash_results)} unique texts in {hash_time:.2f} seconds")

# Compare with all-pairs similarity (small sample only)
sample_size = min(5000, len(df))
sample_df = df.sample(sample_size)

start_time = time.time()
similarity_results = deduplicate_texts(
    sample_df['text'],
    method='cosine',
    threshold=0.7
)
similarity_time = time.time() - start_time
print(f"All-pairs found {len(similarity_results)} unique texts in {similarity_time:.2f} seconds")
```

## Large Dataset Example

For very large datasets, use chunking to process efficiently:

```python
from freamon.deduplication import lsh_deduplication
from freamon.utils.dataframe_utils import process_in_chunks

def deduplicate_chunk(chunk_df):
    """Process a single chunk and return unique indices."""
    local_indices = lsh_deduplication(
        chunk_df['text'],
        threshold=0.8,
        num_minhash_permutations=100,
        num_bands=20
    )
    return chunk_df.iloc[local_indices][['id', 'text']].copy()

# Process in chunks
results = process_in_chunks(
    df,
    chunk_size=10000,
    process_func=deduplicate_chunk,
    combine=True
)

# Final deduplication on combined results
final_indices = lsh_deduplication(
    results['text'],
    threshold=0.8
)

df_unique = results.iloc[final_indices].copy()
```

## Integration with Pipeline

```python
from freamon.pipeline import Pipeline, TextDeduplicationStep

# Create pipeline with deduplication
pipeline = Pipeline()
pipeline.add_step(
    TextDeduplicationStep(
        name="lsh_deduplication",
        method="lsh",
        threshold=0.7,
        text_column="description",
        keep="longest"
    )
)

# Process data through pipeline
result = pipeline.transform(df)
```