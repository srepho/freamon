# LSH Deduplication: Efficient Text Similarity Matching

This document explains Locality-Sensitive Hashing (LSH) deduplication in Freamon, which enables scalable identification of similar texts without comparing all possible pairs.

## Overview

LSH deduplication provides:

1. **Scalability**: Process millions of documents efficiently
2. **Near-Duplicate Detection**: Find similar but not identical texts
3. **Tunable Precision/Recall**: Adjust similarity thresholds for your needs
4. **Memory Efficiency**: Much lower memory footprint than all-pairs comparison

## How LSH Deduplication Works

### The Problem: Similarity Search at Scale

When working with large text collections, finding similar items by comparing every pair becomes computationally prohibitive:
- 10,000 documents require ~50 million comparisons
- 1 million documents require ~500 billion comparisons

LSH solves this by only comparing documents likely to be similar.

### The LSH Process

1. **Shingling**: Convert each document to n-grams (character or word)
   ```
   "hello world" → ["hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"]
   ```

2. **MinHash Signatures**: Create compact numerical representations that preserve similarity
   - Uses multiple hash functions to represent the document
   - Similar documents will have similar signatures
   - Approximates Jaccard similarity (set overlap)

3. **Banding Technique**: Divide signatures into bands and hash them to buckets
   - Documents in the same bucket are candidate pairs
   - Tunable threshold by adjusting bands and rows
   - Only documents that share at least one bucket are compared

4. **Similarity Verification**: Calculate actual similarity for candidate pairs only

5. **Clustering**: Group similar documents into connected components

6. **Deduplication**: Keep one document from each cluster (first, last, or longest)

## Mathematical Foundation

The probability that similar documents will share at least one bucket follows this formula:

P(similar pair shares a bucket) = 1 - (1 - s^r)^b

Where:
- s is the actual similarity between documents
- r is the number of rows per band (signature_size / num_bands)
- b is the number of bands

This creates an S-curve that acts as a similarity threshold function.

## Using LSH Deduplication in Freamon

### Basic Usage

```python
from freamon.deduplication import lsh_deduplication

# Basic usage
unique_indices = lsh_deduplication(
    df['text'],
    threshold=0.7,
    num_minhash_permutations=100,
    num_bands=20
)

# Get unique documents
df_unique = df.iloc[unique_indices].copy()
```

### Key Parameters

- **threshold**: Similarity threshold (0-1). Higher values require more similarity (default=0.7)
- **num_minhash_permutations**: Number of hash functions for signature (default=100)
- **num_bands**: Number of bands to divide signature into (default=20)
- **shingle_size**: Size of n-grams for document representation (default=3)
- **preprocess**: Whether to normalize texts before comparison (default=True)
- **keep**: Which document to keep from each cluster ('first', 'last', 'longest')

### Parameter Relationships

The threshold parameter (t) relates to the number of bands (b) and rows per band (r) with this approximation:

t ≈ (1/b)^(1/r)

For example:
- 100 permutations, 20 bands, 5 rows per band ≈ 0.7 threshold
- 100 permutations, 50 bands, 2 rows per band ≈ 0.5 threshold (more sensitive)
- 100 permutations, 10 bands, 10 rows per band ≈ 0.8 threshold (less sensitive)

### Advanced Usage

```python
# Get both unique indices and similarity information
unique_indices, similarity_dict = lsh_deduplication(
    df['text'],
    threshold=0.8,
    num_minhash_permutations=200,
    num_bands=25,
    shingle_size=4,
    preprocess=True,
    keep='longest',
    return_similarity_dict=True
)

# Check which documents are similar to document #42
similar_docs = similarity_dict.get(42, [])
print(f"Documents similar to #42: {similar_docs}")

# Examine one of the duplicates
doc1 = df.iloc[42]['text']
doc2 = df.iloc[similar_docs[0]]['text']
```

## Custom Text Preprocessing

```python
from freamon.utils.text_utils import TextProcessor

# Create custom text processor
processor = TextProcessor(use_spacy=False)

# Apply in LSH deduplication
unique_indices = lsh_deduplication(
    df['text'],
    preprocess=True,
    text_processor=processor
)
```

## Performance Benchmarking

```python
import time
import matplotlib.pyplot as plt
from freamon.deduplication import (
    hash_deduplication,
    lsh_deduplication,
    deduplicate_texts
)

# Compare methods
methods = {
    "Hash-based (exact matches)": lambda texts: hash_deduplication(texts),
    "LSH (threshold=0.7)": lambda texts: lsh_deduplication(
        texts, threshold=0.7, num_minhash_permutations=100, num_bands=20
    ),
    "LSH (threshold=0.9)": lambda texts: lsh_deduplication(
        texts, threshold=0.9, num_minhash_permutations=100, num_bands=20
    ),
}

# For small datasets only, add exact similarity comparison
if len(df) <= 5000:
    methods["Cosine Similarity"] = lambda texts: deduplicate_texts(
        texts, method='cosine', threshold=0.7
    )

results = {}
for name, method in methods.items():
    start_time = time.time()
    unique_indices = method(df['text'])
    elapsed = time.time() - start_time
    
    results[name] = {
        'time': elapsed,
        'unique_count': len(unique_indices),
        'dedup_count': len(df) - len(unique_indices)
    }
    
    print(f"{name}: {elapsed:.2f}s, {results[name]['dedup_count']} duplicates found")

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(results.keys(), [r['time'] for r in results.values()])
plt.title('Processing Time (seconds)')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
plt.bar(results.keys(), [r['dedup_count'] for r in results.values()])
plt.title('Duplicates Detected')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('deduplication_comparison.png')
```

## Large Dataset Handling

For very large datasets, process in chunks:

```python
from freamon.deduplication import lsh_deduplication
from freamon.utils.dataframe_utils import process_in_chunks

def deduplicate_chunk(chunk_df):
    """Process a single chunk."""
    local_indices = lsh_deduplication(
        chunk_df['text'],
        threshold=0.8,
        num_minhash_permutations=100,
        num_bands=20
    )
    return chunk_df.iloc[local_indices].copy()

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

## Practical Tips

1. **Tuning for Recall vs Precision**:
   - Higher threshold (0.8-0.9): High precision, may miss some duplicates
   - Lower threshold (0.6-0.7): Higher recall, may include false positives

2. **Performance Optimization**:
   - More permutations = more accurate but slower
   - More bands (with fixed permutations) = more sensitive to small similarities
   - Fewer rows per band = more candidate pairs (higher recall, lower precision)

3. **Shingling Choices**:
   - Character n-grams (default): Good for general text, robust to minor variations
   - Word n-grams: Better for detecting semantic similarity in well-formed text
   - Larger shingle size: More specific matching, less sensitive to small changes

4. **Memory Considerations**:
   - MinHash signatures require O(n) memory where n is number of documents
   - Candidate pairs may require significant memory for very similar collections
   - For extreme cases, use chunking as shown above

5. **Preprocessing Recommendations**:
   - Always normalize case and whitespace
   - Consider removing punctuation for general text comparison
   - For specialized text (code, structured data), keep original structure
   - Use custom preprocessing for domain-specific needs

## Comparison with Other Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hash-based** | Extremely fast, minimal memory | Only finds exact duplicates | Simple exact deduplication |
| **LSH** | Scalable, tunable, finds near-duplicates | Requires parameter tuning | Large datasets with textual variations |
| **Cosine Similarity** | Most accurate, no false negatives | O(n²) complexity, high memory | Small datasets, high precision needed |
| **MinHash without LSH** | More accurate than LSH | Less scalable | Medium datasets, accuracy over speed |

## Under the Hood

The LSH implementation uses:
1. Character or word shingling for document representation
2. Multiple hash functions to approximate MinHash signatures
3. Banded Matrix approach for efficient candidate pair generation
4. Connected component analysis to identify duplicate clusters

This makes it possible to efficiently find similar documents in collections of millions of texts without the quadratic complexity of all-pairs comparison.