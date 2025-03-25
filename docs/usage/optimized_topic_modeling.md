# Optimized Topic Modeling

The freamon package provides enhanced topic modeling capabilities designed for large datasets with configurable preprocessing and deduplication. This guide explains how to use the optimized topic modeling functionality to analyze text data effectively.

## Overview

The `create_topic_model_optimized` function in the `freamon.utils.text_utils` module provides a comprehensive solution for topic modeling that handles:

- Configurable text preprocessing
- Automatic deduplication (exact or fuzzy)
- Smart sampling for very large datasets
- Parallel processing for performance
- Full dataset coverage with efficient batch processing
- Mapping between deduplicated and original documents

## Basic Usage

```python
import pandas as pd
from freamon.utils.text_utils import create_topic_model_optimized

# Prepare your data
df = pd.DataFrame({
    'text': ['text document 1', 'text document 2', ...],
    'category': ['cat1', 'cat2', ...]
})

# Run optimized topic modeling
result = create_topic_model_optimized(
    df, 
    text_column='text',
    n_topics=5,
    method='nmf',
    preprocessing_options={'enabled': True},
    deduplication_options={'enabled': True},
    return_full_data=True
)

# Access the results
topics = result['topics']
document_topics = result['document_topics']
topic_model = result['topic_model']
processing_info = result['processing_info']

# Print the topics
for topic_idx, words in topics:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
```

## Configuration Options

### Preprocessing Options

The `preprocessing_options` parameter allows fine-grained control over text preprocessing:

```python
preprocessing_options = {
    'enabled': True,               # Whether to perform preprocessing
    'use_lemmatization': True,     # Whether to use lemmatization
    'remove_stopwords': True,      # Whether to remove stopwords
    'remove_punctuation': True,    # Whether to remove punctuation
    'min_token_length': 3,         # Minimum token length to keep
    'custom_stopwords': ['said'],  # Additional stopwords to remove
    'batch_size': 1000             # Batch size for preprocessing
}
```

### Deduplication Options

The `deduplication_options` parameter controls how duplicate documents are handled:

```python
deduplication_options = {
    'enabled': True,              # Whether to deduplicate
    'method': 'exact',            # 'exact', 'fuzzy', or 'none'
    'hash_method': 'hash',        # 'hash' or 'ngram' (for exact)
    'similarity_threshold': 0.85, # Threshold for fuzzy deduplication
    'similarity_method': 'cosine', # 'cosine', 'jaccard', 'levenshtein'
    'keep': 'first'               # 'first' or 'last' document to keep
}
```

### Other Parameters

- `n_topics`: Number of topics to extract (default: 5)
- `method`: Topic modeling method ('nmf' or 'lda', default: 'nmf')
- `max_docs`: Maximum number of documents to process for topic modeling (default: auto)
- `return_full_data`: Whether to return topic distributions for all documents (default: True)
- `return_original_mapping`: Whether to return mapping from deduplicated to original documents (default: False)
- `use_multiprocessing`: Whether to use multiprocessing for text preprocessing (default: True)

## Return Value

The function returns a dictionary with the following keys:

- `topic_model`: Dictionary with the trained model and topics
- `document_topics`: DataFrame with document-topic distributions
- `topics`: List of (topic_idx, words) tuples
- `processing_info`: Dict with processing statistics
- `deduplication_map`: Dict mapping deduplicated to original indices (if requested)

## Working with Large Datasets

The optimized topic modeling functionality is designed to handle large datasets efficiently:

```python
# For a large dataset (e.g., 100,000+ documents)
result = create_topic_model_optimized(
    large_df,
    text_column='text',
    n_topics=10,
    method='nmf',
    preprocessing_options={
        'enabled': True,
        'use_lemmatization': False,  # Disable for speed
        'batch_size': 5000           # Larger batches for efficiency
    },
    max_docs=25000,  # Sample 25K docs for model building
    deduplication_options={
        'enabled': True,
        'method': 'exact'  # Faster than fuzzy
    },
    return_full_data=True,  # Get topics for all documents
    use_multiprocessing=True
)
```

## Integrating with DataFrames

You can easily add topic distributions back to your original DataFrame:

```python
# Get document-topic distribution
doc_topics = result['document_topics']

# Add topics to the original DataFrame
df_with_topics = df.copy()
for col in doc_topics.columns:
    df_with_topics[col] = doc_topics[col]

# Find dominant topic for each document
import numpy as np

def get_dominant_topic(row, topic_cols):
    topic_values = [row[col] for col in topic_cols]
    if all(x == 0 for x in topic_values):
        return -1  # No dominant topic
    return np.argmax(topic_values) + 1  # 1-based indexing

# Add dominant topic column
topic_columns = doc_topics.columns.tolist()
df_with_topics['dominant_topic'] = df_with_topics.apply(
    lambda row: get_dominant_topic(row, topic_columns), axis=1
)
```

## Performance Considerations

- **Lemmatization**: Enabling lemmatization significantly improves topic quality but is much slower. For very large datasets, consider disabling it.
- **Deduplication**: Exact deduplication is much faster than fuzzy deduplication. Use fuzzy only when needed.
- **Multiprocessing**: Recommended for datasets with more than 10,000 documents.
- **Sampling**: For extremely large datasets (100K+), the function will automatically sample a subset for model training.

## Example: Topic Distribution Analysis

```python
# Analyze topic distribution by category
category_topic_dist = df_with_topics.groupby('category')[topic_columns].mean()

# Visualize with matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(category_topic_dist.values, cmap='viridis', aspect='auto')
plt.colorbar(label='Topic Probability')
plt.xticks(range(len(topic_columns)), topic_columns, rotation=45, ha='right')
plt.yticks(range(len(category_topic_dist.index)), category_topic_dist.index)
plt.xlabel('Topics')
plt.ylabel('Categories')
plt.title('Topic Distribution by Category')
plt.tight_layout()
plt.savefig('category_topic_distribution.png')
```

## Advanced: Fuzzy Deduplication

For cases where exact matching is not sufficient, fuzzy deduplication can identify similar documents:

```python
result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    preprocessing_options={'enabled': True},
    deduplication_options={
        'enabled': True,
        'method': 'fuzzy',
        'similarity_threshold': 0.8,  # Documents with similarity >= 0.8 are considered duplicates
        'similarity_method': 'cosine'  # 'cosine', 'jaccard', or 'levenshtein'
    },
    return_original_mapping=True  # Get mapping between duplicates
)

# Access the deduplication mapping
dedup_map = result['deduplication_map']

# Example: For each kept document, show its duplicates
for kept_idx, duplicates in dedup_map.items():
    if len(duplicates) > 1:  # Has duplicates
        print(f"Document {kept_idx} has {len(duplicates)-1} duplicates: {duplicates}")
```

## Complete Example

For a complete working example, see the example scripts in the package:

- `examples/optimized_topic_modeling_example.py`: Basic usage
- `examples/dataframe_topic_modeling_example.py`: Advanced DataFrame integration