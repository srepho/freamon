# Optimized Topic Modeling

This guide explains how to use the optimized topic modeling functionality in the Freamon package to efficiently analyze large text collections.

## Overview

The optimized topic modeling pipeline provides a streamlined way to:

1. Preprocess text data with configurable options
2. Remove duplicates (exact or fuzzy)
3. Intelligently sample very large datasets
4. Create topic models using LDA or NMF
5. Map topics back to the full dataset
6. Use parallel processing for better performance

## Basic Usage

Here's a minimal example:

```python
from freamon.utils.text_utils import create_topic_model_optimized
import pandas as pd

# Create your DataFrame with a text column
df = pd.DataFrame({"text": ["Your documents here..."], "category": ["example"]})

# Run topic modeling with default settings
result = create_topic_model_optimized(
    df,
    text_column="text",
    n_topics=5,
    method="nmf"  # or "lda"
)

# Access the results
topics = result["topics"]
doc_topics = result["document_topics"]
info = result["processing_info"]

# Print top words for each topic
for topic_idx, words in topics:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
```

## Advanced Configuration

### Preprocessing Options

Control how texts are preprocessed:

```python
preprocessing_options = {
    'enabled': True,               # Set to False to skip preprocessing
    'use_lemmatization': True,     # Use lemmatization (requires spaCy)
    'remove_stopwords': True,      # Remove common stopwords 
    'remove_punctuation': True,    # Remove punctuation
    'min_token_length': 3,         # Minimum token length to keep
    'custom_stopwords': ['said'],  # Additional stopwords
    'batch_size': 1000             # Batch size for processing
}

result = create_topic_model_optimized(
    df,
    text_column="text",
    n_topics=5,
    preprocessing_options=preprocessing_options
)
```

### Deduplication Options

Configure how duplicates are handled:

```python
deduplication_options = {
    'enabled': True,               # Set to False to keep all documents
    'method': 'exact',             # 'exact', 'fuzzy', or 'none'
    'hash_method': 'hash',         # For exact: 'hash' or 'ngram'
    'similarity_threshold': 0.85,  # For fuzzy: similarity threshold
    'similarity_method': 'cosine', # For fuzzy: 'cosine', 'jaccard', 'levenshtein'
    'keep': 'first'                # Which duplicate to keep: 'first' or 'last'
}

result = create_topic_model_optimized(
    df,
    text_column="text",
    n_topics=5,
    deduplication_options=deduplication_options,
    return_original_mapping=True  # Get mapping from deduplicated docs to originals
)

# Access deduplication mapping
if 'deduplication_map' in result:
    dedup_map = result['deduplication_map']
    # dedup_map maps kept indices to lists of all duplicate indices
```

### Large Dataset Handling

For large datasets (up to 100K documents):

```python
result = create_topic_model_optimized(
    large_df,
    text_column="text",
    n_topics=5,
    max_docs=25000,           # Maximum docs to use for model training
    return_full_data=True,    # Apply model to all documents
    use_multiprocessing=True  # Use parallel processing
)
```

## Adding Topics to Your DataFrame

Add dominant topics to your original DataFrame:

```python
# Get document-topic distributions
doc_topics_df = result["document_topics"]

# Find dominant topic for each document
topic_cols = [col for col in doc_topics_df.columns if col.startswith('Topic')]

# Get dominant topic and probability
def get_dominant_topic(row):
    max_topic = row[topic_cols].idxmax()
    max_prob = row[max_topic]
    return int(max_topic.split()[1]), max_prob

# Apply to get dominant topics and probabilities
dominant_topics = pd.DataFrame(
    doc_topics_df.apply(get_dominant_topic, axis=1).tolist(),
    index=doc_topics_df.index,
    columns=['dominant_topic', 'topic_probability']
)

# Add to original DataFrame
enhanced_df = df.copy()
enhanced_df['dominant_topic'] = dominant_topics['dominant_topic'].reindex(df.index)
enhanced_df['topic_probability'] = dominant_topics['topic_probability'].reindex(df.index)
```

## Processing Information

The result includes detailed processing statistics:

```python
info = result["processing_info"]
print(f"Original documents: {info['original_doc_count']}")
print(f"Duplicates removed: {info['duplicates_removed']}")
print(f"Processed documents: {info['processed_doc_count']}")
print(f"Sample size: {info['sample_size']}")
print(f"Preprocessing time: {info.get('preprocessing_time', 'N/A')}")
print(f"Multiprocessing: {info.get('multiprocessing_enabled', False)}")
```

## Performance Considerations

- For datasets under 10K documents: all documents are used
- For datasets 10K-100K: automatic sampling with `max_docs` parameter
- Multiprocessing is enabled automatically for datasets >10K documents 
- Batch processing is used throughout for memory efficiency
- Deduplication significantly improves performance for redundant data

## Complete Example

See the examples directory for complete examples:
- `optimized_topic_modeling_example.py`: Basic usage
- `dataframe_topic_modeling_example.py`: DataFrame integration