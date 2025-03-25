# Optimized Topic Modeling

The Freamon package includes an optimized topic modeling workflow that supports enhanced text preprocessing, automatic deduplication, and intelligent sampling for large datasets. This module is designed for efficient text analysis with a focus on performance and flexibility.

## Features

- **Configurable text preprocessing options**: Customize how texts are cleaned and normalized
- **Smart automatic deduplication**: Remove duplicate or near-duplicate documents
- **Intelligent sampling for large datasets**: Process very large datasets efficiently
- **Parallel processing support**: Leverage multiple cores for faster processing
- **Full dataset coverage**: Apply topic modeling to the entire dataset efficiently
- **PII anonymization**: Optional anonymization of personally identifiable information

## Basic Usage

```python
from freamon.utils.text_utils import create_topic_model_optimized
import pandas as pd

# Prepare your data
df = pd.DataFrame({
    'text': ['Document 1 about science', 'Document 2 about art', ...],
    'category': ['science', 'art', ...]
})

# Run the optimized topic modeling
result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf'  # 'nmf' or 'lda'
)

# Access the results
topic_model = result['topic_model']  # The trained model
document_topics = result['document_topics']  # Document-topic distributions
topics = result['topics']  # List of topics with their top words
```

## Advanced Configuration

### Preprocessing Options

```python
preprocessing_options = {
    'enabled': True,  # Whether to preprocess texts
    'use_lemmatization': True,  # Use spaCy lemmatization
    'remove_stopwords': True,  # Remove common stopwords
    'remove_punctuation': True,  # Remove punctuation
    'min_token_length': 3,  # Minimum token length to keep
    'custom_stopwords': ['custom', 'words'],  # Additional stopwords
    'batch_size': 1000  # Batch size for processing (auto-calculated if None)
}

result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    preprocessing_options=preprocessing_options
)
```

### Deduplication Options

```python
deduplication_options = {
    'enabled': True,  # Whether to deduplicate texts
    'method': 'exact',  # 'exact', 'fuzzy', or 'none'
    'hash_method': 'hash',  # 'hash' or 'ngram'
    'similarity_threshold': 0.85,  # For fuzzy deduplication
    'similarity_method': 'cosine',  # 'cosine', 'jaccard', 'levenshtein'
    'keep': 'first'  # 'first' or 'last' duplicate to keep
}

result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    deduplication_options=deduplication_options
)
```

### PII Anonymization

```python
# Basic anonymization with default settings
result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    anonymize=True  # Enable anonymization
)

# Customized anonymization with configuration
anonymization_config = {
    'use_all_patterns': True,  # Use all available PII patterns
    'custom_patterns': {
        'CUSTOM_ENTITY': r'your-regex-pattern'
    },
    'replacement_strategy': 'entity_type'  # Replace with entity type label
}

result = create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    anonymize=True,
    anonymization_config=anonymization_config
)
```

## Complete Example

```python
from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('your_data.csv')

# Configure options
preprocessing_options = {
    'enabled': True,
    'use_lemmatization': True,
    'remove_stopwords': True,
    'custom_stopwords': ['specific', 'terms', 'to', 'exclude']
}

deduplication_options = {
    'enabled': True,
    'method': 'fuzzy',
    'similarity_threshold': 0.85,
    'similarity_method': 'cosine'
}

anonymization_config = {
    'use_all_patterns': True
}

# Run the optimized topic modeling
result = create_topic_model_optimized(
    df, 
    text_column='text',
    n_topics=8,
    method='nmf',
    preprocessing_options=preprocessing_options,
    deduplication_options=deduplication_options,
    max_docs=None,  # Process all documents
    anonymize=True,
    anonymization_config=anonymization_config,
    return_full_data=True,
    return_original_mapping=True,
    use_multiprocessing=True
)

# Print processing information
info = result['processing_info']
print(f"Original documents: {info['original_doc_count']}")
print(f"Duplicates removed: {info['duplicates_removed']}")
print(f"Anonymization enabled: {info['anonymization_enabled']}")
if 'anonymization_time' in info:
    print(f"Anonymization time: {info['anonymization_time']:.2f} seconds")

# Print topics
print("\nTop 10 words for each topic:")
for topic_idx, words in result['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Visualize topic distributions
doc_topics = result['document_topics']

# Add category if available
if 'category' in df.columns:
    doc_topics_with_category = doc_topics.copy()
    doc_topics_with_category['category'] = df.loc[doc_topics.index, 'category'].values
    
    # Calculate average topic distribution by category
    category_topic_dist = doc_topics_with_category.groupby('category').mean()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(category_topic_dist.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Topic Probability')
    plt.xticks(range(len(doc_topics.columns)), doc_topics.columns, rotation=45, ha='right')
    plt.yticks(range(len(category_topic_dist.index)), category_topic_dist.index)
    plt.xlabel('Topics')
    plt.ylabel('Categories')
    plt.title('Topic Distribution by Category')
    plt.tight_layout()
    plt.savefig('category_topic_distribution.png')

# Get top documents for each topic
for topic_idx in range(len(result['topics'])):
    topic_col = f"Topic {topic_idx + 1}"
    top_docs = doc_topics.sort_values(by=topic_col, ascending=False).head(5)
    print(f"\nTop documents for Topic {topic_idx + 1}:")
    for idx in top_docs.index:
        print(f"  Document {idx}: Score {top_docs.loc[idx, topic_col]:.4f}")
```

## Performance Considerations

- For datasets with more than 25,000 documents, the function automatically samples documents for topic modeling (configurable with `max_docs`).
- Multiprocessing is enabled by default for datasets with more than 10,000 documents.
- Fuzzy deduplication is more CPU-intensive than exact deduplication, especially for large datasets.
- Using lemmatization requires spaCy and is more computationally expensive than simple preprocessing.
- Anonymization adds an additional processing step but can improve topic quality by removing personally identifiable information.

## Return Values

The function returns a dictionary with the following keys:

- `topic_model`: Dictionary with the trained model and related data
- `document_topics`: DataFrame with document-topic distributions
- `topics`: List of (topic_idx, words) tuples with top words for each topic
- `processing_info`: Dict with processing statistics and settings
- `deduplication_map`: Dict mapping deduplicated to original indices (if `return_original_mapping=True`)

## Anonymization Integration

The topic modeling workflow integrates with the Allyanonimiser package to provide PII (Personally Identifiable Information) anonymization. This feature helps protect sensitive information while maintaining the semantic structure needed for effective topic modeling.

When anonymization is enabled:

1. The text data is processed through Allyanonimiser before any other preprocessing steps
2. PII entities like names, email addresses, phone numbers, etc. are replaced with entity type labels
3. The anonymized text then goes through the normal preprocessing and topic modeling pipeline

This integration requires the Allyanonimiser package to be installed. If the package is not available, the function will issue a warning and continue without anonymization.

### Anonymization Configuration

You can customize the anonymization process with the `anonymization_config` parameter:

```python
anonymization_config = {
    'use_all_patterns': True,  # Use all available patterns
    'custom_patterns': {  # Add your own regex patterns
        'CUSTOM_TYPE': r'pattern-here'
    },
    'replacement_strategy': 'entity_type',  # Replace with type labels
    'obfuscation_level': 'medium'  # How aggressively to anonymize
}
```

For more anonymization options, refer to the Allyanonimiser documentation.