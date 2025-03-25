# Optimized Topic Modeling

This guide covers optimized techniques for topic modeling with Freamon, addressing performance and usability challenges with larger text datasets.

## Simple One-Line Approach

Instead of manually handling text cleaning, lemmatization, and deduplication, Freamon now offers an optimized workflow function:

```python
from freamon.utils.text_utils import TextProcessor
import pandas as pd

# Load your data
df = pd.DataFrame({
    'text': ["First document...", "Second document...", "..."],
    'metadata': [1, 2, 3]
})

# Initialize the text processor
processor = TextProcessor()

# Run optimized topic modeling in one call
result = processor.create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',              # 'nmf' is faster than 'lda'
    use_lemmatization=True,    # Use lemmatization for better topics
    max_docs=None,             # Auto-sample if dataset is very large
    remove_duplicates=True,    # Automatically deduplicate
    return_full_data=True      # Apply model to all docs, not just sample
)

# Access results
topics = result['topics']              # List of (topic_idx, words) tuples
topic_model = result['topic_model']    # The trained model
doc_topics = result['document_topics'] # Document-topic distributions
info = result['processing_info']       # Processing statistics

# Print topics
for topic_idx, words in topics:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Visualize topics
html = processor.plot_topics(topic_model, figsize=(15, 10), return_html=True)
with open("topic_visualization.html", "w") as f:
    f.write(html)
```

## Key Optimizations

The optimized workflow incorporates several performance improvements:

1. **Automatic Deduplication**: Removes exact duplicates before processing
2. **Smart Sampling**: For very large datasets, uses a representative sample for model building
3. **Batch Processing**: Processes documents in batches with progress reporting
4. **Full Dataset Coverage**: Applies the trained model to all documents even when sampling is used
5. **Progress Reporting**: Shows progress for long-running operations
6. **Optimized Parameters**: Uses efficient default settings for vectorization and modeling

## Performance Improvements

This approach offers significant performance improvements:

- **Reduced Memory Usage**: By deduplicating and processing in batches
- **Faster Model Training**: Through sampling of large datasets 
- **Complete Results**: Returns topic distributions for all documents
- **Simplified Interface**: Single function call instead of multiple steps

## Customization Options

The optimized workflow is highly customizable:

- **Topic Count**: Adjust `n_topics` to control the number of topics
- **Algorithm**: Choose between faster `'nmf'` and more interpretable `'lda'`
- **Lemmatization**: Toggle `use_lemmatization` based on language needs
- **Dataset Size**: Control maximum documents to process with `max_docs`
- **Deduplication**: Enable/disable with `remove_duplicates`
- **Result Scope**: Choose between sample-only or full data with `return_full_data`

## Processing Information

The `processing_info` dictionary provides useful statistics:

```python
info = result['processing_info']
print(f"Original documents: {info['original_doc_count']}")
print(f"Duplicates removed: {info['duplicates_removed']}")
print(f"Processed documents: {info['processed_doc_count']}")
if info['sampled']:
    print(f"Sample size used for modeling: {info['sample_size']}")
```

## Working with Topic Results

The document-topic distribution can be easily integrated with your original data:

```python
# Get topic distribution as dataframe
doc_topics = result['document_topics']

# Combine with original metadata
combined_df = pd.concat([df, doc_topics], axis=1)

# Find dominant topic for each document
combined_df['dominant_topic'] = doc_topics.idxmax(axis=1)

# Group by metadata and analyze topic distribution
topic_by_group = combined_df.groupby('metadata')[doc_topics.columns].mean()
```

## Full Example

For a complete working example, see `examples/optimized_topic_modeling_example.py` which demonstrates:

- Full implementation of the optimized workflow
- Processing of a realistic dataset
- Visualization of topic-category relationships
- Performance statistics reporting