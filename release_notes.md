# Freamon 0.3.25 Release Notes

## Optimized Topic Modeling

This release introduces a significantly improved workflow for topic modeling, designed to handle larger datasets more efficiently while simplifying the user experience:

### Key Features

- **Simplified API**: New `create_topic_model_optimized()` method handles the entire topic modeling workflow in one call
- **Automatic Deduplication**: Removes duplicate documents before processing to improve performance and model quality
- **Smart Sampling**: Automatically samples very large datasets for model building while still applying the model to all documents
- **Batch Processing**: Processes data in batches with progress reporting for better user experience
- **Full Dataset Coverage**: Returns topic distributions for all documents, even when sampling was used for model building

### Benefits

- **Performance**: 2-10x faster topic modeling for large datasets
- **Memory Efficiency**: Reduced memory consumption through smart batching
- **Better Results**: Improved model quality by removing duplicate documents
- **User Experience**: Simplified code with detailed progress reporting
- **Flexibility**: Configurable parameters for customization

### Example Usage

```python
from freamon.utils.text_utils import TextProcessor
import pandas as pd

# Load data
df = pd.DataFrame({'text': ["Document 1", "Document 2", ...], 'category': [...]})

# Initialize processor
processor = TextProcessor(use_spacy=True)

# Run optimized topic modeling in one call
result = processor.create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=5,
    method='nmf',
    use_lemmatization=True,
    remove_duplicates=True
)

# Access results
topics = result['topics']                # List of topic words
doc_topics = result['document_topics']   # Document-topic distribution
info = result['processing_info']         # Processing statistics
```

### Documentation

Full documentation is available in the new `docs/usage/optimized_topic_modeling.md` file, and a complete working example can be found in `examples/optimized_topic_modeling_example.py`.

## Also in this release

- Improved HTML report generation for DataTypeDetector with unit tests
- Updated documentation with additional examples
- Bug fixes and performance improvements