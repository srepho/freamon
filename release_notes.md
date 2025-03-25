# Freamon Version 0.3.26 Release Notes

We are excited to announce version 0.3.26 of the Freamon package, which introduces enhanced topic modeling workflows and improved integration with pandas DataFrames.

## Key Features

### Topic Modeling Integration with pandas DataFrames
- New workflow for adding topic information directly to pandas DataFrames
- Easily extract the dominant topic for each document
- Add topic probability distributions as DataFrame columns
- Simple API for topic model creation and application

### Optimized Topic Modeling
- Support for both fuzzy and exact deduplication in topic modeling pipeline
- Automatic handling of large datasets with smart sampling
- Configurable similarity thresholds for fuzzy matching
- Memory-efficient processing with batch operations

### Improved Error Handling and Performance
- Better multiprocessing support for topic modeling operations
- Enhanced error handling in text processing modules
- Progress reporting for long-running operations

## Example Usage

```python
# Create and apply a topic model to a DataFrame
from freamon.utils.text_utils import TextProcessor

# Initialize text processor
processor = TextProcessor(use_spacy=True)

# Create topic model
topic_model = processor.create_topic_model(
    texts=df['text'].tolist(),
    n_topics=5,
    method='nmf'
)

# Get document-topic distribution
doc_topics = processor.get_document_topics(topic_model)

# Add dominant topic to the DataFrame
df['dominant_topic'] = doc_topics.idxmax(axis=1)
```

## New Examples
- `dataframe_topic_modeling_example.py`: Complete workflow for topic modeling with DataFrames
- `optimized_topic_modeling_example.py`: Demonstrates optimized processing for large datasets

Check out our [GitHub repository](https://github.com/your-org/freamon) for more examples and documentation.

## Installation

```bash
pip install freamon==0.3.26
```

Or upgrade your existing installation:

```bash
pip install --upgrade freamon
```