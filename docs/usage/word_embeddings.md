# Word Embeddings

Freamon provides comprehensive word embedding capabilities through its `TextProcessor` class. Word embeddings are vector representations of words that capture semantic relationships, allowing you to perform advanced NLP tasks such as similarity analysis, classification, and clustering.

## Supported Embedding Types

- **Word2Vec**: Train custom embeddings on your own text data
- **GloVe**: Load pre-trained Global Vectors embeddings
- **FastText**: Load pre-trained FastText embeddings that handle subwords

## Basic Usage

```python
from freamon.utils.text_utils import TextProcessor
import pandas as pd

# Initialize the TextProcessor
processor = TextProcessor()

# Sample data
df = pd.DataFrame({
    'text': [
        "This is a document about science and medicine",
        "Astronomy and space exploration are fascinating",
        "Sports news: The hockey team won the championship"
    ],
    'category': ['science', 'space', 'sports']
})

# Train Word2Vec model on your text data
word2vec = processor.create_word2vec_embeddings(
    texts=df['text'],
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10
)

# Create document-level embeddings
doc_embeddings = processor.create_document_embeddings(
    texts=df['text'],
    word_vectors=word2vec['wv'],
    method='mean'  # Average of word vectors
)

# Calculate similarity between documents
similarity = processor.calculate_embedding_similarity(
    doc_embeddings[0],  # First document
    doc_embeddings[1],  # Second document
    method='cosine'     # Cosine similarity
)

print(f"Similarity between documents: {similarity:.4f}")
```

## Working with Duplicate Texts

When working with text data that contains duplicates, it's more efficient to:
1. Deduplicate the texts before creating embeddings
2. Map the embeddings back to all instances of each text

```python
from freamon.data_quality.duplicates import remove_duplicates
import numpy as np

# Detect and remove duplicates
df_unique = remove_duplicates(df, subset=['text'], keep='first')

# Create embeddings on deduplicated data
word2vec = processor.create_word2vec_embeddings(
    texts=df_unique['text'],
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10
)

# Generate document embeddings
doc_embeddings = processor.create_document_embeddings(
    texts=df_unique['text'],
    word_vectors=word2vec['wv'],
    method='mean'
)

# Create a mapping from texts to embeddings
text_to_embedding = {}
for idx, text in enumerate(df_unique['text']):
    text_to_embedding[text] = doc_embeddings[idx]

# Create a DataFrame to store the embeddings for all documents (including duplicates)
embedding_df = pd.DataFrame(index=df.index)
for dim in range(5):  # First 5 dimensions for example
    embedding_df[f'emb_dim_{dim}'] = np.nan

# Map embeddings back to all rows including duplicates
def is_empty_text(text):
    """Check if text is empty, None, NaN, or just whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return True
    return len(text.strip()) == 0

for idx, row in df.iterrows():
    text = row['text']
    if not is_empty_text(text) and text in text_to_embedding:
        embedding = text_to_embedding[text]
        for dim in range(5):
            embedding_df.loc[idx, f'emb_dim_{dim}'] = embedding[dim]
    else:
        # Handle empty texts with zeros
        for dim in range(5):
            embedding_df.loc[idx, f'emb_dim_{dim}'] = 0.0

# Join with original dataframe
result_df = df.join(embedding_df)
```

## Handling Empty Texts

Empty or missing text values are automatically handled by the word embedding functions:

```python
# DataFrame with some empty texts
df_with_empty = pd.DataFrame({
    'text': [
        "This is a valid document",
        "",  # Empty string
        None,  # None value
        "Another valid document"
    ]
})

# Create document embeddings with empty text handling
doc_embeddings = processor.create_document_embeddings(
    texts=df_with_empty['text'],
    word_vectors=word2vec['wv'],
    method='mean',
    handle_empty_texts=True  # Will use zero vectors for empty texts
)

# All rows will have embeddings, with zeros for empty texts
print(f"Embeddings shape: {doc_embeddings.shape}")
```

## Using Pre-trained Embeddings

Freamon supports loading pre-trained embeddings:

```python
# Load GloVe embeddings
glove = processor.load_pretrained_embeddings(
    embedding_type='glove',
    dimension=100,
    limit=50000  # Limit vocabulary size
)

# Load FastText embeddings
fasttext = processor.load_pretrained_embeddings(
    embedding_type='fasttext',
    dimension=300,
    limit=50000
)

# Create document embeddings with pre-trained vectors
glove_doc_embeddings = processor.create_document_embeddings(
    texts=df['text'],
    word_vectors=glove['wv'],
    method='mean'
)
```

## Offline Mode

For environments without internet access, you can save and load embeddings locally:

```python
# Save Word2Vec model to disk
model_path = "path/to/word2vec_model.model"
word2vec['model'].save(model_path)

# Save word vectors in word2vec format
vectors_path = "path/to/word_vectors.txt"
processor.save_word_vectors(word2vec['wv'], vectors_path)

# Later, load the saved model
loaded_model = processor.load_word2vec_model(model_path)

# Or load just the word vectors
loaded_vectors = processor.load_word_vectors(vectors_path)

# Load pre-trained embeddings from a local file
local_glove = processor.load_pretrained_embeddings(
    embedding_type='glove',
    local_path="path/to/glove.6B.100d.txt"
)

# Try with offline mode (will only use cached embeddings)
try:
    offline_embeddings = processor.load_pretrained_embeddings(
        embedding_type='glove',
        dimension=100,
        offline_mode=True  # Won't attempt to download
    )
except FileNotFoundError:
    print("No cached embeddings available")
```

## Integration with Text Feature Engineering

Word embeddings can be integrated into Freamon's text feature engineering pipeline:

```python
# Create text features including embedding-based features
features_df = processor.create_text_features(
    df,
    'text',
    include_stats=True,
    include_readability=True,
    include_sentiment=True,
    include_embeddings=True,  # Add embedding features
    embedding_type='word2vec',
    embedding_dimension=100,
    embedding_components=10  # Apply PCA to reduce to 10 components
)
```

## Benchmarking

The word embeddings functionality includes benchmarking capabilities:

```python
from freamon.examples.word_embeddings_benchmark import run_benchmarks

# Run standard benchmarks for word embedding components
run_benchmarks()
```

The benchmarks measure:
- Training time for Word2Vec models with different dimensions
- Document embedding creation time with different aggregation methods
- Similarity calculation performance
- Classification performance using embeddings