# Word Embeddings

Freamon offers powerful word embedding capabilities through the `TextProcessor` class. Word embeddings are vector representations of words that capture semantic relationships, enabling advanced text analysis and feature engineering.

## Supported Embedding Types

The `TextProcessor` supports three main types of word embeddings:

1. **Word2Vec**: Train embeddings directly on your corpus
2. **GloVe** (Global Vectors for Word Representation): Use pretrained embeddings
3. **FastText**: Use pretrained embeddings with subword information

## Working with Word2Vec

### Training a Word2Vec Model

```python
from freamon.utils.text_utils import TextProcessor

# Initialize TextProcessor
processor = TextProcessor()

# Train Word2Vec on your texts
word2vec = processor.create_word2vec_embeddings(
    texts=df['text_column'],
    vector_size=100,  # Vector dimension
    window=5,         # Context window size
    min_count=5,      # Minimum word frequency
    epochs=10,        # Training epochs
    sg=0              # 0=CBOW, 1=Skip-gram
)

# Access the model and word vectors
model = word2vec['model']
word_vectors = word2vec['wv']

# Get vector for a specific word
vector = word_vectors['example']

# Find similar words
similar_words = word_vectors.most_similar('example', topn=10)
```

## Using Pretrained Embeddings

### Loading GloVe Embeddings

```python
# Load pretrained GloVe embeddings
glove = processor.load_pretrained_embeddings(
    embedding_type='glove',
    dimension=100,     # Options: 50, 100, 200, 300
    limit=50000        # Limit vocabulary size
)

# Access word vectors
word_vectors = glove['wv']
```

### Loading FastText Embeddings

```python
# Load pretrained FastText embeddings
fasttext = processor.load_pretrained_embeddings(
    embedding_type='fasttext',
    dimension=300,     # Only 300 is available for FastText
    limit=50000        # Limit vocabulary size
)

# Access word vectors
word_vectors = fasttext['wv']
```

## Document-Level Embeddings

### Creating Document Embeddings

```python
# Create document embeddings from word embeddings
doc_embeddings = processor.create_document_embeddings(
    texts=df['text_column'],
    word_vectors=word_vectors,
    method='mean'      # Options: 'mean', 'weighted', 'idf'
)

# doc_embeddings is a numpy array with shape (n_documents, vector_size)
```

### Document Similarity

```python
# Calculate similarity between two document embeddings
similarity = processor.calculate_embedding_similarity(
    embedding1=doc_embeddings[0],
    embedding2=doc_embeddings[1],
    method='cosine'    # Options: 'cosine', 'euclidean', 'dot'
)

# Find most similar documents to a query document
similar_docs = processor.find_most_similar_documents(
    query_embedding=doc_embeddings[0],
    document_embeddings=doc_embeddings,
    top_n=5,
    similarity_method='cosine'
)

# similar_docs is a list of tuples (document_index, similarity_score)
```

## Feature Engineering with Embeddings

### Adding Embedding Features

```python
# Create text features including embeddings
features = processor.create_text_features(
    df,
    'text_column',
    include_stats=True,
    include_readability=True,
    include_sentiment=True,
    include_embeddings=True,           # Enable embedding features
    embedding_type='word2vec',         # Options: 'word2vec', 'glove', 'fasttext'
    embedding_dimension=100,
    embedding_components=5             # Number of PCA components to extract
)

# The resulting dataframe includes embedding features named 'text_emb_word2vec_1', etc.
```

## Visualization Example

You can visualize document embeddings using dimensionality reduction techniques:

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reduce dimensions for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(doc_embeddings)

# Plot with categories
plt.figure(figsize=(10, 8))
for category in df['category'].unique():
    indices = df['category'] == category
    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        label=category,
        alpha=0.7
    )
plt.legend()
plt.title("Document Embeddings")
plt.show()
```

## Performance Considerations

- For large datasets, limit the vocabulary size when loading pretrained embeddings
- Use a smaller `embedding_dimension` if memory is a concern
- Consider using `method='idf'` in `create_document_embeddings()` for better representation
- Embeddings are cached in `~/.freamon/embeddings/` to avoid repeated downloads

## Advanced Usage

### Custom Word Weights

```python
# Create document embeddings with custom word weights
weights = {'important': 2.0, 'critical': 3.0, 'key': 1.5}
doc_embeddings = processor.create_document_embeddings(
    texts=df['text_column'],
    word_vectors=word_vectors,
    method='weighted',
    weights=weights
)
```

### IDF Weighting

```python
# Create document embeddings with IDF weighting
doc_embeddings = processor.create_document_embeddings(
    texts=df['text_column'],
    word_vectors=word_vectors,
    method='idf'  # Automatically calculates IDF weights
)
```