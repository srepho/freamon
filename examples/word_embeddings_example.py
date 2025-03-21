"""
Example demonstrating the word embedding capabilities in freamon.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from freamon.utils.text_utils import TextProcessor

# Load subset of the 20 newsgroups dataset
print("Loading 20 newsgroups dataset...")
categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create a DataFrame with the text data
df = pd.DataFrame({
    'text': newsgroups.data[:200],  # Using first 200 documents for brevity
    'category': [newsgroups.target_names[target] for target in newsgroups.target[:200]]
})

# Initialize the TextProcessor
processor = TextProcessor(use_spacy=False)

print("\n=== Word2Vec Embeddings ===")
print("Training Word2Vec model...")
word2vec = processor.create_word2vec_embeddings(
    texts=df['text'],
    vector_size=100,
    window=5,
    min_count=5,
    epochs=10,
    seed=42
)

# Print vocabulary stats
print(f"Vocabulary size: {word2vec['vocab_size']}")
print(f"Vector dimension: {word2vec['vector_size']}")

# Find similar words
word = 'science'
if word in word2vec['wv']:
    print(f"\nWords most similar to '{word}':")
    similar_words = word2vec['wv'].most_similar(word, topn=5)
    for w, score in similar_words:
        print(f"  {w}: {score:.4f}")

# Create document embeddings
print("\nCreating document embeddings...")
doc_embeddings = processor.create_document_embeddings(
    texts=df['text'],
    word_vectors=word2vec['wv'],
    method='mean'
)
print(f"Document embeddings shape: {doc_embeddings.shape}")

# Visualize embeddings with t-SNE
print("\nVisualizing document embeddings with t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
doc_embeddings_2d = tsne.fit_transform(doc_embeddings)

# Plot embeddings by category
plt.figure(figsize=(10, 8))
colors = {'sci.med': 'red', 'sci.space': 'blue', 'rec.autos': 'green', 'rec.sport.hockey': 'purple'}
for category in df['category'].unique():
    indices = df['category'] == category
    plt.scatter(
        doc_embeddings_2d[indices, 0],
        doc_embeddings_2d[indices, 1],
        c=colors[category],
        label=category,
        alpha=0.7,
        s=50
    )
plt.legend()
plt.title("t-SNE visualization of document embeddings")
plt.savefig('document_embeddings_tsne.png')
print("Saved t-SNE visualization to document_embeddings_tsne.png")

# Document similarity example
print("\n=== Document Similarity ===")
print("Finding similar documents...")

# Pick a document
query_idx = 0
query_doc = df.iloc[query_idx]['text']
query_category = df.iloc[query_idx]['category']
print(f"Query document category: {query_category}")
print(f"Query text (truncated): {query_doc[:100]}...")

# Find similar documents
similar_docs = processor.find_most_similar_documents(
    query_embedding=doc_embeddings[query_idx],
    document_embeddings=doc_embeddings,
    top_n=5,
    similarity_method='cosine'
)

print("\nTop 5 most similar documents:")
for i, (doc_idx, similarity) in enumerate(similar_docs):
    category = df.iloc[doc_idx]['category']
    print(f"{i+1}. Document {doc_idx} (Category: {category}, Similarity: {similarity:.4f})")
    print(f"   Text (truncated): {df.iloc[doc_idx]['text'][:100]}...")

# Try working with pretrained embeddings
print("\n=== Using GloVe Embeddings ===")
try:
    print("Loading GloVe embeddings (this might take a while)...")
    glove = processor.load_pretrained_embeddings(
        embedding_type='glove',
        dimension=100,
        limit=50000  # Limit to 50k words for faster loading
    )
    
    print(f"GloVe vocabulary size: {glove['vocab_size']}")
    print(f"GloVe vector dimension: {glove['vector_size']}")
    
    # Create document embeddings with GloVe
    print("\nCreating document embeddings with GloVe...")
    glove_doc_embeddings = processor.create_document_embeddings(
        texts=df['text'].head(10),  # Just use first 10 docs for example
        word_vectors=glove['wv'],
        method='mean'
    )
    
    print(f"GloVe document embeddings shape: {glove_doc_embeddings.shape}")
    
    # Compare similarities
    print("\nCalculating document similarity matrix...")
    similarity_matrix = cosine_similarity(glove_doc_embeddings)
    
    print("Document similarity heatmap (first 10 documents):")
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Document Similarity Matrix")
    plt.xlabel("Document Index")
    plt.ylabel("Document Index")
    plt.savefig('document_similarity_matrix.png')
    print("Saved similarity matrix visualization to document_similarity_matrix.png")
    
except Exception as e:
    print(f"Error loading GloVe embeddings: {str(e)}")
    print("Skipping GloVe example.")

# Feature generation with embeddings example
print("\n=== Text Feature Engineering with Embeddings ===")
print("Creating text features with embeddings...")

# Create text features with embeddings
features = processor.create_text_features(
    df.head(10),  # Just use 10 documents for example
    'text',
    include_stats=True,
    include_readability=True,
    include_sentiment=True,
    include_embeddings=True,
    embedding_type='word2vec',
    embedding_dimension=50,
    embedding_components=5
)

# Print feature statistics
feature_types = {
    'Statistics': 'text_stat_',
    'Readability': 'text_read_',
    'Sentiment': 'text_sent_',
    'Embeddings': 'text_emb_'
}

print("\nFeature summary:")
for feature_type, prefix in feature_types.items():
    feature_count = sum(1 for col in features.columns if col.startswith(prefix))
    print(f"{feature_type}: {feature_count} features")

print("\nFirst 5 embedding feature values:")
embedding_cols = [col for col in features.columns if 'emb_word2vec' in col]
if embedding_cols:
    print(features[embedding_cols].head())

print("\nExample complete!")