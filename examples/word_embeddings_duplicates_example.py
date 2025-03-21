"""
Example demonstrating handling duplicate texts when creating word embeddings.
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity

from freamon.utils.text_utils import TextProcessor
from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates

# Load subset of the 20 newsgroups dataset
print("Loading 20 newsgroups dataset...")
categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create a DataFrame with the text data (add some duplicates for demonstration)
newsgroup_data = newsgroups.data[:200]  # Starting with 200 documents
newsgroup_targets = newsgroups.target[:200]

# Create duplicate texts by copying some entries
duplicate_indices = [5, 10, 15, 20, 25]
duplicates = [(newsgroup_data[i], newsgroup_targets[i]) for i in duplicate_indices]

# Add duplicates to the dataset
for text, target in duplicates * 2:  # Add each duplicate twice
    newsgroup_data.append(text)
    newsgroup_targets.append(target)

df = pd.DataFrame({
    'text': newsgroup_data,
    'category': [newsgroups.target_names[target] for target in newsgroup_targets]
})

print(f"Total dataset size: {len(df)} documents")

# Add empty text rows for demonstration
empty_rows = 5
for _ in range(empty_rows):
    df = pd.concat([df, pd.DataFrame({
        'text': ['', None, np.nan, ' ', '\n'],
        'category': ['sci.med'] * 5
    })])

print(f"After adding empty texts: {len(df)} documents")

# Initialize the TextProcessor
processor = TextProcessor(use_spacy=False)

# Step 1: Detect duplicates and empty texts
print("\n=== Analyzing Duplicate and Empty Texts ===")
duplicates_info = detect_duplicates(df, subset=['text'], return_counts=True)
print(f"Duplicates found: {duplicates_info['duplicate_count']}")
print(f"Duplicate percentage: {duplicates_info['duplicate_percentage']:.2f}%")

# Check for empty texts
def is_empty_text(text):
    """Check if text is empty, None, NaN, or just whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return True
    return len(text.strip()) == 0

empty_mask = df['text'].apply(is_empty_text)
print(f"Empty texts found: {empty_mask.sum()}")

# Step 2: Create a deduplicated version for training word embeddings
print("\n=== Creating Deduplicated Dataset ===")
start_time = time.time()

# Remove duplicates
df_unique = remove_duplicates(df, subset=['text'], keep='first')
print(f"After removing duplicates: {len(df_unique)} documents")

# Remove empty texts
df_clean = df_unique[~df_unique['text'].apply(is_empty_text)].copy()
print(f"After removing empty texts: {len(df_clean)} documents")

processing_time = time.time() - start_time
print(f"Processing time: {processing_time:.4f} seconds")

# Step 3: Create a lookup mapping from text to document ID for later use
print("\n=== Creating Text to ID Mapping ===")
text_to_id = {}
for idx, row in df_clean.iterrows():
    text = row['text']
    if text not in text_to_id:
        text_to_id[text] = []
    text_to_id[text].append(idx)

# Step 4: Train Word2Vec on clean, deduplicated data
print("\n=== Training Word2Vec on Deduplicated Data ===")
start_time = time.time()

word2vec = processor.create_word2vec_embeddings(
    texts=df_clean['text'],
    vector_size=100,
    window=5,
    min_count=5,
    epochs=5,
    seed=42
)

training_time = time.time() - start_time
print(f"Word2Vec training time: {training_time:.4f} seconds")
print(f"Vocabulary size: {word2vec['vocab_size']}")

# Step 5: Create document embeddings for the clean dataset only
print("\n=== Creating Document Embeddings ===")
start_time = time.time()

doc_embeddings = processor.create_document_embeddings(
    texts=df_clean['text'],
    word_vectors=word2vec['wv'],
    method='mean'
)

embedding_time = time.time() - start_time
print(f"Document embedding creation time: {embedding_time:.4f} seconds")
print(f"Document embeddings shape: {doc_embeddings.shape}")

# Step 6: Map embeddings back to original dataset (including duplicates)
print("\n=== Mapping Embeddings Back to Full Dataset ===")
start_time = time.time()

# Create a dataframe to hold all the embeddings
embedding_df = pd.DataFrame(index=df.index)

# Add embedding features
for dim in range(5):  # Use first 5 dimensions for demonstration
    embedding_df[f'emb_dim_{dim}'] = np.nan

# Map deduplicated embeddings back to full dataset
for clean_idx, embedding_vector in zip(df_clean.index, doc_embeddings):
    text = df_clean.loc[clean_idx, 'text']
    
    # Find all instances of this text in original dataframe
    matches = df[df['text'] == text].index
    
    # Apply the embedding to all matching rows
    for match_idx in matches:
        for dim in range(5):  # Use first 5 dimensions
            embedding_df.loc[match_idx, f'emb_dim_{dim}'] = embedding_vector[dim]

# Handle empty texts - use zero vectors
for idx in df[empty_mask].index:
    for dim in range(5):
        embedding_df.loc[idx, f'emb_dim_{dim}'] = 0.0

mapping_time = time.time() - start_time
print(f"Embedding mapping time: {mapping_time:.4f} seconds")

# Check for any unmapped embeddings (should be none)
unmapped = embedding_df.isna().any(axis=1).sum()
print(f"Unmapped rows: {unmapped}")

# Step 7: Demonstrate using the embeddings with the full dataset
print("\n=== Using Embeddings for Document Similarity ===")

# Join the embeddings with the original data
result_df = df.join(embedding_df)

# Select 3 documents of different categories
sample_indices = []
for category in categories[:3]:  # Get 3 different categories
    idx = df[df['category'] == category].index[0]
    sample_indices.append(idx)

# Calculate similarities between these documents
print("\nSimilarity between documents of different categories:")
for i, idx1 in enumerate(sample_indices):
    for j, idx2 in enumerate(sample_indices[i+1:], i+1):
        vec1 = result_df.loc[idx1, [f'emb_dim_{d}' for d in range(5)]].values
        vec2 = result_df.loc[idx2, [f'emb_dim_{d}' for d in range(5)]].values
        
        similarity = processor.calculate_embedding_similarity(
            vec1, vec2, method='cosine'
        )
        
        cat1 = result_df.loc[idx1, 'category']
        cat2 = result_df.loc[idx2, 'category']
        print(f"Similarity between {cat1} and {cat2}: {similarity:.4f}")

# Compare with a duplicate to verify same embeddings were assigned
if duplicates_info['duplicate_count'] > 0:
    print("\nVerifying duplicate texts have identical embeddings:")
    # Find a pair of duplicate texts
    for text, count_info in zip(df['text'], duplicates_info.get('value_counts', [])):
        if count_info['count'] > 1 and not is_empty_text(text):
            # Found a duplicate text
            dup_indices = df[df['text'] == text].index[:2]  # Get the first two instances
            
            vec1 = result_df.loc[dup_indices[0], [f'emb_dim_{d}' for d in range(5)]].values
            vec2 = result_df.loc[dup_indices[1], [f'emb_dim_{d}' for d in range(5)]].values
            
            # These should be identical
            is_identical = np.allclose(vec1, vec2)
            print(f"Duplicate embeddings identical: {is_identical}")
            if is_identical:
                print(f"Values: {vec1}")
            break

print("\nExample complete!")