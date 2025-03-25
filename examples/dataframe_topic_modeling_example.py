"""
Example showing how to integrate topic modeling with pandas DataFrames.

This example demonstrates:
1. Creating a topic model with configurable preprocessing
2. Working with large datasets using optimized processing
3. Adding topic distributions back to the original DataFrame
4. Visualizing and analyzing topic distributions by categories
5. Flexible deduplication options for text data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Import the optimized topic modeling function
from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor

print("=== DataFrame Topic Modeling Example ===")

# Load 20 newsgroups dataset
print("Loading 20 newsgroups dataset...")
categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey', 
              'talk.politics.guns', 'comp.graphics']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create a DataFrame with the text data
df = pd.DataFrame({
    'text': newsgroups.data,
    'category': [newsgroups.target_names[target] for target in newsgroups.target]
})

# Add some metadata for demonstration
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['text'].apply(lambda x: len(str(x)))

print(f"Dataset contains {len(df)} documents across {df['category'].nunique()} categories")
print("\nCategory distribution:")
print(df['category'].value_counts())

print("\nSample from the dataset:")
print(df.head(2)[['category', 'word_count', 'char_count']])

# Configure preprocessing options
preprocessing_options = {
    'enabled': True,
    'use_lemmatization': True,  # Set to False for faster processing if needed
    'remove_stopwords': True,
    'remove_punctuation': True,
    'min_token_length': 3,
    'custom_stopwords': ['said', 'would', 'could']  # Domain-specific stopwords
}

# Configure deduplication options
deduplication_options = {
    'enabled': True,
    'method': 'exact',  # 'exact', 'fuzzy', or 'none'
    'hash_method': 'hash',  # 'hash' or 'ngram'
    'similarity_threshold': 0.85,  # Only used for fuzzy deduplication
    'similarity_method': 'cosine',  # 'cosine', 'jaccard', 'levenshtein'
    'keep': 'first'  # 'first' or 'last'
}

print("\nRunning optimized topic modeling...")
result = create_topic_model_optimized(
    df, 
    text_column='text',
    n_topics=6,  # One per category
    method='nmf',  # 'nmf' or 'lda'
    preprocessing_options=preprocessing_options,
    max_docs=500,  # Use a subset for faster processing
    deduplication_options=deduplication_options,
    return_full_data=True,
    return_original_mapping=True,
    use_multiprocessing=True
)

# Extract the topics
print("\nTop words for each topic:")
for topic_idx, words in result['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Get document-topic distribution
doc_topics = result['document_topics']

# Print processing information
info = result['processing_info']
print("\nProcessing Information:")
print(f"  Original documents: {info['original_doc_count']}")
print(f"  Duplicates removed: {info['duplicates_removed']}")
print(f"  Processed documents: {info['processed_doc_count']}")
if 'preprocessing_time' in info:
    print(f"  Preprocessing time: {info['preprocessing_time']:.2f} seconds")
if 'multiprocessing_enabled' in info:
    print(f"  Multiprocessing: {info['multiprocessing_enabled']}")

# Add topic distributions back to the original DataFrame
print("\nAdding topic distributions to the original DataFrame...")
topic_columns = doc_topics.columns.tolist()

# Merge the topic distributions back to the original DataFrame
df_with_topics = df.copy()
for col in topic_columns:
    df_with_topics[col] = doc_topics[col]

# Find the dominant topic for each document
def get_dominant_topic(row, topic_cols):
    """Get the dominant topic for a document based on topic distribution"""
    topic_values = [row[col] for col in topic_cols]
    if all(x == 0 for x in topic_values):
        return -1  # No dominant topic
    max_topic_idx = np.argmax(topic_values)
    return max_topic_idx + 1  # 1-based indexing for readability

# Add dominant topic column
df_with_topics['dominant_topic'] = df_with_topics.apply(
    lambda row: get_dominant_topic(row, topic_columns), axis=1
)

# Print sample of the enhanced DataFrame
print("\nSample of DataFrame with topic distributions:")
display_cols = ['category', 'word_count'] + topic_columns[:2] + ['dominant_topic']
print(df_with_topics[display_cols].head(3))

# Analyze the relationship between categories and topics
print("\nAnalyzing the relationship between categories and topics...")

# Category-topic distribution
category_topic_dist = df_with_topics.groupby('category')[topic_columns].mean()
print("\nAverage topic distribution by category:")
print(category_topic_dist)

# Create heatmap of category-topic relationships
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
print("Saved category-topic distribution heatmap to category_topic_distribution.png")

# Topic-dominant analysis
print("\nDominant topic distribution:")
topic_counts = df_with_topics['dominant_topic'].value_counts().sort_index()
print(topic_counts)

# Create bar chart of dominant topics
plt.figure(figsize=(10, 6))
topic_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Topic Number')
plt.ylabel('Number of Documents')
plt.title('Number of Documents per Dominant Topic')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('dominant_topic_distribution.png')
print("Saved dominant topic distribution chart to dominant_topic_distribution.png")

# Topic distribution by category
print("\nDominant topic distribution by category:")
topic_by_category = pd.crosstab(
    df_with_topics['category'], 
    df_with_topics['dominant_topic']
)
print(topic_by_category)

# Visualize topic distribution by category
plt.figure(figsize=(12, 8))
topic_by_category.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Category')
plt.ylabel('Number of Documents')
plt.title('Dominant Topic Distribution by Category')
plt.legend(title='Topic')
plt.tight_layout()
plt.savefig('topic_by_category.png')
print("Saved topic by category chart to topic_by_category.png")

# Analyze document characteristics by topic
print("\nAnalyzing document characteristics by topic...")
topic_characteristics = df_with_topics.groupby('dominant_topic').agg({
    'word_count': 'mean',
    'char_count': 'mean'
}).reset_index()

print("Average document characteristics by dominant topic:")
print(topic_characteristics)

# Create a visualization of document characteristics by topic
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Word count by topic
topic_characteristics.plot(
    x='dominant_topic', 
    y='word_count', 
    kind='bar', 
    ax=axes[0], 
    color='skyblue'
)
axes[0].set_title('Average Word Count by Topic')
axes[0].set_xlabel('Topic')
axes[0].set_ylabel('Average Word Count')

# Character count by topic
topic_characteristics.plot(
    x='dominant_topic', 
    y='char_count', 
    kind='bar', 
    ax=axes[1], 
    color='lightgreen'
)
axes[1].set_title('Average Character Count by Topic')
axes[1].set_xlabel('Topic')
axes[1].set_ylabel('Average Character Count')

plt.tight_layout()
plt.savefig('topic_characteristics.png')
print("Saved topic characteristics chart to topic_characteristics.png")

print("\nExample complete!")