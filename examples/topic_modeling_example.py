"""
Example demonstrating the topic modeling capabilities in freamon.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from freamon.utils.text_utils import TextProcessor

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

# Sample a subset for faster processing in this example
print(f"Sampling 500 documents from {len(df)} total documents...")
df = df.sample(500, random_state=42)

# Initialize the TextProcessor
processor = TextProcessor(use_spacy=False)

print("\n=== Basic Topic Modeling ===")
print("Creating LDA topic model with 6 topics...")
topic_model = processor.create_topic_model(
    texts=df['text'],
    n_topics=6,
    method='lda',
    max_features=1000,
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2),  # Include bigrams
    random_state=42
)

# Print topics
print("\nTop words for each topic:")
for topic_idx, words in topic_model['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Visualize topics
print("\nGenerating topic visualization...")
html = processor.plot_topics(topic_model, figsize=(15, 10), return_html=True)

# Save visualization
with open("topic_model_visualization.html", "w") as f:
    f.write(f"<html><body>{html}</body></html>")
print("Saved topic visualization to topic_model_visualization.html")

# Calculate topic coherence
coherence = topic_model.get('coherence_score')
print(f"\nTopic coherence (c_v): {coherence:.4f}")

# Get document-topic distribution
print("\nGetting document-topic distribution...")
doc_topics = processor.get_document_topics(topic_model)

# Summary statistics of topic distribution
print("\nTopic distribution summary:")
topic_means = doc_topics.mean(axis=0)
for col, mean in topic_means.items():
    print(f"  {col}: {mean:.4f}")

print("\n=== Topic Distribution by Category ===")
# Combine with original categories
topic_with_category = doc_topics.copy()
topic_with_category['category'] = df['category'].values

# Calculate average topic distribution by category
print("\nAverage topic distribution by category:")
category_topic_dist = topic_with_category.groupby('category').mean()
print(category_topic_dist)

# Create heatmap of category-topic relationships
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
print("Saved category-topic distribution heatmap to category_topic_distribution.png")

print("\n=== Finding Optimal Number of Topics ===")
print("Searching for optimal number of topics (this may take a while)...")
optimal = processor.find_optimal_topics(
    texts=df['text'].sample(200, random_state=42),  # Use a smaller sample for speed
    min_topics=3,
    max_topics=12,
    step=3,        # Step by 3 for faster processing
    method='lda',
    max_features=1000,
    plot_results=True
)

print(f"\nOptimal number of topics: {optimal['optimal_topics']}")
print(f"Best coherence score: {optimal['best_coherence']:.4f}")

print("\n=== Comparing LDA and NMF ===")
# Create NMF model for comparison
print("Creating NMF topic model...")
nmf_model = processor.create_topic_model(
    texts=df['text'],
    n_topics=6,
    method='nmf',
    max_features=1000,
    max_df=0.7,
    min_df=5,
    random_state=42
)

print("\nNMF Topics:")
for topic_idx, words in nmf_model['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Get document-topic distribution
nmf_doc_topics = processor.get_document_topics(nmf_model)

# Create feature dataset with topics
print("\n=== Creating Text Features with Topics ===")
text_features = processor.create_text_features(
    df,
    'text',
    include_stats=True,
    include_readability=True,
    include_sentiment=True,
    include_topics=True,
    n_topics=6,
    topic_method='lda'
)

print(f"Generated {text_features.shape[1]} features")
print("Feature types:")
feature_groups = {
    'Statistics': 'text_stat_',
    'Readability': 'text_read_',
    'Sentiment': 'text_sent_',
    'Topics': 'text_topic_'
}

for group, prefix in feature_groups.items():
    count = sum(1 for col in text_features.columns if col.startswith(prefix))
    print(f"  {group}: {count} features")

# Preview some features
print("\nSample topic features (first 3 rows, first 4 topic features):")
topic_cols = [col for col in text_features.columns if 'topic_' in col][:4]
print(text_features[topic_cols].head(3))

print("\nExample complete!")