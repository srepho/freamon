"""
Example demonstrating automatic topic number detection in freamon's topic modeling.

This example shows how to use the 'auto' option for n_topics to automatically determine
the optimal number of topics based on coherence scores. The script:

1. Loads a sample dataset
2. Preprocesses the text data
3. Performs automatic topic detection
4. Visualizes the coherence scores and optimal topic number
5. Displays the selected topics

The automatic detection helps find the right granularity of topics without
manual experimentation with different topic numbers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Import freamon utilities
from freamon.utils.text_utils import create_topic_model_optimized

# Load sample dataset from sklearn (20 newsgroups)
print("Loading sample dataset (20 newsgroups)...")
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=['comp.graphics', 'rec.motorcycles', 'sci.space', 'talk.politics.guns'],
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create a DataFrame with the text data
df = pd.DataFrame({
    'text': newsgroups.data,
    'category': [newsgroups.target_names[i] for i in newsgroups.target]
})

# Sample a smaller subset for faster execution
sample_df = df.sample(1000, random_state=42)
print(f"Using {len(sample_df)} documents for topic modeling")

# Set preprocessing options with strong cleanup
preprocessing_options = {
    'enabled': True,
    'use_lemmatization': True,  # Set to False if spaCy is not available
    'remove_stopwords': True,
    'remove_punctuation': True,
    'min_token_length': 3,
    'custom_stopwords': ['say', 'one', 'would', 'could', 'get', 'also']
}

# Set deduplication options
deduplication_options = {
    'enabled': True,
    'method': 'exact',
    'hash_method': 'hash'
}

# Run automatic topic detection with different methods and algorithms
results = {}
for method in ['nmf', 'lda']:
    for topic_selection_method in ['coherence', 'stability']:
        key = f"{method}_{topic_selection_method}"
        print(f"\n{'-'*60}\nRunning automatic topic detection with {method.upper()} and {topic_selection_method} method...")
        result = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',                       # Use automatic topic detection
            method=method,                         # Try both NMF and LDA
            preprocessing_options=preprocessing_options,
            deduplication_options=deduplication_options,
            auto_topics_range=(2, 12),            # Try between 2 and 12 topics
            auto_topics_method=topic_selection_method,  # Try both methods
            use_multiprocessing=True
        )
        results[key] = result

# Visualize the coherence and stability scores
plt.figure(figsize=(14, 10))

# Set up subplots grid (2x2 for NMF/LDA and coherence/stability)
for i, method in enumerate(['nmf', 'lda']):
    # Coherence method plot (top row)
    coherence_key = f"{method}_coherence"
    topic_selection = results[coherence_key]['topic_selection']
    
    plt.subplot(2, 2, i+1)
    plt.plot(topic_selection['topic_range'], topic_selection['coherence_scores'], 'o-', 
             linewidth=2, color='blue', label='Coherence Score')
    
    # Mark the best number of topics
    best_idx = topic_selection['topic_range'].index(topic_selection['best_n_topics'])
    best_score = topic_selection['coherence_scores'][best_idx]
    plt.plot(topic_selection['best_n_topics'], best_score, 'ro', markersize=10)
    
    plt.title(f"{method.upper()} - Coherence Method")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(topic_selection['topic_range'][::2])  # Skip some ticks for readability
    
    # Add annotation for best number of topics
    plt.annotate(f"Optimal: {topic_selection['best_n_topics']} topics\nScore: {best_score:.4f}",
                xy=(topic_selection['best_n_topics'], best_score),
                xytext=(topic_selection['best_n_topics'] + 0.5, best_score - 0.01),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Stability method plot (bottom row)
    stability_key = f"{method}_stability"
    topic_selection = results[stability_key]['topic_selection']
    
    plt.subplot(2, 2, i+3)  # Bottom row (i+3 gives us indices 3 and 4)
    
    # Plot coherence scores for reference
    plt.plot(topic_selection['topic_range'], topic_selection['coherence_scores'], 's--', 
             linewidth=1, alpha=0.5, color='blue', label='Coherence Score')
    
    # Plot stability scores
    plt.plot(topic_selection['topic_range'], topic_selection['stability_scores'], 'o-', 
             linewidth=2, color='green', label='Stability Score')
    
    # Mark the best number of topics
    best_idx = topic_selection['topic_range'].index(topic_selection['best_n_topics'])
    best_score = topic_selection['stability_scores'][best_idx]
    plt.plot(topic_selection['best_n_topics'], best_score, 'ro', markersize=10)
    
    plt.title(f"{method.upper()} - Stability Method")
    plt.xlabel("Number of Topics")
    plt.ylabel("Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(topic_selection['topic_range'][::2])  # Skip some ticks for readability
    plt.legend()
    
    # Add annotation for best number of topics
    plt.annotate(f"Optimal: {topic_selection['best_n_topics']} topics\nScore: {best_score:.4f}",
                xy=(topic_selection['best_n_topics'], best_score),
                xytext=(topic_selection['best_n_topics'] + 0.5, best_score - 0.01),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.tight_layout()
plt.show()

# Print the results of the different methods and compare them
print("\n" + "="*80)
print("COMPARISON OF TOPIC DETECTION METHODS")
print("="*80)

for method in ['nmf', 'lda']:
    coherence_key = f"{method}_coherence"
    stability_key = f"{method}_stability"
    
    coherence_topics = results[coherence_key]['topic_selection']['best_n_topics']
    stability_topics = results[stability_key]['topic_selection']['best_n_topics']
    
    coherence_score = results[coherence_key]['topic_selection']['coherence_scores'][
        results[coherence_key]['topic_selection']['topic_range'].index(coherence_topics)
    ]
    stability_score = results[stability_key]['topic_selection']['stability_scores'][
        results[stability_key]['topic_selection']['topic_range'].index(stability_topics)
    ]
    
    print(f"\n{method.upper()} RESULTS:")
    print(f"  Coherence method: {coherence_topics} topics (score: {coherence_score:.4f})")
    print(f"  Stability method: {stability_topics} topics (score: {stability_score:.4f})")
    print(f"  {'Agreement' if coherence_topics == stability_topics else 'Disagreement'} between methods")

# Determine the best overall model based on stability-adjusted coherence
best_key = None
best_score = -1

for key in results:
    method, selection_method = key.split('_')
    
    if selection_method == 'stability':
        topic_selection = results[key]['topic_selection']
        best_idx = topic_selection['topic_range'].index(topic_selection['best_n_topics'])
        score = topic_selection['stability_scores'][best_idx]
        
        if score > best_score:
            best_score = score
            best_key = key

# Print the best model overall
method, selection_method = best_key.split('_')
print(f"\nBEST MODEL OVERALL: {method.upper()} with {results[best_key]['topic_selection']['best_n_topics']} topics")
print(f"Selected using {selection_method} method (score: {best_score:.4f})")

# Print the topics from the best model
print(f"\nTop words for each topic ({method.upper()}):")
best_model = results[best_key]
for i, (topic_idx, words) in enumerate(best_model['topics']):
    print(f"Topic {topic_idx+1}: {', '.join(words[:10])}")

print("\nTopic distribution across the dataset:")
# Calculate and print the topic distribution as percentage
topic_distribution = best_model['document_topics'].sum(axis=0)
topic_distribution = topic_distribution / topic_distribution.sum() * 100

for col, percentage in topic_distribution.items():
    print(f"{col}: {percentage:.1f}%")

# Visualize the topic distribution
plt.figure(figsize=(10, 6))
topic_distribution.plot(kind='bar', color='skyblue')
plt.title(f"Topic Distribution ({method.upper()} - {best_model['topic_selection']['best_n_topics']} topics)")
plt.ylabel("Percentage of Documents")
plt.xlabel("Topics")
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Show what topics were assigned to which documents (sample)
print("\nSample document topic assignments:")
sample_docs = min(5, len(sample_df))  # Show at most 5 documents
for i in range(sample_docs):
    doc_text = sample_df.iloc[i]['text']
    # Get top 2 topics for this document
    doc_topics = best_model['document_topics'].iloc[i]
    top_topics = doc_topics.sort_values(ascending=False).head(2)
    
    # Print truncated document and its top topics
    max_len = 80  # Truncate text for display
    truncated_text = doc_text[:max_len] + "..." if len(doc_text) > max_len else doc_text
    print(f"\nDocument {i+1}:\n{truncated_text}")
    print("Top topics:")
    for topic, weight in top_topics.items():
        topic_idx = int(topic.split()[1]) - 1  # Convert "Topic X" to index
        topic_words = best_model['topics'][topic_idx][1][:5]  # Get top 5 words
        print(f"  {topic} ({weight:.2f}): {', '.join(topic_words)}")

print("\nAutomatic topic detection complete!")