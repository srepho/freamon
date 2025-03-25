"""
Example demonstrating the optimized topic modeling with DataFrame integration.

This example shows how to use the enhanced topic modeling functionality on a DataFrame
with configurable preprocessing, deduplication, and multiprocessing options.
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import multiprocessing

# Import necessary components
from freamon.utils.text_utils import TextProcessor, create_topic_model_optimized

# Configure multiprocessing
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Method already set
    multiprocessing.freeze_support()
    
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
    
    # Add some duplicate texts to demonstrate deduplication
    print("Adding some duplicate texts to demonstrate deduplication...")
    duplicate_indices = np.random.choice(len(df), size=min(100, len(df) // 5), replace=False)
    
    # Create additional rows with duplicated text (but potentially different categories)
    duplicate_rows = []
    for idx in duplicate_indices:
        row = df.iloc[idx].copy()
        # Sometimes add a small modification to test fuzzy matching
        if np.random.random() < 0.3:
            row['text'] = row['text'] + " " + np.random.choice(["", ".", "!", "?"])
        duplicate_rows.append(row)
    
    # Add duplicates to the dataframe
    duplicates_df = pd.DataFrame(duplicate_rows)
    df = pd.concat([df, duplicates_df], ignore_index=True)
    print(f"Dataset shape after adding duplicates: {df.shape}")
    
    # Configure preprocessing options
    preprocessing_options = {
        'enabled': True,  # Set to False to use raw texts without preprocessing
        'use_lemmatization': True,
        'remove_stopwords': True,
        'remove_punctuation': True,
        'min_token_length': 3,
        'custom_stopwords': ['said', 'would', 'could', 'also'],  # Domain-specific stopwords
        'batch_size': 500  # Customize batch size for your hardware
    }
    
    # Configure deduplication options
    deduplication_options = {
        'enabled': True,  # Set to False to keep duplicates
        'method': 'exact',  # 'exact', 'fuzzy', or 'none'
        'hash_method': 'hash',  # For exact: 'hash' or 'ngram'
        'similarity_threshold': 0.85,  # For fuzzy: threshold for similarity
        'similarity_method': 'cosine',  # For fuzzy: 'cosine', 'jaccard', 'levenshtein'
        'keep': 'first'  # 'first' or 'last' duplicate to keep
    }
    
    # Run the enhanced topic modeling
    print("\nRunning enhanced topic modeling on DataFrame...")
    result = create_topic_model_optimized(
        df, 
        text_column='text',
        n_topics=6,
        method='nmf',  # 'nmf' or 'lda'
        preprocessing_options=preprocessing_options,
        max_docs=500,  # Limit for this example; set to None for auto-sizing
        deduplication_options=deduplication_options,
        return_full_data=True,  # Get topics for all documents, not just the sample
        return_original_mapping=True,  # Get mapping from deduplicated docs to originals
        use_multiprocessing=True  # Enable multiprocessing for large datasets
    )
    
    # Print processing information
    info = result['processing_info']
    print("\nProcessing Information:")
    print(f"  Original documents: {info['original_doc_count']}")
    print(f"  Duplicates removed: {info['duplicates_removed']}")
    print(f"  Processed documents: {info['processed_doc_count']}")
    if info['sampled']:
        print(f"  Sample size used for modeling: {info['sample_size']}")
    print(f"  Lemmatization used: {info['used_lemmatization']}")
    print(f"  Preprocessing enabled: {info['preprocessing_enabled']}")
    if 'preprocessing_time' in info:
        print(f"  Preprocessing time: {info['preprocessing_time']:.2f} seconds")
    if 'multiprocessing_enabled' in info:
        print(f"  Multiprocessing: {info['multiprocessing_enabled']}")
    
    # Print deduplication mapping size
    if 'deduplication_map' in result:
        dedup_map = result['deduplication_map']
        total_duplicates = sum(len(indices) - 1 for indices in dedup_map.values())
        print(f"  Total duplicates mapped: {total_duplicates}")
        # Show a sample of the deduplication mapping
        sample_key = next(iter(dedup_map))
        if len(dedup_map[sample_key]) > 1:
            print(f"  Sample duplicate mapping: {sample_key} -> {dedup_map[sample_key]}")
    
    # Print topics
    print("\nTop 10 words for each topic:")
    for topic_idx, words in result['topics']:
        print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
    
    # Get document-topic distributions
    doc_topics_df = result['document_topics']
    
    # Add dominant topic for each document
    print("\nAdding dominant topic to each document...")
    topic_cols = [col for col in doc_topics_df.columns if col.startswith('Topic')]
    
    # Function to get dominant topic
    def get_dominant_topic(row):
        max_topic = row[topic_cols].idxmax()
        max_prob = row[max_topic]
        return int(max_topic.split()[1]), max_prob
    
    # Apply to get dominant topics and probabilities
    dominant_topics = pd.DataFrame(
        doc_topics_df.apply(get_dominant_topic, axis=1).tolist(),
        index=doc_topics_df.index,
        columns=['dominant_topic', 'topic_probability']
    )
    
    # Add to original DataFrame
    enhanced_df = df.copy()
    enhanced_df['dominant_topic'] = dominant_topics['dominant_topic'].reindex(df.index)
    enhanced_df['topic_probability'] = dominant_topics['topic_probability'].reindex(df.index)
    
    # Show sample of enhanced DataFrame
    print("\nSample of enhanced DataFrame with dominant topics:")
    print(enhanced_df[['category', 'dominant_topic', 'topic_probability']].sample(5))
    
    # Calculate topic distribution by category
    print("\nTopic distribution by category:")
    topic_by_category = pd.crosstab(
        enhanced_df['category'], 
        enhanced_df['dominant_topic'],
        normalize='index'
    )
    print(topic_by_category)
    
    # Visualize topic distribution by category
    plt.figure(figsize=(12, 8))
    topic_by_category.plot(
        kind='bar', 
        stacked=True,
        colormap='viridis',
        figsize=(12, 6)
    )
    plt.title('Topic Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.legend(title='Topic')
    plt.tight_layout()
    plt.savefig('topic_distribution_by_category.png')
    print("Saved topic distribution visualization to topic_distribution_by_category.png")
    
    # Visualize topics with probabilities
    print("\nGenerating topic probability distribution...")
    
    # Plot histogram of topic probabilities
    plt.figure(figsize=(10, 6))
    enhanced_df['topic_probability'].hist(bins=20)
    plt.title('Distribution of Topic Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('topic_probability_distribution.png')
    print("Saved topic probability distribution to topic_probability_distribution.png")
    
    # Save enhanced DataFrame to CSV for later use
    enhanced_df.to_csv('topics_enhanced_dataset.csv', index=False)
    print("Saved enhanced dataset to topics_enhanced_dataset.csv")
    
    print("\nExample complete!")