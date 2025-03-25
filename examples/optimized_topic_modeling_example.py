"""
Example demonstrating optimized topic modeling workflow in freamon.
This example shows how to efficiently process text data for topic modeling
with automatic optimization for large datasets.
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from freamon.utils.text_utils import TextProcessor
# Import necessary components
# Import fallback implementation since deduplicate_exact doesn't exist
try:
    from freamon.deduplication.fuzzy_deduplication import deduplicate_texts
except ImportError:
    # Fallback deduplication implementation
    def deduplicate_texts(texts, threshold=0.8, method='cosine', preprocess=True, keep='first'):
        print(f"Using fallback fuzzy deduplication with {method} method and threshold {threshold}")
        return list(range(len(texts)))  # Return all indices as if no duplicates were found

def deduplicate_exact(df, col, method='hash', keep='first'):
    print(f"Using fallback deduplication - removing exact duplicates in {col}")
    return df.drop_duplicates(subset=[col], keep=keep)

# Add freeze_support for multiprocessing
import multiprocessing
from multiprocessing import set_start_method

if __name__ == '__main__':
    # Needed for multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    multiprocessing.freeze_support()
    
    print("=== Optimized Topic Modeling Example ===")
    
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

    # Utility function for timing operations
    def time_operation(operation_name, func, *args, **kwargs):
        """Run a function and print its execution time"""
        print(f"Starting: {operation_name}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Completed: {operation_name} in {elapsed:.2f} seconds")
        return result

# Import the optimized topic modeling function
from freamon.utils.text_utils import create_topic_model_optimized

if __name__ == '__main__':
    # Run the optimized topic modeling
    print("\nRunning optimized topic modeling workflow...")
    
    # Create preprocessing and deduplication options
    preprocessing_options = {
        'enabled': True,
        'use_lemmatization': True,
        'remove_stopwords': True,
        'remove_punctuation': True,
        'min_token_length': 3,
        'custom_stopwords': []
    }
    
    deduplication_options = {
        'enabled': True,
        'method': 'exact',  # 'exact', 'fuzzy', or 'none'
        'hash_method': 'hash',
        'similarity_threshold': 0.85,
        'similarity_method': 'cosine'
    }
    
    result = create_topic_model_optimized(
        df, 
        text_column='text',
        n_topics=6,
        method='nmf',  # NMF is faster than LDA
        preprocessing_options=preprocessing_options,
        max_docs=200,  # Lower limit for faster execution
        deduplication_options=deduplication_options,
        return_full_data=True,
        return_original_mapping=True,
        use_multiprocessing=True
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
    print(f"  Deduplication method: {info['deduplication_method']}")
    if 'preprocessing_time' in info:
        print(f"  Preprocessing time: {info['preprocessing_time']:.2f} seconds")
    if 'multiprocessing_enabled' in info:
        print(f"  Multiprocessing: {info['multiprocessing_enabled']}")
        if info['multiprocessing_enabled'] and 'num_workers' in info:
            print(f"  Workers: {info['num_workers']}")

    # Print topics
    print("\nTop 10 words for each topic:")
    for topic_idx, words in result['topics']:
        print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
    
    # Get topic distribution dataframe
    doc_topics = result['document_topics']
    print(f"\nTopic distribution shape: {doc_topics.shape}")
    print("First 5 documents topic distribution:")
    print(doc_topics.head(5))
    
    # Combine with original categories
    topic_with_category = doc_topics.copy()
    topic_with_category['category'] = df.loc[topic_with_category.index, 'category'].values
    
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
    plt.savefig('optimized_category_topic_distribution.png')
    print("Saved category-topic distribution heatmap to optimized_category_topic_distribution.png")

    # Visualize topics
    print("\nGenerating topic visualization...")
    processor = TextProcessor(use_spacy=True)
    html = result['topic_model'].get('visualizer') or processor.plot_topics(
        result['topic_model'], 
        figsize=(15, 10), 
        return_html=True
    )
    
    # Save visualization
    with open("optimized_topic_model_visualization.html", "w") as f:
        f.write(f"<html><body>{html}</body></html>")
    print("Saved topic visualization to optimized_topic_model_visualization.html")
    
    print("\nExample complete!")