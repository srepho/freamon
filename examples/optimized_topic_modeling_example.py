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
from freamon.deduplication.exact_deduplication import deduplicate_exact

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

# Define the new optimized function for topic modeling
def create_topic_model_optimized(df, text_column, n_topics=5, method='nmf', 
                               use_lemmatization=True, max_docs=None,
                               remove_duplicates=True, return_full_data=True):
    """
    Optimized topic modeling workflow that handles preprocessing, deduplication,
    and smart sampling automatically.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the text data
    text_column : str
        Name of the column containing text to analyze
    n_topics : int, default=5
        Number of topics to extract
    method : str, default='nmf'
        Topic modeling method ('nmf' or 'lda')
    use_lemmatization : bool, default=True
        Whether to use lemmatization (requires spaCy)
    max_docs : int, default=None
        Maximum number of documents to process (None = use all if < 10000, else use 10000)
    remove_duplicates : bool, default=True
        Whether to remove duplicate documents before processing
    return_full_data : bool, default=True
        Whether to return topic distributions for all documents, not just the sample
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'topic_model': Dictionary with the trained model and topics
        - 'document_topics': DataFrame with document-topic distributions
        - 'topics': List of (topic_idx, words) tuples
        - 'processing_info': Dict with processing statistics
    """
    processing_info = {
        'original_doc_count': len(df),
        'processed_doc_count': len(df),
        'duplicates_removed': 0,
        'sampled': False,
        'sample_size': len(df),
        'used_lemmatization': use_lemmatization
    }
    
    # Step 1: Initialize the processor
    processor = TextProcessor(use_spacy=use_lemmatization)
    
    # Make a copy to avoid modifying the original
    working_df = df.copy()
    
    # Step 2: Optional deduplication
    if remove_duplicates:
        deduped_df = time_operation(
            "Removing duplicate documents",
            deduplicate_exact,
            working_df, 
            col=text_column, 
            method='hash',
            keep='first'
        )
        processing_info['duplicates_removed'] = len(working_df) - len(deduped_df)
        working_df = deduped_df
        processing_info['processed_doc_count'] = len(working_df)
        
    # Step 3: Smart sampling for very large datasets
    if max_docs is None:
        # Default behavior: process all docs if <10K, otherwise sample 10K
        max_docs = 10000
        
    if len(working_df) > max_docs:
        processing_info['sampled'] = True
        processing_info['sample_size'] = max_docs
        print(f"Dataset has {len(working_df)} documents, sampling {max_docs} for topic modeling...")
        sample_df = working_df.sample(max_docs, random_state=42)
    else:
        sample_df = working_df
    
    # Step 4: Preprocess texts (with progress reporting)
    print(f"Preprocessing {len(sample_df)} documents...")
    start_time = time.time()
    
    # Process in batches for better progress reporting
    batch_size = max(1, min(1000, len(sample_df) // 10))
    cleaned_texts = []
    
    for i in range(0, len(sample_df), batch_size):
        batch = sample_df.iloc[i:i+batch_size]
        batch_texts = batch[text_column].tolist()
        
        # Process batch
        processed_batch = [processor.preprocess_text(
            text, 
            remove_stopwords=True, 
            remove_punctuation=True,
            lemmatize=use_lemmatization
        ) for text in batch_texts]
        
        cleaned_texts.extend(processed_batch)
        
        # Report progress
        progress = min(100, (i + len(batch)) * 100 // len(sample_df))
        elapsed = time.time() - start_time
        print(f"  Progress: {progress}% ({i + len(batch)}/{len(sample_df)}) - {elapsed:.1f}s", end='\r')
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds                      ")
    
    # Step 5: Create the topic model
    print(f"Creating {method.upper()} topic model with {n_topics} topics...")
    topic_model = time_operation(
        f"Creating {n_topics}-topic model",
        processor.create_topic_model,
        texts=cleaned_texts,
        n_topics=n_topics,
        method=method,
        max_features=min(1000, len(cleaned_texts) // 2),
        max_df=0.7,
        min_df=3,
        ngram_range=(1, 2),
        random_state=42
    )
    
    # Step 6: Get document-topic distribution for the sample
    doc_topics = processor.get_document_topics(topic_model)
    
    # Step 7: If requested and we've sampled, process the full dataset
    if return_full_data and processing_info['sampled']:
        print("Generating topic distributions for all documents...")
        
        # First, create a mapping to store all results
        all_doc_topics = pd.DataFrame(index=working_df.index)
        
        # Add the sample results we already calculated
        for col in doc_topics.columns:
            all_doc_topics.loc[sample_df.index, col] = doc_topics[col].values
        
        # Process remaining documents in batches
        remaining_idx = working_df.index.difference(sample_df.index)
        remaining_df = working_df.loc[remaining_idx]
        
        if len(remaining_df) > 0:
            batch_size = max(1, min(1000, len(remaining_df) // 10))
            
            for i in range(0, len(remaining_df), batch_size):
                batch = remaining_df.iloc[i:i+batch_size]
                
                # Preprocess batch
                batch_texts = [processor.preprocess_text(
                    text, 
                    remove_stopwords=True, 
                    remove_punctuation=True,
                    lemmatize=use_lemmatization
                ) for text in batch[text_column]]
                
                # Get topic distribution
                batch_vectors = topic_model['vectorizer'].transform(batch_texts)
                
                if method == 'lda':
                    batch_topics = topic_model['model'].transform(batch_vectors)
                else:  # nmf
                    batch_topics = topic_model['model'].transform(batch_vectors)
                
                # Store results
                for j, idx in enumerate(batch.index):
                    for topic_idx in range(n_topics):
                        col_name = f"Topic {topic_idx+1}"
                        all_doc_topics.loc[idx, col_name] = batch_topics[j, topic_idx]
                
                # Report progress
                progress = min(100, (i + len(batch)) * 100 // len(remaining_df))
                print(f"  Generating topics: {progress}%", end='\r')
            
            print("Topic generation completed                      ")
            doc_topics = all_doc_topics
    
    result = {
        'topic_model': topic_model,
        'document_topics': doc_topics,
        'topics': topic_model['topics'],
        'processing_info': processing_info
    }
    
    return result

# Run the optimized topic modeling
print("\nRunning optimized topic modeling workflow...")
result = create_topic_model_optimized(
    df, 
    text_column='text',
    n_topics=6,
    method='nmf',  # NMF is faster than LDA
    use_lemmatization=True,
    max_docs=1000,  # Limit for demo purposes
    remove_duplicates=True,
    return_full_data=True
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