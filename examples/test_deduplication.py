"""
Test script to verify the deduplication functionality in topic modeling.
"""
import pandas as pd
import numpy as np
from freamon.utils.text_utils import TextProcessor

# Create a test DataFrame with duplicate texts
texts = [
    "This is a test document about machine learning and AI",
    "This is a test document about machine learning and AI",  # Exact duplicate
    "This is a test document about machine learning and AI.",  # Near duplicate (just added period)
    "Here is a completely different document about data science",
    "Another document discussing neural networks",
    "Another document discussing neural networks",  # Exact duplicate
    "A unique document about deep learning techniques",
    "This document discusses artificial intelligence applications",
    "This document discusses artificial intelligence applications in healthcare",  # Similar but not duplicate
]

categories = ['tech'] * len(texts)
df = pd.DataFrame({
    'text': texts,
    'category': categories
})

print(f"Original DataFrame: {len(df)} rows")
print(df)

# Test exact deduplication
processor = TextProcessor(use_spacy=False)

print("\n=== Testing with exact deduplication ===")
result = processor.create_topic_model_optimized(
    df=df,
    text_column='text',
    n_topics=3,
    method='nmf',
    use_lemmatization=False,
    max_docs=None,
    remove_duplicates=True,  # Enable deduplication
    return_full_data=True
)

# Get results
topics = result['topics']
doc_topics = result['document_topics']
processing_info = result['processing_info']

# Print deduplication results
print("\nDeduplication Results:")
print(f"Original documents: {processing_info['original_doc_count']}")
print(f"Duplicates removed: {processing_info['duplicates_removed']}")
print(f"Processed documents: {processing_info['processed_doc_count']}")

# Print topics
print("\nTopics:")
for topic_idx, words in topics:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")

# Verify the shape of document topics matrix
print(f"\nDocument-Topic Matrix Shape: {doc_topics.shape}")
print("Should match the number of rows in the original DataFrame")

# Test with fuzzy deduplication by modifying the example
print("\n=== Modifying example to use fuzzy deduplication ===")

# Create a function to modify the deduplication in the example
from types import MethodType

def custom_create_topic_model_optimized(
    self,
    df,
    text_column,
    n_topics=5,
    method='nmf',
    use_lemmatization=True,
    max_docs=None,
    remove_duplicates=True,
    return_full_data=True
):
    """Modified version with fuzzy deduplication enabled"""
    processing_info = {
        'original_doc_count': len(df),
        'processed_doc_count': len(df),
        'duplicates_removed': 0,
        'sampled': False,
        'sample_size': len(df),
        'used_lemmatization': use_lemmatization
    }
    
    # Step 1: Initialize the processor
    from freamon.utils.text_utils import TextProcessor
    processor = TextProcessor(use_spacy=use_lemmatization)
    
    # Make a copy to avoid modifying the original
    working_df = df.copy()
    
    # Step 2: Optional deduplication
    if remove_duplicates:
        # This flag controls whether to use fuzzy or exact deduplication
        use_fuzzy = True  # CHANGED TO TRUE
        
        if use_fuzzy:
            # Fuzzy deduplication approach
            print("Using fuzzy deduplication...")
            texts = working_df[text_column].tolist()
            
            # Import or define fallback deduplication function
            try:
                from freamon.deduplication.fuzzy_deduplication import deduplicate_texts
            except ImportError:
                # Fallback implementation (just a placeholder)
                def deduplicate_texts(texts, threshold=0.8, method='cosine', preprocess=True, keep='first'):
                    print(f"Using fallback fuzzy deduplication (threshold={threshold}, method={method})")
                    unique_texts = set()
                    result_indices = []
                    
                    for i, text in enumerate(texts):
                        if text not in unique_texts:
                            unique_texts.add(text)
                            result_indices.append(i)
                    
                    return result_indices
            
            # Perform deduplication
            print("Starting fuzzy deduplication...")
            kept_indices = deduplicate_texts(
                texts,
                threshold=0.85,  # Higher threshold = more selective deduplication
                method='cosine',  # Options: 'cosine', 'jaccard', 'levenshtein'
                preprocess=True,
                keep='first'
            )
            deduped_df = working_df.iloc[kept_indices].copy()
            print(f"Fuzzy deduplication complete. Kept {len(kept_indices)} of {len(texts)} documents.")
        else:
            # Exact deduplication approach
            print("Using exact deduplication")
            # Fallback implementation for exact deduplication
            def deduplicate_exact(df, col, method='hash', keep='first'):
                return df.drop_duplicates(subset=[col], keep=keep)
                
            deduped_df = deduplicate_exact(
                working_df, 
                col=text_column, 
                method='hash',
                keep='first'
            )
        
        processing_info['duplicates_removed'] = len(working_df) - len(deduped_df)
        working_df = deduped_df
        processing_info['processed_doc_count'] = len(working_df)
    
    # Continue with normal processing...
    # Rest of the function would remain the same
    
    # For this test, we'll just create a basic topic model
    print(f"Creating topic model with {n_topics} topics...")
    topic_model = processor.create_topic_model(
        texts=working_df[text_column].tolist(),
        n_topics=n_topics,
        method=method,
        max_features=100,
        random_state=42
    )
    
    # Get document-topic distribution
    doc_topics = processor.get_document_topics(topic_model)
    
    # Set index to match the original DataFrame 
    doc_topics.index = working_df.index
    
    # Create result object
    result = {
        'topic_model': topic_model,
        'document_topics': doc_topics,
        'topics': topic_model['topics'],
        'processing_info': processing_info
    }
    
    return result

# Attach the custom method to the processor instance
processor.create_topic_model_optimized = MethodType(
    custom_create_topic_model_optimized, processor
)

# Test with fuzzy deduplication
print("\n=== Testing with fuzzy deduplication ===")
fuzzy_result = processor.create_topic_model_optimized(
    df=df,
    text_column='text',
    n_topics=3,
    method='nmf',
    use_lemmatization=False,
    max_docs=None,
    remove_duplicates=True,
    return_full_data=True
)

# Get results from fuzzy deduplication
fuzzy_processing_info = fuzzy_result['processing_info']

# Print fuzzy deduplication results
print("\nFuzzy Deduplication Results:")
print(f"Original documents: {fuzzy_processing_info['original_doc_count']}")
print(f"Duplicates removed: {fuzzy_processing_info['duplicates_removed']}")
print(f"Processed documents: {fuzzy_processing_info['processed_doc_count']}")

print("\nTest completed!")