"""
Test script for demonstrating the optimized topic modeling functionality.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import time

from freamon.utils.text_utils import TextProcessor

print("=== Optimized Topic Modeling Test ===")

# Load a sample of the 20 newsgroups dataset
print("Loading 20 newsgroups dataset sample...")
categories = ['rec.autos', 'sci.med', 'comp.graphics']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create sample DataFrame
df = pd.DataFrame({
    'text': newsgroups.data[:300],  # Just take a small sample
    'category': [newsgroups.target_names[target] for target in newsgroups.target[:300]]
})

print(f"Created sample dataset with {len(df)} documents across {len(categories)} categories")

# Create duplicate documents for testing deduplication
print("Adding duplicate documents for testing deduplication...")
duplicates = df.sample(50, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)
print(f"Dataset now has {len(df)} documents (including {len(duplicates)} duplicates)")

# Initialize TextProcessor
processor = TextProcessor(use_spacy=True)

print("\n=== Testing Basic Topic Modeling ===")
# Time standard approach
start_time = time.time()

# Step 1: Clean and lemmatize texts
print("Processing with standard approach...")
cleaned_texts = []
for text in df['text']:
    cleaned = processor.preprocess_text(
        text,
        remove_stopwords=True,
        remove_punctuation=True,
        lemmatize=True
    )
    cleaned_texts.append(cleaned)

# Step 2: Create topic model
standard_model = processor.create_topic_model(
    texts=cleaned_texts,
    n_topics=3,
    method='nmf',
    max_features=500
)

# Step 3: Get document-topic distribution
standard_doc_topics = processor.get_document_topics(standard_model)

standard_time = time.time() - start_time
print(f"Standard approach completed in {standard_time:.2f} seconds")

print("\n=== Testing Optimized Topic Modeling ===")
# Time optimized approach
start_time = time.time()

# One-line optimized approach
optimized_result = processor.create_topic_model_optimized(
    df,
    text_column='text',
    n_topics=3,
    method='nmf',
    use_lemmatization=True,
    remove_duplicates=True,
    return_full_data=True
)

optimized_time = time.time() - start_time
print(f"Optimized approach completed in {optimized_time:.2f} seconds")

# Print processing information
info = optimized_result['processing_info']
print("\nProcessing Information:")
print(f"  Original documents: {info['original_doc_count']}")
print(f"  Duplicates removed: {info['duplicates_removed']}")
print(f"  Documents after deduplication: {info['processed_doc_count']}")

# Print topics from both approaches
print("\nStandard Approach Topics:")
for topic_idx, words in standard_model['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:5])}")

print("\nOptimized Approach Topics:")
for topic_idx, words in optimized_result['topics']:
    print(f"Topic {topic_idx + 1}: {', '.join(words[:5])}")

# Compare performance
print("\nPerformance Comparison:")
print(f"  Standard approach: {standard_time:.2f} seconds")
print(f"  Optimized approach: {optimized_time:.2f} seconds")
if standard_time > 0:
    speedup = standard_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"  Speedup factor: {speedup:.2f}x")

print("\nTest completed successfully!")