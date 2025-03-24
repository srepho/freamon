"""
Example demonstrating text deduplication functionality.
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from freamon.utils.text_utils import TextProcessor
from freamon.deduplication import (
    hash_deduplication,
    ngram_fingerprint_deduplication,
    find_similar_texts,
    deduplicate_texts,
    lsh_deduplication,
    cluster_texts_hierarchical
)

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

# Create exact duplicates by copying some entries
duplicate_indices = [5, 10, 15, 20, 25]
duplicates = [(newsgroup_data[i], newsgroup_targets[i]) for i in duplicate_indices]

# Add exact duplicates to the dataset
for text, target in duplicates * 2:  # Add each duplicate twice
    newsgroup_data.append(text)
    newsgroup_targets.append(target)

# Create near duplicates by modifying some entries slightly
near_dup_indices = [30, 35, 40, 45, 50]
near_duplicates = []

for i in near_dup_indices:
    text = newsgroup_data[i]
    # Add some typos or modifications
    modified_text = text.replace("the", "teh").replace("a", "aa").replace("and", "adn")
    near_duplicates.append((modified_text, newsgroup_targets[i]))

# Add near duplicates to the dataset
for text, target in near_duplicates:
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

print(f"After adding empty texts: {len(df)} documents\n")

# Initialize the TextProcessor
processor = TextProcessor(use_spacy=False)

# Benchmark different deduplication methods
methods = {
    "Hash-based (Exact)": hash_deduplication,
    "N-gram Fingerprint": ngram_fingerprint_deduplication,
    "Text Similarity (Cosine)": lambda texts: deduplicate_texts(texts, method='cosine'),
    "Text Similarity (Jaccard)": lambda texts: deduplicate_texts(texts, method='jaccard'),
    "Text Similarity (Levenshtein)": lambda texts: deduplicate_texts(texts, method='levenshtein'),
    "Embedding-based": lambda texts: deduplicate_texts(texts, method='embedding'),
    "LSH (MinHash)": lsh_deduplication
}

# Run benchmark
results = {}
print("Benchmarking deduplication methods...")
print("-" * 50)

for name, method in methods.items():
    print(f"Testing {name}...")
    start_time = time.time()
    
    try:
        unique_indices = method(df['text'])
        df_unique = df.iloc[unique_indices].copy()
        elapsed = time.time() - start_time
        
        # Record results
        results[name] = {
            'time': elapsed,
            'unique_count': len(df_unique),
            'dedup_count': len(df) - len(df_unique),
            'dedup_percentage': (len(df) - len(df_unique)) / len(df) * 100
        }
        
        print(f"  Time: {elapsed:.4f} seconds")
        print(f"  Unique documents: {len(df_unique)}")
        print(f"  Removed duplicates: {len(df) - len(df_unique)} ({results[name]['dedup_percentage']:.2f}%)")
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print()

# Example of using clustering for duplicate detection
print("\nExploring text clusters with hierarchical clustering...")
clusters = cluster_texts_hierarchical(
    df['text'],
    distance_threshold=0.3,  # Adjust based on desired similarity
    method='cosine'
)

print(f"Number of clusters: {len(clusters)}")

# Show some statistics about clusters
cluster_sizes = [len(indices) for indices in clusters.values()]
print(f"Largest cluster size: {max(cluster_sizes)}")
print(f"Average cluster size: {sum(cluster_sizes) / len(clusters):.2f}")

# Show a specific cluster example
for cluster_id, indices in clusters.items():
    if len(indices) > 1:  # Show a non-singleton cluster
        print(f"\nCluster {cluster_id} with {len(indices)} documents:")
        for i, idx in enumerate(indices[:3]):  # Show the first 3 documents
            doc_preview = df.iloc[idx]['text'][:100].replace('\n', ' ').strip()
            category = df.iloc[idx]['category']
            print(f"  {i+1}. [{category}] {doc_preview}...")
        
        if len(indices) > 3:
            print(f"  ... and {len(indices) - 3} more")
        break

# Visualize benchmark results
if results:
    plt.figure(figsize=(12, 6))
    
    # Time comparison
    plt.subplot(1, 2, 1)
    times = [results[method]['time'] for method in results]
    plt.bar(results.keys(), times)
    plt.title('Processing Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Deduplication percentage
    plt.subplot(1, 2, 2)
    percentages = [results[method]['dedup_percentage'] for method in results]
    plt.bar(results.keys(), percentages)
    plt.title('Deduplication Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('deduplication_benchmark.png')
    print("\nBenchmark visualization saved to 'deduplication_benchmark.png'")

print("\nExample complete!")