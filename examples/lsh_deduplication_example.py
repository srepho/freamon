"""
Example demonstrating the LSH (Locality Sensitive Hashing) deduplication functionality.

This example shows:
1. How LSH efficiently finds similar texts without comparing all document pairs
2. Parameter tuning for different similarity thresholds
3. Performance comparison with other deduplication methods
4. Visualization of results
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import random
from sklearn.datasets import fetch_20newsgroups

from freamon.utils.text_utils import TextProcessor
from freamon.deduplication import (
    hash_deduplication,
    ngram_fingerprint_deduplication,
    deduplicate_texts,
    lsh_deduplication
)

def create_sample_dataset(size=1000, duplicate_rate=0.2, near_duplicate_rate=0.1):
    """Create a sample dataset with exact and near duplicates."""
    print(f"Creating sample dataset with {size} documents...")
    
    # Load subset of 20 newsgroups
    categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Take a subset as base documents
    base_size = int(size * (1 - duplicate_rate - near_duplicate_rate))
    base_indices = np.random.choice(len(newsgroups.data), min(base_size, len(newsgroups.data)), replace=False)
    
    # Create dataframe
    texts = [newsgroups.data[i] for i in base_indices]
    categories = [newsgroups.target_names[newsgroups.target[i]] for i in base_indices]
    ids = list(range(len(texts)))
    
    # Add exact duplicates
    dup_size = int(size * duplicate_rate)
    dup_indices = np.random.choice(len(texts), dup_size)
    for idx in dup_indices:
        texts.append(texts[idx])
        categories.append(categories[idx])
        ids.append(len(texts) - 1)
    
    # Add near duplicates
    near_dup_size = int(size * near_duplicate_rate)
    near_dup_indices = np.random.choice(len(base_indices), near_dup_size)
    
    for idx in near_dup_indices:
        original = texts[idx]
        
        # Modify text slightly
        words = original.split()
        if len(words) > 10:
            # Replace 10% of words with modified versions
            num_to_modify = max(1, int(len(words) * 0.1))
            for _ in range(num_to_modify):
                pos = random.randint(0, len(words) - 1)
                words[pos] = words[pos][::-1]  # Reverse the word
            
            # Add a random sentence
            words.append("This sentence was added to create a near duplicate.")
            
            modified = ' '.join(words)
            texts.append(modified)
            categories.append(categories[idx])
            ids.append(len(texts) - 1)
    
    print(f"Created dataset with {len(texts)} documents")
    print(f"- Base documents: {base_size}")
    print(f"- Exact duplicates: {dup_size}")
    print(f"- Near duplicates: {near_dup_size}")
    
    return pd.DataFrame({
        'id': range(len(texts)),
        'original_id': ids,
        'text': texts,
        'category': categories,
        'length': [len(t) for t in texts]
    })

def compare_deduplication_methods(df):
    """Compare different deduplication methods."""
    print("\n===== Comparing Deduplication Methods =====")
    
    methods = {
        "Hash-based (MD5)": lambda texts: hash_deduplication(texts, hash_func='md5'),
        "N-gram Fingerprint (n=3)": lambda texts: ngram_fingerprint_deduplication(texts, n=3),
        "LSH (threshold=0.7)": lambda texts: lsh_deduplication(
            texts, threshold=0.7, num_minhash_permutations=100, num_bands=20
        ),
        "LSH (threshold=0.8)": lambda texts: lsh_deduplication(
            texts, threshold=0.8, num_minhash_permutations=100, num_bands=20
        ),
        "LSH (threshold=0.9)": lambda texts: lsh_deduplication(
            texts, threshold=0.9, num_minhash_permutations=100, num_bands=20
        ),
    }
    
    # Add similarity-based method only for smaller datasets
    if len(df) <= 5000:
        methods["Cosine Similarity"] = lambda texts: deduplicate_texts(texts, method='cosine', threshold=0.7)
    
    results = {}
    for name, method in methods.items():
        print(f"\nTesting {name}...")
        
        start_time = time.time()
        unique_indices = method(df['text'])
        elapsed = time.time() - start_time
        
        # Get results
        df_unique = df.iloc[unique_indices].copy()
        
        # Calculate how many duplicates were detected
        total_dupes = len(df) - len(df_unique)
        exact_dupes = len(df) - len(df['original_id'].unique())
        near_dupes = total_dupes - exact_dupes
        
        results[name] = {
            'time': elapsed,
            'unique_count': len(df_unique),
            'dedup_count': total_dupes,
            'exact_dupes': exact_dupes,
            'near_dupes': near_dupes,
            'dedup_percentage': total_dupes / len(df) * 100
        }
        
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Unique documents: {len(df_unique)}")
        print(f"Removed duplicates: {total_dupes} ({results[name]['dedup_percentage']:.2f}%)")
        print(f"Exact duplicates found: {exact_dupes}")
        print(f"Near duplicates found: {near_dupes}")
    
    return results

def analyze_lsh_parameters(df, sample_size=1000):
    """Analyze the effect of LSH parameters on performance and accuracy."""
    print("\n===== Analyzing LSH Parameters =====")
    
    # Use a sample for parameter testing
    df_sample = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    
    # Parameter combinations to test
    param_tests = [
        {"permutations": 50, "bands": 10, "threshold": 0.7},
        {"permutations": 100, "bands": 20, "threshold": 0.7},
        {"permutations": 200, "bands": 40, "threshold": 0.7},
        {"permutations": 100, "bands": 10, "threshold": 0.7},
        {"permutations": 100, "bands": 50, "threshold": 0.7},
    ]
    
    results = {}
    for params in param_tests:
        name = f"LSH (p={params['permutations']}, b={params['bands']}, t={params['threshold']})"
        print(f"\nTesting {name}...")
        
        start_time = time.time()
        unique_indices = lsh_deduplication(
            df_sample['text'],
            num_minhash_permutations=params['permutations'],
            num_bands=params['bands'],
            threshold=params['threshold']
        )
        elapsed = time.time() - start_time
        
        # Get results
        df_unique = df_sample.iloc[unique_indices].copy()
        
        # Calculate duplicates
        total_dupes = len(df_sample) - len(df_unique)
        exact_dupes = len(df_sample) - len(df_sample['original_id'].unique())
        near_dupes = total_dupes - exact_dupes
        
        results[name] = {
            'time': elapsed,
            'unique_count': len(df_unique),
            'dedup_count': total_dupes,
            'exact_dupes': exact_dupes,
            'near_dupes': near_dupes,
            'dedup_percentage': total_dupes / len(df_sample) * 100
        }
        
        print(f"Time: {elapsed:.4f} seconds")
        print(f"Unique documents: {len(df_unique)}")
        print(f"Removed duplicates: {total_dupes} ({results[name]['dedup_percentage']:.2f}%)")
    
    return results

def visualize_results(standard_results, parameter_results=None):
    """Visualize the deduplication results."""
    # Create output directory
    output_dir = Path("dedup_output")
    output_dir.mkdir(exist_ok=True)
    
    # Plot standard results
    plt.figure(figsize=(14, 6))
    
    # Time comparison
    plt.subplot(1, 3, 1)
    times = [standard_results[method]['time'] for method in standard_results]
    plt.bar(standard_results.keys(), times)
    plt.title('Processing Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Deduplication percentage
    plt.subplot(1, 3, 2)
    percentages = [standard_results[method]['dedup_percentage'] for method in standard_results]
    plt.bar(standard_results.keys(), percentages)
    plt.title('Deduplication Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Near duplicate detection
    plt.subplot(1, 3, 3)
    near_dupes = [standard_results[method]['near_dupes'] for method in standard_results]
    plt.bar(standard_results.keys(), near_dupes)
    plt.title('Near-Duplicates Detected')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'deduplication_methods.png')
    print(f"Method comparison saved to '{output_dir}/deduplication_methods.png'")
    
    # Plot parameter results if available
    if parameter_results:
        plt.figure(figsize=(14, 6))
        
        # Time comparison
        plt.subplot(1, 3, 1)
        times = [parameter_results[method]['time'] for method in parameter_results]
        plt.bar(parameter_results.keys(), times)
        plt.title('Processing Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Deduplication percentage
        plt.subplot(1, 3, 2)
        percentages = [parameter_results[method]['dedup_percentage'] for method in parameter_results]
        plt.bar(parameter_results.keys(), percentages)
        plt.title('Deduplication Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Near duplicate detection
        plt.subplot(1, 3, 3)
        near_dupes = [parameter_results[method]['near_dupes'] for method in parameter_results]
        plt.bar(parameter_results.keys(), near_dupes)
        plt.title('Near-Duplicates Detected')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_dir / 'lsh_parameters.png')
        print(f"Parameter comparison saved to '{output_dir}/lsh_parameters.png'")

def explore_similar_documents(df, text_index=None, threshold=0.7):
    """Explore documents similar to a given one using LSH."""
    print("\n===== Exploring Similar Documents =====")
    
    # If no index is provided, choose a random one
    if text_index is None:
        # Try to find a document with duplicates
        original_ids = df['original_id'].value_counts()
        duplicated_ids = original_ids[original_ids > 1].index.tolist()
        
        if duplicated_ids:
            # Find a document that has duplicates
            sample_id = random.choice(duplicated_ids)
            text_index = df[df['original_id'] == sample_id].index[0]
        else:
            # Just choose a random document
            text_index = random.randint(0, len(df) - 1)
    
    print(f"Exploring similarities for document {text_index}:")
    original_text = df.iloc[text_index]['text']
    preview = original_text[:150].replace('\n', ' ').strip() + "..."
    print(f"Text preview: {preview}")
    
    # Run LSH with similarity dict
    _, similarity_dict = lsh_deduplication(
        df['text'],
        threshold=threshold,
        num_minhash_permutations=100,
        num_bands=20,
        return_similarity_dict=True
    )
    
    # Find similar documents
    similar_docs = similarity_dict.get(text_index, [])
    print(f"Found {len(similar_docs)} similar documents")
    
    if similar_docs:
        print("\nSimilar documents:")
        for i, doc_idx in enumerate(similar_docs[:5]):  # Show up to 5 similar docs
            doc = df.iloc[doc_idx]
            preview = doc['text'][:100].replace('\n', ' ').strip() + "..."
            print(f"{i+1}. Document {doc_idx} (Original ID: {doc['original_id']})")
            print(f"   Preview: {preview}")
            
            # Check if this is an exact or near duplicate
            if doc['original_id'] == df.iloc[text_index]['original_id']:
                print("   Type: Exact duplicate (same original ID)")
            else:
                print("   Type: Near duplicate (different original ID)")
            print()
            
        if len(similar_docs) > 5:
            print(f"... and {len(similar_docs) - 5} more similar documents")

def main():
    """Run the complete LSH deduplication example."""
    print("===== LSH Deduplication Example =====")
    
    # Create sample dataset
    df = create_sample_dataset(size=2000, duplicate_rate=0.2, near_duplicate_rate=0.1)
    
    # Compare deduplication methods
    standard_results = compare_deduplication_methods(df)
    
    # Analyze LSH parameters
    parameter_results = analyze_lsh_parameters(df)
    
    # Visualize results
    visualize_results(standard_results, parameter_results)
    
    # Explore similar documents for a specific document
    explore_similar_documents(df)
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()