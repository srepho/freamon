"""
Example demonstrating text deduplication functionality with large datasets.

This example shows how to:
1. Generate a large synthetic text dataset with duplicates
2. Process text data in chunks
3. Compare the performance of different deduplication methods
4. Handle memory-efficient deduplication of large text collections
5. Integrate with efficient data storage and processing
"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import string
import random
import gc
import tempfile

from freamon.utils.dataframe_utils import (
    process_in_chunks,
    save_to_chunks,
    load_from_chunks
)
from freamon.utils.text_utils import TextProcessor
from freamon.deduplication import (
    hash_deduplication,
    ngram_fingerprint_deduplication,
    deduplicate_texts,
    lsh_deduplication
)

def generate_text(length=100, include_special=True):
    """Generate a random text string with specified length."""
    chars = string.ascii_letters + string.digits + ' ' * 10  # Add extra spaces for realism
    if include_special:
        chars += string.punctuation
        
    return ''.join(random.choice(chars) for _ in range(length))

def generate_large_text_dataset(size=100000, duplicate_rate=0.2, near_duplicate_rate=0.1):
    """
    Generate a large synthetic dataset with text data and duplicates.
    
    Parameters
    ----------
    size : int, default=100000
        Number of rows to generate
    duplicate_rate : float, default=0.2
        Percentage of exact duplicates to introduce
    near_duplicate_rate : float, default=0.1
        Percentage of near-duplicates (slightly modified versions)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with text data and metadata
    """
    print(f"Generating text dataset with {size} rows...")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate base texts
    unique_size = int(size * (1 - duplicate_rate - near_duplicate_rate))
    texts = []
    categories = []
    
    # Text length variations for more realism
    text_lengths = np.random.choice([50, 100, 200, 500], unique_size)
    
    print(f"Generating {unique_size} unique base texts...")
    for i, length in enumerate(text_lengths):
        texts.append(generate_text(length))
        categories.append(random.choice(['news', 'social', 'blog', 'review', 'comment']))
    
    # Generate exact duplicates
    dup_size = int(size * duplicate_rate)
    print(f"Adding {dup_size} exact duplicates...")
    
    dup_indices = np.random.choice(range(len(texts)), dup_size)
    for idx in dup_indices:
        texts.append(texts[idx])
        categories.append(categories[idx])
    
    # Generate near-duplicates (with small modifications)
    near_dup_size = int(size * near_duplicate_rate)
    print(f"Adding {near_dup_size} near-duplicates...")
    
    near_dup_indices = np.random.choice(range(unique_size), near_dup_size)
    for idx in near_dup_indices:
        original = texts[idx]
        
        # Apply random modifications
        modification_type = random.choice(['typos', 'additions', 'deletions', 'substitutions'])
        
        if modification_type == 'typos':
            # Introduce random typos
            chars = list(original)
            for _ in range(min(3, len(original) // 20)):
                pos = random.randint(0, len(chars) - 1)
                chars[pos] = random.choice(string.ascii_letters)
            modified = ''.join(chars)
        
        elif modification_type == 'additions':
            # Add a few words
            additions = " " + generate_text(20, include_special=False)
            insert_pos = random.randint(0, len(original))
            modified = original[:insert_pos] + additions + original[insert_pos:]
        
        elif modification_type == 'deletions':
            # Delete a small portion
            if len(original) > 30:
                start = random.randint(0, len(original) - 20)
                modified = original[:start] + original[start+10:]
            else:
                modified = original  # Too short to delete from
        
        else:  # substitutions
            # Replace some words
            words = original.split()
            if len(words) > 5:
                for _ in range(min(3, len(words) // 10)):
                    pos = random.randint(0, len(words) - 1)
                    words[pos] = generate_text(len(words[pos]), include_special=False)
                modified = ' '.join(words)
            else:
                modified = original  # Too few words to substitute
        
        texts.append(modified)
        categories.append(categories[idx])
    
    # Create dataframe with IDs
    df = pd.DataFrame({
        'id': range(1, len(texts) + 1),
        'text': texts,
        'category': categories,
        'length': [len(t) for t in texts],
        'created_at': pd.date_range(start='2023-01-01', periods=len(texts))
    })
    
    # Shuffle the dataframe to mix duplicates
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def example_deduplication_in_chunks(df, chunk_size=10000, output_dir=None):
    """
    Demonstrate deduplication of large text dataset using chunking.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text data
    chunk_size : int, default=10000
        Size of each processing chunk
    output_dir : str, optional
        Directory to save chunks and results
    """
    print(f"\n===== Deduplicating Large Text Dataset ({len(df)} rows) =====")
    
    if output_dir is None:
        output_dir = Path("dedup_results")
        output_dir.mkdir(exist_ok=True)
    
    # First, save the data in chunks for efficient processing
    print(f"Saving data in chunks to {output_dir}...")
    chunk_files = save_to_chunks(
        df,
        output_dir=output_dir,
        base_filename='text_data',
        chunk_size=chunk_size,
        file_format='parquet'
    )
    print(f"Created {len(chunk_files)} chunk files.")
    
    # Method 1: Hash-based deduplication on full dataset
    # This method is memory-efficient and can handle large datasets
    print("\n1. Hash-based deduplication (full dataset):")
    start_time = time.time()
    
    # Use hash deduplication with custom keep strategy
    unique_indices = hash_deduplication(
        df['text'],
        hash_func='md5',
        case_sensitive=False,
        keep='longest',  # Keep the longest instance of each duplicate
        preprocess=True
    )
    
    df_unique_hash = df.iloc[unique_indices].copy()
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Unique documents: {len(df_unique_hash)}")
    print(f"Removed duplicates: {len(df) - len(df_unique_hash)} ({(len(df) - len(df_unique_hash)) / len(df) * 100:.2f}%)")
    
    # Method 2: Chunked processing for memory efficiency
    print("\n2. Processing in chunks with hash deduplication:")
    start_time = time.time()
    
    # Step 1: Process each chunk to get locally unique texts
    def deduplicate_chunk(chunk_df):
        """Deduplicate a single chunk and return unique texts with indices."""
        local_indices = hash_deduplication(
            chunk_df['text'],
            hash_func='md5',
            case_sensitive=False,
            preprocess=True
        )
        return chunk_df.iloc[local_indices][['id', 'text']].copy()
    
    # Process chunks to get locally unique texts
    chunks = load_from_chunks(
        input_dir=output_dir,
        pattern='text_data_*.parquet',
        combine=False,
        output_type='pandas'
    )
    
    # Process each chunk
    local_unique_dfs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        local_unique = deduplicate_chunk(chunk)
        local_unique_dfs.append(local_unique)
    
    # Combine locally unique texts
    combined_local_unique = pd.concat(local_unique_dfs).reset_index(drop=True)
    print(f"After local deduplication: {len(combined_local_unique)} texts")
    
    # Step 2: Deduplicate the combined local results
    final_indices = hash_deduplication(
        combined_local_unique['text'],
        hash_func='md5',
        case_sensitive=False,
        preprocess=True
    )
    
    df_unique_chunked = combined_local_unique.iloc[final_indices].copy()
    
    # Map back to original dataframe to get full records
    df_unique_chunked_full = df[df['id'].isin(df_unique_chunked['id'])].copy()
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Unique documents: {len(df_unique_chunked_full)}")
    print(f"Removed duplicates: {len(df) - len(df_unique_chunked_full)} ({(len(df) - len(df_unique_chunked_full)) / len(df) * 100:.2f}%)")
    
    # Method 3: LSH for near-duplicate detection (on a sample for demonstration)
    print("\n3. LSH for near-duplicate detection (on a sample):")
    # Use a sample to demonstrate LSH for near-duplicate detection
    sample_size = min(20000, len(df))
    df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    start_time = time.time()
    try:
        unique_indices_lsh = lsh_deduplication(
            df_sample['text'],
            threshold=0.8,
            minhash_sigs=100,
            bands=20
        )
        
        df_unique_lsh = df_sample.iloc[unique_indices_lsh].copy()
        
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Unique documents: {len(df_unique_lsh)}")
        print(f"Removed duplicates: {len(df_sample) - len(df_unique_lsh)} ({(len(df_sample) - len(df_unique_lsh)) / len(df_sample) * 100:.2f}%)")
    except Exception as e:
        print(f"Error with LSH deduplication: {str(e)}")
    
    # Save the deduplicated datasets
    df_unique_hash.to_parquet(output_dir / "deduplicated_hash.parquet")
    df_unique_chunked_full.to_parquet(output_dir / "deduplicated_chunked.parquet")
    
    print("\nDeduplicated datasets saved to output directory.")
    
    # Return stats for visualization
    return {
        "original_size": len(df),
        "hash_dedup_size": len(df_unique_hash),
        "chunked_dedup_size": len(df_unique_chunked_full),
        "hash_dedup_time": elapsed
    }

def benchmark_deduplication_methods(df):
    """
    Benchmark different deduplication methods on a dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with text data
    """
    print("\n===== Benchmarking Deduplication Methods =====")
    
    # Use a sample for benchmarking
    sample_size = min(50000, len(df))
    df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    methods = {
        "Hash (MD5)": lambda texts: hash_deduplication(texts, hash_func='md5'),
        "Hash (SHA1)": lambda texts: hash_deduplication(texts, hash_func='sha1'),
        "N-gram Fingerprint": lambda texts: ngram_fingerprint_deduplication(texts, n=3),
        "Similarity (Cosine)": lambda texts: deduplicate_texts(texts, method='cosine', threshold=0.9),
        "Similarity (Jaccard)": lambda texts: deduplicate_texts(texts, method='jaccard', threshold=0.8),
    }
    
    results = {}
    for name, method in methods.items():
        print(f"\nTesting {name}...")
        
        try:
            start_time = time.time()
            unique_indices = method(df_sample['text'])
            elapsed = time.time() - start_time
            
            # Get results
            df_unique = df_sample.iloc[unique_indices].copy()
            dedup_percentage = (len(df_sample) - len(df_unique)) / len(df_sample) * 100
            
            results[name] = {
                'time': elapsed,
                'unique_count': len(df_unique),
                'dedup_count': len(df_sample) - len(df_unique),
                'dedup_percentage': dedup_percentage
            }
            
            print(f"Time: {elapsed:.2f} seconds")
            print(f"Unique documents: {len(df_unique)}")
            print(f"Removed duplicates: {len(df_sample) - len(df_unique)} ({dedup_percentage:.2f}%)")
            
            # Clear memory
            del df_unique
            gc.collect()
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Visualize results
    if results:
        output_dir = Path("dedup_results")
        output_dir.mkdir(exist_ok=True)
        
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
        
        plt.savefig(output_dir / 'deduplication_benchmark.png')
        print("\nBenchmark visualization saved to 'dedup_results/deduplication_benchmark.png'")
    
    return results

def main():
    """Run the example."""
    print("Starting big data text deduplication example...")
    
    # Create output directory
    output_dir = Path("dedup_results")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset size - adjust based on available memory
    dataset_size = 100000  # Increase for more realistic big data scenario
    
    # Generate large text dataset with duplicates
    df = generate_large_text_dataset(
        size=dataset_size,
        duplicate_rate=0.2,
        near_duplicate_rate=0.1
    )
    
    # Show dataset info
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    
    # Overview of text lengths
    length_stats = df['length'].describe()
    print("\nText length statistics:")
    print(f"Min: {length_stats['min']:.0f}")
    print(f"Max: {length_stats['max']:.0f}")
    print(f"Mean: {length_stats['mean']:.0f}")
    
    # Example 1: Deduplication in chunks for large dataset
    dedup_stats = example_deduplication_in_chunks(
        df,
        chunk_size=10000,
        output_dir=output_dir
    )
    
    # Example 2: Benchmark different methods
    benchmark_results = benchmark_deduplication_methods(df)
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()