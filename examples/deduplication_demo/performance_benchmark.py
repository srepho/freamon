"""
Performance Benchmark for Optimized Duplicate Flagging

This script benchmarks the performance of the optimized duplicate flagging functions
for large datasets, comparing:
1. Memory usage
2. Processing time
3. Accuracy

The benchmark covers:
- Standard vs. Chunked implementations for record similarity
- Standard vs. Streaming implementations for text duplicate detection
- Pandas vs. Polars implementations
"""

import pandas as pd
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Any

# Add the parent directory to the path so we can import from freamon
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from freamon.deduplication.flag_duplicates import (
    flag_text_duplicates,
    flag_similar_records,
    add_duplicate_detection_columns
)


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_function(
    func: callable,
    *args,
    iterations: int = 1,
    **kwargs
) -> Tuple[float, float, Any]:
    """
    Benchmark a function for time and memory usage
    
    Parameters
    ----------
    func : callable
        The function to benchmark
    iterations : int
        Number of times to run the function (for averaging)
    *args, **kwargs
        Arguments to pass to the function
    
    Returns
    -------
    Tuple[float, float, Any]
        Average execution time (seconds), peak memory usage (MB), function result
    """
    times = []
    memory_usage = []
    result = None
    
    # Collect initial memory usage
    initial_memory = get_memory_usage()
    
    for _ in range(iterations):
        start_time = time.time()
        memory_before = get_memory_usage()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        end_time = time.time()
        memory_after = get_memory_usage()
        
        times.append(end_time - start_time)
        memory_usage.append(max(memory_after, memory_before) - initial_memory)
    
    return sum(times) / iterations, sum(memory_usage) / iterations, result


def generate_synthetic_dataframe(
    n_rows: int = 10000,
    n_duplicates: int = 1000,
    text_length: int = 100,
    n_similar: int = 500
) -> pd.DataFrame:
    """
    Generate a synthetic dataframe with text data, names, and numerical features
    with a controlled number of duplicates and similar records
    
    Parameters
    ----------
    n_rows : int
        Number of rows in the dataframe
    n_duplicates : int
        Number of exact duplicate texts to add
    text_length : int
        Average length of text items
    n_similar : int
        Number of similar (but not exact) records to add
    
    Returns
    -------
    pd.DataFrame
        Synthetic dataframe
    """
    # Generate random data
    np.random.seed(42)
    
    # Create base dataframe
    df = pd.DataFrame({
        'id': range(n_rows),
        'name': [f"Person {i}" for i in range(n_rows)],
        'email': [f"person{i}@example.com" for i in range(n_rows)],
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.randint(20000, 150000, n_rows),
        'text': [
            " ".join(np.random.choice(
                ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
                 "adipiscing", "elit", "sed", "do", "eiusmod", "tempor", 
                 "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua"],
                size=text_length // 5
            )) for _ in range(n_rows)
        ]
    })
    
    # Add exact duplicates
    for i in range(n_duplicates):
        original_idx = np.random.randint(0, n_rows)
        duplicate_idx = n_rows + i
        
        # Create exact copy but with different ID
        new_row = df.iloc[original_idx].copy()
        new_row['id'] = duplicate_idx
        
        # Add to dataframe
        df.loc[duplicate_idx] = new_row
    
    # Add similar records (not exact duplicates)
    for i in range(n_similar):
        original_idx = np.random.randint(0, n_rows)
        similar_idx = n_rows + n_duplicates + i
        
        # Create similar record (change a few words in text, slight age/income difference)
        new_row = df.iloc[original_idx].copy()
        new_row['id'] = similar_idx
        
        # Modify text slightly (replace ~20% of words)
        words = new_row['text'].split()
        for j in range(len(words) // 5):
            replace_idx = np.random.randint(0, len(words))
            words[replace_idx] = np.random.choice(
                ["new", "different", "altered", "modified", "changed", "unique"]
            )
        new_row['text'] = " ".join(words)
        
        # Modify numeric values slightly
        new_row['age'] = max(18, new_row['age'] + np.random.randint(-3, 4))
        new_row['income'] = max(20000, new_row['income'] + np.random.randint(-5000, 5001))
        
        # Add to dataframe
        df.loc[similar_idx] = new_row
    
    return df


def benchmark_text_duplicates(
    df: pd.DataFrame,
    method: str = 'lsh',
    chunk_sizes: List[int] = None,
    use_polars: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark text duplicate detection with different configurations
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'text' column
    method : str
        Method to use for text duplicate detection
    chunk_sizes : List[int]
        List of chunk sizes to test
    use_polars : bool
        Whether to test Polars implementations
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with benchmark results
    """
    if chunk_sizes is None:
        chunk_sizes = [None]
    
    results = {}
    
    # Benchmark standard implementation
    print(f"Benchmarking standard implementation (method={method})...")
    time_standard, memory_standard, _ = benchmark_function(
        flag_text_duplicates,
        df,
        text_column='text',
        method=method,
        threshold=0.9,
        inplace=False,
        group_column='text_group',
        similarity_column='text_similarity',
        preprocess=True,
        use_polars=False
    )
    
    results['standard'] = {
        'time': time_standard,
        'memory': memory_standard
    }
    
    print(f"  Time: {time_standard:.2f}s, Memory: {memory_standard:.2f}MB")
    
    # Benchmark Polars implementation if requested
    if use_polars:
        print(f"Benchmarking Polars implementation (method={method})...")
        time_polars, memory_polars, _ = benchmark_function(
            flag_text_duplicates,
            df,
            text_column='text',
            method=method,
            threshold=0.9,
            inplace=False,
            group_column='text_group',
            similarity_column='text_similarity',
            preprocess=True,
            use_polars=True
        )
        
        results['polars'] = {
            'time': time_polars,
            'memory': memory_polars
        }
        
        print(f"  Time: {time_polars:.2f}s, Memory: {memory_polars:.2f}MB")
    
    # Benchmark chunked implementations with different chunk sizes
    for chunk_size in chunk_sizes:
        if chunk_size is None:
            continue
            
        print(f"Benchmarking chunked implementation (chunk_size={chunk_size})...")
        time_chunked, memory_chunked, _ = benchmark_function(
            flag_text_duplicates,
            df,
            text_column='text',
            method=method,
            threshold=0.9,
            inplace=False,
            group_column='text_group',
            similarity_column='text_similarity',
            preprocess=True,
            chunk_size=chunk_size,
            use_polars=False
        )
        
        results[f'chunked_{chunk_size}'] = {
            'time': time_chunked,
            'memory': memory_chunked
        }
        
        print(f"  Time: {time_chunked:.2f}s, Memory: {memory_chunked:.2f}MB")
        
        # Benchmark Polars chunked implementation if requested
        if use_polars:
            print(f"Benchmarking Polars chunked implementation (chunk_size={chunk_size})...")
            time_polars_chunked, memory_polars_chunked, _ = benchmark_function(
                flag_text_duplicates,
                df,
                text_column='text',
                method=method,
                threshold=0.9,
                inplace=False,
                group_column='text_group',
                similarity_column='text_similarity',
                preprocess=True,
                chunk_size=chunk_size,
                use_polars=True
            )
            
            results[f'polars_chunked_{chunk_size}'] = {
                'time': time_polars_chunked,
                'memory': memory_polars_chunked
            }
            
            print(f"  Time: {time_polars_chunked:.2f}s, Memory: {memory_polars_chunked:.2f}MB")
    
    return results


def benchmark_similar_records(
    df: pd.DataFrame,
    chunk_sizes: List[int] = None,
    use_polars: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark similar records detection with different configurations
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multiple columns for similarity comparison
    chunk_sizes : List[int]
        List of chunk sizes to test
    use_polars : bool
        Whether to test Polars implementations
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary with benchmark results
    """
    if chunk_sizes is None:
        chunk_sizes = [None]
    
    results = {}
    
    # Define columns and weights for similarity comparison
    columns = ['name', 'email', 'age', 'income']
    weights = {'name': 0.4, 'email': 0.4, 'age': 0.1, 'income': 0.1}
    
    # Benchmark standard implementation
    print("Benchmarking standard implementation...")
    time_standard, memory_standard, _ = benchmark_function(
        flag_similar_records,
        df,
        columns=columns,
        weights=weights,
        threshold=0.8,
        method='composite',
        inplace=False,
        group_column='similarity_group',
        similarity_column='similarity_score',
        use_polars=False
    )
    
    results['standard'] = {
        'time': time_standard,
        'memory': memory_standard
    }
    
    print(f"  Time: {time_standard:.2f}s, Memory: {memory_standard:.2f}MB")
    
    # Benchmark Polars implementation if requested
    if use_polars:
        print("Benchmarking Polars implementation...")
        time_polars, memory_polars, _ = benchmark_function(
            flag_similar_records,
            df,
            columns=columns,
            weights=weights,
            threshold=0.8,
            method='composite',
            inplace=False,
            group_column='similarity_group',
            similarity_column='similarity_score',
            use_polars=True
        )
        
        results['polars'] = {
            'time': time_polars,
            'memory': memory_polars
        }
        
        print(f"  Time: {time_polars:.2f}s, Memory: {memory_polars:.2f}MB")
    
    # Benchmark chunked implementations with different chunk sizes
    for chunk_size in chunk_sizes:
        if chunk_size is None:
            continue
            
        print(f"Benchmarking chunked implementation (chunk_size={chunk_size})...")
        time_chunked, memory_chunked, _ = benchmark_function(
            flag_similar_records,
            df,
            columns=columns,
            weights=weights,
            threshold=0.8,
            method='composite',
            inplace=False,
            group_column='similarity_group',
            similarity_column='similarity_score',
            chunk_size=chunk_size,
            n_jobs=1,
            use_polars=False
        )
        
        results[f'chunked_{chunk_size}'] = {
            'time': time_chunked,
            'memory': memory_chunked
        }
        
        print(f"  Time: {time_chunked:.2f}s, Memory: {memory_chunked:.2f}MB")
        
        # Benchmark chunked implementation with multiple jobs
        print(f"Benchmarking parallel chunked implementation (chunk_size={chunk_size}, n_jobs=4)...")
        time_parallel, memory_parallel, _ = benchmark_function(
            flag_similar_records,
            df,
            columns=columns,
            weights=weights,
            threshold=0.8,
            method='composite',
            inplace=False,
            group_column='similarity_group',
            similarity_column='similarity_score',
            chunk_size=chunk_size,
            n_jobs=4,
            use_polars=False
        )
        
        results[f'parallel_chunked_{chunk_size}'] = {
            'time': time_parallel,
            'memory': memory_parallel
        }
        
        print(f"  Time: {time_parallel:.2f}s, Memory: {memory_parallel:.2f}MB")
        
        # Benchmark Polars chunked implementation if requested
        if use_polars:
            print(f"Benchmarking Polars chunked implementation (chunk_size={chunk_size})...")
            time_polars_chunked, memory_polars_chunked, _ = benchmark_function(
                flag_similar_records,
                df,
                columns=columns,
                weights=weights,
                threshold=0.8,
                method='composite',
                inplace=False,
                group_column='similarity_group',
                similarity_column='similarity_score',
                chunk_size=chunk_size,
                n_jobs=1,
                use_polars=True
            )
            
            results[f'polars_chunked_{chunk_size}'] = {
                'time': time_polars_chunked,
                'memory': memory_polars_chunked
            }
            
            print(f"  Time: {time_polars_chunked:.2f}s, Memory: {memory_polars_chunked:.2f}MB")
    
    return results


def plot_benchmark_results(
    text_results: Dict[str, Dict[str, float]],
    record_results: Dict[str, Dict[str, float]],
    title: str = "Duplicate Detection Performance Benchmark"
) -> None:
    """
    Plot the benchmark results
    
    Parameters
    ----------
    text_results : Dict[str, Dict[str, float]]
        Results from benchmark_text_duplicates
    record_results : Dict[str, Dict[str, float]]
        Results from benchmark_similar_records
    title : str
        Plot title
    """
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    # Text duplicates - Time
    ax = axs[0, 0]
    labels = list(text_results.keys())
    times = [text_results[label]['time'] for label in labels]
    ax.bar(labels, times)
    ax.set_title('Text Duplicate Detection - Time (s)')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Text duplicates - Memory
    ax = axs[0, 1]
    memory = [text_results[label]['memory'] for label in labels]
    ax.bar(labels, memory)
    ax.set_title('Text Duplicate Detection - Memory (MB)')
    ax.set_ylabel('Memory (MB)')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Record similarity - Time
    ax = axs[1, 0]
    labels = list(record_results.keys())
    times = [record_results[label]['time'] for label in labels]
    ax.bar(labels, times)
    ax.set_title('Record Similarity Detection - Time (s)')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Record similarity - Memory
    ax = axs[1, 1]
    memory = [record_results[label]['memory'] for label in labels]
    ax.bar(labels, memory)
    ax.set_title('Record Similarity Detection - Memory (MB)')
    ax.set_ylabel('Memory (MB)')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig('duplicate_detection_benchmark.png')
    plt.close()


def main():
    """Main function to run the benchmark"""
    print("Generating synthetic dataset...")
    
    # Generate a smaller dataset for initial tests
    small_df = generate_synthetic_dataframe(
        n_rows=5000,
        n_duplicates=500,
        text_length=100,
        n_similar=250
    )
    
    print(f"Generated dataset with {len(small_df)} rows")
    
    # Benchmark text duplicates with small dataset
    print("\n=== Benchmarking Text Duplicate Detection (Small Dataset) ===")
    small_text_results = benchmark_text_duplicates(
        small_df,
        method='lsh',
        chunk_sizes=[None, 1000, 2000],
        use_polars=True
    )
    
    # Benchmark similar records with small dataset
    print("\n=== Benchmarking Record Similarity Detection (Small Dataset) ===")
    small_record_results = benchmark_similar_records(
        small_df,
        chunk_sizes=[None, 1000, 2000],
        use_polars=True
    )
    
    # Plot results for small dataset
    plot_benchmark_results(
        small_text_results,
        small_record_results,
        title="Duplicate Detection Performance Benchmark (Small Dataset)"
    )
    
    # Generate a larger dataset for more realistic tests
    print("\nGenerating larger synthetic dataset...")
    large_df = generate_synthetic_dataframe(
        n_rows=20000,
        n_duplicates=2000,
        text_length=200,
        n_similar=1000
    )
    
    print(f"Generated dataset with {len(large_df)} rows")
    
    # Benchmark text duplicates with large dataset
    print("\n=== Benchmarking Text Duplicate Detection (Large Dataset) ===")
    large_text_results = benchmark_text_duplicates(
        large_df,
        method='lsh',
        chunk_sizes=[2000, 5000, 10000],
        use_polars=True
    )
    
    # Benchmark similar records with large dataset
    print("\n=== Benchmarking Record Similarity Detection (Large Dataset) ===")
    large_record_results = benchmark_similar_records(
        large_df,
        chunk_sizes=[2000, 5000, 10000],
        use_polars=True
    )
    
    # Plot results for large dataset
    plot_benchmark_results(
        large_text_results,
        large_record_results,
        title="Duplicate Detection Performance Benchmark (Large Dataset)"
    )
    
    print("\nBenchmark complete. Results saved to duplicate_detection_benchmark.png")


if __name__ == "__main__":
    main()