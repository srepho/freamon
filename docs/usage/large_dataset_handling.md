# Large Dataset Handling

Freamon provides several utilities for working with large datasets efficiently. These tools allow you to process datasets that would normally exceed available memory by using techniques like:

1. Chunking (processing in smaller pieces)
2. Memory optimization
3. Integration with specialized libraries (Dask, Polars)
4. Stream processing

## Basic Usage

### Processing in Chunks

The most common way to handle large datasets is by processing them in chunks:

```python
from freamon.utils.dataframe_utils import process_in_chunks
import pandas as pd
import numpy as np

# Load or create a large dataframe
df = pd.DataFrame(np.random.randn(1_000_000, 10))

# Define a function to process each chunk
def process_chunk(chunk):
    return chunk.mean()

# Define how to combine results from all chunks
def combine_results(results):
    return pd.concat(results).mean()

# Process the dataframe in chunks of 100,000 rows
result = process_in_chunks(
    df=df,
    func=process_chunk,
    chunk_size=100_000,
    combine_func=combine_results,
    show_progress=True
)
```

### Iterating Through Chunks

You can also iterate through chunks directly:

```python
from freamon.utils.dataframe_utils import iterate_chunks

# Initialize a result container
total_sum = 0

# Process each chunk
for chunk in iterate_chunks(df, chunk_size=100_000):
    # Process this chunk
    chunk_sum = chunk.sum().sum()
    total_sum += chunk_sum
    
print(f"Total sum: {total_sum}")
```

### Saving and Loading in Chunks

For very large datasets, you can save and load in chunks:

```python
from freamon.utils.dataframe_utils import save_to_chunks, load_from_chunks
import os

# Save a large dataframe to disk in chunks
chunk_files = save_to_chunks(
    df=df,
    output_dir="data/chunks",
    base_filename="large_dataset",
    chunk_size=100_000,
    file_format="parquet"  # Options: 'csv', 'parquet', 'feather'
)

# Later, load the chunks
# As a combined dataframe
df_combined = load_from_chunks(
    input_dir="data/chunks",
    pattern="large_dataset_*.parquet",
    combine=True,
    output_type="pandas"  # Options: 'pandas', 'polars', 'dask'
)

# Or as separate chunks
chunks = load_from_chunks(
    input_dir="data/chunks",
    pattern="large_dataset_*.parquet",
    combine=False
)
```

## Memory Optimization

Freamon can optimize memory usage by choosing appropriate data types:

```python
from freamon.utils.dataframe_utils import optimize_dtypes

# Optimize memory usage
df_optimized = optimize_dtypes(df)

# Check memory savings
from freamon.utils.dataframe_utils import estimate_memory_usage

before = estimate_memory_usage(df)
after = estimate_memory_usage(df_optimized)

print(f"Before: {before['total_mb']:.2f} MB")
print(f"After: {after['total_mb']:.2f} MB")
print(f"Savings: {(1 - after['total_mb']/before['total_mb'])*100:.1f}%")
```

## Integration with Specialized Libraries

### Using Dask for Distributed Computing

Freamon integrates with Dask for distributed computing:

```python
import dask.dataframe as dd
from freamon.utils.dataframe_utils import process_in_chunks

# Convert pandas to dask
dask_df = dd.from_pandas(df, npartitions=10)

# Process with Dask (automatically uses all cores)
result = process_in_chunks(dask_df, process_chunk)
```

### Using Polars for Fast Processing

Freamon also integrates with Polars for high-performance data processing:

```python
import polars as pl
from freamon.utils.dataframe_utils import convert_dataframe

# Convert pandas to polars
polars_df = convert_dataframe(df, to_type="polars")

# Use Freamon functions with polars
polars_result = process_in_chunks(polars_df, process_chunk)
```

## Using with EDA Analyzer

The EDA Analyzer automatically uses chunking for large datasets:

```python
from freamon.eda import EDAAnalyzer

# Create analyzer with a large dataset
analyzer = EDAAnalyzer(df)

# These methods automatically use chunking for large datasets
basic_stats = analyzer.analyze_basic_stats()
univariate = analyzer.analyze_univariate()
bivariate = analyzer.analyze_bivariate()

# Generate a report (also uses chunking internally)
analyzer.generate_report("large_data_report.html")
```

## Optimizing Multivariate Analysis for Large Datasets

For multivariate analysis on large datasets:

```python
# Select a subset of important features
numeric_cols = df.select_dtypes(include=['number']).columns

# Option 1: Use variance-based feature selection
variances = df[numeric_cols].var().sort_values(ascending=False)
top_features = variances.index[:50].tolist()  # Top 50 by variance

# Option 2: Sample rows for initial analysis
sampled_df = df.sample(n=10000, random_state=42)

# Perform multivariate analysis on the sample or selected features
analyzer = EDAAnalyzer(sampled_df)  # or EDAAnalyzer(df)
multivariate_results = analyzer.analyze_multivariate(columns=top_features)
```

## Performance Recommendations

For optimal performance with large datasets:

1. **Choose the right format**: Use Parquet for disk storage (smaller, faster)
2. **Balance chunk size**: Too small = overhead, too large = memory issues
3. **Use specialized libraries**: Dask for parallel processing, Polars for speed
4. **Optimize early**: Convert data types before heavy processing
5. **Sample during development**: Work with sample data while building pipelines

## Example: End-to-End Large Dataset Pipeline

Here's a complete example showing an end-to-end pipeline for large datasets:

```python
from freamon.utils.dataframe_utils import (
    optimize_dtypes,
    process_in_chunks,
    save_to_chunks,
    load_from_chunks
)
from freamon.eda import EDAAnalyzer
import pandas as pd
import numpy as np
import os

# Step 1: Load and optimize data types (for demonstration)
df = pd.DataFrame(np.random.randn(1_000_000, 20))
df_optimized = optimize_dtypes(df)

# Step 2: Save to disk in chunks
os.makedirs("data", exist_ok=True)
chunk_files = save_to_chunks(
    df_optimized, 
    "data", 
    "large_dataset",
    chunk_size=100_000
)

# Step 3: Process each chunk with a custom function
def process_chunk(chunk):
    # Example: Calculate statistics per chunk
    return {
        'mean': chunk.mean().to_dict(),
        'min': chunk.min().to_dict(),
        'max': chunk.max().to_dict()
    }

# Combine results from all chunks
def combine_results(results):
    combined = {
        'mean': {},
        'min': {},
        'max': {}
    }
    
    # Calculate weighted means
    for chunk_result in results:
        for k, v in chunk_result['mean'].items():
            if k not in combined['mean']:
                combined['mean'][k] = 0
            combined['mean'][k] += v / len(results)
            
        # Take the global min/max
        for k, v in chunk_result['min'].items():
            if k not in combined['min'] or v < combined['min'][k]:
                combined['min'][k] = v
                
        for k, v in chunk_result['max'].items():
            if k not in combined['max'] or v > combined['max'][k]:
                combined['max'][k] = v
    
    return combined

# Step 4: Load and process chunks
chunks = load_from_chunks("data", "large_dataset_*.parquet", combine=False)
results = process_in_chunks(
    df_optimized,
    process_chunk,
    chunk_size=100_000,
    combine_func=combine_results
)

# Step 5: Sample data for exploratory analysis
sample = df_optimized.sample(n=10000, random_state=42)
analyzer = EDAAnalyzer(sample)
analyzer.run_full_analysis("sample_analysis.html")

print("Pipeline completed successfully.")
```

## Resource Management

When working with very large datasets, consider these additional tips:

1. **Monitor memory usage**: Track memory consumption during processing
2. **Use system resources wisely**: Set an appropriate chunk size based on available RAM
3. **Clean up**: Delete temporary files and clear variables when no longer needed
4. **Consider cloud computing**: For datasets that exceed local resources, consider distributed processing in the cloud