# Freamon Deduplication and Enhanced Reporting

This document provides an overview of the new deduplication functionality and enhanced EDA reporting features added to Freamon.

## New Features

### 1. Enhanced EDA Reporting
- Fixed accordion functionality in HTML reports with Bootstrap 5 compatibility
- Implemented image optimization to reduce report file sizes
- Added lazy loading for improved performance with large reports
- Improved HTML structure with proper CSS and JavaScript
- Added Jupyter notebook export functionality
- Enhanced handling of special characters in column names
- Better error handling for analysis section failures

### 2. Deduplication Capabilities
- Added exact deduplication with multiple methods:
  - Hash-based deduplication (MD5, SHA1, SHA256)
  - N-gram fingerprinting
- Added fuzzy deduplication with similarity metrics:
  - Cosine similarity
  - Jaccard similarity
  - Levenshtein distance
- Added text clustering for duplicate detection:
  - Hierarchical clustering
  - DBSCAN clustering
- LSH (Locality Sensitive Hashing) for scalable similarity search
- Memory-efficient processing for large datasets

### 3. Big Data Handling
- Chunked processing for memory efficiency
- Optimized storage formats (Parquet)
- Support for distributed processing with Dask (optional)
- Support for Polars dataframes (optional)

## Example Usage

### Basic Duplicate Detection

```python
from freamon.data_quality.duplicates import detect_duplicates

# Detect duplicates in a dataframe
result = detect_duplicates(df, subset=['column1', 'column2'], return_counts=True)

print(f"Duplicates found: {result['duplicate_count']}")
print(f"Duplicate percentage: {result['duplicate_percentage']:.2f}%")
```

### Removing Duplicates

```python
from freamon.data_quality.duplicates import remove_duplicates

# Remove duplicates
df_unique = remove_duplicates(df, subset=['column1'], keep='first')
```

### Text Deduplication

```python
from freamon.deduplication import hash_deduplication, deduplicate_texts

# Exact deduplication with hashing
unique_indices = hash_deduplication(
    df['text'],
    hash_func='md5',
    keep='longest'
)

# Fuzzy deduplication with similarity
unique_indices = deduplicate_texts(
    df['text'],
    threshold=0.8,
    method='cosine'
)

# Get deduplicated dataframe
df_unique = df.iloc[unique_indices].copy()
```

### Enhanced EDA Reporting

```python
from freamon.eda.analyzer import EDAAnalyzer

# Create analyzer
analyzer = EDAAnalyzer(df, target_column='target')

# Generate enhanced report
analyzer.run_full_analysis(
    output_path="report.html",
    title="My EDA Report",
    include_multivariate=True,
    lazy_loading=True,
    include_export_button=True
)
```

## Examples

We've included several example scripts demonstrating these features:

1. **EDA Report Test Suite**: `examples/eda_report_test_suite.py` - Tests reporting with various data types and edge cases

2. **Big Data Deduplication**: `examples/big_data_deduplication_example.py` - Demonstrates deduplication of large text datasets

3. **Integrated Deduplication and Reporting**: `examples/deduplication_reporting_example.py` - Shows how to combine deduplication with EDA reporting

## Test Suite

A comprehensive test suite for the deduplication functionality is available in `tests/test_deduplication_examples.py`.

## Documentation

For detailed documentation of all functions and classes, please refer to the docstrings in the code or the generated API documentation.