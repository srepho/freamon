# Freamon Examples: Deduplication and Enhanced EDA Reporting

This document provides an overview of the new examples created to demonstrate the deduplication functionality and enhanced EDA reporting features in Freamon.

## Overview of Examples

### 1. EDA Report Test Suite
**File:** `examples/eda_report_test_suite.py`

This script tests the report generation with various types of data:
- Numeric data (different distributions)
- Categorical data (different cardinalities)
- Date/time data (different formats and frequencies)
- Text data
- Currency data with special characters
- Missing data scenarios
- Various column names (including spaces, special characters)
- Different sized datasets

The test suite includes tests for:
- Reports with different dataset sizes
- Reports with different column subsets
- Reports with edge cases (high missing values, constant columns, high cardinality)

### 2. Big Data Deduplication
**File:** `examples/big_data_deduplication_example.py`

This example demonstrates:
- Generating a large synthetic text dataset with duplicates
- Processing text data in chunks for memory efficiency
- Comparing the performance of different deduplication methods
- Memory-efficient deduplication of large text collections
- Integration with efficient data storage and processing

The example includes:
- Hash-based deduplication for exact matches
- N-gram fingerprinting for near-duplicate detection
- LSH (Locality Sensitive Hashing) for scalable similarity search
- Chunked processing for large datasets

### 3. Integrated Deduplication and Reporting
**File:** `examples/deduplication_reporting_example.py`

This example showcases:
- Data quality analysis with duplicate detection
- Different deduplication methods with comparison
- Enhanced EDA reports before and after deduplication
- Visualization of the impact of deduplication on data analysis
- Optimized reporting of large datasets with deduplication insights

## Test Suite
**File:** `tests/test_deduplication_examples.py`

A comprehensive test suite for the deduplication functionality, including tests for:
- Duplicate detection
- Duplicate removal
- Hash-based deduplication
- N-gram fingerprint deduplication
- Integration with the example scripts

## Running the Examples

To run the examples, use the following commands:

```bash
# Run the EDA report test suite
python examples/eda_report_test_suite.py

# Run the big data deduplication example
python examples/big_data_deduplication_example.py

# Run the integrated deduplication and reporting example
python examples/deduplication_reporting_example.py
```

## Generated Output

The examples generate:

1. **HTML Reports**: Located in `eda_test_reports/` and `dedup_reports/` directories.
2. **Visualizations**: Charts comparing deduplication methods and performance.
3. **Deduplicated Data**: Saved as Parquet files for further analysis.

## Requirements

These examples require the following dependencies:
- pandas
- numpy
- matplotlib
- networkx (for graph-based deduplication)
- scikit-learn (for some similarity calculations)

All dependencies are included in the standard Freamon installation.

## Key Features Demonstrated

1. **Enhanced HTML Reporting:**
   - Bootstrap 5 accordions with proper JavaScript
   - Image optimization for smaller report file sizes
   - Lazy loading for performance with large reports
   - Export to Jupyter Notebook functionality

2. **Deduplication Capabilities:**
   - Exact deduplication with hashing
   - Fuzzy deduplication with similarity metrics
   - Performance optimizations for large datasets
   - Chunked processing for memory efficiency

3. **Big Data Handling:**
   - Efficient processing of large datasets
   - Chunk-based operations for memory management
   - Optimized storage with Parquet format
   - Scalable algorithms for large text collections