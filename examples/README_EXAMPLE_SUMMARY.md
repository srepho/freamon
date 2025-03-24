# New Examples for Freamon v0.3.19

## Overview
This document summarizes the new example scripts created to demonstrate the enhanced reporting and deduplication features in Freamon v0.3.18 and v0.3.19.

## New Example Scripts

### 1. `eda_report_test_suite.py`
A comprehensive test suite for the enhanced EDA reporting functionality with:
- Tests for different data types (numeric, categorical, datetime, text, currency)
- Tests for different column name formats (spaces, special characters)
- Tests for different dataset sizes (100, 1000, 10000 rows)
- Tests for edge cases (high missing values, constant columns, high cardinality)
- Tests for different column subsets

### 2. `big_data_deduplication_example.py`
A detailed example demonstrating deduplication of large text datasets with:
- Generation of large synthetic text datasets with duplicates
- Chunked processing for memory efficiency
- Performance benchmarking of different deduplication methods
- Scalable algorithms for large text collections
- Memory-efficient storage and processing

### 3. `deduplication_reporting_example.py`
An integrated example showing:
- The impact of deduplication on data analysis
- Comparison of different deduplication strategies
- Generation of before/after EDA reports
- Visualization of deduplication results
- Structured data deduplication workflows

### 4. `test_deduplication_examples.py`
A test suite to verify the functionality of:
- Duplicate detection methods
- Duplicate removal methods
- Hash-based deduplication
- N-gram fingerprint deduplication
- Integration with example scripts

## Key Features Demonstrated

### Enhanced Reporting (v0.3.19)
- Fixed accordion functionality with proper Bootstrap 5 JavaScript
- Image optimization for reduced file sizes
- Lazy loading for improved performance
- Enhanced HTML structure with better CSS and JavaScript
- Jupyter notebook export functionality
- Special character handling in column names
- Error handling for analysis section failures

### Deduplication Module (v0.3.18)
- Exact deduplication methods (hash-based, n-gram fingerprinting)
- Fuzzy deduplication with similarity metrics (cosine, Jaccard, Levenshtein)
- Text clustering for duplicate detection
- LSH (Locality Sensitive Hashing) for scalable similarity search
- Memory-efficient processing for large datasets

## Usage
Run the examples with:

```bash
python examples/eda_report_test_suite.py
python examples/big_data_deduplication_example.py
python examples/deduplication_reporting_example.py
```

Run the tests with:

```bash
python -m unittest tests/test_deduplication_examples.py
```

Generated reports will be saved in the `eda_test_reports/` and `dedup_reports/` directories.

## Conclusion
These examples demonstrate the significant improvements in reporting and deduplication capabilities in the latest versions of Freamon, with a focus on performance, usability, and large dataset handling.