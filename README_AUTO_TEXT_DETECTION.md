# Automatic Text Column Detection in flag_similar_records

The `flag_similar_records` function now features automatic text column detection, making it easier to find duplicate or similar records in mixed datasets containing both structured data and text content.

## Overview

When working with datasets that contain text columns alongside regular structured data, detecting similar records requires different algorithms for different column types. Text similarity needs specialized approaches like TF-IDF, n-grams, or fuzzy matching, while numeric data might use normalized distances and categorical data might use exact matching.

Previously, users needed to manually specify which columns were text and required special handling. The new automatic text detection feature removes this burden by:

1. Automatically detecting which columns contain text data
2. Applying optimal text similarity algorithms for those columns
3. Properly weighting text and structured columns in the overall similarity calculation
4. Allowing specialized thresholds for text similarity

## Key Features

- **Auto Column Detection**: Identifies text columns based on content characteristics
- **Multiple Text Similarity Methods**: Choose from fuzzy, TF-IDF, n-gram, or LSH methods
- **Specialized Text Thresholding**: Set different thresholds for text vs. structured data
- **Automatic Weighting**: Apply different weights to text columns with optional boosting
- **Integration with DataTypeDetector**: Uses semantic type information when available
- **Backwards Compatibility**: Works seamlessly with existing code

## Usage Examples

### Basic Usage with Auto-Detection

```python
from freamon.deduplication import flag_similar_records

# Automatically detect and handle text columns
result = flag_similar_records(
    df,
    threshold=0.7,
    auto_detect_columns=True,
    flag_column="is_similar"
)
```

### Specifying Text Columns Explicitly

```python
result = flag_similar_records(
    df,
    threshold=0.7,
    auto_detect_columns=True,
    text_columns=["description", "notes"],  # Explicitly specify text columns
    text_method="tfidf",                   # Use TF-IDF for text similarity
    text_threshold=0.6,                    # Lower threshold for text matching
    flag_column="is_similar"
)
```

### Using Text Weight Boosting

```python
result = flag_similar_records(
    df,
    threshold=0.7,
    auto_detect_columns=True,
    text_weight_boost=2.0,  # Give text columns twice the weight
    flag_column="is_similar"
)
```

### Comparing Different Text Methods

```python
# Try different text similarity methods
for method in ["fuzzy", "tfidf", "ngram", "lsh"]:
    result = flag_similar_records(
        df,
        threshold=0.7,
        auto_detect_columns=True,
        text_method=method,
        flag_column=f"is_similar_{method}"
    )
    print(f"{method}: Found {result[f'is_similar_{method}'].sum()} similar records")
```

## Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `auto_detect_columns` | bool | Enable automatic column detection (default: False) |
| `text_columns` | List[str] | Explicitly specify which columns contain text data |
| `text_method` | str | Text similarity method: 'fuzzy', 'tfidf', 'ngram', or 'lsh' (default: 'fuzzy') |
| `text_threshold` | float | Similarity threshold for text columns (default: same as main threshold) |
| `min_text_length` | int | Minimum average length to consider a string column as text (default: 20) |
| `text_weight_boost` | float | Factor to boost text column weights (default: 1.0) |

## How It Works

### Text Column Detection

When `auto_detect_columns=True`, the function:

1. First checks if the DataTypeDetector has been run and uses its semantic type information
2. For columns without semantic types, analyzes the string content and statistics:
   - Average string length exceeding `min_text_length`
   - Ratio of unique values to total values (to exclude high-cardinality IDs)
   - Exclusion of datetime and boolean columns
3. Identifies a suitable set of text columns for specialized processing

### Text Similarity Methods

- **fuzzy**: Uses Levenshtein distance for short to medium text (default)
- **tfidf**: Uses TF-IDF vectorization with cosine similarity for longer text
- **ngram**: Uses character n-gram Jaccard similarity for faster processing
- **lsh**: Uses MinHash Locality Sensitive Hashing for very large text corpora

### Weight Assignment

- Regular columns get their user-defined weights or equal weights if not specified
- Text columns can be boosted with `text_weight_boost` to prioritize text matches
- All weights are normalized to sum to 1.0 for consistent thresholding

## Integration with Existing Features

The automatic text detection works seamlessly with all existing `flag_similar_records` methods:

- **composite**: Weighted average of similarities across all columns
- **exact_subset**: Match if a subset of columns match exactly
- **fuzzy_subset**: Match if a subset of columns have high similarity

## Requirements

For full functionality, the following optional dependencies are recommended:

- **scikit-learn**: Required for TF-IDF vectorization
- **datasketch**: Required for LSH similarity
- **pandas**: Required for DataFrame handling

## Performance Considerations

- For very large datasets, using the 'lsh' method for text is recommended
- For speed with moderate datasets, 'ngram' offers a good balance
- For highest accuracy but slower processing, use 'tfidf'