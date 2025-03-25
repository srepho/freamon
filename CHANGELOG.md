# Changelog

## Version 0.3.27 (2025-03-25)

### Added
- Integrated enhanced topic modeling functionality into main `text_utils` module
- Added configurable preprocessing options for topic modeling
- Added flexible deduplication options (exact/fuzzy/none) for topic modeling
- Optimized for large datasets up to 100K documents
- Added comprehensive documentation in `docs/usage/optimized_topic_modeling.md`
- New test suite for the optimized topic modeling functionality

### Improved
- Enhanced performance with automatic multiprocessing support
- Better error handling and fallbacks for missing functionality
- Improved progress reporting during processing steps
- More flexible configuration options for text preprocessing
- Enhanced deduplication with mapping to original documents

## Version 0.3.26 (2025-03-25)

### Added
- New topic modeling integration with pandas DataFrame workflow
- `dataframe_topic_modeling_example.py` showing how to add topics as columns to DataFrames
- Optimized topic modeling with automatic deduplication support (fuzzy/exact)
- Support for large dataset processing with smart sampling in topic modeling

### Improved
- Better error handling in text processing modules
- Enhanced multiprocessing support for topic modeling

## Version 0.3.25

### Added
- Optimized topic modeling workflow with automatic deduplication
- Smart sampling for large datasets in topic modeling
- Batch processing with progress reporting
- Full dataset topic mapping for sampled processing

## Version 0.3.24

### Added
- HTML report generation for DataTypeDetector
- Color-coded data type visualization in reports
- Statistical information in type detection reports