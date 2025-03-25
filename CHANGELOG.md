# Changelog

## Version 0.3.32 (2025-03-25)

### Added
- Improved EDA module with split report generation capability
- Added separate univariate and bivariate reports for better performance
- Enhanced bivariate analysis with feature importance metrics
- Added ability to run a complete analysis with separate reports
- Added convenience methods for generating univariate and bivariate reports

### Improved
- Better performance for bivariate analysis by moving it to a separate report
- Enhanced feature importance display in bivariate analysis
- More detailed statistical measures for feature-target relationships

## Version 0.3.31 (2025-03-25)

### Fixed
- Renamed `CrossValidatedTrainer` class to `CrossValidationTrainer` in cv_trainer.py to match import in AutoModelFlow
- Fixed import errors in model selection module
- Added tests to verify correct imports and functionality

## Version 0.3.30 (2025-03-25)

### Fixed
- Added missing `create_time_series_cv` function to cross_validation module
- Added missing `create_stratified_cv` and `create_kfold_cv` helper functions
- Fixed import error in AutoModelFlow when using time series cross-validation

## Version 0.3.29 (2025-03-25)

### Added
- New AutoModelFlow for end-to-end automated modeling workflows
- Automatic handling of text data with topic modeling and feature extraction
- Automatic time series feature creation for date-based columns
- Intelligent cross-validation selection based on data type
- High-level `auto_model` function for simplified usage
- Comprehensive documentation in `docs/usage/automated_modeling.md` 
- New example script demonstrating automated modeling with text and time series data

### Improved
- Enhanced integration between text processing, feature engineering, and modeling
- Better handling of different types of columns with automatic detection
- Automated hyperparameter tuning with time series cross-validation support

## Version 0.3.28 (2025-03-25)

### Added
- Added anonymization support to optimized topic modeling
- Integrated with Allyanonimiser package for PII detection and anonymization
- Added graceful fallback when anonymization libraries aren't available
- Created new example script demonstrating topic modeling with anonymization
- Added comprehensive documentation on anonymization configuration options

### Improved
- Enhanced topic modeling workflow with privacy-preserving features
- Better handling of personally identifiable information in text data
- Optimized anonymization process with batch processing

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