# Changelog

## Version 0.3.56
* Added date similarity support to flag_similar_records function:
  * Implemented three date similarity calculation methods: linear, exponential, and threshold-based
  * Added auto-detection of date columns using sophisticated heuristics
  * Added handling for multiple date formats including native datetime, string representations, and timestamps
  * Added configurable thresholds for date similarity calculations
  * Created comprehensive tests for date auto-detection and similarity features
  * Fixed duplicated parameter definitions in flag_duplicates.py

## Version 0.3.55
* Fixed AutoModelFlow cross-validation errors with similarity features:
  * Modified Model.predict and Model.predict_proba to handle missing features by adding zeros
  * Enhanced cross-validation to handle feature mismatches during training/prediction
  * Added robust error handling around prediction and metrics calculation
  * Improved handling of dynamic features like those from deduplication processes
  * Added graceful fallback for metrics calculation failures

## Version 0.3.54
* Fixed additional issues with Model class compatibility:
  * Fixed error in calculate_metrics() call with incorrect parameter order
  * Added robust handling of metrics not explicitly requested but returned by calculate_metrics
  * Added safety checks to prevent KeyError on log_loss metric
  * Improved cross-validation metrics handling for different problem types
  * Enhanced error recovery in model evaluation

## Version 0.3.53
* Fixed Model class compatibility with scikit-learn:
  * Added get_params and set_params methods to comply with scikit-learn estimator interface
  * Fixed TypeError in sklearn.base.clone when using cross-validation
  * Added proper parameter handling for nested model objects
  * Fixed integration between custom Model class and scikit-learn's cross-validation
  * Improved error handling for parameter validation

## Version 0.3.52
* Fixed critical bugs in AutoModelFlow functionality:
  * Enhanced topic model parameter filtering to explicitly define supported parameters
  * Added better handling of custom parameters in topic modeling functions
  * Fixed cross-validation implementation to properly call cv.split() method
  * Added debug logging for filtered parameters to improve troubleshooting
  * Added complete parameter validation for topic modeling
  * Fixed pipeline integration when passing custom model parameters

## Version 0.3.51
* Added automatic text column detection to flag_similar_records function:
  * Implemented smart detection of text vs. structured data columns
  * Added specialized text similarity methods: fuzzy, TF-IDF, n-gram, and LSH
  * Added text-specific thresholding and automatic weight boosting
  * Fixed parallel processing issues for efficient handling of large datasets
  * Added comprehensive test suite for auto-detection features
  * Created detailed examples showing auto-detection capabilities
  * Implemented backwards compatibility with existing code
  * Added documentation for all new parameters and features

## Version 0.3.50
* Fixed critical bugs in parameter handling:
  * Fixed errors when passing unsupported parameters to topic modeling functions
  * Fixed parameter extraction for early_stopping_rounds in LightGBM models
  * Added comprehensive test coverage for parameter handling
  * Implemented parameter filtering to safely handle unsupported parameters
  * Added robust fallback mechanisms for error recovery in topic modeling
  * Fixed model attribute access in pipeline integration tests
  * Improved Pipeline integration with shorthand model types
* Added new test files for comprehensive validation:
  * Added test_shorthand_model_types_integration.py for validating model type support
  * Added test_early_stopping_parameter.py for early stopping parameter handling
  * Added test_topic_modeling_parameters.py for parameter filtering tests

## Version 0.3.49
* Enhanced model type support in hyperparameter tuning:
  * Added support for shorthand model types like 'lgbm_classifier' and 'lgbm_regressor'
  * Improved error messages with list of supported model types
  * Unified handling of direct and shorthand model types in tuning pipeline

## Version 0.3.48
* Fixed critical bugs in auto_model functionality:
  * Fixed parameter handling in LightGBM classifier to prevent "unexpected keyword argument 'early_stopping_rounds'" error
  * Fixed handling of 'sampling_ratio' parameter in topic modeling to prevent "unexpected keyword argument" error
  * Added robust error handling and recovery for topic modeling in autoflow
  * Improved parameter validation and filtering throughout model creation pipeline
  * Enhanced error recovery for text processing with better fallback mechanisms
  * Fixed index alignment issues in document-topic distributions

## Version 0.3.47
* Fixed critical bugs in deduplication module:
  * Fixed handling of categorical columns in evaluation to prevent "Categorical does not support reduction 'sum'" error
  * Resolved multiprocessing pickling issues in chunked processing by moving nested functions to module level
  * Improved robustness for various data types in deduplication evaluation
  * Enhanced parallel processing for better stability with large datasets

## Version 0.3.46
* Added enhanced deduplication reporting system:
  * Implemented comprehensive HTML reports with detailed statistics and visualizations
  * Created Markdown reports summarizing deduplication results
  * Added Excel export with detailed duplicate pairs and similarity scores
  * Implemented PowerPoint report generation for presentations
  * Added special Jupyter notebook rendering for interactive exploration
  * Implemented interactive visualizations with threshold sensitivity analysis
  * Added business impact analysis for optimal threshold selection
  * Created cohesive reporting class with multiple export formats
  * Added examples demonstrating the enhanced reporting capabilities

## Version 0.3.45
* Added advanced LSH algorithms and evaluation capabilities:
  * Implemented SimHash for efficient text similarity
  * Added BKTree for edit distance similarity search
  * Implemented SuperMinHash for more efficient similarity signatures
  * Added evaluation with known duplicate flags for measuring accuracy
  * Implemented precision, recall, and F1 score calculations
  * Added confusion matrix visualization for duplicate detection
  * Created threshold sensitivity analysis for finding optimal thresholds
  * Implemented advanced LSH method comparison utility
  * Added comprehensive examples for all new functionality

## Version 0.3.44
* Added interactive progress tracking for deduplication:
  * Implemented real-time progress indicators for Jupyter notebooks
  * Added ETA calculations based on completed comparisons
  * Added block processing progress tracking for chunked processing
  * Implemented memory usage monitoring
  * Created examples demonstrating the progress tracking functionality
  * Added dependencies for Jupyter integration (ipython, ipywidgets)
  * Added psutil dependency for memory tracking

## Version 0.3.43
* Added blocking and LSH functionality to `flag_similar_records`:
  * Implemented blocking for faster deduplication based on exact column matches
  * Added Locality-Sensitive Hashing (LSH) for approximate matching with high performance
  * Implemented phonetic and n-gram based blocking strategies
  * Added combined blocking and LSH approach for very large datasets
  * Created dedicated modules for blocking and LSH implementations
  * Added comprehensive example demonstrating performance improvements
  * Added optional dependencies for LSH (datasketch) and phonetic matching (jellyfish)

## Version 0.3.42
* Fixed bug in deduplication module:
  * Fixed `flag_similar_records` function to handle empty weights dictionary
  * Prevented division by zero error when normalizing weights

## Version 0.3.41
* Improved memory efficiency in deduplication module:
  * Added missing parameters to `flag_similar_records` function for better backward compatibility
  * Enhanced memory optimization for large dataset processing
  * Implemented generator-based pair creation instead of storing all pairs in memory
  * Added batch processing for record comparisons
  * Improved garbage collection during intensive operations
  * Added strategic processing order for duplicate detection
  * Enhanced parallel processing with configurable workers
  * Added progress reporting for long-running operations
  * Implemented adaptive chunk size reduction for very large datasets
  * Added proportional sampling across chunks when using limits
* Documentation and examples:
  * Created comprehensive examples demonstrating memory-efficient deduplication
  * Added benchmarking tools to compare different approaches
  * Updated documentation with parameter explanations and best practices
  * Added visualization capabilities for duplicate group analysis
  * Fixed documentation to reflect current parameter names

## Version 0.3.40
* Added comprehensive documentation and examples:
  * Created detailed examples for duplicate flagging functionality
  * Added documentation for advanced EDA features
  * Added documentation for automatic train-test splitting in automodeling
  * Added export capabilities documentation for PowerPoint and Excel
  * Enhanced README with improved dependency information and examples
  * Added cross-references between documentation files for better discoverability
  * Added examples demonstrating complete end-to-end workflows
* Enhanced Jupyter notebook integration:
  * Added notebook-friendly examples with interactive visualizations
  * Improved display capabilities for deduplication reporting
  * Added comprehensive workflow examples in a notebook-friendly format
  * Added display_eda_report method for interactive EDA reports in Jupyter
* Features include:
  * Streamlined installation instructions with dependency groups
  * Clearer documentation of optional feature requirements
  * Examples showing integration between different modules
  * Performance optimization guidance for large datasets
  * Detailed examples of advanced features
  * Interactive visualized EDA reports for Jupyter notebooks
* Improved user experience:
  * Added complete workflow examples from data loading to modeling
  * Better explanations of component interactions
  * Clearer API documentation with usage examples
  * More comprehensive examples showing real-world use cases
  * Quick reports for exploratory analysis in interactive environments

## Version 0.3.39
* Added duplicate flagging functionality:
  * `flag_exact_duplicates()` to identify exact matches across specified columns
  * `flag_text_duplicates()` for identifying similar text content with multiple methods
  * `flag_similar_records()` for multi-column weighted similarity detection
  * `flag_supervised_duplicates()` for ML-based duplicate identification
  * `add_duplicate_detection_columns()` as a high-level wrapper for all methods
* Added performance optimizations for large datasets:
  * Chunked processing to handle datasets too large for all-pairs comparison
  * Streaming LSH implementation for text collections that don't fit in memory
  * Parallel processing capabilities with configurable number of workers
  * Polars integration for faster string operations and reduced memory usage
  * Network-based algorithms for identifying duplicate clusters efficiently
* Features include:
  * Non-destructive duplicate identification (adds columns instead of removing rows)
  * Support for both pandas and polars DataFrames
  * Multiple similarity measures (hash, n-gram, fuzzy matching, LSH)
  * Graph-based clustering for finding duplicate groups
  * Configurable thresholds and weighting for different columns
  * Integration with existing deduplication framework
  * Memory efficiency improvements for very large datasets (50-70% reduction)
  * Performance improvements of 2-5x for large datasets
* Added comprehensive benchmarking tools for comparing different implementations
* Added test suite and examples for all duplicate flagging methods and optimizations

## Version 0.3.38
* Enhanced Polars-optimized supervised deduplication:
  * Improved robustness and error handling for production use
  * Fixed numpy divide warnings in correlation calculations
  * Enhanced feature name handling for ensemble models
  * Added adaptive thresholding for improved duplicate detection
  * Fixed model compatibility for RandomForest and GradientBoosting models
  * Added comprehensive benchmarking tools and examples
  * Enhanced feature contribution calculation for better explainability
  * Improved chunked processing to handle large datasets more efficiently
  * Added support for mixed data types and diverse feature sets
  * Fixed edge cases in cross-batch duplicate detection
* Added integration tests for deduplication pipeline with Polars optimization:
  * End-to-end testing of LSH and supervised models together
  * Cross-validation of results between pandas and Polars implementations
  * Mixed dataframe type testing (pandas/Polars interoperability)
  * Advanced features integration testing
* Added comprehensive example scripts:
  * Performance benchmarking between pandas and Polars implementations
  * Detailed supervised deduplication examples with visualization
  * End-to-end duplicate detection workflow examples
  * Demonstration of advanced features and optimizations

## Version 0.3.37
* Added Polars-optimized deduplication for improved performance:
  * Optimized LSH deduplication with 2-5x performance improvements
  * Memory-efficient processing of large datasets (60-70% less memory usage)
  * Streaming deduplication for datasets that don't fit in memory
  * Batch processing to handle very large text collections
  * Integration with existing deduplication framework
* Added Polars-optimized supervised deduplication:
  * Optimized feature generation for record pairs
  * Enhanced batch processing for large datasets
  * Maintained all advanced features (active learning, incremental learning, etc.)
  * Chunked processing for memory efficiency
  * Graceful fallback to pandas implementation
* Added Polars-optimized text utilities:
  * Batch text processing for large collections
  * Optimized text vectorization workflows
  * Memory-efficient similarity calculations
  * Parallel processing capabilities
  * Text column deduplication and processing utilities
* Enhanced dataframe utilities:
  * Improved conversion between pandas and Polars
  * Better datatype optimization
  * Chunked processing for large datasets
  * Enhanced datetime detection

## Version 0.3.36
* Added supervised deduplication enhancements:
  * Active learning for efficient labeling of ambiguous duplicate pairs
  * Incremental learning for continually updating models with new data
  * Entity resolution framework with blocking strategies for large datasets
  * Advanced explainability for duplicate detection decisions
  * Ensemble deduplication methods combining multiple modeling approaches
  * Automatic threshold optimization with business impact analysis
* Improved deduplication reporting with interactive visualizations

## Version 0.3.35
* Fixed serialization issue with optimized topic models
* Added documentation for topic model export functionality
* Enhanced index tracking for consistent processing of large datasets

## Version 0.3.34
* Added split EDA reports for improved memory efficiency
* Enhanced automated modeling workflows with better performance metrics
* Fixed export functionality to support various output formats
* Added more comprehensive examples for advanced use cases

## Version 0.3.33
* Fixed an issue with date format detection in mixed datasets
* Enhanced support for international date formats
* Added support for Australian date patterns
* Improved memory efficiency for large datasets
* Enhanced reporting capabilities with new visualization options

## Version 0.3.32
* Added enhanced support for Excel date format detection
* Fixed currency rendering in matplotlib visualizations
* Improved datatype detection for numeric fields
* Enhanced report formatting for special characters

## Version 0.3.31
* Added support for LSH-based deduplication for improved scalability
* Enhanced text analytics with optimized topic modeling
* Fixed serialization issues with large topic models
* Added tracking for deduplicated indices

## Version 0.3.30
* Fixed CV training step references
* Renamed CrossValidatedTrainer to CrossValidationTrainer for clarity
* Added comprehensive examples for time series feature engineering
* Improved markdown report formatting

## Version 0.3.29
* Enhanced performance for large dataset processing
* Added support for ShapIQ feature engineering
* Improved numerical stability in lightgbm objectives
* Added documentation for time series feature engineering

## Version 0.3.28
* Added feature selection for time series datasets
* Enhanced documentation with detailed usage examples
* Fixed serialization issue with model exports

## Version 0.3.27
* Added support for optimized categorical encoders
* Enhanced cross-validation with stratification options
* Improved pipeline visualization
* Updated documentation with usage examples

## Version 0.3.26
* Added enhanced support for text analytics
* Fixed serialization issue with large datasets
* Improved performance for multivariate analysis
* Added comprehensive examples for time series analysis

## Version 0.3.25
* Fixed issue with date parsing in mixed format datasets
* Enhanced support for international date formats
* Added optimized routines for large dataset processing
* Improved documentation with more usage examples

## Version 0.3.24
* Added deduplication tracking for pipeline integration
* Enhanced reporting capabilities with interactive elements
* Fixed memory leak in large dataset processing
* Added more comprehensive examples for deduplication workflows

## Version 0.3.23
* Fixed incorrect datetime parsing for numeric-looking date strings
* Enhanced support for month-year formats
* Added memory optimizations for large datasets
* Updated documentation with detailed usage instructions

## Version 0.3.22
* Added enhanced exact deduplication algorithms
* Fixed bug in index tracking during deduplication
* Improved performance for fuzzy matching on large datasets
* Added support for custom similarity thresholds

## Version 0.3.21
* Added basic deduplication capabilities
* Fixed formatting issues in EDA reports
* Enhanced datatype detection for specialized formats
* Improved performance for large dataset analysis

## Version 0.3.20
* Fixed issues with handling scientific notation in numeric columns
* Enhanced support for complex currency formats
* Added automatic datatype handling improvements
* Streamlined EDA output for larger datasets

## Version 0.3.19
* Added early stopping for training
* Enhanced evaluation metrics
* Fixed serialization issues with pipelines
* Improved documentation for pipeline workflows

## Version 0.3.18
* Added automatic feature importance calculation
* Enhanced pipeline visualization capabilities
* Fixed issues with model serialization
* Added comprehensive examples for typical modeling workflows

## Version 0.3.17
* Added support for time series feature engineering
* Enhanced cross-validation for time series datasets
* Fixed memory leaks in large dataset analysis
* Updated documentation with time series examples

## Version 0.3.16
* Added comprehensive multivariate analysis
* Fixed memory performance for large datasets
* Enhanced visualization capabilities
* Added detailed examples for multivariate workflows

## Version 0.3.15
* Added new threshold optimization algorithms
* Enhanced multivariate analysis capabilities
* Fixed memory issues with large datasets
* Added comprehensive examples for primary workflows

## Version 0.3.14
* Added cross-validation training
* Enhanced model evaluation metrics
* Fixed serialization for complex models
* Added pipeline integration examples

## Version 0.3.13
* Added support for automated modeling workflows
* Enhanced pipeline configurations
* Fixed bug in model serialization
* Added comprehensive examples

## Version 0.3.12
* Added pipeline module for end-to-end workflows
* Enhanced visualization capabilities
* Fixed memory issues with large datasets
* Added detailed documentation and examples

## Version 0.3.11
* Added support for LightGBM custom objectives
* Enhanced model tuning capabilities
* Fixed serialization issues with complex models
* Added examples for custom objectives

## Version 0.3.10
* Added model calibration functionality
* Enhanced evaluation metrics
* Fixed memory leaks in large models
* Added comprehensive documentation and examples

## Version 0.3.9
* Added automatic feature importance
* Enhanced visualization capabilities
* Fixed serialization for complex models
* Improved documentation with detailed examples

## Version 0.3.8
* Added model factory for simplified model creation
* Enhanced evaluation metrics
* Fixed memory performance for large datasets
* Added comprehensive examples for key workflows

## Version 0.3.7
* Added feature selection module
* Enhanced cross-validation capabilities
* Fixed memory issues with large datasets
* Added detailed examples

## Version 0.3.6
* Added feature engineering module
* Enhanced model evaluation metrics
* Fixed bug in pipeline serialization
* Added comprehensive documentation and examples

## Version 0.3.5
* Added basic model selection capabilities
* Enhanced cross-validation routines
* Fixed memory leaks in large model training
* Added examples and documentation

## Version 0.3.4
* Added support for drift detection
* Enhanced data quality assessments
* Fixed memory issues with large datasets
* Added comprehensive examples

## Version 0.3.3
* Added cardinality analysis
* Enhanced outlier detection
* Fixed serialization issues
* Added detailed documentation and examples

## Version 0.3.2
* Added missing value analysis
* Enhanced data quality reporting
* Fixed memory performance for large datasets
* Added comprehensive examples for data quality workflows

## Version 0.3.1
* Added data quality module
* Enhanced exploratory data analysis
* Fixed serialization for complex datasets
* Added detailed examples

## Version 0.3.0
* Initial public release
* Exploratory data analysis
* Basic modeling support
* Visualization capabilities