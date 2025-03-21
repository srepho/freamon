# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2025-04-29

### Added
- Advanced data type detection with Australian data patterns:
  - Added DataTypeDetector class for semantic type detection
  - Added support for Australian postcodes, phone numbers, ABNs, ACNs, and TFNs
  - Added custom pattern handling with prioritization
  - Added type conversion suggestions based on content analysis
  - Added datetime detection with various formats
  - Added Excel date detection for numeric columns (supports dates exported from Excel to CSV)
  - Added comprehensive test suite with performance tests
  - Added documentation for data type detection in docs/usage/data_type_detection.md
  - Added example scripts for data type detection and custom patterns

### Fixed
- Fixed dependency issues with `pip install freamon[all]` option:
  - Separated development tools from the `all` option to avoid confusion
  - Added `full` option for dependencies including development tools
  - Updated documentation to clarify installation options

## [0.3.2] - 2025-04-25

### Added
- Word embeddings capabilities in TextProcessor:
  - Added create_word2vec_embeddings() method for training Word2Vec models
  - Added load_pretrained_embeddings() for GloVe and FastText support
  - Added create_document_embeddings() for document-level representations
  - Added calculate_embedding_similarity() for comparing embeddings
  - Added find_most_similar_documents() for document similarity search
  - Enhanced create_text_features() to include embedding-based features
  - Added save_word_vectors() and load_word_vectors() for offline mode support
  - Added load_word2vec_model() for loading entire models from disk
  - Added comprehensive documentation for word embedding functions
  - Added word_embeddings_example.py showcasing new capabilities
  - Added test_word_embeddings.py for testing core functionality
  - Added new word_embeddings optional dependency group

### Improved
- Enhanced text feature engineering pipeline:
  - Integrated word embeddings with existing text features
  - Added PCA dimensionality reduction for embedding features
  - Improved error handling for optional dependencies
  - Added support for document similarity visualization
  - Added integration tests for word embeddings with Pipeline system
  - Added efficient handling of duplicate texts to improve performance
  - Added robust handling of blank/null text values with fallback strategies
  - Added benchmarking utilities for word embedding components
  - Added examples of offline usage in environments without internet access

## [0.3.1] - 2025-04-22

### Added
- Advanced topic modeling capabilities in TextProcessor:
  - Added create_topic_model() method supporting LDA and NMF algorithms
  - Added plot_topics() for visualizing topic models with horizontal bar charts
  - Added get_document_topics() for extracting document-topic distributions
  - Added calculate_topic_coherence() for evaluating topic model quality
  - Added find_optimal_topics() to determine the optimal number of topics
  - Enhanced create_text_features() to include topic modeling features
  - Added comprehensive documentation for topic modeling functions
  - Added topic_modeling_example.py showcasing new capabilities
  - Added tests for topic modeling functionality

### Improved
- Enhanced text processing capabilities:
  - Better integration with spaCy for advanced NLP tasks
  - Improved error handling for optional dependencies
  - Added support for interactive visualizations with HTML export
  - Enhanced documentation with more examples

## [0.3.0] - 2025-04-15

### Added
- Enhanced time series regression capabilities:
  - Improved LightGBM integration for time series modeling
  - Added helper functions for easier model creation with sensible defaults
  - Added visualization tools for model evaluation and interpretation
  - Added feature importance analysis with grouping by feature types
  - Added time series cross-validation with prediction saving
  - Added visualization for CV metrics, feature importance, and time series predictions
  - Updated text_time_series_regression_example.py to demonstrate these capabilities

### Changed
- Refactored model creation with helper functions:
  - Added create_lightgbm_regressor() and create_lightgbm_classifier() functions
  - Added create_sklearn_model() function for scikit-learn models
  - Improved parameter handling for direct model creation vs. tuning
- Enhanced visualization capabilities:
  - Added plot_cv_metrics() for visualizing cross-validation results
  - Added plot_feature_importance() with flexible display options
  - Added plot_importance_by_groups() for grouped feature importance analysis
  - Added plot_time_series_predictions() for visualization of predictions over time
  - Added summarize_feature_importance_by_groups() for aggregated importance analysis

### Fixed
- Fixed LightGBMModel compatibility with direct LightGBM usage
- Fixed eval_set handling in Model class
- Fixed issues with feature importance extraction and visualization
- Fixed serialization of datetime objects in visualization functions

## [0.2.9] - 2025-04-10

### Added
- Enhanced text analytics capabilities:
  - Added text statistics extraction for basic feature engineering
  - Added readability metrics calculation (Flesch Reading Ease, Flesch-Kincaid Grade Level, etc.)
  - Added RAKE (Rapid Automatic Keyword Extraction) for keyword extraction
  - Added lexicon-based sentiment analysis with customizable lexicons
  - Added document similarity calculation using cosine, Jaccard, and overlap methods
  - Added comprehensive text feature creation combining multiple analysis methods
  - Added example script demonstrating text analytics capabilities
  - Added documentation for text analytics features

## [0.2.8] - 2025-04-05

### Fixed
- Added missing `Callable` imports in several modules:
  - Added to `eda/analyzer.py`, `eda/report.py`, and `data_quality/analyzer.py`
  - Added to `modeling/lightgbm.py`
  - Ensures proper type checking and IDE support for function types

## [0.2.7] - 2025-04-01

### Added
- Classification threshold optimization capabilities:
  - Added `find_optimal_threshold` function to evaluate and select optimal probability thresholds
  - Updated `LightGBMModel` to support custom probability thresholds for binary classification
  - Added threshold-aware predictions with customizable cutoffs
  - Added utility for optimizing various metrics: F1, precision, recall, accuracy, and more
  - Added support for visualizing threshold effects on model performance
  - Added example script demonstrating threshold optimization functionality

## [0.2.6] - 2025-03-30

### Fixed
- Fixed multivariate analysis functionality:
  - Added support for 'both' method parameter to perform PCA and t-SNE together
  - Updated matplotlib API calls to use current non-deprecated functions
- Fixed ShapIQ integration with more robust error handling:
  - Added compatibility with different versions of ShapIQ API
  - Improved error detection and fallback mechanisms
  - Enhanced interaction detection with better property checking
- Fixed Polars integration for datetime detection:
  - Added support for multiple Polars versions with better conversion strategy
  - Fixed timestamp conversion with appropriate time units
- Improved test coverage and mocking for optional dependencies

## [0.2.5] - 2025-03-28

### Added
- Cross-validation integration in the pipeline system
  - Added `CrossValidatedTrainer` for unified CV training
  - Added `CrossValidationTrainingStep` for pipeline integration
  - Added support for multiple ensemble methods: best, average, weighted, stacking
  - Added support for various CV strategies: kfold, stratified, timeseries, walk_forward
- Enhanced model selection capabilities
  - Added unified cross-validation interface
  - Improved support for time series data
  - Added ensemble model creation from multiple folds

### Fixed
- Fixed circular import issues in the pipeline system
- Fixed parameter compatibility with scikit-learn models
- Improved error handling in cross-validation
- Fixed feature importance extraction for ensemble models

## [0.2.4] - 2025-03-25

### Added
- Enhanced multivariate analysis
  - Added correlation network visualization for identifying feature relationships
  - Added interactive heatmaps for feature interactions
  - Added community detection for identifying feature clusters
  - Improved PCA and t-SNE visualizations
- Large dataset handling
  - Added chunk-based processing for handling large datasets efficiently
  - Implemented functions for processing, iterating, saving, and loading data in chunks
  - Added integration with Dask for distributed processing
  - Improved Polars integration with better time unit handling

### Fixed
- Fixed pandas DeprecationWarning for is_categorical_dtype
- Fixed Polars integration with time_unit parameter compatibility across versions
- Improved error handling in multivariate analysis functions

### Documentation
- Added comprehensive documentation for multivariate analysis features
- Added documentation for large dataset handling
- Created example scripts for multivariate analysis and large dataset processing

## [0.2.3] - 2025-03-20

### Added
- Automated end-to-end pipeline example that integrates:
  - Data quality analysis and cleaning
  - Drift detection between datasets
  - Exploratory data analysis (EDA)
  - Feature engineering with ShapIQ integration
  - Model training with LightGBM and hyperparameter tuning
  - Comprehensive HTML reporting with interactive dashboard
- Documentation for the automated pipeline in docs/usage/automated_pipeline.md
- Integrated dashboard that connects all reports through a single interface

## [0.2.2] - 2025-03-15

### Fixed
- Fixed pandas deprecation warnings in data quality and EDA modules
  - Updated deprecated `pd.api.types.is_*_dtype()` functions to modern `select_dtypes()` approach
  - Affected modules: drift analysis, univariate analysis, data quality
- Fixed LightGBM tuning issues with log-scale parameters
  - Modified tuning to use small positive values (1e-5) instead of zeros
  - Updated categorical data type handling in tests
- Improved pipeline visualization with proper matplotlib fallback
  - Made visualization work without Graphviz dependency
  - Updated tests to handle Graphviz absence gracefully

### Added
- Enhanced property-based testing for text processing utilities
- Added proper dependency handling for optional packages
- Improved documentation for optional dependencies

## [0.2.0] - 2025-03-10

### Added
- ShapIQ integration for explainability and feature engineering
  - Added `ShapIQExplainer` for detecting feature interactions
  - Added `ShapIQFeatureEngineer` for automatic feature engineering based on detected interactions
  - Added HTML reporting for interaction analysis
- Advanced categorical encoders from the category_encoders package
  - Added `BinaryEncoderWrapper` for binary encoding of categorical variables
  - Added `HashingEncoderWrapper` for hashing encoding of high-cardinality categorical variables
  - Added `WOEEncoderWrapper` for weight of evidence encoding
- Pipeline system
  - Added unified pipeline interface for connecting all steps
  - Added pipeline persistence capabilities
  - Added pipeline visualization with multiple backends
  - Added interactive HTML reports for pipeline workflows
- Data drift detection
  - Added drift detection for numeric, categorical, and datetime features
  - Added statistical measures of drift significance
  - Added visual reporting of data drift
- LightGBM optimizations
  - Added intelligent hyperparameter tuning with optuna
  - Added custom objectives support
  - Added early stopping callbacks
  - Added automatic feature importance calculation
- Model calibration for classification models
  - Added isotonic and sigmoid calibration
  - Added calibration visualization
- Example scripts demonstrating all new functionality
  - Added examples for pipelines, LightGBM, ShapIQ, and data drift
  
### Changed
- Enhanced explainability module with SHAP and ShapIQ integration
- Improved feature engineering capabilities
- Updated README with comprehensive feature list and examples
- Updated ROADMAP to reflect implemented features
- Bumped version to 0.2.0 in all relevant files

## [0.1.0] - 2025-02-05

### Added
- Initial project structure
- Data quality module
  - Missing value analysis and handling
  - Outlier detection using IQR, Z-score, and modified Z-score methods
  - Data type analysis
- Utils module
  - Dataframe type checking and conversion
  - Memory usage optimization
  - Memory usage estimation
  - Categorical encoders (One-hot, Ordinal, Target)
  - Text processing utilities with optional spaCy integration
  - Automatic date/time detection and conversion
  - Support for multiple dataframe backends (Pandas, Polars, Dask)
- Model selection module
  - Train/test splitting
  - Time-series aware splitting
  - Stratified time-series splitting
  - Cross-validation
  - Time-series cross-validation
- Modeling module
  - Model creation factory
  - Support for scikit-learn, LightGBM, XGBoost, and CatBoost
  - Unified model interface
  - Model evaluation metrics
  - Model training and evaluation
  - Feature importance extraction
  - Model persistence
- Test framework
- Documentation
  - README with usage examples
  - Development roadmap