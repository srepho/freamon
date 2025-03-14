# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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