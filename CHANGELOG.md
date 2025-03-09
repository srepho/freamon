# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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