# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

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