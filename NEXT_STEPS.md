# Freamon Development: Next Steps

## Project Status

Freamon is a comprehensive package for data science and machine learning on tabular data. Current version: 0.2.0

The package has recently completed several major features:
- Pipeline system with visualization and persistence
- Data drift detection and monitoring
- ShapIQ integration for feature engineering and explainability
- Advanced category encoders (binary, hashing, WOE)
- LightGBM optimizations and hyperparameter tuning
- Model calibration and importance calculations
- Documentation and examples for all new features

## Priority Development Tasks

Based on the roadmap and current status, here are the next development priorities:

### 1. EDA Module Enhancements

- **Multivariate Analysis**
  - Implement PCA visualization for high-dimensional data
  - Add correlation networks for identifying feature clusters
  - Create interactive heatmaps for feature interactions

- **Time Series EDA**
  - Add seasonality detection
  - Implement decomposition plots (trend, seasonal, residual)
  - Add autocorrelation and partial autocorrelation visualizations

### 2. Feature Selection Methods

- **Filter Methods**
  - Implement chi-square feature selection for categorical features
  - Add ANOVA F-value selection for regression problems

- **Wrapper Methods**
  - Implement recursive feature elimination (RFE)
  - Add forward/backward selection methods

- **Embedded Methods**
  - Implement Lasso and Ridge Regression based feature selection
  - Add regularization-based feature selection for tree models

### 3. Large Dataset Handling

- **Chunking Mechanisms**
  - Implement streaming data processing for large datasets
  - Add chunk-based processing for memory-intensive operations

- **Distributed Processing**
  - Add Dask integration for large-scale data processing
  - Implement parallel processing for intensive operations

### 4. AutoML Capabilities

- **Automatic Hyperparameter Optimization**
  - Expand existing LightGBM tuning to other models
  - Implement pipeline-level optimization

- **Model Selection**
  - Add automatic model selection based on problem type
  - Implement ensemble methods for combining multiple models

### 5. Quality Improvements

- **Testing**
  - Fix failing tests in test suite
  - Improve test coverage for new features
  - Add integration tests for complete workflows

- **Documentation**
  - Complete documentation for all modules
  - Add more examples showcasing feature combinations
  - Create tutorials for common workflows

## Bug Fixes and Technical Debt

### Completed Fixes (v0.2.1 patch)

- ✅ Fixed pipeline tests by:
  - Making FeatureEngineer accept an optional dataframe
  - Adding fit() method to FeatureEngineer and ModelTrainer
  - Fixing MockPipelineStep for testing
  - Improving robustness of test assertions

- ✅ Fixed datetime detection tests by:
  - Converting DatetimeIndex to Series for assertions
  - Excluding ID-like columns from timestamp detection
  - Fixing custom date format tests

### Remaining Issues to Address

- Text processing features (CountVectorizer and TfidfVectorizer parameters)
- Polars integration (time_unit parameter in datetime conversion)
- ShapIQ-related tests (graceful handling when library not available)
- Pipeline visualization (handle FeatureEngineer initialization)
- Early stopping issue with cosine annealing
- Charts/HTML generation in EDA module
- Fix pandas DeprecationWarning for is_categorical_dtype

## Release Planning

### Version 0.2.x Patches

- Version 0.2.1: Fix failing tests and known bugs
- Version 0.2.2: Address technical debt and improve documentation

### Version 0.3.0

- Implement EDA module enhancements
- Add feature selection methods
- Improve large dataset handling

### Version 0.4.0

- Add AutoML capabilities
- Implement ensemble methods
- Add web-based dashboard for exploration

## Getting Involved

Contributors can help with:

1. Fixing failing tests
2. Improving documentation
3. Adding examples demonstrating features
4. Implementing new features from the roadmap

To get started, install the development version:

```bash
# Clone the repository
git clone https://github.com/yourusername/freamon.git
cd freamon

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests to identify issues
pytest
```