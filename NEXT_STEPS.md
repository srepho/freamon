# Freamon Development: Next Steps

## Project Status

Freamon is a comprehensive package for data science and machine learning on tabular data. Current version: 0.3.0

The package has recently completed several major features:
- Pipeline system with visualization and persistence
- Data drift detection and monitoring
- ShapIQ integration for feature engineering and explainability
- Advanced category encoders (binary, hashing, WOE)
- LightGBM optimizations and hyperparameter tuning
- Model calibration and importance calculations
- Text processing optimizations with multiple backends (pandas, polars, pyarrow)
- Time series regression capabilities with visualization tools
- Documentation and examples for all new features

## Priority Development Tasks

Based on the roadmap and current status, here are the next development priorities:

### 1. Advanced NLP Capabilities

- **Topic Modeling** ✅
  - Implement Latent Dirichlet Allocation (LDA) for topic extraction ✅
  - Add Non-Negative Matrix Factorization (NMF) as alternative algorithm ✅
  - Create topic coherence metrics and visualization ✅
  - Add topic-based document clustering ✅

- **Word Embeddings Integration** ✅
  - Add support for pre-trained word embeddings (Word2Vec, GloVe, FastText) ✅
  - Implement document-level embeddings for similarity and clustering ✅
  - Create visualization tools for word and document embeddings ✅
  - Add embedding-based feature engineering ✅

- **Advanced Text Classification**
  - Implement pipeline for text classification tasks
  - Add support for multi-label and hierarchical classification
  - Create evaluation metrics specific to text classification

### 2. EDA Module Enhancements

- **Multivariate Analysis**
  - Implement PCA visualization for high-dimensional data
  - Add correlation networks for identifying feature clusters
  - Create interactive heatmaps for feature interactions

- **Time Series EDA**
  - Add seasonality detection
  - Implement decomposition plots (trend, seasonal, residual)
  - Add autocorrelation and partial autocorrelation visualizations

### 3. Feature Selection Methods

- **Filter Methods**
  - Implement chi-square feature selection for categorical features
  - Add ANOVA F-value selection for regression problems

- **Wrapper Methods**
  - Implement recursive feature elimination (RFE)
  - Add forward/backward selection methods

- **Embedded Methods**
  - Implement Lasso and Ridge Regression based feature selection
  - Add regularization-based feature selection for tree models

### 4. Large Dataset Handling

- **Chunking Mechanisms**
  - Implement streaming data processing for large datasets
  - Add chunk-based processing for memory-intensive operations

- **Distributed Processing**
  - Add Dask integration for large-scale data processing
  - Implement parallel processing for intensive operations

### 5. AutoML Capabilities

- **Automatic Hyperparameter Optimization**
  - Expand existing LightGBM tuning to other models
  - Implement pipeline-level optimization

- **Model Selection**
  - Add automatic model selection based on problem type
  - Implement ensemble methods for combining multiple models

### 6. Quality Improvements

- **Testing**
  - Fix failing tests in test suite
  - Improve test coverage for new features
  - Add integration tests for complete workflows

- **Documentation**
  - Complete documentation for all modules
  - Add more examples showcasing feature combinations
  - Create tutorials for common workflows

## Bug Fixes and Technical Debt

### Completed Fixes (v0.3.0 patch)

- ✅ Enhanced text processing with multiple backends:
  - Added Polars integration for high-performance string operations
  - Added PyArrow backend for optimized memory usage
  - Implemented parallel processing for large datasets
  - Added batch processing for spaCy operations
  - Implemented auto-selection of optimal backend based on data size

- ✅ Improved time series regression capabilities:
  - Added helper functions for model creation
  - Added visualization tools for model evaluation
  - Added feature importance grouping by feature types

### Remaining Issues to Address

- Topic modeling integration (planned for v0.3.1)
- Word embedding integration (planned for v0.3.1)
- Polars integration (time_unit parameter in datetime conversion)
- ShapIQ-related tests (graceful handling when library not available)
- Pipeline visualization (handle FeatureEngineer initialization)
- Early stopping issue with cosine annealing
- Charts/HTML generation in EDA module
- Fix pandas DeprecationWarning for is_categorical_dtype

## Release Planning

### Version 0.3.x Patches

- Version 0.3.1: Implement advanced NLP capabilities (topic modeling, word embeddings)
- Version 0.3.2: Address technical debt and improve documentation

### Version 0.4.0

- Implement EDA module enhancements
- Add feature selection methods
- Improve large dataset handling

### Version 0.5.0

- Add AutoML capabilities
- Implement ensemble methods
- Add web-based dashboard for exploration

## Getting Involved

Contributors can help with:

1. Implementing advanced NLP capabilities
2. Fixing failing tests
3. Improving documentation
4. Adding examples demonstrating features

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