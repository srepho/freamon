# Freamon Development Roadmap

This document outlines the planned development phases and milestones for the Freamon package.

## Phase 1: Core Foundation (Current)

### Data Quality Module
- [x] Missing value analysis and handling
- [x] Outlier detection
- [x] Data type analysis
- [x] Duplicate detection
- [x] Cardinality analysis
- [x] HTML report generation
- [x] Data drift detection

### Utils Module
- [x] Dataframe type handling
- [x] Memory optimization
- [x] Categorical encoders
  - [x] One-hot encoding
  - [x] Ordinal encoding
  - [x] Target encoding
  - [x] Binary encoding (via category_encoders)
  - [x] Hashing encoding (via category_encoders)
  - [x] Weight of Evidence (WOE) encoding (via category_encoders)
- [x] Basic text processing
  - [x] Support for spaCy integration
  - [x] Text preprocessing
  - [x] Bag-of-words features
  - [x] TF-IDF features

### Model Selection Module
- [x] Train/test splitting
- [x] Time-series aware splitting
- [x] Cross-validation
- [x] Time-series cross-validation
- [ ] Hyperparameter optimization

### Modeling Module
- [x] Model creation factory
- [x] Support for scikit-learn models
- [x] Support for LightGBM
- [x] Support for XGBoost
- [x] Support for CatBoost
- [x] Model evaluation metrics
- [x] Feature importance extraction
- [x] Model persistence

## Phase 2: Expanded Functionality

### EDA Module
- [x] Univariate analysis
  - [x] Distribution plots for numerical features
  - [x] Bar plots for categorical features
- [x] Bivariate analysis
  - [x] Correlation analysis
  - [x] Feature vs. target plots
- [x] Multivariate analysis
  - [x] PCA visualization
  - [x] t-SNE visualization
- [x] Time series analysis
  - [x] Trend analysis
  - [x] Seasonality detection
  - [x] Autocorrelation plots
- [x] Interactive HTML reports

### Features Module
- [x] Automated feature engineering
  - [x] ShapIQ-based interaction detection
  - [x] Automatic creation of interaction features
- [x] Time series feature engineering
  - [x] Automated lag detection and generation
  - [x] Intelligent rolling window features
  - [x] Smart differencing features
  - [x] Multiple time series support
- [x] Polynomial features
- [x] Standard interaction terms
- [x] Time-based features from datetime columns
- [x] Function to automatically create variables based on differences between date columns
- [ ] Feature scaling and normalization
- [x] Feature selection methods
  - [x] Filter methods
  - [x] Wrapper methods
  - [x] Embedded methods

### Large Dataset Handling
- [ ] Chunking for out-of-core processing
- [ ] Lazy evaluation support
- [ ] Memory usage monitoring
- [ ] Integration with distributed computing frameworks

## Phase 3: Advanced Features

### AutoML Capabilities
- [ ] Automated model selection
- [ ] Automated hyperparameter tuning
- [ ] Automated feature selection
- [ ] Ensemble model building

### Advanced Time Series Support
- [x] Seasonal decomposition (STL, classical)
- [x] Stationarity testing (ADF, KPSS)
- [x] Multiple seasonality detection
- [x] Feature engineering for time series
  - [x] Automated lag detection
  - [x] Intelligent rolling features
  - [x] Smart differencing
- [x] Forecast performance evaluation
- [ ] Forecasting models
- [ ] Anomaly detection

### Model Explainability
- [x] SHAP value integration
- [x] ShapIQ integration for interaction detection
- [x] Automatic feature engineering based on interactions
- [x] Interactive explainability reports
- [ ] Partial dependence plots
- [ ] Summary visualizations

### Text Analytics Expansion
- [ ] Advanced NLP capabilities
  - [ ] Named entity recognition
  - [ ] Topic modeling
  - [ ] Sentiment analysis
  - [ ] Word embedding features

## Phase 4: User Experience & Integration

### Pipeline Building
- [x] Unified pipeline interface
- [x] Pipeline persistence
- [x] Pipeline visualization
- [x] Interactive HTML reports

### Interactive Dashboards
- [ ] Web-based dashboard for data exploration
- [ ] Real-time model monitoring
- [ ] Interactive model comparison

### Integration with ML Platforms
- [ ] MLflow integration
- [ ] Weights & Biases integration
- [ ] Optuna integration

### Documentation & Examples
- [ ] Comprehensive API reference
- [ ] User guides and tutorials
- [ ] Interactive examples
- [ ] Case studies

## Next Steps (v0.2.5+)

### Completed in v0.2.4
- [x] Added multivariate analysis with correlation network visualization
- [x] Integrated PCA and t-SNE for dimensionality reduction
- [x] Implemented chunk-based processing for large datasets
- [x] Added utilities for saving/loading chunked data
- [x] Improved Polars integration with enhanced time_unit handling
- [x] Created example script and documentation for large dataset handling

### Current Focus (v0.2.5)
- [x] Enhanced time series analysis
  - [x] Advanced seasonality detection and decomposition (STL, classical)
  - [x] Stationarity testing with ADF and KPSS tests
  - [x] Multiple seasonality detection and visualization
  - [x] Forecast performance evaluation framework
- [x] Time series feature engineering automation
  - [x] Intelligent lag feature detection and generation
  - [x] Auto-detection of optimal rolling window sizes
  - [x] Smart differencing features based on stationarity analysis
  - [x] Support for multiple time series (panel data)

### Completed in v0.2.6
- [x] Feature selection module expansion
  - [x] Recursive feature elimination with cross-validation
  - [x] Stability selection implementation
  - [x] Genetic algorithm-based feature selection
  - [x] Support for multi-objective feature selection
  - [x] Time series-specific feature selection methods

### Coming Soon (v0.2.7)
- CI/CD examples for automated pipeline deployment
- Enhanced dashboard with interactive visualizations

### Medium-term (v0.3.0)
- Integrate more hyperparameter tuning strategies
- Integrate with MLflow and other tracking platforms
- Create a web-based UI for pipeline configuration
- Expand the automated feature engineering capabilities

### Long-term (v0.4.0+)
- Deep learning integration
- Support for more data sources (databases, APIs, etc.)
- Deployment helpers
- Specialized modules for specific industries
- Automated reporting through visualization dashboards

## Community Feedback & Future Directions

We welcome input from users to guide future development priorities. Please share your feedback and feature requests through GitHub issues or discussions.