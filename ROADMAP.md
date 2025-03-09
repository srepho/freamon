# Freamon Development Roadmap

This document outlines the planned development phases and milestones for the Freamon package.

## Phase 1: Core Foundation (Current)

### Data Quality Module
- [x] Missing value analysis and handling
- [x] Outlier detection
- [x] Data type analysis
- [ ] Duplicate detection
- [ ] Cardinality analysis
- [ ] HTML report generation

### Utils Module
- [x] Dataframe type handling
- [x] Memory optimization
- [x] Categorical encoders
  - [x] One-hot encoding
  - [x] Ordinal encoding
  - [x] Target encoding
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
- [ ] Univariate analysis
  - [ ] Distribution plots for numerical features
  - [ ] Bar plots for categorical features
- [ ] Bivariate analysis
  - [ ] Correlation analysis
  - [ ] Feature vs. target plots
- [ ] Multivariate analysis
  - [ ] PCA visualization
  - [ ] t-SNE visualization
- [ ] Time series analysis
  - [ ] Trend analysis
  - [ ] Seasonality detection
  - [ ] Autocorrelation plots
- [ ] Interactive HTML reports

### Features Module
- [ ] Automated feature engineering
- [ ] Polynomial features
- [ ] Interaction terms
- [ ] Time-based features from datetime columns
- [ ] Feature scaling and normalization
- [ ] Feature selection methods
  - [ ] Filter methods
  - [ ] Wrapper methods
  - [ ] Embedded methods

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
- [ ] Forecasting models
- [ ] Feature engineering for time series
- [ ] Seasonal decomposition
- [ ] Anomaly detection

### Model Explainability
- [ ] SHAP value integration
- [ ] Partial dependence plots
- [ ] Feature interaction analysis
- [ ] Interactive explainability reports

### Text Analytics Expansion
- [ ] Advanced NLP capabilities
  - [ ] Named entity recognition
  - [ ] Topic modeling
  - [ ] Sentiment analysis
  - [ ] Word embedding features

## Phase 4: User Experience & Integration

### Pipeline Building
- [ ] Unified pipeline interface
- [ ] Pipeline persistence
- [ ] Pipeline visualization

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

## Community Feedback & Future Directions

We welcome input from users to guide future development priorities. Some areas we're considering:

- Deep learning integration
- Support for more data sources (databases, APIs, etc.)
- Deployment helpers
- Data drift detection
- Specialized modules for specific industries

Please share your feedback and feature requests through GitHub issues or discussions.