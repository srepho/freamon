# freamon

<p align="center">
  <img src="package_logo.webp" alt="Freamon Logo" width="250"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/freamon.svg)](https://pypi.org/project/freamon/)
[![GitHub release](https://img.shields.io/github/v/release/srepho/freamon)](https://github.com/srepho/freamon/releases)

A package to make data science projects on tabular data easier. Named after the great character from The Wire played by Clarke Peters. Featuring advanced data type detection with support for Australian data patterns.

## Features

- **Data Quality Assessment:** Missing values, outliers, data types, duplicates
- **Advanced Data Type Detection:** 
  - Semantic type identification including Australian-specific data patterns (postcodes, ABNs, ACNs, phone numbers)
  - Mixed date format detection and intelligent multi-pass conversion
  - Scientific notation detection and preservation
- **Exploratory Data Analysis (EDA):** Statistical analysis and visualizations
- **Feature Engineering:** 
  - **Standard Features:** Polynomial, interaction, datetime, binned features
  - **Automatic Interaction Detection:** ShapIQ-based automatic feature engineering  
  - **Time Series Feature Engineering:** Automated lag detection, rolling windows, differencing
- **Categorical Encoding:** 
  - **Basic Encoders:** One-hot, ordinal, target encoding
  - **Advanced Encoders:** Binary, hashing, weight of evidence (WOE) encoding
- **Text Processing:** Basic NLP with optional spaCy integration
- **Model Selection:** Train/test splitting with time-series awareness
- **Modeling:** Training, evaluation, and validation
  - **Support for Multiple Libraries:** scikit-learn, LightGBM, XGBoost, CatBoost
  - **Intelligent Hyperparameter Tuning:** Parameter-importance aware tuning for LightGBM
  - **Cross-Validation:** Training with cross-validation as the standard approach
    - **Multiple Strategies:** K-fold, stratified, time series, and walk-forward validation
    - **Ensemble Methods:** Combine models from different folds for improved performance
- **Explainability:** 
  - **SHAP Support:** Feature importance and explanations
  - **ShapIQ Integration:** Feature interactions detection and visualization
  - **Interactive Reports:** HTML reports for explainability findings
  - **Permutation Importance:** Better feature importance for black-box models
- **Pipeline System:**
  - **Integrated Workflow:** Connect feature engineering, selection, and modeling
  - **Modular Design:** Mix and match steps for custom workflows
  - **Persistence:** Save and load complete pipelines
  - **Visualization:** Pipeline visualization with multiple backends
- **Multiple DataFrame Backends:** 
  - **Pandas:** Standard interface
  - **Polars:** High-performance alternative
  - **Dask:** Out-of-core processing for large datasets

## Installation

```bash
# Basic installation
pip install freamon==0.3.16

# With all optional dependencies (no development tools)
pip install freamon[all]==0.3.16

# With all dependencies including development tools
pip install freamon[full]

# With specific optional dependencies
pip install freamon[lightgbm]        # For LightGBM support
pip install freamon[xgboost]         # For XGBoost support
pip install freamon[catboost]        # For CatBoost support
pip install freamon[nlp]             # For NLP capabilities with spaCy
pip install freamon[polars]          # For Polars support
pip install freamon[dask]            # For Dask support
pip install freamon[explainability]  # For SHAP and ShapIQ integration
pip install freamon[visualization]   # For pipeline visualization with Graphviz
pip install freamon[tuning]          # For hyperparameter tuning support

# Development installation
git clone https://github.com/yourusername/freamon.git
cd freamon
pip install -e ".[dev,all]"  # Or use pip install -e ".[full]" for all dependencies
```

## Quick Start

### Optimized Data Type Detection (New!)

```python
import pandas as pd
import numpy as np
from datetime import datetime
from freamon.utils.datatype_detector import DataTypeDetector

# Sample data with various data types
data = {
    'id': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
    'date_iso': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'date_us': ['01/06/2023', '01/07/2023', '01/08/2023', '01/09/2023', '01/10/2023'],
    'date_uk': ['11/01/2023', '12/01/2023', '13/01/2023', '14/01/2023', '15/01/2023'],
    'date_mixed': ['2023-01-16', '01/17/2023', '18/01/2023', '2023-01-19', '01/20/2023'],
    'excel_date': [44927, 44928, 44929, 44930, 44931],  # Excel dates (days since 1899-12-30)
    'month_year': ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023'],
    'australian_postcode': ['2000', '3000', '4000', '5000', '6000'],
    'australian_phone': ['02 9876 5432', '03 8765 4321', '08 7654 3210', '07 6543 2109', '04 5432 1098'],
    'continuous': [1.23, 4.56, 7.89, 10.11, 12.13],
    'categorical': [1, 2, 1, 3, 2],
    'scientific': ['1.2e-3', '3.4e-4', '5.6e-5', '7.8e-6', '9.0e-7'],
    'text_data': ['Sample text 1', 'Sample text 2', 'Sample text 3', 'Sample text 4', 'Sample text 5'],
    'mixed_with_nan': [44927, np.nan, 44929, np.nan, 44931]  # Excel dates with missing values
}

# Create DataFrame
df = pd.DataFrame(data)
print(f"Data shape: {df.shape}")

# Initialize detector with optimizations enabled
detector = DataTypeDetector(
    df,
    optimized=True,              # Enable performance optimizations
    use_pyarrow=True,            # Use PyArrow for faster processing
    sample_size=1000,            # Set maximum sample size for large datasets
    detect_semantic_types=True,  # Enable semantic type detection
    distinguish_numeric=True     # Distinguish between categorical and continuous numeric
)

# Detect all types
detector.detect_all_types()

# View detected types
print("\nDetected Column Types:")
for col, type_info in detector.column_types.items():
    print(f"{col}: {type_info}")

# View semantic types
print("\nSemantic Types:")
for col, sem_type in detector.semantic_types.items():
    if sem_type:  # Only show columns with detected semantic types
        print(f"{col}: {sem_type}")

# Get conversion suggestions
print("\nSuggested Conversions:")
conversions = detector.generate_conversion_suggestions()
for col, suggestion in conversions.items():
    print(f"{col}: {suggestion}")

# Convert columns to appropriate types
converted_df = detector.convert_types()
print("\nDataFrame info after conversion:")
print(converted_df.dtypes)

# Display visual report (in Jupyter notebooks)
# detector.display_detection_report()

# For standard Python environments, save HTML report
report_html = detector.get_column_report_html()
with open("datatype_detection_report.html", "w") as f:
    f.write(report_html)
print("\nSaved HTML report to 'datatype_detection_report.html'")
```

### Time Series Modeling and Visualization

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from freamon.modeling.helpers import create_lightgbm_regressor
from freamon.model_selection.cross_validation import time_series_cross_validate
from freamon.modeling.visualization import (
    plot_cv_metrics,
    plot_feature_importance, 
    plot_importance_by_groups,
    plot_time_series_predictions
)

# Create time series data with date, target and text features
def create_sample_data(n_days=365):
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create trend and seasonal components
    trend = np.linspace(100, 200, n_days)
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    monthly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 30)
    noise = np.random.normal(0, 10, n_days)
    
    # Target variable
    values = trend + weekly + monthly + noise
    
    # Text features (simulated)
    sentiments = np.random.choice(['positive', 'neutral', 'negative'], n_days, p=[0.3, 0.5, 0.2])
    topics = np.random.choice(['finance', 'technology', 'retail', 'healthcare'], n_days)
    
    return pd.DataFrame({
        'date': dates,
        'target': values,
        'sentiment': sentiments,
        'topic': topics,
        'text_length': np.random.randint(50, 500, n_days)
    })

# Create dataset and prepare features
df = create_sample_data()
print(f"Dataset created with {len(df)} observations")

# Feature engineering - create time-based features
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# Create lag features
for lag in [1, 2, 3, 7, 14]:
    df[f'target_lag_{lag}'] = df['target'].shift(lag)

# Create rolling window features
for window in [3, 7, 14]:
    df[f'target_roll_mean_{window}'] = df['target'].rolling(window=window).mean()
    df[f'target_roll_std_{window}'] = df['target'].rolling(window=window).std()

# Encode categorical variables
df = pd.get_dummies(df, columns=['sentiment', 'topic'], drop_first=True)

# Drop NAs from lag features
df = df.dropna()

# Define features and target
features = [col for col in df.columns if col not in ['date', 'target']]
X = df[features]
y = df['target']
date_col = df['date']

# Create LightGBM regressor with simplified helper function
model = create_lightgbm_regressor(num_leaves=31, learning_rate=0.1)

# Perform time series cross-validation with expanding window
cv_results = time_series_cross_validate(
    model, X, y, date_col,
    initial_window=0.5, 
    step=30,
    save_predictions=True
)

# Visualize cross-validation metrics
plot_cv_metrics(cv_results)

# Visualize feature importance from the last fold's model
plot_feature_importance(cv_results['models'][-1], X.columns, top_n=15)

# Group features by type and visualize importance by groups
feature_groups = {
    'time_features': ['dayofweek', 'month', 'day', 'is_weekend'],
    'lag_features': [col for col in X.columns if 'lag' in col],
    'rolling_features': [col for col in X.columns if 'roll' in col],
    'text_features': ['text_length'] + [col for col in X.columns if 'sentiment' in col or 'topic' in col]
}

plot_importance_by_groups(cv_results['models'][-1], X.columns, feature_groups)

# Visualize predictions from cross-validation
plot_time_series_predictions(
    date_col[cv_results['test_indices'][-1]],
    y.iloc[cv_results['test_indices'][-1]],
    cv_results['predictions'][-1]
)

# Final model training on all data
final_model = create_lightgbm_regressor(num_leaves=31, learning_rate=0.1)
final_model.fit(X, y)
```

### Time Series Feature Engineering

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer
from freamon.eda.time_series import analyze_seasonality, analyze_stationarity

# Create a synthetic time series dataset with daily data
def create_sample_data(n_days=365):
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create trend component
    trend = np.linspace(100, 200, n_days)
    
    # Weekly seasonality
    weekly_seasonality = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Monthly seasonality
    monthly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n_days) / 30)
    
    # Random noise
    noise = np.random.normal(0, 10, n_days)
    
    # Combine components
    values = trend + weekly_seasonality + monthly_seasonality + noise
    
    return pd.DataFrame({'date': dates, 'value': values})

# Create the dataset
df = create_sample_data()
print(f"Created time series data with {len(df)} observations")

# 1. Analyze the time series
print("\n1. Time Series Analysis")
seasonality = analyze_seasonality(df, 'date', 'value')
print(f"Detected periods: {seasonality['detected_periods']}")

# Check stationarity
stationarity = analyze_stationarity(df, 'date', 'value')
print(f"Stationarity status: {stationarity['stationarity_status']}")
if not stationarity['is_stationary'] and 'recommendations' in stationarity:
    print("Recommendations:")
    for rec in stationarity['recommendations']:
        print(f"- {rec}")

# 2. Automatically engineer time series features
print("\n2. Automated Feature Engineering")
ts_engineer = TimeSeriesFeatureEngineer(df, 'date', 'value')

# Add feature creation steps
result_df = (ts_engineer
    .create_lag_features(strategy='auto')  # Auto-detect optimal lags
    .create_rolling_features(
        metrics=['mean', 'std', 'min', 'max'],
        auto_detect=True  # Auto-detect optimal window sizes
    )
    .create_differential_features()
    .transform()
)

# Display resulting features
print(f"Original dataframe shape: {df.shape}")
print(f"After automatic feature engineering: {result_df.shape}")
print("\nGenerated features:")
new_columns = [col for col in result_df.columns if col not in df.columns]
for col in new_columns[:5]:  # Show first 5 features
    print(f"- {col}")
if len(new_columns) > 5:
    print(f"... and {len(new_columns) - 5} more features")

# 3. Use these features for forecasting or classification
print("\n3. Ready for modeling")
print("The engineered features can now be used for forecasting or other ML tasks")
```

### LightGBM with Intelligent Hyperparameter Tuning

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from freamon import LightGBMModel

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Add a categorical feature
X['category'] = pd.qcut(X['mean radius'], 4, labels=['A', 'B', 'C', 'D'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model with automatic hyperparameter tuning
model = LightGBMModel(
    problem_type='classification',
    metric='auc',
    tuning_trials=50,  # Number of hyperparameter trials
    random_state=42
)

# Fit the model with automatic hyperparameter tuning
model.fit(
    X_train, y_train,
    categorical_features=['category'],  # List categorical features
    validation_size=0.2,  # Create validation set from training data
    tune_hyperparameters=True  # Enable intelligent tuning
)

# Get feature importance
importance = model.get_feature_importance(method='native')
print("Top 5 features:", importance.head(5))

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")

# Save model for later use
model.save("breast_cancer_model.joblib")

# Load the saved model
loaded_model = LightGBMModel.load("breast_cancer_model.joblib")
```

### Pipeline Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    CrossValidationTrainingStep,
    EvaluationStep
)

# Load and split your data
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline steps
feature_step = FeatureEngineeringStep(name="feature_engineering")
feature_step.add_operation(
    method="add_polynomial_features",
    columns=["feature1", "feature2"],
    degree=2
)
feature_step.add_operation(
    method="add_binned_features",
    columns=["feature3"],
    n_bins=5
)

model_step = ModelTrainingStep(
    name="model",
    model_type="lightgbm",
    problem_type="classification",
    hyperparameters={"num_leaves": 31, "learning_rate": 0.05}
)

eval_step = EvaluationStep(
    name="evaluation",
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

# Create and fit pipeline
pipeline = Pipeline()
pipeline.add_step(feature_step)
pipeline.add_step(model_step)
pipeline.add_step(eval_step)
pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipeline.predict(X_test)
metrics = eval_step.evaluate(y_test, y_pred, model_step.predict_proba(X_test))
print(f"Evaluation metrics: {metrics}")

# Save pipeline for later use
pipeline.save("my_pipeline")
```

### Traditional Workflow

```python
import pandas as pd
from freamon.data_quality import DataQualityAnalyzer
from freamon.modeling import ModelTrainer
from freamon.model_selection import train_test_split
from freamon.utils import OneHotEncoderWrapper
from freamon.utils.dataframe_utils import detect_datetime_columns

# Load your data
df = pd.read_csv("your_data.csv")

# Automatically detect and convert datetime columns
df = detect_datetime_columns(df)

# Analyze data quality
analyzer = DataQualityAnalyzer(df)
analyzer.generate_report("data_quality_report.html")

# Handle missing values
from freamon.data_quality import handle_missing_values
df_clean = handle_missing_values(df, strategy="mean")

# Encode categorical features
encoder = OneHotEncoderWrapper()
df_encoded = encoder.fit_transform(df_clean)

# Split data
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

# Train a model
feature_cols = [col for col in train_df.columns if col != "target"]
trainer = ModelTrainer(
    model_type="lightgbm",
    model_name="LGBMClassifier",
    problem_type="classification",
)
metrics = trainer.train(
    train_df[feature_cols],
    train_df["target"],
    X_val=test_df[feature_cols],
    y_val=test_df["target"],
)

# View the results
print(f"Validation metrics: {metrics}")
```

## Version 0.3.16 Highlights

- **Enhanced EDA for Large Datasets:**
  - Smart data table rendering showing first/last rows with row count indicators
  - Lazy loading for images to improve performance with many visualizations
  - Adaptive sampling based on dataset size
  
- **Export to Jupyter Notebook:**
  - One-click export of EDA reports to Jupyter notebooks 
  - Client-side conversion without additional dependencies
  - Preserves all visualizations with code to reproduce them
  
- **Currency and Special Character Handling:**
  - Comprehensive fixes for matplotlib rendering of currency symbols
  - Improved LaTeX handling for special characters (_, $, ^, etc.)
  - Simple API to apply all patches with a single function call

## Module Overview

- **data_quality:** Tools for assessing and improving data quality
  - **drift:** Data drift detection and monitoring
  - **outliers:** Outlier detection and handling
  - **missing_values:** Missing value analysis and imputation
- **eda:** Exploratory data analysis tools
  - **analyzer:** Comprehensive EDA with HTML reporting and Jupyter export
  - **time_series:** Enhanced time series analysis, seasonality, stationarity, and forecasting
  - **report:** Interactive HTML reports with lazy loading for large datasets
- **features:** Feature engineering utilities
  - **engineer:** Standard feature transformations
  - **shapiq_engineer:** Automatic feature interaction detection
  - **time_series_engineer:** Automated time series feature generation
- **utils:** Utility functions for working with dataframes and encoders
  - **dataframe_utils:** Tools for different dataframe backends and date detection
  - **encoders:** Categorical variable encoding tools with cross-validation support
  - **text_utils:** Text processing utilities with fit/transform capability for TF-IDF and bag-of-words features
- **model_selection:** Methods for splitting data and cross-validation
  - **cross_validation:** Standard and time series cross-validation tools
  - **cv_trainer:** Cross-validated model training with ensemble methods
  - **splitter:** Train/test splitting with special modes for time series
- **modeling:** Model training, evaluation, and comparison
  - **model:** Base model class with consistent interface
  - **factory:** Model creation utilities for multiple libraries
  - **trainer:** Training and evaluation tools
  - **lightgbm:** High-level LightGBM interface with intelligent tuning
  - **tuning:** Hyperparameter optimization with parameter importance awareness
  - **importance:** Permutation-based feature importance
  - **calibration:** Probability calibration for classification models
  - **helpers:** Simplified model creation functions with sensible defaults
  - **visualization:** Tools for visualizing model performance, feature importance, and predictions
- **pipeline:** Integrated workflow system connecting feature engineering with model training
  - **pipeline:** Core Pipeline interface
  - **steps:** Reusable pipeline steps for different tasks
  - **visualization:** Pipeline visualization tools
  - **cross_validation:** Cross-validation training in pipelines

Check out the [ROADMAP.md](ROADMAP.md) file for information on planned features and development phases.

## Example Scripts

The package includes several example scripts to demonstrate its functionality:

- **text_time_series_regression_example.py** - Advanced time series modeling with LightGBM and visualization
- **lightgbm_simplified_example.py** - Simplified LightGBM model creation with helper functions
- **threshold_optimization_example.py** - Classification threshold optimization
- **shapiq_example.py** - Feature interaction detection with ShapIQ
- **pipeline_example.py** - Complete modeling pipeline
- **automated_end_to_end_pipeline.py** - Fully automated modeling workflow
- **drift_and_visualization_example.py** - Data drift detection and visualization
- **time_series_enhanced_example.py** - Advanced time series features
- **text_analytics_example.py** - Text processing and feature extraction
- **multivariate_analysis_example.py** - Multivariate feature exploration
- **mixed_date_formats_example.py** - Handling date columns with multiple formats
- **scientific_notation_example.py** - Detecting and visualizing scientific notation data
- **datatype_detector_example.py** - Optimized data type detection for large datasets
- **large_dataset_eda_example.py** - Enhanced EDA reporting for large datasets with lazy loading
- **jupyter_export_example.py** - Export EDA reports to Jupyter notebooks for interactive analysis
- **eda_performance_test.py** - Performance benchmarks for EDA reporting optimization

Run any example by navigating to the examples directory and executing:

```bash
python example_name.py
```

## Development

To contribute to freamon, install the development dependencies:

```bash
# Install development dependencies only
pip install -e ".[dev]"

# Install all dependencies (including dev tools)
pip install -e ".[full]"
```

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=freamon

# Run specific tests
pytest tests/test_datatype_detector_performance.py
```

## License

MIT License