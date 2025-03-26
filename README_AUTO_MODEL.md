# Auto Model Functionality Guide

This guide provides a walkthrough of Freamon's `auto_model` functionality, which offers a simplified interface for automated machine learning workflows.

## Overview

The `auto_model` function in Freamon provides:

- Automated model training and evaluation
- Support for text data with topic modeling
- Time series feature engineering
- Hyperparameter tuning
- Visualization of results
- Classification and regression support

## Setup in Conda Environment

```bash
# Create a new conda environment
conda create -n freamon-test python=3.10
conda activate freamon-test

# Install freamon from the local directory in development mode
pip install -e .

# Install required dependencies
pip install matplotlib scikit-learn pandas numpy lightgbm
```

## Testing Results

When testing the `auto_model` functionality in a conda environment, we confirmed:

- Basic classification functionality works correctly
- Model metrics calculation works (accuracy, precision, recall, etc.)
- Test set evaluation works with proper train/test splitting
- Feature importance calculation works
- Matplotlib plotting works correctly for visualizations

## Basic Usage

### Classification Example

```python
from freamon.modeling.autoflow import auto_model
import pandas as pd

# Prepare your data
df = pd.read_csv("your_data.csv")

# Run auto_model for classification
results = auto_model(
    df=df,
    target_column='target',
    problem_type='classification',
    model_type='lgbm_classifier',
    metrics=['accuracy', 'precision', 'recall', 'f1'],
    tuning=False,  # Set to True to enable hyperparameter tuning
    random_state=42
)

# Access results
model = results['model']  # The trained model
metrics = results['test_metrics']  # Evaluation metrics
importance = results['feature_importance']  # Feature importance

# Plot feature importance manually using the feature importance DataFrame
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sorted_importance = results['feature_importance'].sort_values(by='importance', ascending=False).head(15)
plt.barh(sorted_importance['feature'], sorted_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("feature_importance.png")
```

### Regression with Time Series Features

```python
# Prepare time series data
df = pd.read_csv("time_series_data.csv")
df['date'] = pd.to_datetime(df['date'])  # Ensure date column is datetime type

# Run auto_model with time series options
results = auto_model(
    df=df,
    target_column='target',
    date_column='date',  # Specify date column for time features
    problem_type='regression',
    model_type='lgbm_regressor',
    metrics=['rmse', 'mae', 'r2'],
    time_options={
        'create_target_lags': True,
        'lag_periods': [1, 7, 30],  # Daily, weekly, monthly lags
        'rolling_windows': [7, 30]  # Rolling stats windows
    }
)

# Plot time series predictions
plt.figure(figsize=(12, 6))
test_df = results['test_df']
results['autoflow'].plot_predictions_over_time(test_df)
plt.savefig("time_series_predictions.png")
```

## Advanced Features

### Text Processing

```python
# With text data
results = auto_model(
    df=df,
    target_column='target',
    text_columns=['description', 'comments'],  # Specify text columns
    problem_type='classification',
    model_type='lightgbm'
)

# Access topic models
text_topics = results['text_topics']
```

### Hyperparameter Tuning Options

```python
results = auto_model(
    df=df,
    target_column='target',
    problem_type='classification',
    tuning=True,
    tuning_options={
        'n_trials': 20,  # Number of tuning trials (lower is faster)
        'early_stopping_rounds': 10,  # Stop if no improvement
        'timeout': 300  # Max seconds for tuning
    }
)
```

### Cross-Validation Configuration

```python
results = auto_model(
    df=df,
    target_column='target',
    problem_type='regression',
    cv_folds=5,  # 5-fold cross-validation
    metrics=['rmse', 'mae', 'r2']  # Metrics to evaluate
)
```

## Simple Testing Script

Here's a minimal script to test `auto_model` functionality:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from freamon.modeling.autoflow import auto_model

# Create synthetic dataset
X, y = make_classification(
    n_samples=100,
    n_features=5,
    n_informative=3,
    random_state=42
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['target'] = y

# Run auto_model
results = auto_model(
    df=df,
    target_column='target',
    problem_type='classification',
    model_type='lgbm_classifier',
    metrics=['accuracy'],
    tuning=False,  # Disable tuning for speed
    cv_folds=2,    # Minimal CV
    random_state=42
)

# Print test metrics
print("Test Metrics:")
for metric, value in results['test_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Plot feature importance
plt.figure(figsize=(8, 5))
sorted_importance = results['feature_importance'].sort_values(by='importance', ascending=False)
plt.barh(sorted_importance['feature'], sorted_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("classification_feature_importance.png")
```

## Key Parameters Reference

| Parameter | Description |
|-----------|-------------|
| `df` | The input DataFrame |
| `target_column` | Column name of the target variable |
| `problem_type` | 'classification' or 'regression' |
| `model_type` | Model type (e.g., 'lightgbm', 'lgbm_classifier') |
| `text_columns` | List of text columns for topic modeling |
| `date_column` | Date column for time series features |
| `tuning` | Boolean to enable hyperparameter tuning |
| `tuning_options` | Dict of tuning parameters |
| `time_options` | Dict of time series feature options |
| `cv_folds` | Number of cross-validation folds |
| `metrics` | List of metrics to calculate |
| `test_size` | Proportion of data for testing |
| `auto_split` | Boolean to enable automatic train/test split |
| `random_state` | Random seed for reproducibility |
| `verbose` | Boolean to enable verbose output |

## Return Values

The `auto_model` function returns a dictionary with:

- `model`: The trained model
- `autoflow`: The AutoModelFlow instance
- `metrics`: Cross-validation metrics
- `test_metrics`: Test set metrics
- `feature_importance`: DataFrame of feature importance
- `text_topics`: Topic models for text columns (if text columns provided)
- `test_df`: Test DataFrame with predictions
- `X_train`, `X_test`, `y_train`, `y_test`: Training and test data splits

## Using the Trained Model for Predictions

After training your model with `auto_model`, you can use it to make predictions on new data. There are two ways to do this:

### Method 1: Using the Model Directly

```python
# Get the trained model from results
model = results['model']

# Prepare new data (must have the same columns as training data, except target)
new_data = pd.read_csv("new_data.csv")

# Make predictions
predictions = model.predict(new_data)
print("Predictions:", predictions[:5])

# For classification, you can also get probabilities
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(new_data)
    print("Probabilities for first 5 samples:")
    print(probabilities[:5])
```

### Method 2: Using the AutoModelFlow Instance

The `autoflow` object handles all the preprocessing steps automatically, making it safer when dealing with text or time series data:

```python
# Get the AutoModelFlow instance
autoflow = results['autoflow']

# Prepare new data (must have the same columns as training data, including text/date columns)
new_data = pd.read_csv("new_data.csv")

# For time series data, ensure date column is datetime
if 'date' in new_data.columns:
    new_data['date'] = pd.to_datetime(new_data['date'])

# Make predictions - this will handle text processing and feature engineering automatically
predictions = autoflow.predict(new_data)
print("Predictions:", predictions[:5])

# For classification, get probabilities
if autoflow.problem_type == 'classification':
    probabilities = autoflow.predict_proba(new_data)
    print("Probabilities for first 5 samples:")
    print(probabilities[:5])
```

### Saving and Loading the Model

To save the model for later use:

```python
import pickle

# Save the entire AutoModelFlow object (recommended)
with open('autoflow_model.pkl', 'wb') as f:
    pickle.dump(results['autoflow'], f)

# Later, load the model
with open('autoflow_model.pkl', 'rb') as f:
    loaded_autoflow = pickle.load(f)

# Use the loaded model
predictions = loaded_autoflow.predict(new_data)
```

## Troubleshooting

- If you see LightGBM warnings about "No further splits with positive gain", these are generally harmless and indicate that the model has reached its optimal split point.
- For text processing, ensure you have sufficient text data for good topic modeling.
- When using time series features, ensure your date column is properly formatted as datetime.
- If you're getting memory errors with large datasets, try reducing the `cv_folds` or setting `tuning=False`.
- For better performance, adjust the `tuning_options` parameters to lower values (fewer trials, shorter timeout).