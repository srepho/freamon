# Automated Modeling Flow

Freamon's `AutoModelFlow` provides a high-level interface for automated machine learning workflows that handle text data, time series features, and model training with minimal code.

## Overview

The automated modeling flow handles:

1. **Automatic Dataset Analysis**: Detecting column types (text, categorical, date, numeric)
2. **Text Processing**: Topic modeling, keyword extraction, and text feature generation
3. **Time Series Features**: Automatic creation of lag features, rolling statistics, and date-based features
4. **Model Training**: Cross-validation with appropriate methods for time series or standard data
5. **Feature Importance**: Analysis and visualization of feature importance
6. **Hyperparameter Tuning**: Intelligent parameter optimization

## Simple Usage

The simplest way to use this functionality is through the `auto_model` function:

```python
from freamon import auto_model

# Perform end-to-end modeling
results = auto_model(
    df=train_df,                        # Input DataFrame
    target_column='target',             # Target variable
    date_column='date',                 # Optional date column for time series
    model_type='lightgbm',              # Model type to use
    problem_type='classification',      # 'classification' or 'regression'
    text_columns=['description'],       # Optional, will be auto-detected if not specified
    categorical_columns=['category'],   # Optional, will be auto-detected if not specified
    cv_folds=5,                         # Number of cross-validation folds
    metrics=['accuracy', 'f1', 'auc'],  # Metrics to compute
    tuning=True                         # Whether to perform hyperparameter tuning
)

# Access results
model = results['model']                # Trained model
metrics = results['metrics']            # Cross-validation metrics
feature_importance = results['feature_importance']  # Feature importance
text_topics = results['text_topics']    # Topic models for text columns
autoflow = results['autoflow']          # Full AutoModelFlow instance

# Make predictions
predictions = autoflow.predict(test_df)
```

## Advanced Usage with AutoModelFlow

For more fine-grained control, you can use the `AutoModelFlow` class directly:

```python
from freamon.modeling import AutoModelFlow

# Initialize
model_flow = AutoModelFlow(
    model_type="lightgbm",
    problem_type="classification",
    text_processing=True,
    time_series_features=True,
    feature_selection=True,
    hyperparameter_tuning=True,
    random_state=42
)

# Analyze dataset and identify column types
dataset_info = model_flow.analyze_dataset(
    df=df,
    target_column='target',
    date_column='date'
)

# Fit model with custom options
model_flow.fit(
    df=df,
    target_column='target',
    date_column='date',
    text_columns=['description', 'comments'],
    categorical_columns=['category', 'region'],
    cv_folds=5,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
    text_processing_options={
        'topic_modeling': {
            'method': 'nmf',
            'auto_topics_range': (2, 15),
            'auto_topics_method': 'stability'
        },
        'text_features': {
            'extract_features': True,
            'include_stats': True,
            'include_readability': True,
            'include_sentiment': True,
            'extract_keywords': True
        }
    },
    time_series_options={
        'create_target_lags': True,
        'lag_periods': [1, 7, 14, 30],
        'rolling_windows': [7, 14, 30],
        'include_numeric_columns': True
    },
    tuning_options={
        'n_trials': 100,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }
)

# Make predictions
predictions = model_flow.predict(test_df)
probabilities = model_flow.predict_proba(test_df)  # For classification

# Visualize results
model_flow.plot_metrics()  # Cross-validation metrics
model_flow.plot_importance(top_n=20)  # Feature importance

# For time series data, plot predictions over time
model_flow.plot_predictions_over_time(test_df)

# Access topic model information
topic_terms = model_flow.get_topic_terms('description', n_terms=10)
document_topics = model_flow.get_document_topics(test_df, 'description')
```

## Configuration Options

### Text Processing Options

```python
text_processing_options = {
    # Topic modeling options
    'topic_modeling': {
        'method': 'nmf',            # 'nmf' or 'lda'
        'auto_topics_range': (2, 15),  # Range of topics to try
        'auto_topics_method': 'coherence'  # 'coherence' or 'stability'
    },
    
    # Text feature extraction options
    'text_features': {
        'extract_features': True,       # Whether to extract additional features
        'include_stats': True,          # Text statistics features
        'include_readability': True,    # Readability metrics
        'include_sentiment': True,      # Sentiment analysis
        'extract_keywords': True        # Extract and use keywords as features
    }
}
```

### Time Series Options

```python
time_series_options = {
    'create_target_lags': True,     # Whether to create target lag features
    'lag_periods': [1, 7, 14, 30],  # Periods for lag features
    'rolling_windows': [7, 14, 30], # Window sizes for rolling statistics
    'include_numeric_columns': True, # Create features for numeric columns
    'numeric_lags': [1, 7],         # Lags for numeric columns
    'numeric_rolling_windows': [7, 14] # Rolling windows for numeric columns
}
```

### Hyperparameter Tuning Options

```python
tuning_options = {
    'n_trials': 100,               # Number of tuning trials
    'timeout': None,               # Timeout in seconds (None for no limit)
    'eval_metric': 'auc',          # Metric to optimize 
    'early_stopping_rounds': 50,   # Early stopping parameter
    'use_optuna': True             # Whether to use Optuna for tuning
}
```

## Best Practices

1. **Dataset Size**: For large datasets, consider sampling in your initial explorations.

2. **Text Columns**: Explicitly specifying text columns can be helpful for better control, especially for ambiguous columns.

3. **Time Series**: For time series data, make sure your date column is properly formatted as datetime.

4. **Cross-Validation**: For time series, the function automatically uses time series cross-validation; for standard data, it uses stratified or k-fold CV.

5. **Hyperparameter Tuning**: Enable tuning for better results but be aware it increases computation time.

6. **Model Interpretation**: Always examine feature importance to understand what drives your model's predictions.

## Example Use Cases

### Text Classification

Automatically process text data and create a classification model:

```python
from freamon import auto_model
from sklearn.model_selection import train_test_split

# Prepare data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train model
results = auto_model(
    df=train_df,
    target_column='category',
    text_columns=['text_content'],
    problem_type='classification',
    model_type='lightgbm'
)

# Evaluate on test set
predictions = results['autoflow'].predict(test_df)
from sklearn.metrics import classification_report
print(classification_report(test_df['category'], predictions))

# Get topic insights
topics = results['text_topics']['text_content']['topics']
for topic_id, terms in topics:
    print(f"Topic {topic_id+1}: {', '.join(terms[:10])}")
```

### Time Series Forecasting

Create a forecasting model with automatic feature generation:

```python
from freamon import auto_model

# Train model with time series data
results = auto_model(
    df=train_df,
    target_column='sales',
    date_column='date',
    problem_type='regression',
    model_type='lightgbm',
    time_options={
        'lag_periods': [1, 7, 14, 28],
        'rolling_windows': [7, 14, 28, 90]
    }
)

# Forecast future values
future_predictions = results['autoflow'].predict(future_df)

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
plt.plot(train_df['date'], train_df['sales'], label='Historical')
plt.plot(future_df['date'], future_predictions, label='Forecast')
plt.legend()
plt.show()
```