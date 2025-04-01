# Automatic Train-Test Splitting in AutoModelFlow

The `auto_model` function in Freamon now supports automatic train-test splitting, making it easier to build and evaluate machine learning models with minimal code.

## Key Features

- **Automatic Train-Test Splitting**: Eliminates the need for manual data splitting
- **Intelligent Splitting Strategy**: Uses appropriate splitting method based on the problem type
  - Stratified splitting for classification problems
  - Random splitting for regression problems
  - Time-based splitting for time series data
- **Test Set Metrics**: Returns comprehensive test performance metrics
- **Streamlined Workflow**: End-to-end process from raw data to predictions and evaluation

## Usage Examples

### Basic Classification

```python
from freamon.modeling.autoflow import auto_model

# Run automatic modeling with train-test split
results = auto_model(
    df=data_df,                   # Full dataset (will be split automatically)
    target_column='target',
    problem_type='classification',
    auto_split=True,              # Enable automatic splitting
    test_size=0.2                 # 20% for testing
)

# Access results
model = results['model']
test_metrics = results['test_metrics']
test_df = results['test_df']

# Check performance
print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test AUC: {test_metrics['roc_auc']:.4f}")

# Make predictions on new data
predictions = model.predict(new_data)
```

### Time Series Forecasting

```python
from freamon.modeling.autoflow import auto_model

# Time-based train-test split
results = auto_model(
    df=time_series_df,
    target_column='target',
    date_column='date',           # Date column triggers time-based splitting
    problem_type='regression',
    auto_split=True,
    test_size=0.2,                # Last 20% of timeline used for testing
    time_options={
        'create_target_lags': True,
        'lag_periods': [1, 7, 14, 28],
        'rolling_windows': [7, 14, 30]
    }
)

# Access test results
test_metrics = results['test_metrics']
test_df = results['test_df']

# Check performance
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.4f}")

# Visualize predictions
results['autoflow'].plot_predictions_over_time(test_df)
```

## How It Works

When `auto_split=True`, the function:

1. **Analyzes the dataset** to understand its structure and characteristics
2. **Chooses an appropriate splitting strategy**:
   - For classification: Stratified sampling to maintain class distribution
   - For regression: Random sampling
   - For time series (when date_column is provided): Chronological splitting
3. **Creates train and test sets** with the specified test_size
4. **Trains the model** on the training set with cross-validation
5. **Evaluates on the test set** and returns detailed metrics
6. **Returns both model and test data** for further analysis

## Benefits

- **Proper Evaluation**: Get accurate estimates of model performance on unseen data
- **Reduced Code**: No need to manually implement train-test splitting logic
- **No Data Leakage**: Especially important for time series data, where proper temporal splitting is crucial
- **Standardized Metrics**: Consistent reporting across different modeling tasks

## Parameters

- `auto_split`: Enable/disable automatic splitting (default: True)  
- `test_size`: Proportion of data to use for test set (default: 0.2)

See the full documentation for more details on available parameters and advanced options.

## Text Preprocessing and Topic Modeling

### Advanced Text Options

The `auto_model` function provides comprehensive text processing capabilities through the `text_options` parameter:

```python
text_options = {
    # Preprocessing options
    'preprocessing': {
        'remove_stopwords': True,      # Remove common words like "the", "and", etc.
        'lemmatize': True,             # Convert words to base form (e.g., "running" -> "run")
        'remove_punctuation': True,    # Remove punctuation marks
        'min_token_length': 3,         # Ignore words shorter than 3 characters
        'custom_stopwords': ['said', 'like'],  # Additional words to remove
        'anonymize': False             # Whether to anonymize PII
    },
    
    # Topic modeling options
    'topic_modeling': {
        'method': 'nmf',               # 'nmf' or 'lda'
        'auto_topics_range': (2, 10),  # Try between 2-10 topics
        'auto_topics_method': 'coherence',  # Selection method
        'sampling_ratio': 0.8,         # % of data to use for topic modeling
        'max_sample_size': 5000,       # Cap sample size for large datasets
        'reset_index': True,           # Fix index mismatch issues
        'deduplicate': True            # Remove duplicate texts
    },
}
```

### Avoiding Index Errors

When processing very large datasets with text, you might encounter a "Length mismatch" error. This typically happens when there's an index alignment issue between the sampled data used for topic modeling and the full dataset. To fix this:

1. Set `reset_index=True` in the topic_modeling options
2. Reduce the sample size with `max_sample_size` and/or `sampling_ratio`
3. Set `deduplicate=True` to avoid duplicate texts in the topic model

### Using the Topic Model Directly

If you want to use the optimized topic modeling function directly without the full `auto_model`:

```python
from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor

# Create topic model
result = create_topic_model_optimized(
    df=text_df,
    text_column='text',
    n_topics='auto',  # Automatically determine optimal number
    method='nmf',
    reset_index=True,
    max_sample_size=10000
)

# Access components
topic_model = result['topic_model']
topics = result['topics']
doc_topics = result['document_topics']

# Display topics
for topic_id, words in topics:
    print(f"Topic {topic_id+1}: {', '.join(words[:10])}")
```

## Example Files

For detailed examples of using the automatic train-test splitting and text processing options, see:
- `examples/auto_split_example.py` - Basic automatic train-test splitting
- `examples/text_preprocessing_example.py` - Advanced text processing options