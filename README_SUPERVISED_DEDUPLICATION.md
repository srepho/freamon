# Supervised Deduplication

Freamon now supports supervised learning approaches for detecting duplicate records in datasets.

## Overview

The supervised deduplication module combines machine learning with feature engineering to learn patterns of duplicate records from labeled examples. This approach is particularly effective when:

- You have labeled examples of duplicates
- You need to handle complex data with multiple fields
- Simple rule-based approaches miss too many duplicates
- You want to score records by probability of being duplicates

## Features

- Train models using labeled duplicate data
- Use date/time features with special handling
- Identify which features are most important for duplicate detection
- Score record pairs with a probability of being duplicates
- Apply trained models to new datasets

## Usage

### Basic Example

```python
from freamon.deduplication import SupervisedDeduplicationModel

# Initialize the model with key fields
model = SupervisedDeduplicationModel(
    model_type='lightgbm',  # Options: 'lightgbm', 'random_forest', 'gradient_boosting'
    date_features=['transaction_date', 'created_at'],
    key_features=['name', 'email', 'phone', 'address', 'amount']
)

# Train on labeled data
model.fit(
    df=your_dataframe,
    duplicate_pairs=list_of_duplicate_index_pairs,
    validation_fraction=0.2  # Optional validation split
)

# Find duplicates in new data
duplicates = model.find_duplicates(
    df=new_dataframe,
    threshold=0.7,  # Probability threshold
    return_probabilities=True  # Return scored pairs
)

# View feature importances
importances = model.get_feature_importances()
```

### Training with Labeled Data

To train the model, you need a dataset and a list of known duplicate pairs:

```python
# Your dataframe with records
df = pd.DataFrame({
    'name': ['John Smith', 'J. Smith', 'Jane Doe', 'J Doe'],
    'email': ['john@example.com', 'john@example.com', 'jane@example.com', 'jane@example.com'],
    'amount': [100.00, 100.00, 50.00, 50.50],
    'date': ['2023-01-01', '2023-01-02', '2023-01-15', '2023-01-15']
})

# List of index pairs that are known duplicates
duplicate_pairs = [(0, 1), (2, 3)]

# Train the model
model.fit(df, duplicate_pairs)
```

### Scoring New Record Pairs

Once trained, the model can score any pair of records for similarity:

```python
# Predict duplicate probability for all pairs
result = model.predict_duplicate_probability(df=new_df)

# View high-probability duplicates
high_prob_duplicates = result[result['duplicate_probability'] > 0.8]
print(high_prob_duplicates)
```

### Evaluating Model Performance

The model includes methods to evaluate detection accuracy:

```python
# Evaluate on test data with known duplicates
metrics = model.evaluate(
    df=test_df,
    true_duplicate_pairs=known_test_duplicates,
    threshold=0.7
)

print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
```

## Feature Engineering

The model automatically generates features that capture similarity between records:

- **Text fields**: Cosine and Levenshtein similarity metrics
- **Numeric fields**: Absolute differences and ratios
- **Date fields**: Time differences in days
- **Categorical fields**: Exact match indicators

## Advanced Options

### Customizing the Model

```python
# Custom model parameters
model_params = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 100
}

model = SupervisedDeduplicationModel(
    model_type='lightgbm',
    key_features=['name', 'email', 'phone'],
    model_params=model_params
)
```

### Finding an Optimal Threshold

```python
# Test multiple thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for threshold in thresholds:
    metrics = model.evaluate(
        df=test_df, 
        true_duplicate_pairs=known_test_duplicates,
        threshold=threshold
    )
    results.append({
        'threshold': threshold,
        **metrics
    })
    
# Find threshold with best F1 score
best_result = max(results, key=lambda x: x['f1'])
print(f"Best threshold: {best_result['threshold']}, F1: {best_result['f1']:.4f}")
```

## Performance Considerations

- For large datasets, use the `max_pairs` parameter to limit comparisons
- Use `return_features=True` to inspect generated features
- If memory is a concern, process data in batches

## Full Example

See `examples/supervised_deduplication_example.py` for a complete example with synthetic data generation, model training, evaluation, and visualization.