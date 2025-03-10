# Cross-Validation Training

Freamon provides comprehensive cross-validation capabilities through the `CrossValidatedTrainer` class and `CrossValidationTrainingStep` pipeline step. These components make cross-validation the standard mechanism for model training in your machine learning workflows.

## Why Use Cross-Validation?

Cross-validation offers several advantages over simple train-test splits:

1. **Better model evaluation**: Cross-validation provides a more robust estimate of model performance by using multiple train-test splits.
2. **Reduced variance**: Averaging results across folds reduces the variance in performance estimates.
3. **More efficient use of data**: All data points are used for both training and validation, making it particularly valuable for smaller datasets.
4. **Detection of overfitting**: Cross-validation helps identify models that perform well on training data but generalize poorly.
5. **Ensemble opportunities**: Models trained on different folds can be combined into a powerful ensemble.

## Available Cross-Validation Strategies

Freamon supports multiple cross-validation strategies:

1. **K-Fold Cross-Validation**: Splits the data into k equal-sized folds, using each fold once as validation while training on the others.
2. **Stratified K-Fold**: Maintains the same class distribution in each fold, important for imbalanced classification problems.
3. **Time Series Cross-Validation**: Respects temporal order for time series data, training on past data and validating on future data.
4. **Walk-Forward Validation**: A sliding window approach for time series, particularly useful for financial time series.

## Using the CrossValidatedTrainer

The `CrossValidatedTrainer` class provides a simple interface for training models with cross-validation:

```python
from freamon.model_selection import CrossValidatedTrainer

# Initialize trainer
trainer = CrossValidatedTrainer(
    model_type="lightgbm",
    problem_type="classification",
    cv_strategy="stratified",
    n_splits=5,
    ensemble_method="weighted",
    eval_metric="accuracy",
    random_state=42
)

# Fit with cross-validation
trainer.fit(X_train, y_train)

# Make predictions
y_pred = trainer.predict(X_test)

# For classification, get probabilities
y_prob = trainer.predict_proba(X_test)

# Get cross-validation results
cv_results = trainer.get_cv_results()

# Get feature importances
importances = trainer.get_feature_importances()
```

### Configuration Options

The `CrossValidatedTrainer` offers extensive configuration options:

```python
trainer = CrossValidatedTrainer(
    # Model configuration
    model_type="lightgbm",              # Model type (lightgbm, xgboost, catboost, sklearn)
    problem_type="classification",      # Problem type (classification or regression)
    hyperparameters={                   # Model hyperparameters
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    },
    
    # Cross-validation configuration
    cv_strategy="kfold",                # CV strategy (kfold, stratified, timeseries, walk_forward)
    n_splits=5,                         # Number of folds
    eval_metric="accuracy",             # Metric to optimize
    
    # Time series specific parameters (for timeseries and walk_forward strategies)
    date_column="date",                 # Date column for time series
    gap="1d",                           # Gap between train and validation periods
    expanding_window=True,              # Whether to use expanding window
    
    # Ensemble options
    ensemble_method="weighted",         # Method for combining models (best, average, weighted, stacking)
    
    # Other options
    early_stopping_rounds=10,           # Early stopping rounds
    random_state=42                     # Random seed
)
```

## Ensemble Methods

The `CrossValidatedTrainer` supports multiple ensemble methods for combining models trained on different folds:

1. **Best**: Trains a new model on the full dataset using the best hyperparameters found during cross-validation.
2. **Average**: Creates an ensemble by averaging predictions from models trained on each fold.
3. **Weighted**: Similar to average, but weights each model's contribution based on its validation performance.
4. **Stacking**: Trains a meta-model that learns how to best combine predictions from the fold models.

Example comparing different ensemble methods:

```python
# Compare different ensemble methods
methods = ["best", "average", "weighted", "stacking"]
results = {}

for method in methods:
    trainer = CrossValidatedTrainer(
        model_type="lightgbm",
        cv_strategy="kfold",
        ensemble_method=method,
        n_splits=5
    )
    
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    
    # Calculate metrics
    results[method] = calculate_metrics(y_test, y_pred)
    
    print(f"{method} - Accuracy: {results[method]['accuracy']:.4f}")
```

## Pipeline Integration

The `CrossValidationTrainingStep` allows you to use cross-validation within Freamon's pipeline system:

```python
from freamon.pipeline import Pipeline
from freamon.pipeline import CrossValidationTrainingStep

# Create pipeline
pipeline = Pipeline()

# Add data preparation steps
pipeline.add_step(DataFrameStep(name="data_preparation"))

# Add feature selection
pipeline.add_step(FeatureSelectionStep(name="feature_selection", method="importance"))

# Add cross-validation training step
pipeline.add_step(
    CrossValidationTrainingStep(
        name="model_training",
        model_type="lightgbm",
        problem_type="classification",
        cv_strategy="stratified",
        n_splits=5,
        ensemble_method="weighted"
    )
)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Get cross-validation results
cv_step = pipeline.get_step("model_training")
cv_results = cv_step.get_cv_results()
```

## Time Series Cross-Validation

For time series data, Freamon provides specialized cross-validation strategies that respect temporal order:

```python
# Time series cross-validation
trainer = CrossValidatedTrainer(
    model_type="lightgbm",
    problem_type="regression",
    cv_strategy="timeseries",
    n_splits=5,
    date_column="date",          # Column containing dates
    gap="7d",                    # 7-day gap between train and validation
    expanding_window=True,       # Use expanding window
    ensemble_method="best"
)

# Make sure your DataFrame has a date column
X_train['date'] = pd.to_datetime(X_train['date'])

# Fit with time series cross-validation
trainer.fit(X_train, y_train)
```

## Example: Complete Cross-Validation Workflow

Here's a complete example that demonstrates a cross-validation workflow:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from freamon.model_selection import CrossValidatedTrainer
from freamon.modeling.metrics import calculate_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train with cross-validation
trainer = CrossValidatedTrainer(
    model_type="lightgbm",
    problem_type="classification",
    cv_strategy="stratified",
    n_splits=5,
    ensemble_method="weighted",
    eval_metric="accuracy",
    random_state=42
)

trainer.fit(X_train, y_train)

# Make predictions
y_pred = trainer.predict(X_test)
y_prob = trainer.predict_proba(X_test)

# Calculate metrics
metrics = calculate_metrics(y_test, y_pred, y_prob=y_prob, problem_type="classification")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC AUC: {metrics['roc_auc']:.4f}")

# Get CV results
cv_results = trainer.get_cv_results()
print(f"CV Accuracy: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")

# Get feature importances
importances = trainer.get_feature_importances()
print("Feature importances:")
for feature, importance in importances.items():
    print(f"  {feature}: {importance:.4f}")

# Plot CV results
plt.figure(figsize=(10, 6))
plt.bar(range(len(cv_results['accuracy'])), cv_results['accuracy'])
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Results')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## Best Practices

1. **Choose the right CV strategy**: Use stratified CV for imbalanced classification, time series CV for temporal data.
2. **Balance fold count**: 5-10 folds typically work well. More folds give better estimates but require more computation.
3. **Ensemble methods**: "Weighted" and "stacking" often outperform "best" but take longer to train and predict.
4. **Feature importance**: Cross-validated feature importance is more reliable than from a single model.
5. **Nested CV**: For hyperparameter tuning, consider nested cross-validation (using `HyperparameterTuningStep` inside each fold).

By making cross-validation the standard approach for model training, you can build more robust models that generalize better to unseen data.