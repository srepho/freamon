# Classification Threshold Optimization

## Overview

In binary classification tasks, the default decision threshold (typically 0.5) is often suboptimal. Depending on the specific business context, you might want to prioritize different metrics:

- **Precision**: When false positives are more costly than false negatives
- **Recall**: When false negatives are more costly than false positives
- **F1 Score**: When you need a balance between precision and recall
- **Custom Business Metrics**: When you have domain-specific requirements

The Freamon library provides built-in utilities for finding optimal probability thresholds based on your preferred metric. This can be done either manually or automatically through the pipeline system.

## Basic Usage

```python
from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.metrics import find_optimal_threshold

# Train your model
model = LightGBMModel(problem_type='classification')
model.fit(X_train, y_train)

# Find the optimal threshold on validation data
threshold, score, results = model.find_optimal_threshold(
    X_val, y_val, metric='f1'
)
print(f"Optimal threshold: {threshold:.4f}, F1 score: {score:.4f}")

# Use the optimal threshold for predictions
y_pred = model.predict(X_test, threshold=threshold)

# Or set it as the model's default threshold
model.probability_threshold = threshold
y_pred = model.predict(X_test)  # Will now use the optimal threshold
```

## Available Metrics

You can optimize for different metrics depending on your requirements:

- `'f1'`: F1 score (harmonic mean of precision and recall)
- `'precision'`: Precision score (minimize false positives)
- `'recall'`: Recall score (minimize false negatives)
- `'accuracy'`: Overall accuracy
- `'balanced_accuracy'`: Balanced accuracy (useful for imbalanced datasets)
- `'f_beta'`: F-beta score (weighted F1 score with customizable beta)
- `'precision_recall_product'`: Product of precision and recall
- `'younden_j'`: Younden's J statistic (sensitivity + specificity - 1)
- `'kappa'`: Cohen's kappa score
- `'mcc'`: Matthews correlation coefficient

You can also provide a custom metric function that takes `(y_true, y_pred)` and returns a score to maximize.

## Visualizing Threshold Effects

```python
import matplotlib.pyplot as plt

# Find optimal thresholds for different metrics
metrics = ['f1', 'precision', 'recall', 'accuracy']
results_dict = {}

for metric in metrics:
    _, _, results = model.find_optimal_threshold(X_val, y_val, metric=metric)
    results_dict[metric] = results

# Plot threshold vs. metric score
plt.figure(figsize=(10, 6))
for metric, results in results_dict.items():
    plt.plot(results['threshold'], results['score'], label=f"{metric}")

plt.axvline(x=0.5, color='black', linestyle='--', label='Default (0.5)')
plt.xlabel('Probability Threshold')
plt.ylabel('Metric Score')
plt.title('Effect of Threshold on Different Metrics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Application Scenarios

### Healthcare Screening

In medical screening, false negatives (missing a condition) may be more costly than false positives (additional testing):

```python
# Optimize for high recall to minimize false negatives
threshold, score, _ = model.find_optimal_threshold(X_val, y_val, metric='recall')
model.probability_threshold = threshold
```

### Fraud Detection

In fraud detection, false positives (flagging legitimate transactions) may damage customer trust:

```python
# Optimize for high precision to minimize false positives
threshold, score, _ = model.find_optimal_threshold(X_val, y_val, metric='precision')
model.probability_threshold = threshold
```

### Custom Business Metric

For domain-specific requirements, you can define custom metrics:

```python
def custom_cost_function(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Example: FN costs 5x more than FP
    fp_cost = 1
    fn_cost = 5
    
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    # Return negative cost (since we're maximizing)
    return -total_cost

threshold, score, _ = model.find_optimal_threshold(
    X_val, y_val, metric=custom_cost_function
)
```

## Advanced Usage: Threshold Sweep

To explore many thresholds and their effects on multiple metrics:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Generate fine-grained thresholds
thresholds = np.linspace(0.01, 0.99, 99)
y_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class

# Evaluate metrics at each threshold
results = []
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    results.append({
        'threshold': threshold,
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_val, y_pred)
    })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)
```

## Automatic Threshold Optimization in Pipelines

You can enable automatic threshold optimization in the `HyperparameterTuningStep` of your pipeline:

```python
from freamon.pipeline.pipeline import Pipeline
from freamon.pipeline.steps import HyperparameterTuningStep

# Create hyperparameter tuning step with threshold optimization
tuning_step = HyperparameterTuningStep(
    name="model_tuning",
    model_type="lightgbm",
    problem_type="classification",
    metric="auc",  # Metric for hyperparameter tuning
    optimize_threshold=True,  # Enable threshold optimization
    threshold_metric="f1"  # Metric to optimize threshold for
)

# Create and fit pipeline
pipeline = Pipeline()
pipeline.add_step(tuning_step)
pipeline.fit(X_train, y_train, val_X=X_val, val_y=y_val)

# Make predictions (automatically uses optimized threshold)
y_pred = pipeline.predict(X_test)
```

### Available Metrics for Threshold Optimization

You can choose different metrics to optimize the threshold for:

- `"f1"`: Optimize threshold for F1 score (default)
- `"precision"`: Optimize for precision (minimize false positives)  
- `"recall"`: Optimize for recall (minimize false negatives)
- `"accuracy"`: Optimize for overall accuracy
- `"balanced_accuracy"`: Optimize for balanced accuracy
- Other options: `"f_beta"`, `"precision_recall_product"`, `"younden_j"`, `"kappa"`, `"mcc"`

### Accessing the Optimal Threshold

After fitting the pipeline, you can access the optimal threshold:

```python
tuning_step = pipeline.get_step("model_tuning")
optimal_threshold = tuning_step.optimal_threshold
print(f"Optimal threshold: {optimal_threshold:.4f}")
```

## Best Practices

1. **Use validation data**, not training data, to find the optimal threshold
2. **Monitor multiple metrics** to understand the tradeoffs
3. **Consider business costs** of false positives vs. false negatives
4. **Reassess thresholds** whenever the model is retrained or when data distributions change
5. **Perform threshold optimization after model tuning** to avoid overfitting
6. **Choose the threshold metric** based on your business requirements