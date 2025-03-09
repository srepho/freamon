# Pipeline System

The Freamon pipeline system provides a streamlined way to integrate feature engineering, feature selection, model training, and evaluation into a single workflow. It allows you to create reproducible machine learning pipelines with minimal boilerplate code.

## Overview

The pipeline system consists of:

1. A core `Pipeline` class that manages the execution of steps
2. Multiple specialized `PipelineStep` implementations for different tasks
3. Persistence capabilities for saving and loading pipelines

## Basic Usage

Here's a simple example of how to use the pipeline system:

```python
from freamon.pipeline import (
    Pipeline, 
    FeatureEngineeringStep, 
    FeatureSelectionStep, 
    ModelTrainingStep
)

# Create pipeline steps
feature_step = FeatureEngineeringStep(name="feature_engineering")
feature_step.add_operation(
    method="add_polynomial_features",
    columns=["feature1", "feature2"],
    degree=2
)

selection_step = FeatureSelectionStep(
    name="feature_selection",
    method="model_based",
    n_features=20
)

model_step = ModelTrainingStep(
    name="model_training",
    model_type="lightgbm",
    problem_type="classification"
)

# Create and fit pipeline
pipeline = Pipeline()
pipeline.add_step(feature_step)
pipeline.add_step(selection_step)
pipeline.add_step(model_step)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
```

## Available Pipeline Steps

### FeatureEngineeringStep

Wraps the `FeatureEngineer` class to provide feature engineering capabilities.

```python
from freamon.pipeline import FeatureEngineeringStep

# Create step
feature_step = FeatureEngineeringStep(name="feature_engineering")

# Add operations
feature_step.add_operation(
    method="add_polynomial_features",
    columns=["feature1", "feature2"],
    degree=2,
    interaction_only=True
)

feature_step.add_operation(
    method="add_datetime_features",
    column="date_column",
    features=["month", "dayofweek", "hour"]
)
```

Available operations correspond to methods in the `FeatureEngineer` class:

- `add_polynomial_features`: Create polynomial and interaction features
- `add_datetime_features`: Extract features from datetime columns
- `add_binned_features`: Create bins from numerical features
- `add_target_encoding`: Apply target encoding to categorical features
- `add_text_features`: Extract features from text columns
- And more (see `FeatureEngineer` documentation)

### ShapIQFeatureEngineeringStep

Uses ShapIQ to automatically detect and generate interaction features.

```python
from freamon.pipeline import ShapIQFeatureEngineeringStep

# Create step
shapiq_step = ShapIQFeatureEngineeringStep(
    name="shapiq_interactions",
    model_type="lightgbm",
    n_interactions=10,
    max_interaction_size=2,
    categorical_features=["cat1", "cat2"]
)
```

### FeatureSelectionStep

Wraps the `FeatureSelector` class to provide feature selection capabilities.

```python
from freamon.pipeline import FeatureSelectionStep

# Create step
selection_step = FeatureSelectionStep(
    name="feature_selection",
    method="model_based",    # or "correlation", "variance", "mutual_info"
    n_features=20,
    features_to_keep=["important_feature1"]
)
```

### ModelTrainingStep

Wraps the `ModelTrainer` class to provide model training capabilities.

```python
from freamon.pipeline import ModelTrainingStep

# Create step
model_step = ModelTrainingStep(
    name="model_training",
    model_type="lightgbm",   # or "xgboost", "catboost", "sklearn_rf", etc.
    problem_type="classification",  # or "regression"
    eval_metric="auc",
    hyperparameters={
        "num_leaves": 31,
        "learning_rate": 0.05
    },
    cv_folds=5,  # Set to 0 to disable cross-validation
    early_stopping_rounds=50
)
```

### EvaluationStep

Evaluates model performance using various metrics.

```python
from freamon.pipeline import EvaluationStep

# Create step
eval_step = EvaluationStep(
    name="evaluation",
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    problem_type="classification"
)

# After fitting the pipeline and making predictions:
eval_results = eval_step.evaluate(y_test, y_pred, y_prob)
```

## Pipeline Features

### Step Outputs

You can access the output of any step in the pipeline after fitting:

```python
# Get output from a specific step
feature_eng_output = pipeline.get_step_output("feature_engineering")

# Get final output
final_output = pipeline.get_step_output("final_output")
```

### Saving and Loading Pipelines

Pipelines can be saved to disk and loaded later:

```python
# Save pipeline
pipeline.save("my_pipeline")

# Load pipeline
loaded_pipeline = Pipeline().load("my_pipeline")

# Use loaded pipeline
predictions = loaded_pipeline.predict(new_data)
```

### Feature Importances

You can get feature importances from the model step:

```python
importances = pipeline.get_feature_importances()
```

### Pipeline Summary

Get a summary of the pipeline and its steps:

```python
summary = pipeline.summary()
```

## Advanced Usage: Custom Pipeline Steps

You can create custom pipeline steps by inheriting from the `PipelineStep` abstract base class:

```python
from freamon.pipeline import PipelineStep

class MyCustomStep(PipelineStep):
    def __init__(self, name, param1, param2):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y=None, **kwargs):
        # Implement fit logic
        self._is_fitted = True
        return self
        
    def transform(self, X, **kwargs):
        # Implement transform logic
        return transformed_X
```

## Complete Example

See `examples/pipeline_example.py` for a complete example of using the pipeline system.