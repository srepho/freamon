# LightGBM Intelligent Tuning in Freamon

This document provides a comprehensive overview of the LightGBM tuning capabilities in Freamon, which combines Optuna's powerful optimization framework with sophisticated parameter-importance-aware optimization.

## Overview

Freamon's LightGBM tuning system provides:

1. **Progressive Tuning**: Focuses first on the most impactful parameters
2. **Parameter Importance Analysis**: Identifies which parameters matter most
3. **Cross-Validation Integration**: Ensures robust parameter selection
4. **Early Stopping Support**: Prevents overfitting during tuning
5. **Visualization Tools**: Understand the optimization process
6. **Pipeline Integration**: Seamlessly integrate within ML workflows

## How It Works

### Progressive Tuning Process

1. **Phase 1**: Optimize core parameters first
   - num_leaves, max_depth, learning_rate, subsample, etc.
   - Uses a subset of the total trials budget
   
2. **Parameter Importance Analysis**
   - Analyzes which parameters had the most impact
   - Fixes unimportant parameters at their best values
   
3. **Phase 2**: Fine-tune with expanded parameter set
   - Includes regularization parameters and additional settings
   - Focuses on the most important parameters from Phase 1

### Key Components

- **LightGBMTuner**: Main class for hyperparameter optimization
- **HyperparameterTuningStep**: Pipeline integration of tuning functionality
- **AutoModelFlow**: High-level automated modeling with built-in tuning

## Using LightGBM Tuning

### Basic Usage

```python
from freamon.modeling.tuning import LightGBMTuner

# Initialize tuner
tuner = LightGBMTuner(
    problem_type='classification',  # or 'regression'
    metric='auc',                   # evaluation metric
    cv=5,                           # number of cross-validation folds
    cv_type='stratified',           # cross-validation strategy
    n_trials=100,                   # optimization budget
    random_state=42
)

# Run tuning
best_params = tuner.tune(
    X_train, y_train,
    categorical_features=['category_col'],
    progressive_tuning=True,        # enable progressive tuning
    early_stopping_rounds=50
)

# Create and train a model with the best parameters
model = tuner.create_model()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Pipeline Integration

```python
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    HyperparameterTuningStep,
    EvaluationStep
)

# Create pipeline with tuning step
pipeline = Pipeline()
pipeline.add_step(
    HyperparameterTuningStep(
        name="lgbm_tuning",
        model_type="lightgbm",
        problem_type="classification",
        metric="auc",
        n_trials=50,
        early_stopping_rounds=30,
        progressive_tuning=True
    )
)
pipeline.add_step(EvaluationStep(name="evaluation"))

# Fit pipeline
pipeline.fit(X_train, y_train)

# Get tuned model and evaluate
model = pipeline.get_step("lgbm_tuning").model
feature_importance = pipeline.get_step("lgbm_tuning").get_feature_importances()
```

### Automated Workflow

```python
from freamon.modeling import auto_model

# Full automated workflow with tuning
results = auto_model(
    df=df,
    target_column='target',
    problem_type='classification',
    model_type='lightgbm',
    tuning=True,
    tuning_options={
        'n_trials': 50,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50
    }
)

# Access results
model = results['model']
metrics = results['metrics']
feature_importance = results['feature_importance']
```

## Parameter Configuration

### Core Parameters (Phase 1)

```python
{
    'num_leaves': {
        'type': 'int',
        'low': 20,
        'high': 150,
        'log': True
    },
    'max_depth': {
        'type': 'int',
        'low': 3,
        'high': 12
    },
    'learning_rate': {
        'type': 'continuous',
        'low': 0.01,
        'high': 0.3,
        'log': True
    },
    'subsample': {
        'type': 'continuous',
        'low': 0.5,
        'high': 1.0
    },
    'colsample_bytree': {
        'type': 'continuous',
        'low': 0.5,
        'high': 1.0
    },
    'min_child_samples': {
        'type': 'int',
        'low': 5,
        'high': 100,
        'log': True
    }
}
```

### Additional Parameters (Phase 2)

```python
{
    'reg_alpha': {
        'type': 'continuous',
        'low': 1e-5,
        'high': 10.0,
        'log': True
    },
    'reg_lambda': {
        'type': 'continuous',
        'low': 1e-5,
        'high': 10.0,
        'log': True
    },
    'min_split_gain': {
        'type': 'continuous',
        'low': 0.0,
        'high': 1.0
    },
    'feature_fraction': {
        'type': 'continuous',
        'low': 0.5,
        'high': 1.0
    },
    'bagging_fraction': {
        'type': 'continuous',
        'low': 0.5,
        'high': 1.0
    },
    'bagging_freq': {
        'type': 'int',
        'low': 0,
        'high': 10
    },
    'min_data_in_leaf': {
        'type': 'int',
        'low': 10,
        'high': 200,
        'log': True
    }
}
```

## Advanced Features

### Parameter Importance Visualization

```python
# Get parameter importance
param_importance = tuner.param_importance

# Print top parameters by importance
for param, importance in sorted(param_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:5]:
    print(f"{param}: {importance:.4f}")

# Visualize with Optuna
fig = tuner.plot_param_importances()
fig.savefig('param_importance.png')
```

### Optimization History Visualization

```python
# Plot optimization progress
fig = tuner.plot_optimization_history()
fig.savefig('optimization_history.png')
```

### Cross-Validation Types

- **kfold**: Standard K-fold cross-validation (default)
- **stratified**: Stratified K-fold for classification problems
- **timeseries**: Time-based splits for temporal data

```python
# Time series cross-validation
tuner = LightGBMTuner(
    problem_type='regression',
    cv_type='timeseries',
    cv=5
)
```

### Categorical Feature Handling

LightGBM has special handling for categorical features, which is fully supported in the tuning process.

```python
best_params = tuner.tune(
    X_train, y_train,
    categorical_features=['category1', 'category2'],
    early_stopping_rounds=50
)
```

## Practical Tips

1. **Number of Trials**: Start with a smaller budget (30-50) for rapid testing, increase to 100+ for production models

2. **Progressive Tuning**: Almost always beneficial, especially for larger parameter spaces

3. **Early Stopping**: Always use to prevent overfitting during tuning

4. **Metric Choice**:
   - Classification: 'auc', 'binary_logloss', 'accuracy', 'f1'
   - Regression: 'rmse', 'mae', 'mape'

5. **Fixed Parameters**: If you already know good values for certain parameters, you can fix them:

```python
fixed_params = {
    'n_estimators': 200,
    'objective': 'binary',
    'verbosity': -1
}

best_params = tuner.tune(
    X_train, y_train,
    fixed_params=fixed_params
)
```

6. **Memory Considerations**: For large datasets, reduce the number of cross-validation folds or use a subset of the data for tuning

## Performance Comparison

In benchmark studies, this progressive tuning approach typically produces models that are:
- **More accurate** than default parameters or random search
- **More efficient** than standard Bayesian optimization (by focusing on important parameters)
- **More robust** due to the cross-validation integration

## Under the Hood

The tuning process leverages [Optuna](https://optuna.org/), enhancing it with:
1. Progressive parameter exploration
2. Parameter importance-based optimization
3. LightGBM-specific configuration handling
4. Cross-validation integration

This creates a more effective and efficient parameter search than standard random, grid, or basic Bayesian optimization methods.