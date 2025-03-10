# Feature Selection

Freamon provides a comprehensive set of feature selection methods to help you identify the most relevant features for your machine learning models. This module includes basic statistical methods, advanced wrapper methods, and specialized approaches for time series data.

## Overview

Feature selection is a critical step in the machine learning pipeline that offers several benefits:

- **Improved Model Performance**: By removing irrelevant features, you can reduce overfitting and improve generalization.
- **Faster Training**: Fewer features mean faster training times and lower computational requirements.
- **Better Interpretability**: Models with fewer features are typically easier to understand and explain.
- **Reduced Storage Requirements**: Smaller datasets require less storage space.

## Basic Feature Selection Methods

### Correlation-based Selection

Select features based on their correlation with the target variable:

```python
from freamon.features import select_by_correlation

# Select features with correlation above 0.3
selected_features = select_by_correlation(
    df, target='target', threshold=0.3, return_names_only=True
)
```

### Importance-based Selection

Select features based on importance scores from a model:

```python
from freamon.features import select_by_importance

# Select top 10 features by importance
selected_features = select_by_importance(
    df, target='target', k=10, return_names_only=True
)
```

### Variance-based Selection

Select features based on their variance:

```python
from freamon.features import select_by_variance

# Remove features with variance below threshold
selected_features = select_by_variance(
    df, threshold=0.01, return_names_only=True
)
```

### Mutual Information-based Selection

Select features based on mutual information with the target:

```python
from freamon.features import select_by_mutual_info

# Select top 15 features by mutual information
selected_features = select_by_mutual_info(
    df, target='target', k=15, return_names_only=True
)
```

### K-Best Selection

Select the k best features based on statistical tests:

```python
from freamon.features import select_by_kbest

# Select top 10 features
selected_features = select_by_kbest(
    df, target='target', k=10, return_names_only=True
)
```

### Percentile Selection

Select the top percentile of features:

```python
from freamon.features import select_by_percentile

# Select top 20% of features
selected_features = select_by_percentile(
    df, target='target', percentile=20, return_names_only=True
)
```

## Advanced Feature Selection Methods

### Recursive Feature Elimination with Cross-Validation (RFECV)

RFECV uses a model's feature importance to recursively remove features while using cross-validation to determine the optimal number of features:

```python
from freamon.features import select_features_rfecv
from sklearn.ensemble import RandomForestRegressor

# Select features using RFECV
selected_features = select_features_rfecv(
    df, 
    target='target',
    estimator=RandomForestRegressor(n_estimators=100),
    cv=5,
    return_names_only=True
)
```

You can also use the `RecursiveFeatureEliminationCV` class directly for more control:

```python
from freamon.features import RecursiveFeatureEliminationCV
from sklearn.ensemble import RandomForestRegressor

# Create RFECV instance
rfecv = RecursiveFeatureEliminationCV(
    estimator=RandomForestRegressor(n_estimators=100),
    cv=5
)

# Fit RFECV
rfecv.fit(X, y)

# Get selected features
selected_features = rfecv.selected_features_

# Plot cross-validation results
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
rfecv.plot_cv_results(ax=ax)
plt.show()
```

### Stability Selection

Stability selection addresses the instability of feature selection by running selection algorithms on different subsamples of the data:

```python
from freamon.features import select_features_stability

# Select features using stability selection
selected_features = select_features_stability(
    df,
    target='target',
    threshold=0.6,
    n_subsamples=100,
    return_names_only=True
)
```

You can also use the `StabilitySelector` class directly:

```python
from freamon.features import StabilitySelector
from sklearn.ensemble import RandomForestRegressor

# Create stability selector
stability_selector = StabilitySelector(
    estimator=RandomForestRegressor(n_estimators=100),
    threshold=0.6,
    n_subsamples=100
)

# Fit stability selector
stability_selector.fit(X, y)

# Get selected features
selected_features = stability_selector.selected_features_

# Plot stability path
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
stability_selector.plot_stability_path(ax=ax)
plt.show()
```

### Genetic Algorithm-based Selection

Genetic algorithms evolve a population of feature subsets to find the optimal subset:

```python
from freamon.features import select_features_genetic

# Select features using genetic algorithm
selected_features = select_features_genetic(
    df,
    target='target',
    n_features_to_select=10,
    population_size=50,
    n_generations=40,
    return_names_only=True
)
```

You can also use the `GeneticFeatureSelector` class directly:

```python
from freamon.features import GeneticFeatureSelector
from sklearn.ensemble import RandomForestRegressor

# Create genetic selector
genetic_selector = GeneticFeatureSelector(
    estimator=RandomForestRegressor(n_estimators=100),
    n_features_to_select=10,
    population_size=50,
    n_generations=40
)

# Fit genetic selector
genetic_selector.fit(X, y)

# Get selected features
selected_features = genetic_selector.selected_features_

# Plot fitness evolution
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
genetic_selector.plot_fitness_evolution(ax=ax)
plt.show()
```

### Multi-objective Feature Selection

Multi-objective optimization finds the best trade-offs between multiple competing objectives, such as model performance and feature count:

```python
from freamon.features import select_features_multi_objective

# Select features using multi-objective optimization
selected_features = select_features_multi_objective(
    df,
    target='target',
    objectives=['score', 'n_features'],
    selected_solution_method='knee',
    return_names_only=True
)
```

You can also use the `MultiObjectiveFeatureSelector` class directly:

```python
from freamon.features import MultiObjectiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor

# Create multi-objective selector
mo_selector = MultiObjectiveFeatureSelector(
    estimator=RandomForestRegressor(n_estimators=100),
    objectives=['score', 'n_features'],
    selected_solution_method='knee'
)

# Fit multi-objective selector
mo_selector.fit(X, y)

# Get selected features
selected_features = mo_selector.selected_features_

# Plot Pareto front
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
mo_selector.plot_pareto_front(ax=ax)
plt.show()

# Select a different solution from the Pareto front
solution = mo_selector.select_from_pareto(method='best_score')
```

## Time Series Feature Selection

Freamon provides specialized feature selection methods for time series data:

```python
from freamon.features import select_features_time_series

# Select features for time series data
selected_features = select_features_time_series(
    df,
    target='target',
    time_col='date',
    method='combined',
    max_lag=5,
    return_names_only=True
)
```

You can also use the `TimeSeriesFeatureSelector` class directly:

```python
from freamon.features import TimeSeriesFeatureSelector

# Create time series selector
ts_selector = TimeSeriesFeatureSelector(
    method='combined',
    target='target',
    time_col='date',
    max_lag=5
)

# Fit time series selector
ts_selector.fit(df)

# Get selected features
selected_features = ts_selector.selected_features_

# Plot feature scores
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ts_selector.plot_feature_scores(ax=ax)
plt.show()
```

### Time Series Selection Methods

The time series feature selector supports several methods:

- **'causality'**: Select features based on Granger causality tests
- **'autocorrelation'**: Select features based on autocorrelation with the target
- **'forecasting_impact'**: Select features based on their impact on forecasting performance
- **'combined'**: Use a combination of causality and autocorrelation methods

## Using the High-Level API

Freamon provides a high-level API for all feature selection methods through the `select_features` function:

```python
from freamon.features import select_features

# Basic methods
selected_features = select_features(
    df, 'target', method='correlation', threshold=0.3, return_names_only=True
)

# Advanced methods
selected_features = select_features(
    df, 'target', method='rfecv', cv=5, return_names_only=True
)

# Time series methods
selected_features = select_features(
    df, 'target', method='time_series', time_col='date', return_names_only=True
)
```

## Dependencies

While the basic feature selection methods are available with the standard Freamon installation, some advanced methods require additional dependencies:

- **Genetic algorithm-based selection**: Requires the DEAP library (`pip install deap`)
- **Multi-objective feature selection**: Requires the pymoo library (`pip install pymoo`)
- **Time series feature selection**: Requires the statsmodels library (`pip install statsmodels`)

## Complete Example

Here's a complete example showing how to use different feature selection methods:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from freamon.features import (
    select_features,
    select_by_correlation,
    select_by_importance,
    select_features_rfecv,
    select_features_stability,
    select_features_time_series
)

# Load data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Basic selection methods
corr_features = select_by_correlation(
    X_train, y_train, threshold=0.2, return_names_only=True
)

# Advanced selection methods
rfecv_features = select_features_rfecv(
    X_train, y_train, return_names_only=True
)

# Evaluate models
def evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    X_train_sel = X_train[feature_names]
    X_test_sel = X_test[feature_names]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_sel, y_train)
    
    y_pred = model.predict(X_test_sel)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# Compare performance
mse_corr = evaluate_model(X_train, X_test, y_train, y_test, corr_features)
mse_rfecv = evaluate_model(X_train, X_test, y_train, y_test, rfecv_features)

print(f"Correlation-based selection MSE: {mse_corr:.4f}")
print(f"RFECV-based selection MSE: {mse_rfecv:.4f}")
```

See the [feature_selection_example.py](../../examples/feature_selection_example.py) script for a more comprehensive example showing all feature selection methods.