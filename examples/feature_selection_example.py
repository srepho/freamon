"""
Example script for using the feature selection methods in freamon.

This script demonstrates:
1. Basic feature selection methods (correlation, importance, variance, mutual info)
2. Recursive feature elimination with cross-validation
3. Stability selection
4. Genetic algorithm-based feature selection
5. Multi-objective feature selection
6. Time series-specific feature selection
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import freamon
from freamon.features import (
    # Basic feature selection
    select_features,
    select_by_correlation,
    select_by_importance,
    select_by_variance,
    select_by_mutual_info,
    
    # Advanced feature selection classes
    RecursiveFeatureEliminationCV,
    StabilitySelector,
    GeneticFeatureSelector,
    MultiObjectiveFeatureSelector,
    TimeSeriesFeatureSelector,
    
    # Advanced feature selection wrapper functions
    select_features_rfecv,
    select_features_stability,
    select_features_genetic,
    select_features_multi_objective,
    select_features_time_series,
)


def evaluate_model(X_train, X_test, y_train, y_test, feature_names=None):
    """Evaluate a random forest model with the given features."""
    if feature_names is not None:
        # Select only the chosen features
        X_train = X_train[feature_names]
        X_test = X_test[feature_names]
    
    # Create and train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Return number of features and MSE
    n_features = X_train.shape[1]
    return n_features, mse


def print_result(method_name, n_features, mse, selected_features=None):
    """Print the result of a feature selection method."""
    print(f"\n{method_name}:")
    print(f"  Number of features: {n_features}")
    print(f"  Mean squared error: {mse:.4f}")
    if selected_features is not None:
        print(f"  Selected features: {', '.join(selected_features)}")


# Load data
print("Loading diabetes dataset...")
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Available features: {', '.join(X.columns)}")

# Baseline (all features)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test)
print_result("Baseline (all features)", n_features, mse)

# Basic methods
print("\n----- Basic Feature Selection Methods -----")

# Correlation-based selection
selected_features = select_by_correlation(
    X_train, y_train, threshold=0.2, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("Correlation-based selection", n_features, mse, selected_features)

# Importance-based selection
selected_features = select_by_importance(
    X_train, y_train, k=5, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("Importance-based selection", n_features, mse, selected_features)

# Variance-based selection
selected_features = select_by_variance(
    X_train, threshold=0.01, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("Variance-based selection", n_features, mse, selected_features)

# Mutual information-based selection
selected_features = select_by_mutual_info(
    X_train, y_train, k=5, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("Mutual information-based selection", n_features, mse, selected_features)

# Using the high-level select_features function
selected_features = select_features(
    X_train, y_train, method='mutual_info', k=5, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("High-level select_features (mutual_info)", n_features, mse, selected_features)

# Advanced methods
print("\n----- Advanced Feature Selection Methods -----")

# RFECV
print("\nRecursive Feature Elimination with Cross-Validation (this may take a moment)...")
selected_features = select_features_rfecv(
    X_train, y_train, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("RFECV", n_features, mse, selected_features)

# Stability selection
print("\nStability Selection (this may take a moment)...")
selected_features = select_features_stability(
    X_train, y_train, threshold=0.6, n_subsamples=30, return_names_only=True
)
n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
print_result("Stability Selection", n_features, mse, selected_features)

# Genetic algorithm
try:
    print("\nGenetic Algorithm-based Selection (this may take several minutes)...")
    selected_features = select_features_genetic(
        X_train, y_train, n_generations=10, population_size=20, return_names_only=True
    )
    n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
    print_result("Genetic Algorithm", n_features, mse, selected_features)
except ImportError:
    print("\nGenetic Algorithm-based Selection (DEAP library not installed)")
    print("To install, run: pip install deap")

# Multi-objective optimization
try:
    print("\nMulti-objective Optimization (this may take several minutes)...")
    selected_features = select_features_multi_objective(
        X_train, y_train, n_generations=10, population_size=20, return_names_only=True
    )
    n_features, mse = evaluate_model(X_train, X_test, y_train, y_test, selected_features)
    print_result("Multi-objective Optimization", n_features, mse, selected_features)
except ImportError:
    print("\nMulti-objective Optimization (pymoo library not installed)")
    print("To install, run: pip install pymoo")

# Time series example
print("\n----- Time Series Feature Selection Example -----")
print("Creating synthetic time series data...")

# Create synthetic time series data
np.random.seed(42)
n_samples = 1000
dates = pd.date_range('2020-01-01', periods=n_samples)

# Create some features with different levels of predictive power
X_ts = pd.DataFrame({
    'date': dates,
    'feature1': np.sin(np.arange(n_samples) / 20) + np.random.normal(0, 0.1, n_samples),
    'feature2': np.cos(np.arange(n_samples) / 10) + np.random.normal(0, 0.1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),  # random noise
    'feature4': np.sin(np.arange(n_samples) / 50) + np.random.normal(0, 0.1, n_samples),
    'feature5': np.random.normal(0, 1, n_samples),  # random noise
})

# Create target with lag relationships
X_ts['target'] = (
    0.5 * X_ts['feature1'].shift(1) +
    0.3 * X_ts['feature2'].shift(2) +
    0.1 * X_ts['feature4'].shift(3) +
    np.random.normal(0, 0.1, n_samples)
)

# Drop NAs
X_ts = X_ts.dropna()

# Split into train/test
train_size = int(len(X_ts) * 0.8)
X_ts_train = X_ts.iloc[:train_size]
X_ts_test = X_ts.iloc[train_size:]

try:
    # Time series feature selection
    print("\nTime Series Feature Selection (needs statsmodels)...")
    selected_features = select_features_time_series(
        X_ts_train, 'target', time_col='date', 
        method='combined', max_lag=5, return_names_only=True
    )
    
    # Evaluate (simple, just to show the concept)
    X_train_ts = X_ts_train.drop(columns=['target', 'date'])
    X_test_ts = X_ts_test.drop(columns=['target', 'date'])
    y_train_ts = X_ts_train['target']
    y_test_ts = X_ts_test['target']
    
    n_features, mse = evaluate_model(X_train_ts, X_test_ts, y_train_ts, y_test_ts, selected_features)
    print_result("Time Series Selection", n_features, mse, selected_features)
except ImportError:
    print("\nTime Series Feature Selection (statsmodels library not installed)")
    print("To install, run: pip install statsmodels")

print("\nDone! Feature selection examples complete.")