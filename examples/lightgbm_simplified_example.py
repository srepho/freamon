"""
Example demonstrating the simplified LightGBM interface in freamon.

This example demonstrates the high-level LightGBMModel class, which provides
a streamlined interface for training LightGBM models with intelligent
hyperparameter tuning.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error

from freamon.modeling.lightgbm import LightGBMModel


def run_classification_example():
    """Run a classification example using the breast cancer dataset."""
    print("\n" + "=" * 80)
    print("LightGBM Simplified Interface - Classification Example")
    print("=" * 80)
    
    # Load the dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    # Add a categorical feature for demonstration
    X['hospital'] = np.random.choice(['A', 'B', 'C'], size=len(X))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Categorical features: ['hospital']")
    print()
    
    # Create and fit the model
    model = LightGBMModel(
        problem_type='classification',
        metric='auc',
        tuning_trials=20,  # Reduced for example
        random_state=42
    )
    
    print("Fitting the model with automatic hyperparameter tuning...")
    model.fit(
        X_train, y_train,
        categorical_features=['hospital'],
        tune_hyperparameters=True,
        # Fixed parameters
        fixed_params={'n_estimators': 200}
    )
    
    # Evaluate the model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest AUC: {auc:.4f}")
    
    # Get feature importance
    importance = model.get_feature_importance(method='native')
    top_features = importance.head(10)
    
    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind='barh')
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.savefig('lightgbm_top_features_classification.png')
    print("Feature importance plot saved to 'lightgbm_top_features_classification.png'")
    
    # Save the model
    model.save('breast_cancer_model.joblib')
    print("Model saved to 'breast_cancer_model.joblib'")
    
    # Load the model and check predictions match
    loaded_model = LightGBMModel.load('breast_cancer_model.joblib')
    loaded_preds = loaded_model.predict_proba(X_test)[:, 1]
    assert np.allclose(y_pred_proba, loaded_preds)
    print("Model loaded successfully and predictions match")


def run_regression_example():
    """Run a regression example using the California housing dataset."""
    print("\n" + "=" * 80)
    print("LightGBM Simplified Interface - Regression Example")
    print("=" * 80)
    
    # Load the dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="price")
    
    # Add a categorical feature for demonstration
    X['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(X))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Categorical features: ['region']")
    print()
    
    # Create and fit the model (without hyperparameter tuning for simplicity)
    model = LightGBMModel(
        problem_type='regression',
        metric='rmse',
        random_state=42
    )
    
    print("Fitting the model with default parameters (no tuning)...")
    model.fit(
        X_train, y_train,
        categorical_features=['region'],
        tune_hyperparameters=False,
        fixed_params={
            'n_estimators': 200,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'max_depth': 6
        }
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # Get permutation feature importance
    print("\nCalculating permutation importance...")
    perm_importance = model.get_feature_importance(method='permutation', X=X_test)
    top_features = perm_importance.head(10)
    
    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind='barh')
    plt.title('Top 10 Features by Permutation Importance')
    plt.tight_layout()
    plt.savefig('lightgbm_permutation_importance.png')
    print("Permutation importance plot saved to 'lightgbm_permutation_importance.png'")


if __name__ == "__main__":
    # Run classification example
    run_classification_example()
    
    # Run regression example
    run_regression_example()