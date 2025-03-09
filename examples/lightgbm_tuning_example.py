"""
Example showcasing intelligent LightGBM hyperparameter tuning with freamon.

This example demonstrates:
1. Using the LightGBMTuner for intelligent hyperparameter optimization
2. Progressive tuning that focuses on most important parameters
3. Visualization of parameter importance and optimization history
4. Building and evaluating an optimized LightGBM model
5. Handling categorical features properly in LightGBM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score

from freamon.modeling.tuning import LightGBMTuner
from freamon.modeling.factory import create_model


def run_regression_example():
    """
    Run a regression example using California Housing dataset.
    """
    print("=" * 80)
    print("LightGBM Intelligent Tuning - Regression Example")
    print("=" * 80)
    
    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Create a pandas DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features (for demonstration)
    X_df['ocean_proximity'] = np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY'], size=len(X))
    X_df['county'] = np.random.choice(['Los Angeles', 'San Diego', 'San Francisco', 'Alameda', 'Sacramento'], size=len(X))
    
    # Convert target to pandas Series
    y_series = pd.Series(y, name='price')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    # Identify categorical features
    categorical_features = ['ocean_proximity', 'county']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Categorical features: {categorical_features}")
    print()
    
    # Initialize and run the tuner
    tuner = LightGBMTuner(
        problem_type='regression',
        objective='regression',
        metric='rmse',
        cv=5,
        cv_type='kfold',
        random_state=42,
        n_trials=50,  # Reduced for example
        n_jobs=-1,  # Use all cores
        verbose=True
    )
    
    # Set some fixed parameters (optional)
    fixed_params = {
        'n_estimators': 200,
        'verbose': -1
    }
    
    print("Starting hyperparameter tuning...")
    best_params = tuner.tune(
        X_train, y_train,
        categorical_features=categorical_features,
        fixed_params=fixed_params,
        progressive_tuning=True,
        early_stopping_rounds=50
    )
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Print parameter importance
    if tuner.param_importance is not None:
        print("\nParameter importance:")
        for param, importance in sorted(tuner.param_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {importance:.4f}")
    
    # Create and train model with best parameters
    model = tuner.create_model()
    
    print("\nTraining final model with best parameters...")
    model.fit(
        X_train, y_train,
        categorical_feature=categorical_features
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # Plot feature importance
    feature_importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().plot(kind='barh')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png')
    print("Feature importance plot saved to 'lightgbm_feature_importance.png'")
    
    # Try to plot optimization history
    try:
        import plotly.io as pio
        history_fig = tuner.plot_optimization_history()
        if history_fig:
            pio.write_image(history_fig, 'optimization_history.png')
            print("Optimization history plot saved to 'optimization_history.png'")
        
        importance_fig = tuner.plot_param_importances()
        if importance_fig:
            pio.write_image(importance_fig, 'parameter_importance.png')
            print("Parameter importance plot saved to 'parameter_importance.png'")
    except ImportError:
        print("Could not generate Optuna plots. Install plotly to enable visualization.")


def run_classification_example():
    """
    Run a classification example using Breast Cancer dataset.
    """
    print("\n" + "=" * 80)
    print("LightGBM Intelligent Tuning - Classification Example")
    print("=" * 80)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Create a pandas DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features (for demonstration)
    X_df['hospital'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(X))
    X_df['doctor_id'] = np.random.choice(['001', '002', '003', '004', '005'], size=len(X))
    
    # Convert target to pandas Series
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify categorical features
    categorical_features = ['hospital', 'doctor_id']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Categorical features: {categorical_features}")
    print(f"Class distribution: {pd.Series(y).value_counts(normalize=True)}")
    print()
    
    # Initialize and run the tuner
    tuner = LightGBMTuner(
        problem_type='classification',
        objective='binary',
        metric='auc',
        cv=5,
        cv_type='stratified',
        random_state=42,
        n_trials=50,  # Reduced for example
        n_jobs=-1,  # Use all cores
        verbose=True
    )
    
    # Set some fixed parameters (optional)
    fixed_params = {
        'n_estimators': 200,
        'verbose': -1
    }
    
    print("Starting hyperparameter tuning...")
    best_params = tuner.tune(
        X_train, y_train,
        categorical_features=categorical_features,
        fixed_params=fixed_params,
        progressive_tuning=True,
        early_stopping_rounds=50
    )
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Print parameter importance
    if tuner.param_importance is not None:
        print("\nParameter importance:")
        for param, importance in sorted(tuner.param_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {importance:.4f}")
    
    # Create and train model with best parameters
    model = tuner.create_model()
    
    print("\nTraining final model with best parameters...")
    model.fit(
        X_train, y_train,
        categorical_feature=categorical_features
    )
    
    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nTest AUC: {auc:.4f}")
    
    # Plot feature importance
    feature_importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    feature_importance.sort_values().plot(kind='barh')
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance_classification.png')
    print("Feature importance plot saved to 'lightgbm_feature_importance_classification.png'")


if __name__ == "__main__":
    # Run regression example
    run_regression_example()
    
    # Run classification example
    run_classification_example()