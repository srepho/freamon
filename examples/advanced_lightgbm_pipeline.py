"""
Example of advanced LightGBM training with hyperparameter tuning,
early stopping, feature selection, and pipeline integration.

This example demonstrates:
1. Using the HyperparameterTuningStep in a pipeline
2. Implementing smart feature selection based on importance
3. Using enhanced early stopping with multiple strategies
4. Combining all components into a cohesive workflow
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from freamon.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep, 
    FeatureSelectionStep,
    HyperparameterTuningStep,
    EvaluationStep
)
from freamon.modeling.importance import (
    get_permutation_importance_df,
    plot_permutation_importance,
    auto_select_features
)
from freamon.modeling.early_stopping import (
    get_early_stopping_callback,
    get_lr_scheduler
)
from freamon.modeling.lightgbm import LightGBMModel


def run_regression_example():
    """Demonstrate advanced LightGBM workflow on a regression problem."""
    print("\n" + "=" * 80)
    print("Advanced LightGBM Pipeline - Regression Example")
    print("=" * 80)
    
    # Load data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='price')
    
    # Add a categorical feature for demonstration
    X['location_cluster'] = np.random.choice(['coastal', 'inland', 'mountain', 'urban'], size=len(X))
    
    # Perform train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print()
    
    # Create a pipeline with feature engineering, tuning, and evaluation
    feature_step = FeatureEngineeringStep(name="feature_engineering")
    feature_step.add_operation(
        method="add_polynomial_features",
        columns=["MedInc", "AveRooms"],
        degree=2,
        interaction_only=True
    )
    
    # Add hyperparameter tuning step for LightGBM
    tuning_step = HyperparameterTuningStep(
        name="lgbm_tuning",
        model_type="lightgbm",
        problem_type="regression",
        metric="rmse",
        n_trials=30,
        categorical_features=["location_cluster"],
        fixed_params={
            "n_estimators": 300,
            "verbose": -1
        },
        progressive_tuning=True,
        random_state=42
    )
    
    # Add evaluation step
    eval_step = EvaluationStep(
        name="evaluation",
        metrics=["rmse", "mae", "r2"],
        problem_type="regression"
    )
    
    # Create and fit pipeline
    pipeline = Pipeline()
    pipeline.add_step(feature_step)
    pipeline.add_step(tuning_step)
    pipeline.add_step(eval_step)
    
    print("Fitting pipeline with hyperparameter tuning...")
    pipeline.fit(X_train, y_train)
    
    # Get tuned model
    model = tuning_step.model
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate on test set
    metrics = eval_step.evaluate(y_test, y_pred)
    print("\nTest set metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get parameter importance
    print("\nMost important hyperparameters:")
    param_importance = tuning_step.get_param_importances()
    for param, importance in sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {param}: {importance:.4f}")
    
    # Get feature importance
    print("\nFeature importance:")
    feature_importance = tuning_step.get_feature_importances()
    for feature, importance in feature_importance.head(5).items():
        print(f"  {feature}: {importance:.4f}")
    
    # Perform permutation-based feature selection
    print("\nPerforming permutation-based feature selection...")
    selected_features, importance_df = auto_select_features(
        model, X_test, y_test,
        selection_method='cumulative',
        importance_method='permutation',
        cumulative_importance=0.95,
        features_to_keep=["location_cluster"]
    )
    
    print(f"Selected {len(selected_features)} out of {X_test.shape[1]} features:")
    print(f"  {', '.join(selected_features[:5])}...")
    
    # Plot permutation importance
    plot_permutation_importance(importance_df, top_n=10, show=False)
    plt.savefig('permutation_importance.png')
    print("Permutation importance plot saved to 'permutation_importance.png'")


def run_direct_api_example():
    """Demonstrate using the LightGBM API directly with advanced features."""
    print("\n" + "=" * 80)
    print("Using LightGBM API Directly with Advanced Features")
    print("=" * 80)
    
    # Load data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='price')
    
    # Create train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42  # 0.25 of 0.8 = 0.2 of total
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print()
    
    # Create a LightGBM model with minimal hyperparameter tuning
    print("Training LightGBM model with smart hyperparameter tuning...")
    model = LightGBMModel(
        problem_type='regression',
        metric='rmse',
        tuning_trials=30,  # Try 30 different hyperparameter combinations
        early_stopping_rounds=50,
        random_state=42
    )
    
    # Fit model with early stopping
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        tune_hyperparameters=True,
        fixed_params={
            'objective': 'regression',
            'n_estimators': 500
        }
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"\nTest RMSE: {rmse:.4f}")
    
    # Get feature importance
    print("\nFeature importance:")
    importance = model.get_feature_importance(method='native')
    for feature, value in importance.head(5).items():
        print(f"  {feature}: {value:.4f}")
    
    # Get permutation importance
    print("\nPermutation importance:")
    perm_importance = model.get_feature_importance(method='permutation', X=X_test)
    for feature, value in perm_importance.head(5).items():
        print(f"  {feature}: {value:.4f}")
    
    # Save and load model
    model.save("california_housing_model.joblib")
    print("\nModel saved to 'california_housing_model.joblib'")
    
    # Load model
    loaded_model = LightGBMModel.load("california_housing_model.joblib")
    loaded_pred = loaded_model.predict(X_test)
    
    # Verify predictions match
    assert np.allclose(y_pred, loaded_pred)
    print("Model loaded successfully and predictions match")


if __name__ == "__main__":
    run_regression_example()
    run_direct_api_example()