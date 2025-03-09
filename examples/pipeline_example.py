"""Example demonstrating the use of Freamon's pipeline system.

This example shows how to create and use a complete ML pipeline that integrates
feature engineering, feature selection, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    ShapIQFeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    EvaluationStep
)


def load_example_data():
    """Load example data for demonstration."""
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_example_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create pipeline steps
    
    # 1. Feature Engineering Step
    feature_eng_step = FeatureEngineeringStep(name="feature_engineering")
    
    # Add polynomial features
    feature_eng_step.add_operation(
        method="add_polynomial_features",
        columns=["mean radius", "mean texture", "mean perimeter"],
        degree=2,
        interaction_only=True
    )
    
    # Add binned features
    feature_eng_step.add_operation(
        method="add_binned_features",
        columns=["mean area", "mean smoothness"],
        n_bins=5,
        strategy="quantile"
    )
    
    # 2. Feature Selection Step
    feature_select_step = FeatureSelectionStep(
        name="feature_selection",
        method="model_based",
        n_features=20,
        model_type="lightgbm"
    )
    
    # 3. ShapIQ Feature Engineering Step
    shapiq_step = ShapIQFeatureEngineeringStep(
        name="shapiq_interactions",
        model_type="lightgbm",
        n_interactions=5,
        max_interaction_size=2
    )
    
    # 4. Model Training Step
    model_step = ModelTrainingStep(
        name="model_training",
        model_type="lightgbm",
        problem_type="classification",
        eval_metric="auc",
        hyperparameters={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100
        }
    )
    
    # 5. Evaluation Step
    eval_step = EvaluationStep(
        name="evaluation",
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        problem_type="classification"
    )
    
    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_step(feature_eng_step)
    pipeline.add_step(feature_select_step)
    pipeline.add_step(shapiq_step)
    pipeline.add_step(model_step)
    pipeline.add_step(eval_step)
    
    # Fit pipeline
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Get feature importances
    importances = pipeline.get_feature_importances()
    print("\nTop 5 feature importances:")
    print(importances.head(5))
    
    # Make predictions
    print("\nMaking predictions on test data...")
    y_pred = pipeline.predict(X_test)
    y_prob = model_step.predict_proba(X_test)
    
    # Evaluate
    print("\nEvaluating model performance:")
    eval_results = eval_step.evaluate(y_test, y_pred, y_prob)
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Get output from a specific step
    feature_eng_output = pipeline.get_step_output("feature_engineering")
    print(f"\nFeature engineering output shape: {feature_eng_output.shape}")
    
    # Save pipeline
    print("\nSaving pipeline...")
    pipeline.save("saved_pipeline")
    
    # Load pipeline
    print("Loading pipeline...")
    loaded_pipeline = Pipeline().load("saved_pipeline")
    
    # Verify loaded pipeline works
    y_pred_loaded = loaded_pipeline.predict(X_test)
    print(f"Original and loaded pipeline predictions match: {np.array_equal(y_pred, y_pred_loaded)}")
    
    # Print pipeline summary
    print("\nPipeline summary:")
    summary = pipeline.summary()
    for idx, step in enumerate(summary["steps"]):
        print(f"{idx+1}. {step['name']} ({step['type']})")


if __name__ == "__main__":
    main()