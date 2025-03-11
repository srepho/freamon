"""
Example demonstrating automatic threshold optimization in a classification pipeline.

This example shows how to:
1. Create a pipeline with automatic threshold optimization
2. Train a classification model with optimized probability threshold
3. Compare results with and without threshold optimization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

from freamon.pipeline.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    FeatureSelectionStep,
    HyperparameterTuningStep,
    EvaluationStep
)


def main():
    # Load binary classification dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"Dataset: {data.DESCR.split('\n')[0]}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create validation set for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Create pipeline without threshold optimization (default 0.5)
    pipeline_default = create_pipeline(optimize_threshold=False)
    
    # Train pipeline without threshold optimization
    print("\nTraining pipeline with default threshold (0.5)...")
    pipeline_default.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    
    # Create pipeline with threshold optimization (F1)
    pipeline_f1 = create_pipeline(optimize_threshold=True, threshold_metric='f1')
    
    # Train pipeline with F1 threshold optimization
    print("\nTraining pipeline with F1 threshold optimization...")
    pipeline_f1.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    
    # Create pipeline with threshold optimization (precision)
    pipeline_precision = create_pipeline(optimize_threshold=True, threshold_metric='precision')
    
    # Train pipeline with precision threshold optimization
    print("\nTraining pipeline with precision threshold optimization...")
    pipeline_precision.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    
    # Create pipeline with threshold optimization (recall)
    pipeline_recall = create_pipeline(optimize_threshold=True, threshold_metric='recall')
    
    # Train pipeline with recall threshold optimization
    print("\nTraining pipeline with recall threshold optimization...")
    pipeline_recall.fit(X_train, y_train, val_X=X_val, val_y=y_val)
    
    # Evaluate all models on test data
    evaluate_pipeline(pipeline_default, "Default threshold (0.5)", X_test, y_test)
    evaluate_pipeline(pipeline_f1, "F1-optimized threshold", X_test, y_test)
    evaluate_pipeline(pipeline_precision, "Precision-optimized threshold", X_test, y_test)
    evaluate_pipeline(pipeline_recall, "Recall-optimized threshold", X_test, y_test)
    
    # Get the optimal thresholds
    tuning_step_default = pipeline_default.get_step("model_tuning")
    tuning_step_f1 = pipeline_f1.get_step("model_tuning")
    tuning_step_precision = pipeline_precision.get_step("model_tuning")
    tuning_step_recall = pipeline_recall.get_step("model_tuning")
    
    # Compare confusion matrices
    thresholds = {
        "Default (0.5)": 0.5,
        "F1-optimized": tuning_step_f1.optimal_threshold,
        "Precision-optimized": tuning_step_precision.optimal_threshold,
        "Recall-optimized": tuning_step_recall.optimal_threshold
    }
    
    # Get predictions from all models
    all_predictions = {
        name: pipeline.predict(X_test)
        for name, pipeline in [
            ("Default", pipeline_default),
            ("F1", pipeline_f1),
            ("Precision", pipeline_precision),
            ("Recall", pipeline_recall)
        ]
    }
    
    # Compare confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, (name, predictions) in enumerate(all_predictions.items()):
        cm = confusion_matrix(y_test, predictions)
        threshold_str = f"{thresholds.get(name+'-optimized', 0.5):.3f}" if name != "Default" else "0.5"
        
        axes[i].imshow(cm, cmap='Blues')
        axes[i].set_title(f"{name} Threshold ({threshold_str})")
        axes[i].set_xlabel("Predicted Label")
        if i % 2 == 0:
            axes[i].set_ylabel("True Label")
        
        # Add text annotations
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, str(cm[j, k]), 
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png')
    print("\nSaved confusion matrix comparison to 'threshold_comparison.png'")


def create_pipeline(optimize_threshold=False, threshold_metric='f1'):
    """Create a classification pipeline with optional threshold optimization.
    
    Args:
        optimize_threshold: Whether to optimize the classification threshold
        threshold_metric: Metric to optimize threshold for (f1, precision, recall)
        
    Returns:
        Pipeline: Classification pipeline
    """
    # Create pipeline steps
    feature_eng_step = FeatureEngineeringStep(
        name="feature_eng",
        operations=[]  # No feature engineering for this example
    )
    
    feature_selection_step = FeatureSelectionStep(
        name="feature_selection",
        method="model_based",
        n_features=20  # Select top 20 features
    )
    
    tuning_step = HyperparameterTuningStep(
        name="model_tuning",
        model_type="lightgbm",
        problem_type="classification",
        metric="auc",
        n_trials=20,  # Reduced for example
        cv=3,  # Reduced for example
        early_stopping_rounds=20,
        optimize_threshold=optimize_threshold,
        threshold_metric=threshold_metric
    )
    
    eval_step = EvaluationStep(
        name="evaluation",
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        problem_type="classification"
    )
    
    # Create and return pipeline
    pipeline = Pipeline()
    pipeline.add_step(feature_eng_step)
    pipeline.add_step(feature_selection_step)
    pipeline.add_step(tuning_step)
    pipeline.add_step(eval_step)
    
    return pipeline


def evaluate_pipeline(pipeline, name, X_test, y_test):
    """Evaluate a pipeline on test data and print results.
    
    Args:
        pipeline: Trained pipeline
        name: Name for the pipeline
        X_test: Test features
        y_test: Test targets
    """
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Get evaluation step and calculate metrics
    eval_step = pipeline.get_step("evaluation")
    
    # Get probabilities for ROC AUC
    tuning_step = pipeline.get_step("model_tuning")
    y_prob = tuning_step.predict_proba(X_test)
    
    # Evaluate
    metrics = eval_step.evaluate(y_test, y_pred, y_prob)
    
    # Print results
    print(f"\nResults for {name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()