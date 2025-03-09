"""
Example showing the integration between custom objectives and the LightGBMModel class.

This example demonstrates:
1. Using the FocalLoss custom objective with LightGBMModel
2. Using parameter tuning with custom objectives
3. End-to-end workflow for imbalanced classification
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.lightgbm_objectives import FocalLoss, register_with_lightgbm
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import HyperparameterTuningStep, EvaluationStep


def create_imbalanced_dataset(n_samples=50000, imbalance_ratio=0.01):
    """
    Create a highly imbalanced binary classification dataset.
    
    Parameters
    ----------
    n_samples : int, default=50000
        Number of samples to generate.
    imbalance_ratio : float, default=0.01
        Proportion of positive class (1% by default).
        
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    # Generate imbalanced binary classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[1-imbalance_ratio, imbalance_ratio],  # 1% positive class
        flip_y=0.05,  # Add some noise
        random_state=42
    )
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    print(f"Training class distribution:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print(f"Number of positive examples: {sum(y_train == 1)}")
    
    return X_train, X_test, y_train, y_test


def evaluate_with_precision_recall(y_true, y_pred_proba):
    """
    Evaluate model performance with emphasis on imbalanced metrics.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Probability estimates for the positive class.
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    # Calculate AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Calculate precision-recall curve and PR AUC
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Find threshold for 80% precision
    precision_80_idx = np.argmax(precision >= 0.8)
    threshold_80_precision = thresholds[precision_80_idx] if precision_80_idx < len(thresholds) else 1.0
    recall_at_80_precision = recall[precision_80_idx]
    
    # Find F1-optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_f1': best_f1,
        'best_f1_threshold': best_f1_threshold,
        'threshold_80_precision': threshold_80_precision,
        'recall_at_80_precision': recall_at_80_precision
    }


def run_standard_model():
    """
    Train and evaluate a standard LightGBM model without focal loss.
    """
    print("\n" + "=" * 80)
    print("Standard LightGBM Model (Binary Cross-Entropy)")
    print("=" * 80)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_imbalanced_dataset()
    
    # Create and train standard model
    model = LightGBMModel(
        problem_type='classification',
        objective='binary',
        metric='auc',
        tuning_trials=30,  # Reduced for example
        early_stopping_rounds=10,
        random_state=42
    )
    
    # Fit with hyperparameter tuning
    model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        tune_hyperparameters=True
    )
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_with_precision_recall(y_test, y_pred_proba)
    
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, metrics


def run_focal_loss_model():
    """
    Train and evaluate a LightGBM model with Focal Loss for handling imbalance.
    """
    print("\n" + "=" * 80)
    print("LightGBM Model with Focal Loss")
    print("=" * 80)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_imbalanced_dataset()
    
    # Create focal loss objective
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    obj_func, eval_func = register_with_lightgbm(focal_loss)
    
    # Create and train model with focal loss
    model = LightGBMModel(
        problem_type='classification',
        metric='auc',
        tuning_trials=30,  # Reduced for example
        early_stopping_rounds=10,
        random_state=42,
        custom_objective=obj_func,
        custom_eval_metric=eval_func
    )
    
    # Fit with hyperparameter tuning
    model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        tune_hyperparameters=True
    )
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_with_precision_recall(y_test, y_pred_proba)
    
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, metrics


def run_pipeline_example():
    """
    Use Focal Loss with the pipeline API.
    """
    print("\n" + "=" * 80)
    print("Pipeline with Focal Loss and Hyperparameter Tuning")
    print("=" * 80)
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_imbalanced_dataset()
    
    # Create focal loss objective
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    obj_func, eval_func = register_with_lightgbm(focal_loss)
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add hyperparameter tuning step with focal loss
    tuning_step = HyperparameterTuningStep(
        name="focal_tuning",
        model_type="lightgbm",
        problem_type="classification",
        metric="auc",
        n_trials=20,  # Reduced for example
        fixed_params={
            'custom_objective': obj_func,
            'custom_eval_metric': eval_func
        },
        random_state=42
    )
    
    # Add evaluation step
    eval_step = EvaluationStep(
        name="evaluation",
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        problem_type="classification"
    )
    
    # Add steps to pipeline
    pipeline.add_step(tuning_step)
    pipeline.add_step(eval_step)
    
    # Fit pipeline
    print("\nFitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = tuning_step.predict_proba(X_test)[:, 1]
    
    # Evaluate with the evaluation step
    eval_metrics = eval_step.evaluate(y_test, y_pred, y_pred_proba)
    print("\nEvaluation metrics from pipeline:")
    for metric, value in eval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluate with our precision-recall focused metrics
    pr_metrics = evaluate_with_precision_recall(y_test, y_pred_proba)
    print("\nPrecision-Recall focused metrics:")
    for metric, value in pr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get feature importances
    print("\nTop 10 features by importance:")
    importances = tuning_step.get_feature_importances()
    for feature, importance in importances.head(10).items():
        print(f"  {feature}: {importance:.4f}")


def compare_results():
    """
    Compare standard model with focal loss model.
    """
    print("\n" + "=" * 80)
    print("Comparison: Standard vs Focal Loss")
    print("=" * 80)
    
    # Run both models
    standard_model, standard_metrics = run_standard_model()
    focal_model, focal_metrics = run_focal_loss_model()
    
    # Compare metrics
    print("\nMetric comparison:")
    metrics_to_compare = ['roc_auc', 'pr_auc', 'best_f1', 'recall_at_80_precision']
    
    for metric in metrics_to_compare:
        std_value = standard_metrics[metric]
        focal_value = focal_metrics[metric]
        improvement = ((focal_value - std_value) / std_value) * 100
        
        print(f"  {metric}:")
        print(f"    - Standard: {std_value:.4f}")
        print(f"    - Focal:    {focal_value:.4f}")
        print(f"    - Change:   {improvement:+.2f}%")
    
    # Compare feature importance
    std_importance = standard_model.get_feature_importance()
    focal_importance = focal_model.get_feature_importance()
    
    # Merge and compare
    importance_comparison = pd.DataFrame({
        'Standard': std_importance,
        'Focal': focal_importance
    })
    
    importance_comparison['Difference'] = importance_comparison['Focal'] - importance_comparison['Standard']
    importance_comparison['Rel_Difference'] = importance_comparison['Difference'] / importance_comparison['Standard'].replace(0, np.nan)
    
    print("\nTop 5 features with largest relative change in importance:")
    top_changed = importance_comparison.sort_values('Rel_Difference', ascending=False).head(5)
    
    for idx, row in top_changed.iterrows():
        print(f"  {idx}:")
        print(f"    - Standard: {row['Standard']:.4f}")
        print(f"    - Focal:    {row['Focal']:.4f}")
        print(f"    - Change:   {row['Rel_Difference']*100:+.2f}%")


if __name__ == "__main__":
    # Individual examples
    # run_standard_model()
    # run_focal_loss_model()
    # run_pipeline_example()
    
    # Compare results
    compare_results()