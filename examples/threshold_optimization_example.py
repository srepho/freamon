"""
Example demonstrating threshold optimization in classification tasks.

This example shows how to:
1. Train a binary classification model
2. Find the optimal probability threshold to maximize different metrics
3. Visualize the effect of different thresholds on model performance
4. Make predictions with custom thresholds
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

from freamon.modeling.lightgbm import LightGBMModel


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
    
    # Create model
    model = LightGBMModel(
        problem_type='classification',
        tuning_trials=10,  # Low for example purposes
        random_state=42
    )
    
    # Train model
    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        validation_size=0.2,
        tune_hyperparameters=True
    )
    
    # Get default predictions (threshold=0.5)
    y_pred_default = model.predict(X_test)
    default_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_default),
        'precision': precision_score(y_test, y_pred_default),
        'recall': recall_score(y_test, y_pred_default),
        'f1': f1_score(y_test, y_pred_default)
    }
    
    print("\nModel performance with default threshold (0.5):")
    for metric, value in default_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Optimize for different metrics
    metrics_to_optimize = ['f1', 'precision', 'recall', 'accuracy']
    optimal_thresholds = {}
    
    for metric_name in metrics_to_optimize:
        print(f"\nFinding optimal threshold for {metric_name}...")
        threshold, score, results = model.find_optimal_threshold(
            X_test, y_test, metric=metric_name
        )
        optimal_thresholds[metric_name] = {
            'threshold': threshold,
            'score': score,
            'results': results
        }
        print(f"Optimal threshold: {threshold:.4f}, {metric_name}: {score:.4f}")
        
        # Make predictions with optimal threshold
        y_pred_opt = model.predict(X_test, threshold=threshold)
        print("Performance with optimal threshold:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred_opt):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred_opt):.4f}")
        print(f"F1: {f1_score(y_test, y_pred_opt):.4f}")
    
    # Set a specific optimal threshold as default (e.g., F1)
    best_f1_threshold = optimal_thresholds['f1']['threshold']
    model.probability_threshold = best_f1_threshold
    print(f"\nSetting F1 optimal threshold ({best_f1_threshold:.4f}) as default")
    
    # Predictions with default threshold (now the F1-optimal)
    y_pred_new_default = model.predict(X_test)
    print("\nPerformance with new default threshold:")
    print(classification_report(y_test, y_pred_new_default))
    
    # Plot threshold optimization curves
    plt.figure(figsize=(12, 8))
    for i, metric_name in enumerate(metrics_to_optimize):
        plt.subplot(2, 2, i+1)
        results = optimal_thresholds[metric_name]['results']
        plt.plot(results['threshold'], results['score'])
        opt_threshold = optimal_thresholds[metric_name]['threshold']
        opt_score = optimal_thresholds[metric_name]['score']
        plt.scatter([opt_threshold], [opt_score], color='red', s=100, zorder=5)
        
        plt.title(f'Threshold Optimization for {metric_name}')
        plt.xlabel('Probability Threshold')
        plt.ylabel(f'{metric_name.capitalize()} Score')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Default (0.5)')
        plt.axvline(x=opt_threshold, color='red', linestyle='--', alpha=0.5, 
                   label=f'Optimal ({opt_threshold:.3f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_optimization.png')
    print("\nSaved threshold optimization curves to 'threshold_optimization.png'")
    
    # Compare confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default threshold (0.5)
    cm_default = confusion_matrix(y_test, y_pred_default)
    axes[0].imshow(cm_default, cmap='Blues')
    axes[0].set_title('Confusion Matrix (Default Threshold = 0.5)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Add text annotations
    for i in range(cm_default.shape[0]):
        for j in range(cm_default.shape[1]):
            axes[0].text(j, i, str(cm_default[i, j]), 
                         ha="center", va="center", color="black")
    
    # Optimal threshold
    cm_optimal = confusion_matrix(y_test, y_pred_new_default)
    axes[1].imshow(cm_optimal, cmap='Blues')
    axes[1].set_title(f'Confusion Matrix (Optimal Threshold = {best_f1_threshold:.3f})')
    axes[1].set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(cm_optimal.shape[0]):
        for j in range(cm_optimal.shape[1]):
            axes[1].text(j, i, str(cm_optimal[i, j]), 
                         ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png')
    print("\nSaved confusion matrix comparison to 'confusion_matrix_comparison.png'")


if __name__ == "__main__":
    main()