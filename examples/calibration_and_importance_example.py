"""
Example demonstrating permutation importance and probability calibration.

This example shows:
1. How to calculate and visualize permutation importance for model interpretation
2. How to calibrate classification model probabilities using different methods
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from freamon.modeling.importance import (
    calculate_permutation_importance,
    get_permutation_importance_df,
    plot_permutation_importance
)
from freamon.modeling.calibration import (
    ProbabilityCalibrator,
    evaluate_calibration,
    compare_calibration_methods
)

# ===============================
# Permutation Importance Example
# ===============================

def demonstrate_permutation_importance():
    """Demonstrate permutation importance for classification and regression."""
    print("\n== Permutation Importance Demonstration ==")
    
    # Classification example
    print("\nClassification Model Importance:")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Create a pandas DataFrame for better feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42
    )
    
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Calculate permutation importance
    result = calculate_permutation_importance(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )
    
    # Get importance as DataFrame
    importance_df = get_permutation_importance_df(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=42
    )
    
    # Print top 5 important features
    print("Top 5 important features (classification):")
    print(importance_df.head(5))
    
    # Plot importance
    print("\nPlotting classification importance...")
    plot_permutation_importance(importance_df, top_n=5)
    
    # Regression example
    print("\nRegression Model Importance:")
    
    # Generate synthetic regression data
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    # Create a pandas DataFrame
    X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(X_reg.shape[1])])
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_df, y_reg, test_size=0.3, random_state=42
    )
    
    # Train a random forest regressor
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train_reg, y_train_reg)
    
    # Calculate permutation importance for regression
    reg_importance = get_permutation_importance_df(
        regr, X_test_reg, y_test_reg,
        n_repeats=10,
        random_state=42,
        scoring='r2'
    )
    
    # Print top 5 important features for regression
    print("Top 5 important features (regression):")
    print(reg_importance.head(5))
    
    # Plot importance
    print("\nPlotting regression importance...")
    plot_permutation_importance(reg_importance, top_n=5)
    
    # Comparing native importance vs permutation importance
    print("\nComparing native vs. permutation importance:")
    
    # Get native importance from random forest
    native_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Print top 5 features from native importance
    print("Top 5 important features (native importance):")
    print(native_importance.head(5))
    
    # Compare with bar plot
    plt.figure(figsize=(12, 6))
    
    # Plot top 5 features for both methods
    top5_native = native_importance.head(5)
    top5_perm = importance_df.head(5)
    
    # Combine and plot
    combined_df = pd.DataFrame({
        'Native': {row['feature']: row['importance'] for _, row in top5_native.iterrows()},
        'Permutation': {row['feature']: row['importance'] for _, row in top5_perm.iterrows()}
    })
    
    combined_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Native vs. Permutation Feature Importance')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


# ===============================
# Probability Calibration Example
# ===============================

def demonstrate_probability_calibration():
    """Demonstrate probability calibration for classification models."""
    print("\n== Probability Calibration Demonstration ==")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Create a pandas DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42
    )
    
    # Train a random forest classifier (often needs calibration)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate uncalibrated model
    y_prob_uncalibrated = clf.predict_proba(X_test)
    uncalibrated_accuracy = accuracy_score(y_test, clf.predict(X_test))
    uncalibrated_auc = roc_auc_score(y_test, y_prob_uncalibrated[:, 1])
    uncalibrated_brier = brier_score_loss(y_test, y_prob_uncalibrated[:, 1])
    
    print(f"Uncalibrated model performance:")
    print(f"  - Accuracy: {uncalibrated_accuracy:.4f}")
    print(f"  - ROC AUC: {uncalibrated_auc:.4f}")
    print(f"  - Brier score: {uncalibrated_brier:.4f}")
    
    # Evaluate calibration
    print("\nEvaluating calibration of uncalibrated model...")
    uncalibrated_metrics = evaluate_calibration(y_test, y_prob_uncalibrated)
    print(f"  - Expected Calibration Error: {uncalibrated_metrics['ece']:.4f}")
    
    # Calibrate using Platt scaling (sigmoid)
    print("\nCalibrating with Platt scaling (sigmoid)...")
    sigmoid_calibrator = ProbabilityCalibrator(method='sigmoid')
    sigmoid_calibrator.fit(clf, X_train, y_train)
    
    # Get calibrated probabilities
    y_prob_sigmoid = sigmoid_calibrator.predict_proba(X_test)
    
    # Evaluate sigmoid-calibrated model
    sigmoid_accuracy = accuracy_score(y_test, sigmoid_calibrator.predict(X_test))
    sigmoid_auc = roc_auc_score(y_test, y_prob_sigmoid[:, 1])
    sigmoid_brier = brier_score_loss(y_test, y_prob_sigmoid[:, 1])
    
    print(f"Sigmoid-calibrated model performance:")
    print(f"  - Accuracy: {sigmoid_accuracy:.4f}")
    print(f"  - ROC AUC: {sigmoid_auc:.4f}")
    print(f"  - Brier score: {sigmoid_brier:.4f}")
    
    # Evaluate calibration
    print("\nEvaluating calibration of sigmoid-calibrated model...")
    sigmoid_metrics = evaluate_calibration(y_test, y_prob_sigmoid)
    print(f"  - Expected Calibration Error: {sigmoid_metrics['ece']:.4f}")
    
    # Calibrate using isotonic regression
    print("\nCalibrating with isotonic regression...")
    isotonic_calibrator = ProbabilityCalibrator(method='isotonic')
    isotonic_calibrator.fit(clf, X_train, y_train)
    
    # Get calibrated probabilities
    y_prob_isotonic = isotonic_calibrator.predict_proba(X_test)
    
    # Evaluate isotonic-calibrated model
    isotonic_accuracy = accuracy_score(y_test, isotonic_calibrator.predict(X_test))
    isotonic_auc = roc_auc_score(y_test, y_prob_isotonic[:, 1])
    isotonic_brier = brier_score_loss(y_test, y_prob_isotonic[:, 1])
    
    print(f"Isotonic-calibrated model performance:")
    print(f"  - Accuracy: {isotonic_accuracy:.4f}")
    print(f"  - ROC AUC: {isotonic_auc:.4f}")
    print(f"  - Brier score: {isotonic_brier:.4f}")
    
    # Evaluate calibration
    print("\nEvaluating calibration of isotonic-calibrated model...")
    isotonic_metrics = evaluate_calibration(y_test, y_prob_isotonic)
    print(f"  - Expected Calibration Error: {isotonic_metrics['ece']:.4f}")
    
    # Compare calibration methods
    print("\nComparing calibration methods...")
    compare_calibration_methods(X_train, y_train, X_test, y_test, clf)
    
    # Compare different models' calibration
    print("\nComparing calibration across different models...")
    
    # Train a logistic regression model (often well-calibrated out of the box)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    # Get logistic regression probabilities
    y_prob_lr = lr.predict_proba(X_test)
    
    # Evaluate logistic regression performance
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test))
    lr_auc = roc_auc_score(y_test, y_prob_lr[:, 1])
    lr_brier = brier_score_loss(y_test, y_prob_lr[:, 1])
    
    print(f"Logistic Regression model performance:")
    print(f"  - Accuracy: {lr_accuracy:.4f}")
    print(f"  - ROC AUC: {lr_auc:.4f}")
    print(f"  - Brier score: {lr_brier:.4f}")
    
    # Evaluate calibration
    print("\nEvaluating calibration of logistic regression model...")
    lr_metrics = evaluate_calibration(y_test, y_prob_lr)
    print(f"  - Expected Calibration Error: {lr_metrics['ece']:.4f}")
    
    # Plot a comparison of all models' calibration curves
    plt.figure(figsize=(12, 8))
    
    # Plot the perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    
    # Plot each model's calibration curve
    models = [
        ('Uncalibrated RF', y_prob_uncalibrated, uncalibrated_metrics, 'red'),
        ('Sigmoid Calibrated RF', y_prob_sigmoid, sigmoid_metrics, 'green'),
        ('Isotonic Calibrated RF', y_prob_isotonic, isotonic_metrics, 'blue'),
        ('Logistic Regression', y_prob_lr, lr_metrics, 'purple')
    ]
    
    for name, y_prob, metrics, color in models:
        plt.plot(
            metrics['prob_pred'], 
            metrics['prob_true'], 
            marker='o', 
            linewidth=2, 
            color=color,
            label=f"{name} (ECE: {metrics['ece']:.4f})"
        )
    
    # Customize the plot
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves for Different Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ===============================
# Combined Example
# ===============================

def combined_importance_with_calibration():
    """Demonstrate how to combine permutation importance with calibration."""
    print("\n== Combined Permutation Importance and Calibration Example ==")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        random_state=42
    )
    
    # Create a pandas DataFrame with meaningful feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42
    )
    
    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Calculate permutation importance
    print("\nCalculating permutation importance...")
    importance_df = get_permutation_importance_df(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='roc_auc'  # Use ROC AUC for importance
    )
    
    # Print top 10 important features
    print("Top 10 important features:")
    print(importance_df.head(10))
    
    # Select top 5 most important features
    top_features = importance_df.head(5)['feature'].tolist()
    print(f"\nSelected top 5 features: {top_features}")
    
    # Create reduced datasets with only top features
    X_train_reduced = X_train[top_features]
    X_test_reduced = X_test[top_features]
    
    # Train a model on reduced feature set
    clf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_reduced.fit(X_train_reduced, y_train)
    
    # Compare performance
    full_accuracy = accuracy_score(y_test, clf.predict(X_test))
    reduced_accuracy = accuracy_score(y_test, clf_reduced.predict(X_test_reduced))
    
    full_proba = clf.predict_proba(X_test)
    reduced_proba = clf_reduced.predict_proba(X_test_reduced)
    
    full_auc = roc_auc_score(y_test, full_proba[:, 1])
    reduced_auc = roc_auc_score(y_test, reduced_proba[:, 1])
    
    print("\nModel Performance Comparison:")
    print(f"  - Full model (20 features):")
    print(f"    * Accuracy: {full_accuracy:.4f}")
    print(f"    * ROC AUC: {full_auc:.4f}")
    print(f"  - Reduced model (5 features):")
    print(f"    * Accuracy: {reduced_accuracy:.4f}")
    print(f"    * ROC AUC: {reduced_auc:.4f}")
    
    # Calibrate the reduced model
    print("\nCalibrating the reduced model...")
    calibrator = ProbabilityCalibrator(method='sigmoid')
    calibrator.fit(clf_reduced, X_train_reduced, y_train)
    
    # Get calibrated probabilities
    calibrated_proba = calibrator.predict_proba(X_test_reduced)
    calibrated_accuracy = accuracy_score(y_test, calibrator.predict(X_test_reduced))
    calibrated_auc = roc_auc_score(y_test, calibrated_proba[:, 1])
    
    print("\nCalibrated Reduced Model Performance:")
    print(f"  - Accuracy: {calibrated_accuracy:.4f}")
    print(f"  - ROC AUC: {calibrated_auc:.4f}")
    
    # Evaluate and compare calibration
    print("\nComparing calibration between models...")
    
    # Get calibration metrics for each model
    full_cal = evaluate_calibration(y_test, full_proba, show=False)
    reduced_cal = evaluate_calibration(y_test, reduced_proba, show=False)
    calibrated_cal = evaluate_calibration(y_test, calibrated_proba, show=False)
    
    print(f"  - Full model ECE: {full_cal['ece']:.4f}")
    print(f"  - Reduced model ECE: {reduced_cal['ece']:.4f}")
    print(f"  - Calibrated reduced model ECE: {calibrated_cal['ece']:.4f}")
    
    # Plot calibration comparison
    plt.figure(figsize=(12, 8))
    
    # Plot the perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    
    # Plot each model's calibration curve
    models = [
        ('Full Model', full_proba, full_cal, 'red'),
        ('Reduced Model', reduced_proba, reduced_cal, 'blue'),
        ('Calibrated Reduced Model', calibrated_proba, calibrated_cal, 'green')
    ]
    
    for name, y_prob, metrics, color in models:
        plt.plot(
            metrics['prob_pred'], 
            metrics['prob_true'], 
            marker='o', 
            linewidth=2, 
            color=color,
            label=f"{name} (ECE: {metrics['ece']:.4f})"
        )
    
    # Customize the plot
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Permutation importance for the calibrated model
    print("\nCalculating permutation importance for the calibrated model...")
    
    # Define a custom function to get calibrated probabilities
    def get_calibrated_proba(X):
        return calibrator.predict_proba(X)[:, 1]
    
    # Manual permutation importance calculation for calibrated probabilities
    n_repeats = 5
    baseline_score = roc_auc_score(y_test, get_calibrated_proba(X_test_reduced))
    importance_result = {feature: [] for feature in X_test_reduced.columns}
    
    for feature in X_test_reduced.columns:
        for _ in range(n_repeats):
            # Create a permuted copy
            X_permuted = X_test_reduced.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            
            # Score with permuted feature
            permuted_score = roc_auc_score(y_test, get_calibrated_proba(X_permuted))
            
            # Store importance (decrease in performance)
            importance_result[feature].append(baseline_score - permuted_score)
    
    # Calculate mean importance for each feature
    calibrated_importance = pd.DataFrame({
        'feature': list(importance_result.keys()),
        'importance': [np.mean(importance_result[f]) for f in importance_result.keys()],
        'std': [np.std(importance_result[f]) for f in importance_result.keys()]
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\nImportance scores for calibrated model:")
    print(calibrated_importance)
    
    # Plot importance for original vs. calibrated model
    plt.figure(figsize=(12, 6))
    
    # Get importance for original reduced model
    original_importance = get_permutation_importance_df(
        clf_reduced, X_test_reduced, y_test,
        n_repeats=5,
        random_state=42,
        scoring='roc_auc'
    )
    
    # Plot comparison
    features = original_importance['feature'].tolist()
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    original_vals = [original_importance.loc[original_importance['feature'] == f, 'importance'].iloc[0] 
                    for f in features]
    
    calibrated_vals = [calibrated_importance.loc[calibrated_importance['feature'] == f, 'importance'].iloc[0] 
                      for f in features]
    
    ax.bar(x - width/2, original_vals, width, label='Original Model')
    ax.bar(x + width/2, calibrated_vals, width, label='Calibrated Model')
    
    ax.set_ylabel('Permutation Importance')
    ax.set_title('Feature Importance: Original vs. Calibrated Model')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    demonstrate_permutation_importance()
    demonstrate_probability_calibration()
    combined_importance_with_calibration()