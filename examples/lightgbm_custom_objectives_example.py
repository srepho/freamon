"""
Example of using custom objective functions with LightGBM models.

This example demonstrates:
1. Using the FocalLoss custom objective for imbalanced classification
2. Using the HuberLoss custom objective for robust regression
3. Using the TweedieLoss for non-negative regression targets
4. Using the MultiClassFocalLoss for imbalanced multi-class problems
5. Integrating custom objectives with the LightGBM API
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

import lightgbm as lgb
from freamon.modeling.lightgbm_objectives import (
    FocalLoss, 
    HuberLoss, 
    TweedieLoss, 
    MultiClassFocalLoss,
    register_with_lightgbm
)


def binary_classification_example():
    """
    Demonstrate FocalLoss for imbalanced binary classification.
    """
    print("\n" + "=" * 80)
    print("Binary Classification with Focal Loss")
    print("=" * 80)
    
    # Create imbalanced binary classification data
    X, y = make_classification(
        n_samples=10000, 
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2, 
        weights=[0.95, 0.05],  # Imbalanced classes
        random_state=42
    )
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Print class distribution
    print(f"Training class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
    
    # Create focal loss objective (alpha=0.25 means more focus on the minority class)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    obj_func, eval_func = register_with_lightgbm(focal_loss)
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Parameters for LightGBM
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',  # Will be overridden by custom objective
        'metric': 'auc',  # Still use AUC for evaluation
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4
    }
    
    print("\nTraining with standard binary log loss...")
    # Train standard model
    standard_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # Use our custom focal loss
    print("\nTraining with focal loss...")
    focal_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        fobj=obj_func,  # Custom objective
        verbose_eval=20
    )
    
    # Evaluate models
    y_pred_standard = standard_model.predict(X_test)
    y_pred_focal = focal_model.predict(X_test)
    
    # Calculate metrics
    auc_standard = roc_auc_score(y_test, y_pred_standard)
    auc_focal = roc_auc_score(y_test, y_pred_focal)
    
    print(f"\nTest AUC (Standard): {auc_standard:.4f}")
    print(f"Test AUC (Focal Loss): {auc_focal:.4f}")
    
    # Calculate metrics for the minority class
    minority_idx = y_test == 1
    auc_standard_minority = roc_auc_score(y_test[minority_idx], y_pred_standard[minority_idx]) if sum(minority_idx) > 0 else 0
    auc_focal_minority = roc_auc_score(y_test[minority_idx], y_pred_focal[minority_idx]) if sum(minority_idx) > 0 else 0
    
    print(f"\nMinority class AUC (Standard): {auc_standard_minority:.4f}")
    print(f"Minority class AUC (Focal Loss): {auc_focal_minority:.4f}")


def regression_example():
    """
    Demonstrate HuberLoss for robust regression.
    """
    print("\n" + "=" * 80)
    print("Regression with Huber Loss")
    print("=" * 80)
    
    # Load California housing dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='price')
    
    # Add some outliers to demonstrate robustness
    outlier_idx = np.random.choice(len(y), size=int(0.02 * len(y)), replace=False)
    y.iloc[outlier_idx] = y.iloc[outlier_idx] * 5  # Multiply by 5 to create outliers
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create Huber loss objective
    huber_loss = HuberLoss(delta=1.0)
    obj_func, eval_func = register_with_lightgbm(huber_loss)
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Parameters for LightGBM
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',  # Will be overridden by custom objective
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("\nTraining with standard L2 loss...")
    # Train standard model
    standard_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # Use our custom Huber loss
    print("\nTraining with Huber loss...")
    huber_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        fobj=obj_func,  # Custom objective
        verbose_eval=20
    )
    
    # Evaluate models
    y_pred_standard = standard_model.predict(X_test)
    y_pred_huber = huber_model.predict(X_test)
    
    # Calculate metrics
    rmse_standard = np.sqrt(mean_squared_error(y_test, y_pred_standard))
    rmse_huber = np.sqrt(mean_squared_error(y_test, y_pred_huber))
    
    print(f"\nTest RMSE (Standard): {rmse_standard:.4f}")
    print(f"Test RMSE (Huber Loss): {rmse_huber:.4f}")
    
    # Calculate metrics on outliers
    outlier_test_idx = np.intersect1d(outlier_idx, y_test.index)
    if len(outlier_test_idx) > 0:
        rmse_standard_outliers = np.sqrt(mean_squared_error(
            y_test.loc[outlier_test_idx], 
            y_pred_standard[np.isin(y_test.index, outlier_test_idx)]
        ))
        rmse_huber_outliers = np.sqrt(mean_squared_error(
            y_test.loc[outlier_test_idx], 
            y_pred_huber[np.isin(y_test.index, outlier_test_idx)]
        ))
        
        print(f"\nOutlier RMSE (Standard): {rmse_standard_outliers:.4f}")
        print(f"Outlier RMSE (Huber Loss): {rmse_huber_outliers:.4f}")
    
    # Plot predictions vs true values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_standard, alpha=0.5)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Standard L2 Loss')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_huber, alpha=0.5)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Huber Loss')
    
    plt.tight_layout()
    plt.savefig('huber_vs_standard.png')
    print("\nPlot saved as 'huber_vs_standard.png'")


def multiclass_example():
    """
    Demonstrate MultiClassFocalLoss for imbalanced multi-class classification.
    """
    print("\n" + "=" * 80)
    print("Multi-class Classification with Focal Loss")
    print("=" * 80)
    
    # Load Iris dataset (make it imbalanced)
    data = load_iris()
    X = data.data
    y = data.target
    
    # Create imbalanced dataset by removing many samples from class 1 and 2
    keep_idx = np.logical_or(y == 0, np.random.rand(len(y)) < 0.3)
    X = X[keep_idx]
    y = y[keep_idx]
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=data.feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Print class distribution
    print(f"Training class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
    
    # Create multi-class focal loss objective with class weights
    class_counts = pd.Series(y_train).value_counts().sort_index()
    # Inverse frequency weighting
    class_weights = 1 / (class_counts / class_counts.sum())
    class_weights = class_weights / class_weights.sum()
    
    focal_loss = MultiClassFocalLoss(
        alpha=class_weights.tolist(),  # Class weights
        gamma=2.0,
        num_classes=3
    )
    obj_func, eval_func = register_with_lightgbm(focal_loss)
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Parameters for LightGBM
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print("\nTraining with standard multi-class log loss...")
    # Train standard model
    standard_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # Use our custom focal loss
    print("\nTraining with multi-class focal loss...")
    focal_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_eval],
        early_stopping_rounds=10,
        fobj=obj_func,  # Custom objective
        verbose_eval=20
    )
    
    # Evaluate models
    y_pred_standard = standard_model.predict(X_test)
    y_pred_focal = focal_model.predict(X_test)
    
    # Convert probabilities to class predictions
    y_pred_standard_class = np.argmax(y_pred_standard, axis=1)
    y_pred_focal_class = np.argmax(y_pred_focal, axis=1)
    
    # Calculate accuracy
    acc_standard = accuracy_score(y_test, y_pred_standard_class)
    acc_focal = accuracy_score(y_test, y_pred_focal_class)
    
    print(f"\nTest Accuracy (Standard): {acc_standard:.4f}")
    print(f"Test Accuracy (Focal Loss): {acc_focal:.4f}")
    
    # Calculate per-class accuracy
    for cls in range(3):
        cls_idx = y_test == cls
        if sum(cls_idx) > 0:
            acc_standard_cls = accuracy_score(
                y_test[cls_idx], 
                y_pred_standard_class[cls_idx.values]
            )
            acc_focal_cls = accuracy_score(
                y_test[cls_idx], 
                y_pred_focal_class[cls_idx.values]
            )
            print(f"\nClass {cls} accuracy (Standard): {acc_standard_cls:.4f}")
            print(f"Class {cls} accuracy (Focal Loss): {acc_focal_cls:.4f}")


if __name__ == "__main__":
    binary_classification_example()
    regression_example()
    multiclass_example()