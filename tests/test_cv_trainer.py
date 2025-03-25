"""
Tests for the cross-validation trainer and pipeline step.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from freamon.model_selection.cv_trainer import CrossValidationTrainer
from freamon.model_selection.cv_training_step import CrossValidationTrainingStep
from freamon.pipeline import Pipeline


def get_iris_data():
    """Get Iris dataset for testing."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y


def test_cross_validation_trainer_initialization():
    """Test initialization of CrossValidationTrainer."""
    trainer = CrossValidationTrainer(
        model_type="lightgbm",
        problem_type="classification",
        cv_strategy="kfold",
        n_splits=3,
        ensemble_method="best"
    )
    
    assert trainer.model_type == "lightgbm"
    assert trainer.problem_type == "classification"
    assert trainer.cv_strategy == "kfold"
    assert trainer.n_splits == 3
    assert trainer.ensemble_method == "best"
    assert trainer._is_fitted is False


def test_cross_validation_trainer_fit_predict():
    """Test fitting and prediction with CrossValidationTrainer."""
    X, y = get_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    trainer = CrossValidationTrainer(
        model_type="sklearn",
        problem_type="classification",
        cv_strategy="kfold",
        n_splits=3,
        ensemble_method="best",
        random_state=42
    )
    
    # Fit trainer
    trainer.fit(X_train, y_train)
    
    # Check that trainer is fitted
    assert trainer._is_fitted is True
    
    # Check that fold metrics are populated
    assert len(trainer.fold_metrics) > 0
    assert "accuracy" in trainer.fold_metrics
    assert len(trainer.fold_metrics["accuracy"]) == 3  # n_splits
    
    # Make predictions
    y_pred = trainer.predict(X_test)
    
    # Check predictions shape and type
    assert y_pred.shape == (X_test.shape[0],)
    assert isinstance(y_pred, np.ndarray)
    
    # Make probability predictions
    y_prob = trainer.predict_proba(X_test)
    
    # Check probability predictions shape
    assert y_prob.shape == (X_test.shape[0], 3)  # 3 classes in Iris
    
    # Get feature importances
    importances = trainer.get_feature_importances()
    
    # Check importances
    assert importances.shape[0] == X.shape[1]
    assert isinstance(importances, pd.Series)
    
    # Get CV results
    cv_results = trainer.get_cv_results()
    
    # Check CV results
    assert isinstance(cv_results, dict)
    assert "accuracy" in cv_results
    assert len(cv_results["accuracy"]) == 3  # n_splits


def test_cross_validation_trainer_ensemble_methods():
    """Test different ensemble methods in CrossValidationTrainer."""
    X, y = get_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    ensemble_methods = ["best", "average", "weighted", "stacking"]
    
    for method in ensemble_methods:
        # Initialize trainer
        trainer = CrossValidationTrainer(
            model_type="sklearn",
            problem_type="classification",
            cv_strategy="kfold",
            n_splits=3,
            ensemble_method=method,
            random_state=42
        )
        
        # Fit trainer
        trainer.fit(X_train, y_train)
        
        # Make predictions
        y_pred = trainer.predict(X_test)
        
        # Check predictions
        assert y_pred.shape == (X_test.shape[0],)


def test_cross_validation_training_step():
    """Test CrossValidationTrainingStep in a pipeline."""
    X, y = get_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add CV training step
    pipeline.add_step(
        CrossValidationTrainingStep(
            name="cv_training",
            model_type="sklearn",
            problem_type="classification",
            cv_strategy="kfold",
            n_splits=3,
            ensemble_method="best",
            random_state=42
        )
    )
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Get the CV step
    cv_step = pipeline.get_step("cv_training")
    
    # Check that step is fitted
    assert cv_step._is_fitted is True
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Check predictions
    assert y_pred.shape == (X_test.shape[0],)
    
    # Get CV results
    cv_results = cv_step.get_cv_results()
    
    # Check CV results
    assert "accuracy" in cv_results
    assert len(cv_results["accuracy"]) == 3  # n_splits