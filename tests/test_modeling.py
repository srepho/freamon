"""
Tests for the modeling module.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from freamon.modeling import (
    Model,
    ModelTrainer,
    calculate_metrics,
    create_model,
)


class TestModelCreation:
    """Test class for model creation functionality."""
    
    def test_create_model_sklearn(self):
        """Test creating a scikit-learn model."""
        model = create_model(
            model_type="sklearn",
            model_name="RandomForestClassifier",
            n_estimators=10,
            max_depth=5,
            random_state=42,
        )
        
        # Check that the model has the correct type
        assert model.model_type == "sklearn"
        
        # Check that the parameters were set correctly
        assert model.params["n_estimators"] == 10
        assert model.params["max_depth"] == 5
        assert model.params["random_state"] == 42
    
    def test_create_model_invalid_type(self):
        """Test creating a model with an invalid type."""
        with pytest.raises(ValueError):
            create_model(
                model_type="invalid",
                model_name="RandomForestClassifier",
            )
    
    def test_create_model_invalid_name(self):
        """Test creating a model with an invalid name."""
        with pytest.raises(ValueError):
            create_model(
                model_type="sklearn",
                model_name="InvalidModelName",
            )


class TestModelClass:
    """Test class for the Model class."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def regression_data(self):
        """Create regression data for testing."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_model_fit_predict(self, classification_data):
        """Test fitting and predicting with a model."""
        X, y = classification_data
        
        # Create a logistic regression model
        model = Model(
            model=LogisticRegression(random_state=42),
            model_type="sklearn",
            params={"random_state": 42},
        )
        
        # Fit the model
        model.fit(X, y)
        
        # Check that the model is fitted
        assert model.is_fitted
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Check that the predictions have the correct shape
        assert len(y_pred) == len(y)
    
    def test_get_feature_importance(self, regression_data):
        """Test getting feature importances."""
        X, y = regression_data
        
        # Rename columns to check feature names in importances
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create a random forest model (which has feature_importances_)
        model = create_model(
            model_type="sklearn",
            model_name="RandomForestRegressor",
            n_estimators=10,
            random_state=42,
        )
        
        # Fit the model
        model.fit(X, y)
        
        # Get feature importances
        importances = model.get_feature_importance()
        
        # Check that the importances are a Series
        assert isinstance(importances, pd.Series)
        
        # Check that the importances have the correct index
        assert set(importances.index) == set(X.columns)
        
        # Check that the importances sum to approximately 1
        assert abs(importances.sum() - 1.0) < 1e-6
    
    def test_model_save_load(self, regression_data):
        """Test saving and loading a model."""
        X, y = regression_data
        
        # Create and fit a model
        model = Model(
            model=LinearRegression(),
            model_type="sklearn",
            params={},
            feature_names=X.columns.tolist(),
        )
        model.fit(X, y)
        
        # Get predictions from the original model
        y_pred_original = model.predict(X)
        
        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
            model_path = temp.name
        
        try:
            model.save(model_path)
            
            # Load the model
            loaded_model = Model.load(model_path)
            
            # Check that the loaded model has the same attributes
            assert loaded_model.model_type == model.model_type
            assert loaded_model.params == model.params
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.is_fitted == model.is_fitted
            
            # Get predictions from the loaded model
            y_pred_loaded = loaded_model.predict(X)
            
            # Check that the predictions are the same
            np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)


class TestModelTrainer:
    """Test class for the ModelTrainer class."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification data for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df
    
    @pytest.fixture
    def regression_data(self):
        """Create regression data for testing."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df
    
    def test_model_trainer_classification(self, classification_data):
        """Test ModelTrainer with a classification problem."""
        # Split data into features and target
        X = classification_data.drop("target", axis=1)
        y = classification_data["target"]
        
        # Create a trainer for classification
        trainer = ModelTrainer(
            model_type="sklearn",
            model_name="LogisticRegression",
            problem_type="classification",
            random_state=42,
        )
        
        # Train the model
        metrics = trainer.train(X, y)
        
        # Check that metrics were calculated
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Check that the model is fitted
        assert trainer.model.is_fitted
        
        # Make predictions
        y_pred = trainer.predict(X)
        assert len(y_pred) == len(y)
        
        # Get feature importances
        importances = trainer.get_feature_importance()
        assert isinstance(importances, pd.Series)
        assert set(importances.index) == set(X.columns)
    
    def test_model_trainer_regression(self, regression_data):
        """Test ModelTrainer with a regression problem."""
        # Split data into features and target
        X = regression_data.drop("target", axis=1)
        y = regression_data["target"]
        
        # Create a trainer for regression
        trainer = ModelTrainer(
            model_type="sklearn",
            model_name="LinearRegression",
            problem_type="regression",
        )
        
        # Train the model
        metrics = trainer.train(X, y)
        
        # Check that metrics were calculated
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        
        # Check that the model is fitted
        assert trainer.model.is_fitted
        
        # Make predictions
        y_pred = trainer.predict(X)
        assert len(y_pred) == len(y)


class TestMetrics:
    """Test class for metric calculation."""
    
    def test_classification_metrics(self):
        """Test calculating classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, problem_type="classification")
        
        # Check that the expected metrics are present
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Check specific values
        assert metrics["accuracy"] == 4/6  # 4 correct out of 6
    
    def test_regression_metrics(self):
        """Test calculating regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8])
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, problem_type="regression")
        
        # Check that the expected metrics are present
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "explained_variance" in metrics
        
        # Check specific values
        expected_mse = ((1.1-1.0)**2 + (2.2-2.0)**2 + (2.9-3.0)**2 + (3.8-4.0)**2) / 4
        assert abs(metrics["mse"] - expected_mse) < 1e-6