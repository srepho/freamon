"""
Tests for Model class feature mismatch handling.

This module tests the feature mismatch handling behavior added to the Model class,
specifically how it handles missing features during prediction.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import warnings

from freamon.modeling.model import Model
from freamon.modeling.autoflow import AutoModelFlow


class TestModelFeatureMismatch:
    """Test class for Model feature mismatch handling."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Create a dataframe with common features and similarity features
        X_train = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
            "feature_3": np.random.normal(0, 1, 100),
            "Sim-feature_1_0.5": np.random.normal(0, 1, 100),  # Similarity feature
            "Sim-feature_2_0.75": np.random.normal(0, 1, 100),  # Similarity feature
        })
        
        # Create binary target for classification
        y_train_cls = (X_train["feature_1"] + X_train["feature_2"] > 0).astype(int)
        
        # Create continuous target for regression
        y_train_reg = X_train["feature_1"] + 2 * X_train["feature_2"] + np.random.normal(0, 0.1, 100)
        
        # Create test set with missing similarity features
        X_test_missing = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 20),
            "feature_2": np.random.normal(0, 1, 20),
            "feature_3": np.random.normal(0, 1, 20),
            # Note: Similarity features intentionally missing
        })
        
        # Create test set with all features
        X_test_complete = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 20),
            "feature_2": np.random.normal(0, 1, 20),
            "feature_3": np.random.normal(0, 1, 20),
            "Sim-feature_1_0.5": np.random.normal(0, 1, 20),
            "Sim-feature_2_0.75": np.random.normal(0, 1, 20),
        })
        
        return {
            "X_train": X_train, 
            "y_train_cls": y_train_cls,
            "y_train_reg": y_train_reg, 
            "X_test_missing": X_test_missing,
            "X_test_complete": X_test_complete
        }
    
    def test_sklearn_classifier_missing_features(self, sample_data):
        """Test classification with sklearn model and missing features."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        X_test_missing = sample_data["X_test_missing"]
        
        # Create and fit a sklearn model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Model(
            model=clf,
            model_type='sklearn',
            params={'n_estimators': 10, 'random_state': 42}
        )
        model.fit(X_train, y_train)
        
        # Test prediction with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure all warnings are always triggered
            predictions = model.predict(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_missing)
        
        # Test predict_proba with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            probabilities = model.predict_proba(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify probabilities have the right shape
        assert probabilities.shape[0] == len(X_test_missing)
        assert probabilities.shape[1] == 2  # Binary classification
    
    def test_sklearn_regressor_missing_features(self, sample_data):
        """Test regression with sklearn model and missing features."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_reg"]
        X_test_missing = sample_data["X_test_missing"]
        
        # Create and fit a sklearn model
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        model = Model(
            model=reg,
            model_type='sklearn',
            params={'n_estimators': 10, 'random_state': 42}
        )
        model.fit(X_train, y_train)
        
        # Test prediction with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_missing)
    
    def test_lightgbm_classifier_missing_features(self, sample_data):
        """Test classification with lightgbm model and missing features."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        X_test_missing = sample_data["X_test_missing"]
        
        # Create and fit a lightgbm model
        lgb_clf = lgb.LGBMClassifier(n_estimators=10, random_state=42)
        model = Model(
            model=lgb_clf,
            model_type='lightgbm',
            params={'n_estimators': 10, 'random_state': 42}
        )
        model.fit(X_train, y_train)
        
        # Test prediction with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_missing)
        
        # Test predict_proba with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            probabilities = model.predict_proba(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify probabilities have the right shape
        assert probabilities.shape[0] == len(X_test_missing)
        assert probabilities.shape[1] == 2  # Binary classification
    
    def test_lightgbm_regressor_missing_features(self, sample_data):
        """Test regression with lightgbm model and missing features."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_reg"]
        X_test_missing = sample_data["X_test_missing"]
        
        # Create and fit a lightgbm model
        lgb_reg = lgb.LGBMRegressor(n_estimators=10, random_state=42)
        model = Model(
            model=lgb_reg,
            model_type='lightgbm',
            params={'n_estimators': 10, 'random_state': 42}
        )
        model.fit(X_train, y_train)
        
        # Test prediction with missing features
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict(X_test_missing)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_missing)
    
    def test_feature_order_preserved(self, sample_data):
        """Test that feature order is preserved in predictions."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        X_test_missing = sample_data["X_test_missing"]
        
        # Create a model with extra sensitivity to feature order
        # (Use a simple model for testing)
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        model = Model(
            model=clf,
            model_type='sklearn',
            params={'n_estimators': 5, 'random_state': 42}
        )
        
        # Mix up the column order in the training set
        mixed_columns = ['feature_3', 'Sim-feature_2_0.75', 'feature_1', 'Sim-feature_1_0.5', 'feature_2']
        X_train_mixed = X_train[mixed_columns]
        
        model.fit(X_train_mixed, y_train)
        
        # Feature names should be in the mixed order
        assert model.feature_names == mixed_columns
        
        # Test prediction with missing features and different column order
        X_test_different_order = X_test_missing[['feature_3', 'feature_2', 'feature_1']]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict(X_test_different_order)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_different_order)
    
    def test_get_set_params_compatibility(self, sample_data):
        """Test compatibility of get_params and set_params with scikit-learn."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        
        # Create model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Model(
            model=clf,
            model_type='sklearn',
            params={'n_estimators': 10, 'random_state': 42}
        )
        
        # Get parameters
        params = model.get_params()
        
        # Check that model parameters were included
        assert 'model__n_estimators' in params
        assert params['model__n_estimators'] == 10
        
        # Set new parameters
        model.set_params(model__n_estimators=20)
        
        # Verify parameter was changed
        assert model.model.n_estimators == 20
        
        # Fit and ensure it works with new parameters
        model.fit(X_train, y_train)
        
        # Test with missing features to ensure that still works
        X_test_missing = sample_data["X_test_missing"]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            predictions = model.predict(X_test_missing)
            assert len(predictions) == len(X_test_missing)
    
    def test_scikit_learn_clone_compatibility(self, sample_data):
        """Test compatibility with scikit-learn's clone function."""
        from sklearn.base import clone
        
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        
        # Create model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Model(
            model=clf,
            model_type='sklearn',
            params={'n_estimators': 10, 'random_state': 42}
        )
        
        # Clone the model
        cloned_model = clone(model)
        
        # Verify that cloned model has the same parameters
        assert cloned_model.model_type == model.model_type
        assert cloned_model.params == model.params
        
        # Fit the cloned model
        cloned_model.fit(X_train, y_train)
        
        # Test with missing features
        X_test_missing = sample_data["X_test_missing"]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            predictions = cloned_model.predict(X_test_missing)
            assert len(predictions) == len(X_test_missing)
    
    def test_partial_feature_overlap(self, sample_data):
        """Test with partial feature overlap between training and prediction."""
        X_train = sample_data["X_train"]
        y_train = sample_data["y_train_cls"]
        
        # Create and fit a model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Model(
            model=clf,
            model_type='sklearn',
            params={'n_estimators': 10, 'random_state': 42}
        )
        model.fit(X_train, y_train)
        
        # Create test data with some features missing and some extra features
        X_test_partial = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 20),
            "feature_3": np.random.normal(0, 1, 20),
            "Sim-feature_1_0.5": np.random.normal(0, 1, 20),
            "extra_feature": np.random.normal(0, 1, 20),  # Extra feature not in training
        })
        
        # Test prediction
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict(X_test_partial)
            
            # Verify a warning was issued
            assert len(w) > 0
            assert any("missing features during prediction" in str(warning.message) for warning in w)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_partial)
    
    def test_autoflow_with_missing_features(self, sample_data):
        """Test AutoModelFlow with missing features."""
        # Create a dataset with similarity features
        X_train_full = sample_data["X_train"].copy()
        y_train = sample_data["y_train_cls"].copy()
        
        # Create AutoModelFlow instance
        autoflow = AutoModelFlow(
            model_type="lightgbm",
            problem_type="classification",
            hyperparameter_tuning=False,  # Skip tuning for test speed
            random_state=42,
            verbose=False
        )
        
        # Analyze and prepare the dataset
        autoflow.analyze_dataset(
            df=pd.concat([X_train_full, pd.DataFrame({"target": y_train})], axis=1),
            target_column="target"
        )
        
        # Train the model
        X_train_subset = X_train_full.iloc[:80]  # Use subset for faster training
        y_train_subset = y_train.iloc[:80]
        autoflow.fit(X_train_subset, y_train_subset)
        
        # Test prediction with data missing all similarity features
        X_test_missing = sample_data["X_test_missing"]
        
        # If the model was trained with similarity features, predict should work
        # with or without them
        predictions = autoflow.predict(X_test_missing)
        
        # Verify predictions have the right shape
        assert len(predictions) == len(X_test_missing)


if __name__ == "__main__":
    pytest.main()