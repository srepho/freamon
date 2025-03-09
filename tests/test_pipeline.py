"""Tests for the pipeline module."""

import os
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from freamon.pipeline.pipeline import Pipeline, PipelineStep
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    EvaluationStep,
    ShapIQFeatureEngineeringStep
)


class MockPipelineStep(PipelineStep):
    """Mock pipeline step for testing."""
    
    def __init__(self, name, transform_func=None):
        super().__init__(name)
        self.transform_func = transform_func or (lambda X, **kwargs: X)
        self.fit_called = False
        
    def fit(self, X, y=None, **kwargs):
        self.fit_called = True
        self._is_fitted = True
        return self
        
    def transform(self, X, **kwargs):
        return self.transform_func(X, **kwargs)


class TestPipeline:
    """Tests for the Pipeline class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        return X, y
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        # Initialize with no steps
        pipeline = Pipeline()
        assert pipeline.steps == []
        
        # Initialize with steps
        step1 = MockPipelineStep("step1")
        step2 = MockPipelineStep("step2")
        pipeline = Pipeline([step1, step2])
        assert pipeline.steps == [step1, step2]
    
    def test_add_step(self):
        """Test adding steps to the pipeline."""
        pipeline = Pipeline()
        step1 = MockPipelineStep("step1")
        pipeline.add_step(step1)
        assert pipeline.steps == [step1]
        
        # Add another step
        step2 = MockPipelineStep("step2")
        pipeline.add_step(step2)
        assert pipeline.steps == [step1, step2]
        
        # Test method chaining
        step3 = MockPipelineStep("step3")
        result = pipeline.add_step(step3)
        assert result is pipeline
        assert pipeline.steps == [step1, step2, step3]
        
        # Test duplicate step name validation
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_step(MockPipelineStep("step1"))
    
    def test_fit_transform(self, sample_data):
        """Test fitting and transforming with the pipeline."""
        X, y = sample_data
        
        # Create pipeline with mock steps
        step1 = MockPipelineStep("step1", lambda X, **kwargs: X * 2)
        step2 = MockPipelineStep("step2", lambda X, **kwargs: X + 1)
        pipeline = Pipeline([step1, step2])
        
        # Fit the pipeline
        result = pipeline.fit(X, y)
        assert result is pipeline
        assert step1.fit_called
        assert step2.fit_called
        
        # Check step outputs
        assert f"step1_input" in pipeline._step_outputs
        assert f"step1_output" in pipeline._step_outputs
        assert f"step2_input" in pipeline._step_outputs
        assert f"step2_output" in pipeline._step_outputs
        assert "final_output" in pipeline._step_outputs
        
        # Test transform
        transformed = pipeline.transform(X)
        # First multiplied by 2, then added 1
        expected = X * 2 + 1
        pd.testing.assert_frame_equal(transformed, expected)
        
        # Test fit_transform
        pipeline = Pipeline([step1, step2])
        transformed = pipeline.fit_transform(X, y)
        pd.testing.assert_frame_equal(transformed, expected)
    
    def test_get_step_output(self, sample_data):
        """Test retrieving step outputs."""
        X, y = sample_data
        
        # Create pipeline with mock steps
        step1 = MockPipelineStep("step1", lambda X, **kwargs: X * 2)
        pipeline = Pipeline([step1])
        pipeline.fit(X, y)
        
        # Get step output
        output = pipeline.get_step_output("step1", "output")
        pd.testing.assert_frame_equal(output, X * 2)
        
        # Get step input
        input_df = pipeline.get_step_output("step1", "input")
        pd.testing.assert_frame_equal(input_df, X)
        
        # Test invalid output
        with pytest.raises(KeyError, match="not found"):
            pipeline.get_step_output("nonexistent")
    
    def test_save_load(self, sample_data):
        """Test saving and loading the pipeline."""
        X, y = sample_data
        
        # Create and fit pipeline
        step1 = MockPipelineStep("step1", lambda X, **kwargs: X * 2)
        step2 = MockPipelineStep("step2", lambda X, **kwargs: X + 1)
        pipeline = Pipeline([step1, step2])
        pipeline.fit(X, y)
        
        # Get predictions before saving
        before_preds = pipeline.transform(X)
        
        # Save pipeline
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save(save_path)
            
            # Check files were created
            assert (save_path / "metadata.json").exists()
            assert (save_path / "step1.pkl").exists()
            assert (save_path / "step2.pkl").exists()
            
            # Load pipeline
            loaded = Pipeline().load(save_path)
            
            # Check steps were loaded correctly
            assert len(loaded.steps) == 2
            assert loaded.steps[0].name == "step1"
            assert loaded.steps[1].name == "step2"
            
            # Check predictions match
            after_preds = loaded.transform(X)
            pd.testing.assert_frame_equal(before_preds, after_preds)


class TestFeatureEngineeringStep:
    """Tests for the FeatureEngineeringStep class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data[:50], columns=data.feature_names)
        y = pd.Series(data.target[:50], name="target")
        return X, y
    
    def test_add_operation(self):
        """Test adding operations to the feature engineering step."""
        step = FeatureEngineeringStep("feature_eng")
        
        # Add an operation
        result = step.add_operation("add_polynomial_features", columns=["mean radius"], degree=2)
        assert result is step
        assert len(step.operations) == 1
        assert step.operations[0]["method"] == "add_polynomial_features"
        assert step.operations[0]["params"]["columns"] == ["mean radius"]
        assert step.operations[0]["params"]["degree"] == 2
        
        # Add another operation
        step.add_operation("add_binned_features", columns=["mean texture"], n_bins=5)
        assert len(step.operations) == 2
        assert step.operations[1]["method"] == "add_binned_features"
    
    def test_fit_transform(self, sample_data):
        """Test fitting and transforming with the feature engineering step."""
        X, y = sample_data
        
        # Create step with polynomial features operation
        step = FeatureEngineeringStep("feature_eng")
        step.add_operation(
            "add_polynomial_features",
            columns=["mean radius", "mean texture"],
            degree=2,
            interaction_only=True
        )
        
        # Fit the step
        result = step.fit(X, y)
        assert result is step
        assert step.is_fitted
        
        # Transform the data
        transformed = step.transform(X)
        
        # Check new columns were created
        new_cols = list(set(transformed.columns) - set(X.columns))
        assert len(new_cols) > 0
        assert "mean radius_x_mean texture" in transformed.columns
        
        # Test dropping original columns
        step = FeatureEngineeringStep("feature_eng", drop_original=True)
        step.add_operation(
            "add_polynomial_features",
            columns=["mean radius", "mean texture"],
            degree=2,
            interaction_only=True
        )
        step.fit(X, y)
        transformed = step.transform(X)
        
        # Check original columns were dropped
        assert "mean radius" not in transformed.columns
        assert "mean texture" not in transformed.columns


class TestModelTrainingStep:
    """Tests for the ModelTrainingStep class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data[:50], columns=data.feature_names)
        y = pd.Series(data.target[:50], name="target")
        return X, y
    
    def test_fit_predict(self, sample_data):
        """Test fitting and predicting with the model training step."""
        X, y = sample_data
        
        # Create step with LightGBM model
        step = ModelTrainingStep(
            "model",
            model_type="lightgbm",
            problem_type="classification",
            hyperparameters={"n_estimators": 10}
        )
        
        # Fit the step
        result = step.fit(X, y)
        assert result is step
        assert step.is_fitted
        
        # Transform should return the input unchanged
        transformed = step.transform(X)
        pd.testing.assert_frame_equal(transformed, X)
        
        # Test predict
        predictions = step.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        
        # Test predict_proba
        probas = step.predict_proba(X)
        assert isinstance(probas, np.ndarray)
        assert probas.shape[0] == len(X)
        assert probas.shape[1] == 2  # Binary classification
        
        # Test get_feature_importances
        importances = step.get_feature_importances()
        assert isinstance(importances, pd.DataFrame)
        assert len(importances) > 0
        assert "feature" in importances.columns
        assert "importance" in importances.columns


class TestEvaluationStep:
    """Tests for the EvaluationStep class."""
    
    def test_evaluate_classification(self):
        """Test evaluation for classification problems."""
        # Create evaluation step
        step = EvaluationStep(
            "evaluation",
            metrics=["accuracy", "precision", "recall", "f1"],
            problem_type="classification"
        )
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], 
                           [0.6, 0.4], [0.4, 0.6], [0.3, 0.7]])
        
        # Evaluate
        results = step.evaluate(y_true, y_pred, y_prob)
        
        # Check results
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1"] <= 1
    
    def test_evaluate_regression(self):
        """Test evaluation for regression problems."""
        # Create evaluation step
        step = EvaluationStep(
            "evaluation",
            metrics=["mse", "rmse", "mae", "r2"],
            problem_type="regression"
        )
        
        # Create sample data
        y_true = np.array([3.0, 1.0, 2.0, 7.0, 5.0])
        y_pred = np.array([2.5, 0.5, 2.0, 8.0, 4.5])
        
        # Evaluate
        results = step.evaluate(y_true, y_pred)
        
        # Check results
        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results
        assert results["mse"] >= 0
        assert results["rmse"] >= 0
        assert results["mae"] >= 0
        assert results["r2"] <= 1
        assert results["rmse"] == np.sqrt(results["mse"])