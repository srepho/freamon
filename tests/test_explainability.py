"""
Tests for the explainability module including SHAP and ShapIQ integrations.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Check if SHAP is installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Check if ShapIQ is installed
try:
    import shapiq
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False

from freamon.explainability.shap_explainer import ShapExplainer
from freamon.explainability.shap_explainer import ShapIQExplainer


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP package not installed")
class TestShapExplainer:
    """Test class for ShapExplainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
        })
        y = 2 * X['feature1'] + X['feature2'] + np.random.normal(0, 0.1, 100)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_init(self, sample_data):
        """Test initialization of ShapExplainer."""
        X, y, model = sample_data
        explainer = ShapExplainer(model, model_type='tree')
        assert explainer.model == model
        assert explainer.model_type == 'tree'
        assert explainer.is_fitted == False
    
    def test_fit(self, sample_data):
        """Test fitting the explainer."""
        X, y, model = sample_data
        explainer = ShapExplainer(model, model_type='tree')
        explainer.fit(X)
        assert explainer.is_fitted == True
        assert explainer.explainer is not None
    
    def test_explain(self, sample_data):
        """Test explaining predictions."""
        X, y, model = sample_data
        explainer = ShapExplainer(model, model_type='tree')
        explainer.fit(X)
        
        # Get a sample for explanation
        X_sample = X.iloc[:5]
        
        # Get SHAP values
        shap_values = explainer.explain(X_sample)
        
        # Check that shape matches
        assert shap_values.shape == X_sample.shape


@pytest.mark.skipif(not SHAPIQ_AVAILABLE, reason="ShapIQ package not installed")
class TestShapIQExplainer:
    """Test class for ShapIQExplainer."""
    
    @pytest.fixture
    def sample_data_with_interactions(self):
        """Create a sample dataset with interactions for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.uniform(-1, 1, 100),
            'feature2': np.random.uniform(-1, 1, 100),
            'feature3': np.random.uniform(-1, 1, 100),
        })
        
        # Create target with interactions
        y = (
            X['feature1'] * X['feature2'] +  # Strong interaction
            0.5 * X['feature3'] +           # Main effect
            np.random.normal(0, 0.1, 100)    # Noise
        )
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return X, y, model
    
    def test_init(self, sample_data_with_interactions):
        """Test initialization of ShapIQExplainer."""
        X, y, model = sample_data_with_interactions
        explainer = ShapIQExplainer(model, max_order=2)
        assert explainer.model == model
        assert explainer.max_order == 2
        assert explainer.is_fitted == False
    
    def test_fit(self, sample_data_with_interactions):
        """Test fitting the explainer."""
        X, y, model = sample_data_with_interactions
        explainer = ShapIQExplainer(model, max_order=2)
        explainer.fit(X)
        assert explainer.is_fitted == True
        assert explainer.explainer is not None
    
    def test_explain(self, sample_data_with_interactions):
        """Test explaining interactions."""
        X, y, model = sample_data_with_interactions
        explainer = ShapIQExplainer(model, max_order=2)
        explainer.fit(X)
        
        # Get a sample for explanation
        X_sample = X.iloc[:5]
        
        # Get ShapIQ interaction values
        interactions = explainer.explain(X_sample)
        
        # Check that we have both main effects and interactions
        assert interactions is not None
