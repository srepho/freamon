"""
Tests for the ShapIQ-based feature engineering.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Check if ShapIQ is installed
try:
    import shapiq
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False

# Skip all tests in this module if ShapIQ is not available
pytestmark = pytest.mark.skipif(
    not SHAPIQ_AVAILABLE, 
    reason="ShapIQ package not installed"
)

# Conditionally import ShapIQFeatureEngineer to avoid ImportError
if SHAPIQ_AVAILABLE:
    from freamon.features.shapiq_engineer import ShapIQFeatureEngineer


# Only define the ShapIQ test class if ShapIQ is available
if SHAPIQ_AVAILABLE:
    class TestShapIQFeatureEngineer:
        """Test class for ShapIQFeatureEngineer."""
        
        @pytest.fixture
        def sample_data_with_interactions(self):
            """Create a sample dataset with interactions for testing."""
            np.random.seed(42)
            X = pd.DataFrame({
                'feature1': np.random.uniform(-1, 1, 100),
                'feature2': np.random.uniform(-1, 1, 100),
                'feature3': np.random.uniform(-1, 1, 100),
                'feature4': np.random.uniform(-1, 1, 100),
            })
            
            # Create target with interactions
            y = (
                X['feature1'] * X['feature2'] +       # Strong interaction
                0.5 * X['feature2'] * X['feature4'] +  # Medium interaction
                0.3 * X['feature3'] +                 # Main effect
                np.random.normal(0, 0.1, 100)          # Noise
            )
            
            # Train a simple model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            return X, y, model
        
        def test_init(self, sample_data_with_interactions):
            """Test initialization of ShapIQFeatureEngineer."""
            X, y, model = sample_data_with_interactions
            engineer = ShapIQFeatureEngineer(
                model=model,
                X=X,
                y=y,
                max_order=2,
                threshold=0.01,
                max_interactions=5
            )
            
            assert engineer.model == model
            assert engineer.max_order == 2
            assert engineer.threshold == 0.01
            assert engineer.max_interactions == 5
        
        def test_detect_interactions(self, sample_data_with_interactions, monkeypatch):
            """Test detecting interactions."""
            X, y, model = sample_data_with_interactions
            engineer = ShapIQFeatureEngineer(
                model=model,
                X=X,
                y=y,
                max_order=2,
                threshold=0.01,
                max_interactions=5
            )
            
            # Mock the ShapIQExplainer.explain method to return synthetic results
            def mock_explain(*args, **kwargs):
                from types import SimpleNamespace
                # Create a mock object with properties needed for the test
                mock_obj = SimpleNamespace()
                mock_obj.values = np.array([[[0.1, 0.2, 0.01, 0.05],
                                           [0.2, 0.0, 0.3, 0.02],
                                           [0.01, 0.3, 0.0, 0.4],
                                           [0.05, 0.02, 0.4, 0.0]]])
                
                # Add get_order method
                def get_order(order):
                    return mock_obj
                
                mock_obj.get_order = get_order
                return mock_obj
            
            # Apply the monkeypatch
            from freamon.explainability.shap_explainer import ShapIQExplainer
            monkeypatch.setattr(ShapIQExplainer, "explain", mock_explain)
            
            # Detect interactions
            interactions = engineer.detect_interactions()
            
            # Check that we detected some interactions
            assert len(interactions) > 0
            
            # Check that detected interactions are pairs of feature names
            for feature1, feature2 in interactions:
                assert feature1 in X.columns
                assert feature2 in X.columns
        
        def test_create_features(self, sample_data_with_interactions, monkeypatch):
            """Test creating interaction features."""
            X, y, model = sample_data_with_interactions
            engineer = ShapIQFeatureEngineer(
                model=model,
                X=X,
                y=y,
                max_order=2,
                threshold=0.01,
                max_interactions=2
            )
            
            # Mock the detect_interactions method to return synthetic interactions
            def mock_detect_interactions(*args, **kwargs):
                # Return a predefined list of interactions
                mock_interactions = [
                    ('feature1', 'feature2'),
                    ('feature2', 'feature4')
                ]
                # Also set the instance variables directly
                engineer.detected_interactions = mock_interactions
                engineer.interaction_strengths = {
                    ('feature1', 'feature2'): 0.5,
                    ('feature2', 'feature4'): 0.3
                }
                return mock_interactions
                
            # Apply the monkeypatch
            monkeypatch.setattr(engineer, "detect_interactions", mock_detect_interactions)
            
            # First detect interactions (this will use our mock)
            engineer.detect_interactions()
            
            # Create features
            df_with_features = engineer.create_features(operations=['multiply'])
            
            # Check that we have new columns
            assert len(df_with_features.columns) > len(X.columns)
            
            # Check that new columns start with 'shapiq'
            new_cols = set(df_with_features.columns) - set(X.columns)
            for col in new_cols:
                assert col.startswith('shapiq')
        
        def test_pipeline(self, sample_data_with_interactions, monkeypatch):
            """Test the full pipeline."""
            X, y, model = sample_data_with_interactions
            engineer = ShapIQFeatureEngineer(
                model=model,
                X=X,
                y=y,
                max_order=2,
                threshold=0.01,
                max_interactions=2
            )
            
            # Mock the detect_interactions method
            def mock_detect_interactions(*args, **kwargs):
                # Return a predefined list of interactions
                mock_interactions = [
                    ('feature1', 'feature2'),
                    ('feature2', 'feature4')
                ]
                # Also set the instance variables directly
                engineer.detected_interactions = mock_interactions
                engineer.interaction_strengths = {
                    ('feature1', 'feature2'): 0.5,
                    ('feature2', 'feature4'): 0.3
                }
                return mock_interactions
                
            # Apply the monkeypatch
            monkeypatch.setattr(engineer, "detect_interactions", mock_detect_interactions)
            
            # Run the pipeline
            df_with_features, report = engineer.pipeline(operations=['multiply'])
            
            # Check that we have new columns
            assert len(df_with_features.columns) > len(X.columns)
            
            # Check that the report has the expected fields
            assert 'num_interactions' in report
            assert 'interactions' in report
else:
    # Create an empty test class for pytest discovery
    class TestShapIQFeatureEngineer:
        """Placeholder for ShapIQ tests when package is not available."""
        
        @pytest.fixture
        def sample_data_with_interactions(self):
            """Create a sample dataset with interactions for testing."""
            pytest.skip("ShapIQ package not installed")
            return None
            
        def test_placeholder(self):
            """This test is a placeholder when ShapIQ is not available."""
            pytest.skip("ShapIQ package not installed")
        
        def test_init(self, sample_data_with_interactions):
            """Placeholder for init test."""
            pytest.skip("ShapIQ package not installed")
        
        def test_detect_interactions(self, sample_data_with_interactions):
            """Placeholder for detect_interactions test."""
            pytest.skip("ShapIQ package not installed")
        
        def test_create_features(self, sample_data_with_interactions):
            """Placeholder for create_features test."""
            pytest.skip("ShapIQ package not installed")
        
        def test_pipeline(self, sample_data_with_interactions):
            """Placeholder for pipeline test."""
            pytest.skip("ShapIQ package not installed")
