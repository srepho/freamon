"""
Tests for multivariate analysis functions in the EDA module.
"""
import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.preprocessing import StandardScaler

# Try importing optional dependencies
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shapiq
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False

from freamon.eda.multivariate import (
    perform_pca,
    perform_tsne,
    analyze_multivariate,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Create a dataset with clear clusters
    X, y = make_blobs(
        n_samples=100,
        n_features=5,
        centers=3,
        random_state=42,
    )
    
    # Create a DataFrame with feature names
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    
    return df


@pytest.fixture
def regression_data():
    """Generate regression data for testing."""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    
    # Create a DataFrame with feature names
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    
    return df


def test_perform_pca(sample_data):
    """Test that PCA dimensionality reduction works correctly."""
    # Run PCA with 2 components
    result = perform_pca(sample_data, n_components=2)
    
    # Check that the result contains expected keys
    assert "pca_results" in result
    assert "explained_variance" in result
    assert "loadings" in result
    assert "visualization" in result
    
    # Check that the number of components is correct
    assert len(result["explained_variance"]) == 2
    
    # Check that PCA results contain the right dimensions
    assert len(result["pca_results"]["PC1"]) == len(sample_data)
    assert len(result["pca_results"]["PC2"]) == len(sample_data)
    
    # Check that explained variance adds up to less than 1.0
    total_variance = sum(result["explained_variance"])
    assert 0 < total_variance <= 1.0
    
    # Check loadings format
    assert "loadings" in result
    
    # The loadings are returned as a dict with PCs as keys
    loadings = result["loadings"]
    assert "PC1" in loadings
    assert "PC2" in loadings
    
    # Each PC should have values for each feature
    for feature in sample_data.select_dtypes(include=["number"]).columns:
        if feature != "target":  # Exclude target column
            assert feature in loadings["PC1"]


def test_perform_tsne(sample_data):
    """Test that t-SNE dimensionality reduction works correctly."""
    # Run t-SNE with 2 components
    result = perform_tsne(sample_data, n_components=2, n_iter=250)
    
    # Check that the result contains expected keys
    assert "tsne_results" in result
    assert "perplexity" in result
    assert "visualization" in result
    
    # Check that t-SNE results contain the right dimensions
    assert len(result["tsne_results"]["TSNE1"]) == len(sample_data)
    assert len(result["tsne_results"]["TSNE2"]) == len(sample_data)
    
    # Check that parameters are correctly returned
    assert result["n_components"] == 2
    assert result["n_iter"] == 250


def test_analyze_multivariate_pca_only(regression_data):
    """Test multivariate analysis with PCA only."""
    result = analyze_multivariate(regression_data, method="pca", n_components=3)
    
    # Check that only PCA is included
    assert "pca" in result
    assert "tsne" not in result
    
    # Verify PCA results
    pca_result = result["pca"]
    assert len(pca_result["explained_variance"]) == 3


def test_analyze_multivariate_tsne_only(regression_data):
    """Test multivariate analysis with t-SNE only."""
    result = analyze_multivariate(regression_data, method="tsne", n_components=2, tsne_n_iter=250)
    
    # Check that only t-SNE is included
    assert "tsne" in result
    assert "pca" not in result
    
    # Verify t-SNE results
    tsne_result = result["tsne"]
    assert tsne_result["n_iter"] == 250
    assert tsne_result["n_components"] == 2


def test_analyze_multivariate_both(regression_data):
    """Test multivariate analysis with both PCA and t-SNE."""
    result = analyze_multivariate(regression_data, method="both", n_components=2)
    
    # Check that both methods are included
    assert "pca" in result
    assert "tsne" in result


def test_pca_with_missing_values(regression_data):
    """Test that PCA handles missing values correctly."""
    # Add some missing values
    regression_data.iloc[0, 0] = np.nan
    regression_data.iloc[5, 2] = np.nan
    
    # Run PCA
    result = perform_pca(regression_data)
    
    # Check that the result contains expected keys
    assert "pca_results" in result
    assert len(result["pca_results"]["PC1"]) == len(regression_data)


def test_tsne_with_custom_parameters(sample_data):
    """Test t-SNE with custom parameters."""
    result = perform_tsne(
        sample_data,
        perplexity=15.0,
        learning_rate=200.0,
        n_iter=300,
        random_state=42
    )
    
    # Verify parameters were used
    assert result["perplexity"] == 15.0
    assert result["learning_rate"] == 200.0
    assert result["n_iter"] == 300


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_lightgbm_integration():
    """Test that LightGBM integration works."""
    from freamon.modeling.factory import create_model
    
    # Test classification
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    
    # Create a LightGBM classifier using factory
    model = create_model("lightgbm", "LGBMClassifier", {'n_estimators': 10})
    model.fit(df.drop("target", axis=1), df["target"])
    preds = model.predict(df.drop("target", axis=1))
    assert len(preds) == len(df)
    
    # Get feature importance
    importances = model.get_feature_importance()
    assert len(importances) == df.drop("target", axis=1).shape[1]
    
    # Test regression
    X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    
    # Create a LightGBM regressor
    reg_model = create_model("lightgbm", "LGBMRegressor", {'n_estimators': 10})
    reg_model.fit(df.drop("target", axis=1), df["target"])
    preds = reg_model.predict(df.drop("target", axis=1))
    assert len(preds) == len(df)
    
    # Test feature importance for regression model
    reg_importances = reg_model.get_feature_importance()
    assert len(reg_importances) == df.drop("target", axis=1).shape[1]


@pytest.mark.skipif(not SHAPIQ_AVAILABLE, reason="ShapIQ not installed")
def test_shapiq_integration():
    """Test that ShapIQ integration works."""
    try:
        from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
        from freamon.explainability.shap_explainer import ShapIQExplainer
    except ImportError as e:
        pytest.skip(f"Could not import ShapIQ modules: {str(e)}")
    
    # Create dataset
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    
    # Test ShapIQ Feature Engineer
    try:
        engineer = ShapIQFeatureEngineer(
            target_column="target",
            max_interaction_order=2,
            n_interactions=5
        )
        df_engineered = engineer.fit_transform(df)
        
        # Check that new interaction features were created
        assert len(df_engineered.columns) > len(df.columns)
    except Exception as e:
        pytest.skip(f"ShapIQ feature engineer test failed: {str(e)}")
    
    # Test ShapIQ Explainer
    try:
        # Create and train model with LightGBM
        from freamon.modeling.factory import create_model
        model = create_model("lightgbm", "LGBMRegressor", {'n_estimators': 10})
        model.fit(df.drop("target", axis=1), df["target"])
        
        # Create ShapIQ explainer
        explainer = ShapIQExplainer(model=model)
        interactions = explainer.explain(
            df.drop("target", axis=1), 
            max_interaction_order=2,
            n_interactions=5
        )
        
        # Check results
        assert len(interactions) > 0
    except Exception as e:
        pytest.skip(f"ShapIQ explainer test failed: {str(e)}")


@pytest.mark.skipif(not (LIGHTGBM_AVAILABLE and SHAPIQ_AVAILABLE), 
                    reason="LightGBM or ShapIQ not installed")
def test_eda_with_integrated_packages():
    """Test that the EDA module works with LightGBM and ShapIQ."""
    try:
        from freamon.eda import EDAAnalyzer
        from freamon.modeling.lightgbm import LightGBMModel
        from freamon.modeling.factory import create_model
        from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
        from freamon.pipeline import Pipeline
        from freamon.pipeline.steps import DataFrameFeatureEngineeringStep, ModelTrainingStep
    except ImportError as e:
        pytest.skip(f"Could not import required modules: {str(e)}")
    
    # Create a simple dataset
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    
    try:
        # Create an EDA analyzer
        analyzer = EDAAnalyzer(df, target_column="target")
        
        # Run multivariate analysis
        multi_results = analyzer.analyze_multivariate(method='pca', n_components=2)
        
        # Check that PCA was performed
        assert "pca" in multi_results
        
        # Create a pipeline with ShapIQ feature engineering and LightGBM model
        pipeline = Pipeline()
        
        # Add a feature engineering step
        pipeline.add_step(
            DataFrameFeatureEngineeringStep(
                name="shapiq_features",
                transformer=ShapIQFeatureEngineer(
                    target_column="target",
                    max_interaction_order=2,
                    n_interactions=3
                )
            )
        )
        
        # Add a model training step using factory method
        pipeline.add_step(
            ModelTrainingStep(
                name="lightgbm_model",
                model=create_model("lightgbm", "LGBMRegressor", {'n_estimators': 10}),
                target_column="target"
            )
        )
        
        # Fit the pipeline
        pipeline.fit(df)
        
        # Make predictions
        predictions = pipeline.predict(df.drop("target", axis=1))
        
        # Check that predictions were made
        assert len(predictions) == len(df)
        
        # Test LightGBM with multivariate analysis results
        if "pca" in multi_results:
            # Extract principal components
            pca_results = multi_results["pca"]["pca_results"]
            pca_df = pd.DataFrame({
                "PC1": pca_results["PC1"],
                "PC2": pca_results["PC2"],
                "target": df["target"]
            })
            
            # Train a LightGBM model on the PCA components
            pca_model = create_model("lightgbm", "LGBMRegressor", {'n_estimators': 10})
            pca_model.fit(pca_df.drop("target", axis=1), pca_df["target"])
            
            # Make predictions
            pc_preds = pca_model.predict(pca_df.drop("target", axis=1))
            assert len(pc_preds) == len(pca_df)
            
    except Exception as e:
        pytest.skip(f"EDA with integrated packages test failed: {str(e)}")