"""
Tests for the feature importance functionality.
"""
import pandas as pd
import numpy as np
import pytest

from freamon.eda.bivariate import calculate_feature_importance
from freamon.eda.analyzer import EDAAnalyzer


class TestFeatureImportance:
    """Test class for feature importance functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing feature importance."""
        np.random.seed(42)
        n = 1000
        
        # Create a dataframe with a clear relationship between features and target
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        
        # Target has strong dependence on x1, moderate on x2, and weak on x3
        y_numeric = 3*x1 + 1.5*x2 + 0.5*x3 + np.random.normal(0, 0.5, n)
        
        # Create a categorical target based on quantiles
        y_cat = pd.qcut(y_numeric, 3, labels=['Low', 'Medium', 'High'])
        
        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'noise': np.random.normal(0, 1, n),  # Pure noise feature
            'target_numeric': y_numeric,
            'target_cat': y_cat
        })
    
    def test_calculate_feature_importance_numeric_target(self, sample_df):
        """Test feature importance calculation with a numeric target."""
        features = ['x1', 'x2', 'x3', 'noise']
        target = 'target_numeric'
        
        result = calculate_feature_importance(
            df=sample_df,
            features=features,
            target=target,
            method='random_forest',
            n_estimators=100
        )
        
        # Check result structure
        assert 'method' in result
        assert 'target' in result
        assert 'is_classification' in result
        assert 'importances' in result
        assert 'sorted_importances' in result
        
        # Since we created the data with known relationships,
        # feature importance should reflect these relationships
        importances = result['sorted_importances']
        
        # Check that feature order matches our constructed data
        features_by_importance = list(importances.keys())
        
        # x1 should be the most important
        assert features_by_importance[0] == 'x1'
        
        # Noise should be the least important
        assert features_by_importance[-1] == 'noise'
        
        # Check plot generation
        assert 'plot' in result
        assert result['plot'].startswith('data:image/png;base64,')
        
        # Check is_classification flag
        assert result['is_classification'] == False
    
    def test_calculate_feature_importance_categorical_target(self, sample_df):
        """Test feature importance calculation with a categorical target."""
        features = ['x1', 'x2', 'x3', 'noise']
        target = 'target_cat'
        
        result = calculate_feature_importance(
            df=sample_df,
            features=features,
            target=target,
            method='random_forest',
            n_estimators=100
        )
        
        # Check result structure
        assert 'method' in result
        assert 'target' in result
        assert 'is_classification' in result
        assert 'importances' in result
        assert 'sorted_importances' in result
        
        # Since we created the data with known relationships,
        # feature importance should reflect these relationships
        importances = result['sorted_importances']
        
        # Check that feature order matches our constructed data
        features_by_importance = list(importances.keys())
        
        # x1 should be the most important
        assert features_by_importance[0] == 'x1'
        
        # Noise should be the least important
        assert features_by_importance[-1] == 'noise'
        
        # Check plot generation
        assert 'plot' in result
        assert result['plot'].startswith('data:image/png;base64,')
        
        # Check is_classification flag
        assert result['is_classification'] == True
    
    def test_eda_analyzer_feature_importance(self, sample_df):
        """Test the EDAAnalyzer.analyze_feature_importance method."""
        analyzer = EDAAnalyzer(
            df=sample_df,
            target_column='target_numeric'
        )
        
        # Run feature importance analysis
        result = analyzer.analyze_feature_importance(
            method='random_forest',
            n_estimators=50
        )
        
        # Check that results were stored in the analyzer
        assert 'feature_importance' in analyzer.analysis_results
        assert 'target_numeric' in analyzer.analysis_results['feature_importance']
        
        # Check that the result contains the expected data
        assert 'importances' in result
        assert 'sorted_importances' in result
        
        # x1 should be the most important
        features_by_importance = list(result['sorted_importances'].keys())
        assert features_by_importance[0] == 'x1'
        
        # Test with a different target
        result2 = analyzer.analyze_feature_importance(
            target='target_cat',
            method='random_forest',
            n_estimators=50
        )
        
        # Check that results were stored in the analyzer
        assert 'target_cat' in analyzer.analysis_results['feature_importance']
        assert result2['is_classification'] == True