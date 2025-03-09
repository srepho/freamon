"""
Tests for the data_quality module.
"""
import numpy as np
import pandas as pd
import pytest

from freamon.data_quality import (
    DataQualityAnalyzer,
    handle_missing_values,
    detect_outliers,
)


class TestDataQualityAnalyzer:
    """Test class for DataQualityAnalyzer."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "numeric": [1, 2, np.nan, 4, 5],
            "categorical": ["A", "B", np.nan, "D", "E"],
            "mixed": [1, "two", 3, np.nan, 5],
        })
    
    def test_initialization(self, sample_df):
        """Test that the analyzer initializes correctly."""
        analyzer = DataQualityAnalyzer(sample_df)
        assert analyzer.df_type == "pandas"
        
        # Test validation
        with pytest.raises(ValueError):
            DataQualityAnalyzer(pd.DataFrame())
    
    def test_analyze_missing_values(self, sample_df):
        """Test missing value analysis."""
        analyzer = DataQualityAnalyzer(sample_df)
        result = analyzer.analyze_missing_values()
        
        # Check structure
        assert "missing_count" in result
        assert "missing_percent" in result
        assert "total_missing" in result
        assert "total_percent" in result
        
        # Check counts
        assert result["missing_count"]["numeric"] == 1
        assert result["missing_count"]["categorical"] == 1
        assert result["missing_count"]["mixed"] == 1
        assert result["total_missing"] == 3
    
    def test_analyze_data_types(self, sample_df):
        """Test data type analysis."""
        analyzer = DataQualityAnalyzer(sample_df)
        result = analyzer.analyze_data_types()
        
        # Check structure
        assert "dtypes" in result
        assert "type_consistency" in result
        
        # Check type consistency
        assert result["type_consistency"]["numeric"]["consistent"] is True
        assert result["type_consistency"]["categorical"]["consistent"] is True
        assert result["type_consistency"]["mixed"]["consistent"] is False
        assert len(result["type_consistency"]["mixed"]["types"]) > 1


class TestMissingValues:
    """Test class for missing value handling."""
    
    @pytest.fixture
    def missing_df(self):
        """Create a dataframe with missing values for testing."""
        return pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],
            "B": [np.nan, 2, 3, 4, 5],
            "C": ["a", "b", np.nan, "d", "e"],
        })
    
    def test_drop_strategy(self, missing_df):
        """Test the drop strategy."""
        result = handle_missing_values(missing_df, strategy="drop")
        assert len(result) == 2  # Only 2 rows without any missing values
    
    def test_mean_strategy(self, missing_df):
        """Test the mean strategy."""
        result = handle_missing_values(missing_df, strategy="mean")
        assert not pd.isna(result["A"]).any()
        assert not pd.isna(result["B"]).any()
        
        # Check that numeric values are filled with means
        assert result["A"].iloc[2] == 3.0  # Mean of [1, 2, 4, 5]
        assert result["B"].iloc[0] == 3.5  # Mean of [2, 3, 4, 5]
        
        # Check that non-numeric values are still missing
        assert pd.isna(result["C"]).sum() == 1
    
    def test_constant_strategy(self, missing_df):
        """Test the constant strategy."""
        result = handle_missing_values(
            missing_df, strategy="constant", fill_value="MISSING"
        )
        assert not pd.isna(result).any().any()
        assert result["A"].iloc[2] == "MISSING"
        assert result["B"].iloc[0] == "MISSING"
        assert result["C"].iloc[2] == "MISSING"


class TestOutlierDetection:
    """Test class for outlier detection."""
    
    @pytest.fixture
    def outlier_df(self):
        """Create a dataframe with outliers for testing."""
        return pd.DataFrame({
            "A": [1, 2, 3, 4, 100],  # 100 is an outlier
            "B": [5, 6, 7, 8, 9],    # No outliers
        })
    
    def test_iqr_method(self, outlier_df):
        """Test the IQR method."""
        # Get outlier masks
        masks = detect_outliers(
            outlier_df, method="iqr", return_mask=True
        )
        
        # Check that 100 is detected as an outlier in column A
        assert masks["A"][-1] == True
        
        # Check that no outliers are detected in column B
        assert not masks["B"].any()
        
        # Test removing outliers
        result = detect_outliers(outlier_df, method="iqr", return_mask=False)
        assert len(result) == 4  # One row removed
        assert 100 not in result["A"].values
    
    def test_zscore_method(self, outlier_df):
        """Test the Z-score method."""
        # Get outlier masks
        masks = detect_outliers(
            outlier_df, method="zscore", threshold=3.0, return_mask=True
        )
        
        # Check that 100 is detected as an outlier in column A
        assert masks["A"][-1] == True