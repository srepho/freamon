"""
Tests for the data_quality.cardinality module.
"""
import pandas as pd
import numpy as np
import pytest

from freamon.data_quality import analyze_cardinality


class TestCardinality:
    """Test class for cardinality analysis."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with various column types for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            # High cardinality (all unique)
            "id": range(100),
            
            # Medium cardinality (10 unique values)
            "group": np.random.randint(1, 11, 100),
            
            # Low cardinality (3 unique values)
            "category": np.random.choice(["A", "B", "C"], 100),
            
            # Very low cardinality (2 unique values)
            "binary": np.random.choice([0, 1], 100),
            
            # Single value (1 unique value)
            "constant": ["X"] * 100,
            
            # Numeric with some missing values
            "numeric_with_missing": np.append(np.random.normal(0, 1, 95), [np.nan] * 5),
        })
    
    def test_analyze_cardinality_all_columns(self, sample_df):
        """Test analyzing cardinality for all columns."""
        result = analyze_cardinality(sample_df)
        
        # Check that all columns are analyzed
        assert set(result.keys()) == set(sample_df.columns)
        
        # Check structure of results
        for col, info in result.items():
            assert "unique_count" in info
            assert "total_count" in info
            assert "missing_count" in info
            assert "valid_count" in info
            assert "cardinality_ratio" in info
            assert "cardinality_type" in info
            assert "value_counts" in info
    
    def test_analyze_cardinality_column_types(self, sample_df):
        """Test that cardinality analysis correctly identifies column types."""
        result = analyze_cardinality(sample_df)
        
        # Check column types
        assert result["id"]["column_type"] == "numeric"
        assert result["group"]["column_type"] == "numeric"
        assert result["category"]["column_type"] == "categorical"
        assert result["binary"]["column_type"] == "numeric"
        assert result["constant"]["column_type"] == "categorical"
    
    def test_analyze_cardinality_unique_counts(self, sample_df):
        """Test that unique counts are correct."""
        result = analyze_cardinality(sample_df)
        
        # Check unique counts
        assert result["id"]["unique_count"] == 100  # All unique
        assert result["group"]["unique_count"] <= 10  # 10 unique values
        assert result["category"]["unique_count"] == 3  # 3 unique values
        assert result["binary"]["unique_count"] == 2  # 2 unique values
        assert result["constant"]["unique_count"] == 1  # 1 unique value
    
    def test_analyze_cardinality_ratios(self, sample_df):
        """Test that cardinality ratios are correctly calculated."""
        result = analyze_cardinality(sample_df)
        
        # High cardinality (all unique)
        assert result["id"]["cardinality_ratio"] == 1.0
        assert result["id"]["cardinality_type"] == "unique"
        
        # Medium cardinality (10 unique values out of 100)
        assert 0.05 <= result["group"]["cardinality_ratio"] <= 0.2
        assert result["group"]["cardinality_type"] in ["low", "medium"]
        
        # Low cardinality (3 unique values out of 100)
        assert 0.01 <= result["category"]["cardinality_ratio"] <= 0.05
        assert result["category"]["cardinality_type"] in ["very_low", "low"]
        
        # Very low cardinality (2 unique values out of 100)
        assert 0.01 <= result["binary"]["cardinality_ratio"] <= 0.05
        assert result["binary"]["cardinality_type"] in ["very_low", "low"]
        
        # Single value (1 unique value out of 100)
        assert result["constant"]["cardinality_ratio"] == 0.01
        assert result["constant"]["cardinality_type"] == "low"  # 0.01 is classified as low cardinality (0.01-0.05)
    
    def test_analyze_cardinality_missing_values(self, sample_df):
        """Test handling of missing values."""
        result = analyze_cardinality(sample_df)
        
        # Column with missing values
        assert result["numeric_with_missing"]["missing_count"] == 5
        assert result["numeric_with_missing"]["valid_count"] == 95
        assert result["numeric_with_missing"]["total_count"] == 100
    
    def test_analyze_cardinality_value_counts(self, sample_df):
        """Test that value counts are correctly calculated."""
        result = analyze_cardinality(sample_df)
        
        # Check value counts
        assert "value_counts" in result["category"]
        assert len(result["category"]["value_counts"]) == 3  # A, B, C
        
        # Check constant column
        assert len(result["constant"]["value_counts"]) == 1
        assert "X" in result["constant"]["value_counts"]
        assert result["constant"]["value_counts"]["X"] == 100
        
        # Check column with missing values
        if "null" in result["numeric_with_missing"]["value_counts"]:
            assert result["numeric_with_missing"]["value_counts"]["null"] == 5
    
    def test_analyze_cardinality_subset_columns(self, sample_df):
        """Test analyzing cardinality for a subset of columns."""
        columns = ["id", "category", "constant"]
        result = analyze_cardinality(sample_df, columns=columns)
        
        # Check that only the specified columns are analyzed
        assert set(result.keys()) == set(columns)
    
    def test_analyze_cardinality_max_unique(self, sample_df):
        """Test limiting the number of unique values in the result."""
        # Create a column with many unique values
        sample_df["many_values"] = range(100)
        
        # Analyze with a limit of 5 unique values
        result = analyze_cardinality(sample_df, max_unique_to_list=5)
        
        # Check that the result is limited
        assert "many_values" in result
        assert "value_counts" in result["many_values"]
        assert len(result["many_values"]["value_counts"]) <= 6  # 5 values + "other"
        assert result["many_values"]["value_counts_limited"] is True
    
    def test_analyze_cardinality_include_plots(self, sample_df):
        """Test including plots in the results."""
        # With plots
        result_with_plots = analyze_cardinality(sample_df, include_plots=True)
        assert "plot" in result_with_plots["category"]
        assert result_with_plots["category"]["plot"].startswith("data:image/png;base64,")
        
        # Without plots
        result_no_plots = analyze_cardinality(sample_df, include_plots=False)
        assert "plot" not in result_no_plots["category"]
    
    def test_analyze_cardinality_invalid_column(self, sample_df):
        """Test with invalid column names."""
        with pytest.raises(ValueError):
            analyze_cardinality(sample_df, columns=["nonexistent_column"])