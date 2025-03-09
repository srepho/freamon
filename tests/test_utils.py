"""
Tests for the utils module.
"""
import pandas as pd
import pytest

from freamon.utils import (
    check_dataframe_type,
    convert_dataframe,
    optimize_dtypes,
    estimate_memory_usage,
)


class TestDataframeUtils:
    """Test class for dataframe utilities."""
    
    def test_check_dataframe_type(self):
        """Test check_dataframe_type function."""
        # Test pandas dataframe
        df = pd.DataFrame({"A": [1, 2, 3]})
        assert check_dataframe_type(df) == "pandas"
        
        # Test non-dataframe
        assert check_dataframe_type([1, 2, 3]) == "unknown"
    
    def test_optimize_dtypes(self):
        """Test optimize_dtypes function."""
        # Create a dataframe with different data types
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
        })
        
        # Original dtypes
        assert df["int_col"].dtype == "int64"
        assert df["float_col"].dtype == "float64"
        
        # Optimize dtypes
        optimized_df = optimize_dtypes(df)
        
        # Check that dtypes were optimized
        assert optimized_df["int_col"].dtype == "int8"
        assert optimized_df["float_col"].dtype == "float32"
    
    def test_estimate_memory_usage(self):
        """Test estimate_memory_usage function."""
        # Create a simple dataframe
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
        })
        
        # Estimate memory usage
        memory_info = estimate_memory_usage(df)
        
        # Check structure of the result
        assert "total_mb" in memory_info
        assert "columns" in memory_info
        assert set(memory_info["columns"].keys()) == set(df.columns)
        
        # Values should be positive
        assert memory_info["total_mb"] > 0
        for col, size in memory_info["columns"].items():
            assert size > 0