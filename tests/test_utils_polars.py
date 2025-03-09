"""
Tests for Polars integration in utils.dataframe_utils module.
"""
import numpy as np
import pandas as pd
import pytest

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from freamon.utils.dataframe_utils import (
    check_dataframe_type,
    convert_dataframe,
    optimize_dtypes,
    estimate_memory_usage,
    detect_datetime_columns,
)


# Skip all tests if Polars is not available
pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


class TestPolarsIntegration:
    """Test class for Polars integration."""
    
    @pytest.fixture
    def sample_pandas_df(self):
        """Create a sample Pandas dataframe for testing."""
        return pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "str_col": ["A", "B", "C", "D", "E"],
            "date_col": pd.date_range(start="2020-01-01", periods=5),
            "bool_col": [True, False, True, False, True],
        })
    
    @pytest.fixture
    def sample_polars_df(self, sample_pandas_df):
        """Create a sample Polars dataframe for testing."""
        return pl.from_pandas(sample_pandas_df)
    
    def test_check_dataframe_type(self, sample_pandas_df, sample_polars_df):
        """Test detection of dataframe types."""
        assert check_dataframe_type(sample_pandas_df) == "pandas"
        assert check_dataframe_type(sample_polars_df) == "polars"
    
    def test_pandas_to_polars_conversion(self, sample_pandas_df):
        """Test conversion from Pandas to Polars."""
        polars_df = convert_dataframe(sample_pandas_df, "polars")
        
        # Check that the result is a Polars dataframe
        assert check_dataframe_type(polars_df) == "polars"
        
        # Check that the shape is the same
        assert polars_df.shape == sample_pandas_df.shape
        
        # Check that column names are preserved
        assert set(polars_df.columns) == set(sample_pandas_df.columns)
    
    def test_polars_to_pandas_conversion(self, sample_polars_df):
        """Test conversion from Polars to Pandas."""
        pandas_df = convert_dataframe(sample_polars_df, "pandas")
        
        # Check that the result is a Pandas dataframe
        assert check_dataframe_type(pandas_df) == "pandas"
        
        # Check that the shape is the same
        assert pandas_df.shape == sample_polars_df.shape
        
        # Check that column names are preserved
        assert set(pandas_df.columns) == set(sample_polars_df.columns)
    
    def test_optimize_polars_dtypes(self, sample_polars_df):
        """Test optimizing dtypes for a Polars dataframe."""
        # Make a copy with int64 and float64
        df = sample_polars_df.with_columns([
            pl.col("int_col").cast(pl.Int64),
            pl.col("float_col").cast(pl.Float64),
        ])
        
        # Optimize dtypes
        optimized_df = optimize_dtypes(df)
        
        # Check that types were optimized
        schema = dict(optimized_df.schema)
        assert schema["int_col"] in (pl.Int8, pl.Int16, pl.Int32)
        assert schema["float_col"] == pl.Float32
    
    def test_estimate_polars_memory_usage(self, sample_polars_df):
        """Test memory usage estimation for a Polars dataframe."""
        memory_info = estimate_memory_usage(sample_polars_df)
        
        # Check structure of the result
        assert "total_mb" in memory_info
        assert "columns" in memory_info
        assert set(memory_info["columns"].keys()) == set(sample_polars_df.columns)
        
        # Values should be positive
        assert memory_info["total_mb"] > 0
        for col, size in memory_info["columns"].items():
            assert size >= 0
    
    def test_detect_datetime_columns_polars(self):
        """Test datetime detection with Polars dataframes."""
        # Create a Polars dataframe with date strings
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "iso_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            "timestamp": [1577836800, 1580515200, 1583020800, 1585699200, 1588291200],
            "not_date": ["ABC123", "DEF456", "GHI789", "JKL012", "MNO345"],
        })
        
        # Detect datetime columns
        result = detect_datetime_columns(df)
        
        # Check that datetime columns are detected
        schema = dict(result.schema)
        assert schema["iso_date"] == pl.Datetime
        assert schema["timestamp"] == pl.Datetime
        
        # Non-date columns should remain the same
        assert schema["id"] == pl.Int64
        assert schema["not_date"] == pl.Utf8