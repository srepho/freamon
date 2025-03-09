"""
Tests for the date/time detection in utils.dataframe_utils module.
"""
import pandas as pd
import pytest

from freamon.utils.dataframe_utils import detect_datetime_columns


class TestDateTimeDetection:
    """Test class for datetime column detection."""
    
    @pytest.fixture
    def date_samples_df(self):
        """Create a sample dataframe with various date formats."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "iso_date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"],
            "datetime": ["2020-01-01 12:30:45", "2020-02-01 13:30:45", 
                        "2020-03-01 14:30:45", "2020-04-01 15:30:45", "2020-05-01 16:30:45"],
            "us_date": ["01/01/2020", "02/01/2020", "03/01/2020", "04/01/2020", "05/01/2020"],
            "eu_date": ["01-01-2020", "01-02-2020", "01-03-2020", "01-04-2020", "01-05-2020"],
            "timestamp": [1577836800, 1580515200, 1583020800, 1585699200, 1588291200],
            "not_date": ["ABC123", "DEF456", "GHI789", "JKL012", "MNO345"],
            "mixed_data": ["2020-01-01", "not a date", "2020-03-01", "still not a date", "2020-05-01"],
        })
    
    def test_detect_iso_date(self, date_samples_df):
        """Test detection of ISO format dates."""
        result = detect_datetime_columns(date_samples_df)
        
        # Check that the ISO date column is detected as datetime
        assert pd.api.types.is_datetime64_dtype(result["iso_date"])
        
        # Check that the parsed values are correct
        expected = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", 
                                  "2020-04-01", "2020-05-01"])
        pd.testing.assert_series_equal(result["iso_date"], expected)
    
    def test_detect_datetime(self, date_samples_df):
        """Test detection of datetime strings."""
        result = detect_datetime_columns(date_samples_df)
        
        # Check that the datetime column is detected as datetime
        assert pd.api.types.is_datetime64_dtype(result["datetime"])
        
        # Check that the parsed values are correct
        expected = pd.to_datetime(["2020-01-01 12:30:45", "2020-02-01 13:30:45", 
                                  "2020-03-01 14:30:45", "2020-04-01 15:30:45", 
                                  "2020-05-01 16:30:45"])
        pd.testing.assert_series_equal(result["datetime"], expected)
    
    def test_detect_us_date(self, date_samples_df):
        """Test detection of US format dates (MM/DD/YYYY)."""
        result = detect_datetime_columns(date_samples_df)
        
        # Check that the US date column is detected as datetime
        assert pd.api.types.is_datetime64_dtype(result["us_date"])
    
    def test_detect_timestamp(self, date_samples_df):
        """Test detection of Unix timestamps."""
        result = detect_datetime_columns(date_samples_df)
        
        # Check that the timestamp column is detected as datetime
        assert pd.api.types.is_datetime64_dtype(result["timestamp"])
        
        # Check that the parsed values are correct
        expected = pd.to_datetime([1577836800, 1580515200, 1583020800, 
                                  1585699200, 1588291200], unit='s')
        pd.testing.assert_series_equal(result["timestamp"], expected)
    
    def test_non_date_columns_unchanged(self, date_samples_df):
        """Test that non-date columns are not modified."""
        result = detect_datetime_columns(date_samples_df)
        
        # Check that non-date columns remain the same
        assert not pd.api.types.is_datetime64_dtype(result["id"])
        assert not pd.api.types.is_datetime64_dtype(result["not_date"])
        
        # Original values should be preserved
        pd.testing.assert_series_equal(result["not_date"], date_samples_df["not_date"])
    
    def test_threshold_parameter(self, date_samples_df):
        """Test the threshold parameter."""
        # With default threshold (0.9), mixed_data should not be converted
        result_default = detect_datetime_columns(date_samples_df)
        assert not pd.api.types.is_datetime64_dtype(result_default["mixed_data"])
        
        # With lower threshold (0.5), mixed_data should be converted
        result_lower = detect_datetime_columns(date_samples_df, threshold=0.5)
        assert pd.api.types.is_datetime64_dtype(result_lower["mixed_data"])
    
    def test_inplace_parameter(self, date_samples_df):
        """Test the inplace parameter."""
        # Default is not inplace
        result = detect_datetime_columns(date_samples_df)
        assert not pd.api.types.is_datetime64_dtype(date_samples_df["iso_date"])
        
        # With inplace=True
        detect_datetime_columns(date_samples_df, inplace=True)
        assert pd.api.types.is_datetime64_dtype(date_samples_df["iso_date"])
    
    def test_custom_date_formats(self, date_samples_df):
        """Test with custom date formats."""
        # Add a column with a non-standard date format
        df = date_samples_df.copy()
        df["custom_format"] = ["20200101", "20200201", "20200301", "20200401", "20200501"]
        
        # Default formats won't detect this
        result_default = detect_datetime_columns(df)
        assert not pd.api.types.is_datetime64_dtype(result_default["custom_format"])
        
        # With custom format, it should work
        result_custom = detect_datetime_columns(df, date_formats=["%Y%m%d"])
        assert pd.api.types.is_datetime64_dtype(result_custom["custom_format"])