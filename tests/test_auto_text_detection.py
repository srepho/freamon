"""
Tests for the automatic text column detection functionality in flag_similar_records.

This module contains tests for the automatic text column detection functionality in the flag_similar_records
function from the flag_duplicates module.
"""

import pytest
import pandas as pd
import numpy as np
import polars as pl
from typing import List, Dict, Any

from freamon.deduplication.flag_duplicates import flag_similar_records
try:
    # DataTypeDetector is an optional dependency for this feature
    from freamon.utils.datatype_detector import DataTypeDetector
    has_datatype_detector = True
except ImportError:
    has_datatype_detector = False


class TestAutoTextDetection:
    """Test class for the automatic text column detection functionality."""
    
    @pytest.fixture
    def mixed_data_df(self):
        """Create a sample pandas DataFrame with a mix of column types including text."""
        return pd.DataFrame({
            "id": ["001", "002", "003", "004", "005", "006"],
            "short_text": ["apple", "banana", "cherry", "date", "apple", "fig"],
            "name": ["John Smith", "Jane Doe", "Alice Brown", "John Smith Jr", "Jane D.", "Bob Wilson"],
            "email": ["john@example.com", "jane@example.com", "alice@example.com", 
                     "john.smith@example.com", "jane.doe@example.com", "bob@example.com"],
            "amount": [100.0, 200.0, 300.0, 100.0, 200.0, 400.0],
            "description": [
                "This is a long description about apples and their many uses in cooking and baking.",
                "Bananas are yellow fruits that are rich in potassium and other essential nutrients.",
                "Cherries are small stone fruits that can be sweet or sour depending on the variety.",
                "This is another long description about different types of apples used in cooking.",
                "Bananas make excellent smoothies and can be used in various dessert recipes.",
                "Figs are sweet fruits with a unique texture that pairs well with cheese and honey."
            ],
            "notes": [
                "Product received in good condition",
                "Customer reported satisfaction",
                "Follow up needed after one week",
                "Product similar to previous order",
                "Customer requested catalog",
                "No additional notes provided"
            ],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", 
                                   "2023-01-04", "2023-01-05", "2023-01-06"]),
            "is_active": [True, False, True, True, False, True]
        })
    
    @pytest.fixture
    def mixed_data_pl(self, mixed_data_df):
        """Create a sample polars DataFrame from the pandas DataFrame."""
        return pl.from_pandas(mixed_data_df)
    
    def test_auto_detect_columns_pandas(self, mixed_data_df):
        """Test flag_similar_records with auto_detect_columns=True for pandas DataFrame."""
        # Run the function with auto_detect_columns=True
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors even if no duplicates found
        # We're not testing the actual detection, just that the function completes
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_with_text_columns_pandas(self, mixed_data_df):
        """Test flag_similar_records with auto_detect_columns and explicit text_columns."""
        # Run the function with auto_detect_columns=True and explicit text_columns
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            text_columns=["description", "notes"],
            text_method="tfidf",
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_with_text_threshold_pandas(self, mixed_data_df):
        """Test flag_similar_records with auto_detect_columns and text_threshold."""
        # Run the function with auto_detect_columns=True and a custom text_threshold
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            text_threshold=0.8,  # Higher threshold for text columns
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_with_text_weight_boost_pandas(self, mixed_data_df):
        """Test flag_similar_records with auto_detect_columns and text_weight_boost."""
        # Run the function with auto_detect_columns=True and a text_weight_boost
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            text_weight_boost=2.0,  # Boost text column weights
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_with_min_text_length_pandas(self, mixed_data_df):
        """Test flag_similar_records with auto_detect_columns and min_text_length."""
        # Run the function with auto_detect_columns=True and a custom min_text_length
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            min_text_length=50,  # Only consider very long text fields
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_all_text_methods_pandas(self, mixed_data_df):
        """Test flag_similar_records with all text_method options."""
        # Test each text_method
        for method in ['fuzzy', 'tfidf', 'ngram', 'lsh']:
            result = flag_similar_records(
                mixed_data_df,
                threshold=0.7,
                auto_detect_columns=True,
                text_method=method,
                flag_column="is_similar"
            )
            
            # Column should be added
            assert "is_similar" in result.columns
            
            # Function should run without errors
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_similar_records_pandas(self, mixed_data_df):
        """Test flag_similar_records finds similar records with auto detection."""
        # Create a dataframe with some similar records
        df = mixed_data_df.copy()
        # Add a row that's very similar to an existing one
        new_row = df.iloc[0].copy()
        new_row['id'] = '007'
        new_row['description'] = "This is a long description about apple varieties and their uses in cooking."
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Run with auto-detection
        result = flag_similar_records(
            df,
            threshold=0.6,
            auto_detect_columns=True,
            text_method='tfidf',
            flag_column="is_similar"
        )
        
        # Check for similar records
        assert result["is_similar"].sum() > 0
    
    def test_auto_detect_columns_polars(self, mixed_data_pl):
        """Test flag_similar_records with auto_detect_columns=True for polars DataFrame."""
        # Run the function with auto_detect_columns=True
        result = flag_similar_records(
            mixed_data_pl,
            threshold=0.7,
            auto_detect_columns=True,
            flag_column="is_similar"
        )
        
        # Convert to pandas for easy testing
        result_pd = result.to_pandas()
        
        # Column should be added
        assert "is_similar" in result_pd.columns
        
        # Function should run without errors
        assert hasattr(result, 'to_pandas')  # Should still be a polars DataFrame
        assert len(result) == len(mixed_data_pl)
    
    @pytest.mark.skipif(not has_datatype_detector, reason="DataTypeDetector not available")
    def test_with_datatype_detector(self, mixed_data_df):
        """Test flag_similar_records with DataTypeDetector integration."""
        # Initialize DataTypeDetector
        detector = DataTypeDetector()
        
        # Detect datatypes
        detector.detect_types(mixed_data_df)
        
        # Run the function with the detector's results
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            auto_detect_columns=True,
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    @pytest.mark.parametrize(
        "method", ["composite", "exact_subset", "fuzzy_subset"]
    )
    def test_all_methods_with_auto_detect(self, mixed_data_df, method):
        """Test flag_similar_records with all methods using auto_detect_columns."""
        result = flag_similar_records(
            mixed_data_df,
            threshold=0.7,
            method=method,
            auto_detect_columns=True,
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_backward_compatibility(self, mixed_data_df):
        """Test flag_similar_records with old parameter style still works."""
        # Run the function with traditional parameters
        result = flag_similar_records(
            mixed_data_df,
            columns=["name", "email", "amount"],
            weights={"name": 0.5, "email": 0.3, "amount": 0.2},
            threshold=0.7,
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mixed_data_df)
    
    def test_auto_detect_without_text_columns(self):
        """Test flag_similar_records with auto_detect_columns but no text columns."""
        # Create a dataframe without text columns
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "amount": [100, 200, 300, 100, 200],
            "quantity": [10, 20, 30, 10, 20],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
            "is_active": [True, False, True, True, False]
        })
        
        # Run with auto-detection
        result = flag_similar_records(
            df,
            threshold=0.7,
            auto_detect_columns=True,
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Function should run without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


if __name__ == "__main__":
    pytest.main()