"""
Tests for the data_quality.duplicates module.
"""
import pandas as pd
import pytest

from freamon.data_quality import detect_duplicates, remove_duplicates


class TestDuplicates:
    """Test class for duplicate detection and removal."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with duplicates for testing."""
        return pd.DataFrame({
            "A": [1, 2, 1, 3, 4, 2],
            "B": ["x", "y", "x", "z", "x", "y"],
            "C": [0.1, 0.2, 0.1, 0.3, 0.4, 0.5],
        })
    
    def test_detect_duplicates_all_columns(self, sample_df):
        """Test detecting duplicates across all columns."""
        result = detect_duplicates(sample_df)
        
        # Check the result structure
        assert "has_duplicates" in result
        assert "duplicate_count" in result
        assert "duplicate_percentage" in result
        assert "total_rows" in result
        assert "unique_rows" in result
        
        # Check that duplicates are correctly identified
        assert result["has_duplicates"] is True
        assert result["duplicate_count"] == 1
        assert result["duplicate_percentage"] == 1/6 * 100
        assert result["total_rows"] == 6
        assert result["unique_rows"] == 5
    
    def test_detect_duplicates_subset(self, sample_df):
        """Test detecting duplicates in a subset of columns."""
        # Check duplicates in column "B" only
        result = detect_duplicates(sample_df, subset=["B"])
        
        # Column "B" has duplicates: "x" appears 3 times, "y" appears 2 times
        assert result["has_duplicates"] is True
        assert result["duplicate_count"] == 3  # 3 duplicate rows
        
        # Check duplicates with a subset specified as a string
        result_str = detect_duplicates(sample_df, subset="B")
        assert result_str["duplicate_count"] == result["duplicate_count"]
    
    def test_detect_duplicates_return_counts(self, sample_df):
        """Test detection with value counts."""
        result = detect_duplicates(sample_df, subset=["B"], return_counts=True)
        
        # Check that value counts are included
        assert "value_counts" in result
        assert len(result["value_counts"]) > 0
        
        # Verify the counts: "x" appears 3 times, "y" appears 2 times, "z" appears once
        x_count = None
        y_count = None
        for item in result["value_counts"]:
            if item["values"].get("B", "") == "x":
                x_count = item["count"]
            elif item["values"].get("B", "") == "y":
                y_count = item["count"]
        
        assert x_count == 3
        assert y_count == 2
    
    def test_detect_duplicates_no_duplicates(self):
        """Test with a dataframe that has no duplicates."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
        })
        
        result = detect_duplicates(df)
        
        assert result["has_duplicates"] is False
        assert result["duplicate_count"] == 0
        assert result["duplicate_percentage"] == 0.0
    
    def test_remove_duplicates_all_columns(self, sample_df):
        """Test removing duplicates across all columns."""
        result = remove_duplicates(sample_df)
        
        # Check that duplicates were removed
        assert len(result) == 5  # Original had 6 rows, 1 duplicate
        
        # Check that the first occurrence of each value was kept
        assert (1, "x", 0.1) in [tuple(row) for row in result.values]
    
    def test_remove_duplicates_subset(self, sample_df):
        """Test removing duplicates based on a subset of columns."""
        # Remove duplicates based on column "B" only
        result = remove_duplicates(sample_df, subset=["B"])
        
        # Should keep only the first occurrence of each value in "B"
        assert len(result) == 3  # "x", "y", "z"
        
        # Check that we kept the first occurrence of each value
        b_values = result["B"].tolist()
        assert b_values == ["x", "y", "z"]  # First occurrence of each
    
    def test_remove_duplicates_keep_last(self, sample_df):
        """Test removing duplicates but keeping the last occurrence."""
        # Keep the last occurrence of duplicates
        result = remove_duplicates(sample_df, keep="last")
        
        # Should still have 5 rows (1 duplicate removed)
        assert len(result) == 5
        
        # Check that the last row with B="y" was kept (which has C=0.5)
        y_rows = result[result["B"] == "y"]
        assert len(y_rows) == 1
        assert y_rows.iloc[0]["C"] == 0.5
    
    def test_remove_duplicates_keep_false(self, sample_df):
        """Test removing all duplicates including first occurrences."""
        # First, create a dataframe with more duplicates
        df = pd.DataFrame({
            "A": [1, 2, 1, 3],
            "B": ["x", "y", "x", "z"],
        })
        
        # Remove all duplicates including first occurrences
        result = remove_duplicates(df, subset=["A"], keep=False)
        
        # Should only keep rows with unique A values
        assert len(result) == 2
        a_values = set(result["A"].tolist())
        assert a_values == {2, 3}  # Only the unique values remain
    
    def test_remove_duplicates_inplace(self, sample_df):
        """Test removing duplicates in-place."""
        df_copy = sample_df.copy()
        
        # Remove duplicates in-place
        return_val = remove_duplicates(df_copy, inplace=True)
        
        # Should return None when inplace=True
        assert return_val is df_copy
        
        # Original dataframe should be modified
        assert len(df_copy) == 5  # Original had 6 rows, 1 duplicate