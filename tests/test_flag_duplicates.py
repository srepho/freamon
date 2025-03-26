"""
Tests for the freamon.deduplication.flag_duplicates module.

This module contains tests for all functions in the flag_duplicates module,
verifying they work with both pandas and polars DataFrames.
"""

import pytest
import pandas as pd
import numpy as np
import polars as pl
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional

from freamon.deduplication.flag_duplicates import (
    flag_exact_duplicates,
    flag_text_duplicates,
    flag_similar_records,
    flag_supervised_duplicates,
    add_duplicate_detection_columns
)
from freamon.deduplication.polars_supervised_deduplication import PolarsSupervisedDeduplicationModel


class TestFlagDuplicates:
    """Test class for the flag_duplicates module."""
    
    @pytest.fixture
    def sample_df_pandas(self):
        """Create a sample pandas DataFrame with duplicates."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["John Smith", "Jane Doe", "Alice Brown", "John Smith", "Jane Doe", "Bob Wilson"],
            "email": ["john@example.com", "jane@example.com", "alice@example.com", 
                     "john.smith@example.com", "jane.doe@example.com", "bob@example.com"],
            "amount": [100.0, 200.0, 300.0, 100.0, 200.0, 400.0],
            "text": ["This is sample text", "Another example", "Third sample", 
                    "This is sample text", "Another sample", "Final example"]
        })
    
    @pytest.fixture
    def sample_df_polars(self, sample_df_pandas):
        """Create a sample polars DataFrame with duplicates."""
        return pl.from_pandas(sample_df_pandas)
    
    @pytest.fixture
    def empty_df_pandas(self):
        """Create an empty pandas DataFrame."""
        return pd.DataFrame({"id": [], "name": [], "text": []})
    
    @pytest.fixture
    def empty_df_polars(self, empty_df_pandas):
        """Create an empty polars DataFrame."""
        return pl.from_pandas(empty_df_pandas)
    
    @pytest.fixture
    def no_duplicates_df_pandas(self):
        """Create a pandas DataFrame with no duplicates."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["John Smith", "Jane Doe", "Alice Brown", "Bob Wilson", "Carol Taylor"],
            "text": ["Text one", "Text two", "Text three", "Text four", "Text five"]
        })
    
    @pytest.fixture
    def no_duplicates_df_polars(self, no_duplicates_df_pandas):
        """Create a polars DataFrame with no duplicates."""
        return pl.from_pandas(no_duplicates_df_pandas)
    
    # Tests for flag_exact_duplicates
    
    def test_flag_exact_duplicates_pandas(self, sample_df_pandas):
        """Test flag_exact_duplicates with pandas DataFrame."""
        # Test with all columns
        result = flag_exact_duplicates(sample_df_pandas)
        
        # Verify column was added
        assert "is_duplicate" in result.columns
        
        # Verify that duplicates were correctly flagged
        assert result["is_duplicate"].sum() == 0  # No exact duplicates across all columns
        
        # Test with subset of columns
        result = flag_exact_duplicates(sample_df_pandas, subset=["name"])
        
        # Two duplicates of names
        assert result["is_duplicate"].sum() == 2
        assert not result.loc[0, "is_duplicate"]  # First occurrence shouldn't be flagged
        assert not result.loc[1, "is_duplicate"]  # First occurrence shouldn't be flagged
        assert result.loc[3, "is_duplicate"]  # Duplicate of row 0
        assert result.loc[4, "is_duplicate"]  # Duplicate of row 1
        
        # Test with keep='last'
        result = flag_exact_duplicates(sample_df_pandas, subset=["name"], keep="last")
        assert result["is_duplicate"].sum() == 2
        assert result.loc[0, "is_duplicate"]  # First occurrence is flagged
        assert result.loc[1, "is_duplicate"]  # First occurrence is flagged
        assert not result.loc[3, "is_duplicate"]  # Last occurrence not flagged
        assert not result.loc[4, "is_duplicate"]  # Last occurrence not flagged
        
        # Test with keep=False
        result = flag_exact_duplicates(sample_df_pandas, subset=["name"], keep=False)
        assert result["is_duplicate"].sum() == 4
        assert result.loc[0, "is_duplicate"]  # All duplicates flagged
        assert result.loc[1, "is_duplicate"]  # All duplicates flagged
        assert result.loc[3, "is_duplicate"]  # All duplicates flagged
        assert result.loc[4, "is_duplicate"]  # All duplicates flagged
        
        # Test with custom flag column
        result = flag_exact_duplicates(sample_df_pandas, subset=["name"], flag_column="duplicate_flag")
        assert "duplicate_flag" in result.columns
        assert result["duplicate_flag"].sum() == 2
        
        # Test with indicator column
        result = flag_exact_duplicates(
            sample_df_pandas, 
            subset=["name"], 
            indicator_column="duplicate_group"
        )
        
        assert "duplicate_group" in result.columns
        # Two duplicate groups, each with group ID > 0
        assert (result["duplicate_group"] > 0).sum() == 4
        # Same group ID for rows 0 and 3
        assert result.loc[0, "duplicate_group"] == result.loc[3, "duplicate_group"]
        # Same group ID for rows 1 and 4
        assert result.loc[1, "duplicate_group"] == result.loc[4, "duplicate_group"]
        # Different group IDs for different groups
        assert result.loc[0, "duplicate_group"] != result.loc[1, "duplicate_group"]
    
    def test_flag_exact_duplicates_polars(self, sample_df_polars):
        """Test flag_exact_duplicates with polars DataFrame."""
        # Test with all columns
        result = flag_exact_duplicates(sample_df_polars)
        
        # Verify column was added
        assert "is_duplicate" in result.columns
        
        # Verify that duplicates were correctly flagged
        assert result["is_duplicate"].sum() == 0  # No exact duplicates across all columns
        
        # Test with subset of columns
        result = flag_exact_duplicates(sample_df_polars, subset=["name"])
        
        # Convert to pandas for easy row-based access
        result_pd = result.to_pandas()
        
        # Two duplicates of names
        assert result_pd["is_duplicate"].sum() == 2
        assert not result_pd.loc[0, "is_duplicate"]  # First occurrence shouldn't be flagged
        assert not result_pd.loc[1, "is_duplicate"]  # First occurrence shouldn't be flagged
        assert result_pd.loc[3, "is_duplicate"]  # Duplicate of row 0
        assert result_pd.loc[4, "is_duplicate"]  # Duplicate of row 1
        
        # Test with indicator column
        result = flag_exact_duplicates(
            sample_df_polars, 
            subset=["name"], 
            indicator_column="duplicate_group"
        )
        
        result_pd = result.to_pandas()
        assert "duplicate_group" in result_pd.columns
        # Two duplicate groups, each with group ID > 0
        assert (result_pd["duplicate_group"] > 0).sum() == 4
    
    def test_flag_exact_duplicates_empty(self, empty_df_pandas, empty_df_polars):
        """Test flag_exact_duplicates with empty DataFrames."""
        # Test with pandas
        result_pd = flag_exact_duplicates(empty_df_pandas)
        assert "is_duplicate" in result_pd.columns
        assert len(result_pd) == 0
        
        # Test with polars
        result_pl = flag_exact_duplicates(empty_df_polars)
        assert "is_duplicate" in result_pl.columns
        assert len(result_pl) == 0
    
    def test_flag_exact_duplicates_no_duplicates(self, no_duplicates_df_pandas, no_duplicates_df_polars):
        """Test flag_exact_duplicates with DataFrames having no duplicates."""
        # Test with pandas
        result_pd = flag_exact_duplicates(no_duplicates_df_pandas, subset=["name"])
        assert "is_duplicate" in result_pd.columns
        assert result_pd["is_duplicate"].sum() == 0
        
        # Test with polars
        result_pl = flag_exact_duplicates(no_duplicates_df_polars, subset=["name"])
        result_pl_pd = result_pl.to_pandas()  # Convert to pandas for easy testing
        assert "is_duplicate" in result_pl_pd.columns
        assert result_pl_pd["is_duplicate"].sum() == 0
    
    def test_flag_exact_duplicates_inplace(self, sample_df_pandas):
        """Test flag_exact_duplicates with inplace=True."""
        df = sample_df_pandas.copy()
        result = flag_exact_duplicates(df, subset=["name"], inplace=True)
        
        # Should return the same object
        assert result is df
        
        # Column should be added to the original dataframe
        assert "is_duplicate" in df.columns
        assert df["is_duplicate"].sum() == 2
    
    # Tests for flag_text_duplicates
    
    def test_flag_text_duplicates_pandas_hash(self, sample_df_pandas):
        """Test flag_text_duplicates with pandas DataFrame using hash method."""
        result = flag_text_duplicates(
            sample_df_pandas,
            text_column="text",
            method="hash",
            flag_column="is_text_duplicate"
        )
        
        # Column should be added
        assert "is_text_duplicate" in result.columns
        
        # Verify duplicates were correctly flagged
        assert result["is_text_duplicate"].sum() == 1
        assert not result.loc[0, "is_text_duplicate"]  # First occurrence not flagged
        assert result.loc[3, "is_text_duplicate"]  # Duplicate of row 0
        
        # Test with group column
        result = flag_text_duplicates(
            sample_df_pandas,
            text_column="text",
            method="hash",
            flag_column="is_text_duplicate",
            group_column="text_duplicate_group"
        )
        
        # Verify group column was added
        assert "text_duplicate_group" in result.columns
        
        # Rows 0 and 3 should be in the same group
        group_id = result.loc[0, "text_duplicate_group"]
        assert group_id > 0  # Group ID should be positive
        assert result.loc[3, "text_duplicate_group"] == group_id
    
    def test_flag_text_duplicates_pandas_ngram(self, sample_df_pandas):
        """Test flag_text_duplicates with pandas DataFrame using ngram method."""
        result = flag_text_duplicates(
            sample_df_pandas,
            text_column="text",
            method="ngram",
            ngram_size=2,
            flag_column="is_text_duplicate"
        )
        
        # Column should be added
        assert "is_text_duplicate" in result.columns
        
        # Verify duplicates were correctly flagged
        assert result["is_text_duplicate"].sum() == 1
        assert not result.loc[0, "is_text_duplicate"]  # First occurrence not flagged
        assert result.loc[3, "is_text_duplicate"]  # Duplicate of row 0
    
    def test_flag_text_duplicates_pandas_fuzzy(self, sample_df_pandas):
        """Test flag_text_duplicates with pandas DataFrame using fuzzy method."""
        result = flag_text_duplicates(
            sample_df_pandas,
            text_column="text",
            method="fuzzy",
            threshold=0.8,
            flag_column="is_text_duplicate",
            similarity_column="text_similarity"
        )
        
        # Columns should be added
        assert "is_text_duplicate" in result.columns
        assert "text_similarity" in result.columns
        
        # Rows 0 and 3 are exact matches
        assert result.loc[3, "is_text_duplicate"]
        
        # Rows 1 and 4 may be similar enough with threshold 0.8
        if result.loc[4, "is_text_duplicate"]:
            assert result.loc[4, "text_similarity"] >= 0.8
    
    def test_flag_text_duplicates_pandas_lsh(self, sample_df_pandas):
        """Test flag_text_duplicates with pandas DataFrame using lsh method."""
        result = flag_text_duplicates(
            sample_df_pandas,
            text_column="text",
            method="lsh",
            threshold=0.7,
            flag_column="is_text_duplicate"
        )
        
        # Column should be added
        assert "is_text_duplicate" in result.columns
        
        # Rows 0 and 3 are exact matches, should be caught by LSH
        assert result.loc[0, "is_text_duplicate"] or result.loc[3, "is_text_duplicate"]
    
    def test_flag_text_duplicates_polars(self, sample_df_polars):
        """Test flag_text_duplicates with polars DataFrame."""
        result = flag_text_duplicates(
            sample_df_polars,
            text_column="text",
            method="hash",
            flag_column="is_text_duplicate"
        )
        
        # Convert to pandas for easy testing
        result_pd = result.to_pandas()
        
        # Column should be added
        assert "is_text_duplicate" in result_pd.columns
        
        # Verify duplicates were correctly flagged
        assert result_pd["is_text_duplicate"].sum() == 1
        assert not result_pd.loc[0, "is_text_duplicate"]  # First occurrence not flagged
        assert result_pd.loc[3, "is_text_duplicate"]  # Duplicate of row 0
    
    def test_flag_text_duplicates_empty(self, empty_df_pandas, empty_df_polars):
        """Test flag_text_duplicates with empty DataFrames."""
        # Test with pandas
        result_pd = flag_text_duplicates(empty_df_pandas, text_column="text", method="hash")
        assert "is_text_duplicate" in result_pd.columns
        assert len(result_pd) == 0
        
        # Test with polars
        result_pl = flag_text_duplicates(empty_df_polars, text_column="text", method="hash")
        assert "is_text_duplicate" in result_pl.columns
        assert len(result_pl) == 0
    
    def test_flag_text_duplicates_invalid_method(self, sample_df_pandas):
        """Test flag_text_duplicates with invalid method."""
        with pytest.raises(ValueError):
            flag_text_duplicates(
                sample_df_pandas,
                text_column="text",
                method="invalid_method"
            )
    
    # Tests for flag_similar_records
    
    def test_flag_similar_records_pandas_composite(self, sample_df_pandas):
        """Test flag_similar_records with pandas DataFrame using composite method."""
        result = flag_similar_records(
            sample_df_pandas,
            columns=["name", "amount"],
            weights={"name": 0.7, "amount": 0.3},
            threshold=0.7,
            method="composite",
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Rows 0 and 3 have same name and amount, should be flagged as similar
        assert not result.loc[0, "is_similar"]  # First occurrence not flagged
        assert result.loc[3, "is_similar"]  # Should be flagged as similar
        
        # Test with similarity column
        result = flag_similar_records(
            sample_df_pandas,
            columns=["name", "amount"],
            weights={"name": 0.7, "amount": 0.3},
            threshold=0.7,
            method="composite",
            flag_column="is_similar",
            similarity_column="similarity_score"
        )
        
        # Similarity column should be added
        assert "similarity_score" in result.columns
        
        # Rows 0 and 3 should have high similarity
        assert result.loc[3, "similarity_score"] >= 0.7
    
    def test_flag_similar_records_pandas_exact_subset(self, sample_df_pandas):
        """Test flag_similar_records with pandas DataFrame using exact_subset method."""
        result = flag_similar_records(
            sample_df_pandas,
            columns=["name", "email", "amount"],
            weights={"name": 0.6, "email": 0.2, "amount": 0.2},
            threshold=0.6,  # Match if 60% of the weighted columns match exactly
            method="exact_subset",
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Rows 0 and 3 share name and amount (weight 0.8), should be flagged
        assert not result.loc[0, "is_similar"]  # First occurrence not flagged
        assert result.loc[3, "is_similar"]  # Should be flagged as similar
    
    def test_flag_similar_records_pandas_fuzzy_subset(self, sample_df_pandas):
        """Test flag_similar_records with pandas DataFrame using fuzzy_subset method."""
        result = flag_similar_records(
            sample_df_pandas,
            columns=["name", "email", "text"],
            weights={"name": 0.4, "email": 0.3, "text": 0.3},
            threshold=0.4,  # Match if 40% of the weighted columns match with high similarity
            method="fuzzy_subset",
            flag_column="is_similar"
        )
        
        # Column should be added
        assert "is_similar" in result.columns
        
        # Rows 0 and 3 share name and exact text, should be flagged
        assert result.loc[0, "is_similar"] or result.loc[3, "is_similar"]
    
    def test_flag_similar_records_polars(self, sample_df_polars):
        """Test flag_similar_records with polars DataFrame."""
        result = flag_similar_records(
            sample_df_polars,
            columns=["name", "amount"],
            weights={"name": 0.7, "amount": 0.3},
            threshold=0.7,
            method="composite",
            flag_column="is_similar"
        )
        
        # Convert to pandas for easy testing
        result_pd = result.to_pandas()
        
        # Column should be added
        assert "is_similar" in result_pd.columns
        
        # Rows 0 and 3 have same name and amount, should be flagged as similar
        assert not result_pd.loc[0, "is_similar"]  # First occurrence not flagged
        assert result_pd.loc[3, "is_similar"]  # Should be flagged as similar
    
    def test_flag_similar_records_empty(self, empty_df_pandas, empty_df_polars):
        """Test flag_similar_records with empty DataFrames."""
        # Test with pandas
        result_pd = flag_similar_records(
            empty_df_pandas, 
            columns=["name", "text"],
            threshold=0.8
        )
        assert "is_similar" in result_pd.columns
        assert len(result_pd) == 0
        
        # Test with polars
        result_pl = flag_similar_records(
            empty_df_polars, 
            columns=["name", "text"], 
            threshold=0.8
        )
        assert "is_similar" in result_pl.columns
        assert len(result_pl) == 0
    
    def test_flag_similar_records_invalid_method(self, sample_df_pandas):
        """Test flag_similar_records with invalid method."""
        with pytest.raises(ValueError):
            flag_similar_records(
                sample_df_pandas,
                columns=["name", "amount"],
                method="invalid_method",
                threshold=0.7
            )
    
    def test_flag_similar_records_max_comparisons(self, sample_df_pandas):
        """Test flag_similar_records with max_comparisons parameter."""
        result = flag_similar_records(
            sample_df_pandas,
            columns=["name", "amount"],
            threshold=0.7,
            max_comparisons=2  # Limit to just 2 comparisons
        )
        
        # Function should run without errors with limited comparisons
        assert "is_similar" in result.columns
    
    # Tests for flag_supervised_duplicates
    
    def test_flag_supervised_duplicates_pandas(self, sample_df_pandas):
        """Test flag_supervised_duplicates with pandas DataFrame."""
        # Skip testing the actual detection since it depends on trained models
        # Instead, just verify the function sets up the columns correctly
        
        # Create a test dataframe
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "amount": [100.0, 200.0, 300.0],
            "text": ["Sample text A", "Sample text B", "Sample text C"]
        })
        
        # Create very simple training pairs
        duplicate_pairs = [(0, 1)]  # Just need some valid indices
        
        # We're not testing the actual duplicate detection (which depends on ML model quality)
        # but just checking that the function correctly sets up the result dataframe and columns
        result = flag_supervised_duplicates(
            test_df,
            duplicate_pairs=duplicate_pairs,
            key_features=["name", "email"],
            threshold=0.5,
            flag_column="is_duplicate",
            duplicate_of_column="duplicate_of",
            probability_column="duplicate_probability"
        )
        
        # Verify columns were added correctly (this is the core functionality)
        assert "is_duplicate" in result.columns
        assert "duplicate_of" in result.columns
        assert "duplicate_probability" in result.columns
        assert result["is_duplicate"].dtype == bool
        assert all(isinstance(val, bool) for val in result["is_duplicate"])
    
    def test_flag_supervised_duplicates_with_model(self, sample_df_pandas):
        """Test flag_supervised_duplicates with pre-trained model."""
        # Create a simple test dataframe - we're testing interface not detection performance
        test_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "amount": [100.0, 200.0, 300.0],
        })
        
        # Create a mock model with a predict_duplicate_probability method that returns a dataframe
        class MockDuplicateModel:
            def predict_duplicate_probability(self, df, **kwargs):
                # Return a mock result with a few potential pairs
                return pd.DataFrame({
                    'idx1': [0, 1],
                    'idx2': [1, 2],
                    'duplicate_probability': [0.4, 0.6]  # Only the second pair is above threshold
                })
        
        model = MockDuplicateModel()
        
        # Use the mock model
        result = flag_supervised_duplicates(
            test_df,
            model=model,
            threshold=0.5,
            flag_column="is_duplicate"
        )
        
        # Column should be added
        assert "is_duplicate" in result.columns
        
        # Verify the flag column has the right data type
        assert result["is_duplicate"].dtype == bool
    
    def test_flag_supervised_duplicates_polars(self, sample_df_polars):
        """Test flag_supervised_duplicates with polars DataFrame."""
        # Create a simple test dataframe
        test_df_pd = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "amount": [100.0, 200.0, 300.0],
        })
        test_df = pl.from_pandas(test_df_pd)
        
        # Create simple duplicate pairs for basic testing
        duplicate_pairs = [(0, 1)]
        
        # Test the function without expecting specific results
        result = flag_supervised_duplicates(
            test_df,
            duplicate_pairs=duplicate_pairs,
            key_features=["name", "email"],
            threshold=0.5,
            flag_column="is_duplicate"
        )
        
        # Convert to pandas for easy testing
        result_pd = result.to_pandas()
        
        # Column should be added
        assert "is_duplicate" in result_pd.columns
        
        # Verify the flag column has the right data type
        assert result_pd["is_duplicate"].dtype == bool
        
        # Verify the function preserved the polars dataframe type
        assert hasattr(result, 'to_pandas')
    
    def test_flag_supervised_duplicates_validation(self, sample_df_pandas):
        """Test flag_supervised_duplicates with invalid inputs."""
        # Test missing both model and duplicate_pairs
        with pytest.raises(ValueError):
            flag_supervised_duplicates(
                sample_df_pandas,
                key_features=["name", "email"]
            )
        
        # Test missing key_features when training new model
        with pytest.raises(ValueError):
            flag_supervised_duplicates(
                sample_df_pandas,
                duplicate_pairs=[(0, 3)]
            )
    
    # Tests for add_duplicate_detection_columns
    
    def test_add_duplicate_detection_columns_exact(self, sample_df_pandas, sample_df_polars):
        """Test add_duplicate_detection_columns with 'exact' method."""
        # Test with pandas
        result_pd = add_duplicate_detection_columns(
            sample_df_pandas,
            method="exact",
            columns=["name"],
            flag_column="is_duplicate",
            group_column="duplicate_group"
        )
        
        assert "is_duplicate" in result_pd.columns
        assert "duplicate_group" in result_pd.columns
        assert result_pd["is_duplicate"].sum() == 2
        
        # Test with polars
        result_pl = add_duplicate_detection_columns(
            sample_df_polars,
            method="exact",
            columns=["name"],
            flag_column="is_duplicate",
            group_column="duplicate_group"
        )
        
        result_pl_pd = result_pl.to_pandas()
        assert "is_duplicate" in result_pl_pd.columns
        assert "duplicate_group" in result_pl_pd.columns
        assert result_pl_pd["is_duplicate"].sum() == 2
    
    def test_add_duplicate_detection_columns_hash(self, sample_df_pandas):
        """Test add_duplicate_detection_columns with 'hash' method."""
        result = add_duplicate_detection_columns(
            sample_df_pandas,
            method="hash",
            text_column="text",
            flag_column="is_duplicate",
            group_column="duplicate_group"
        )
        
        assert "is_duplicate" in result.columns
        assert "duplicate_group" in result.columns
        assert result["is_duplicate"].sum() == 1
    
    def test_add_duplicate_detection_columns_fuzzy(self, sample_df_pandas):
        """Test add_duplicate_detection_columns with 'fuzzy' method."""
        result = add_duplicate_detection_columns(
            sample_df_pandas,
            method="fuzzy",
            text_column="text",
            threshold=0.8,
            flag_column="is_duplicate",
            group_column="duplicate_group",
            similarity_column="similarity"
        )
        
        assert "is_duplicate" in result.columns
        assert "duplicate_group" in result.columns
        assert "similarity" in result.columns
    
    def test_add_duplicate_detection_columns_similar(self, sample_df_pandas):
        """Test add_duplicate_detection_columns with 'similar' method."""
        result = add_duplicate_detection_columns(
            sample_df_pandas,
            method="similar",
            columns=["name", "amount"],
            threshold=0.7,
            flag_column="is_duplicate",
            group_column="duplicate_group"
        )
        
        assert "is_duplicate" in result.columns
        assert "duplicate_group" in result.columns
    
    def test_add_duplicate_detection_columns_supervised(self, sample_df_pandas):
        """Test add_duplicate_detection_columns with 'supervised' method."""
        duplicate_pairs = [(0, 3), (1, 4)]
        
        result = add_duplicate_detection_columns(
            sample_df_pandas,
            method="supervised",
            threshold=0.5,
            flag_column="is_duplicate",
            duplicate_of_column="duplicate_of",
            similarity_column="probability",
            duplicate_pairs=duplicate_pairs,
            key_features=["name", "email", "amount"]
        )
        
        assert "is_duplicate" in result.columns
        assert "duplicate_of" in result.columns
        assert "probability" in result.columns
    
    def test_add_duplicate_detection_columns_validation(self, sample_df_pandas):
        """Test add_duplicate_detection_columns with invalid inputs."""
        # Test missing text_column for text methods
        with pytest.raises(ValueError):
            add_duplicate_detection_columns(
                sample_df_pandas,
                method="hash"
            )
        
        # Test missing columns for exact and similar methods
        with pytest.raises(ValueError):
            add_duplicate_detection_columns(
                sample_df_pandas,
                method="exact"
            )
        
        # Test missing required parameters for supervised method
        with pytest.raises(ValueError):
            add_duplicate_detection_columns(
                sample_df_pandas,
                method="supervised"
            )
        
        # Test invalid method
        with pytest.raises(ValueError):
            add_duplicate_detection_columns(
                sample_df_pandas,
                method="invalid_method"
            )


if __name__ == "__main__":
    pytest.main()