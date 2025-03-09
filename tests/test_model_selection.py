"""
Tests for the model_selection module.
"""
import pandas as pd
import pytest

from freamon.model_selection import (
    train_test_split,
    time_series_split,
    stratified_time_series_split,
)


class TestSplitters:
    """Test class for data splitting functions."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "id": range(100),
            "value": range(100),
            "category": ["A", "B", "C", "D"] * 25,
            "date": pd.date_range(start="2020-01-01", periods=100),
        })
    
    def test_train_test_split(self, sample_df):
        """Test basic train/test splitting."""
        train_df, test_df = train_test_split(sample_df, test_size=0.2, random_state=42)
        
        # Check that the proportions are correct
        assert len(train_df) == 80
        assert len(test_df) == 20
        
        # Check that all rows are accounted for
        assert len(train_df) + len(test_df) == len(sample_df)
        
        # Check that there's no overlap between train and test
        assert set(train_df.index).isdisjoint(set(test_df.index))
    
    def test_train_test_split_stratified(self, sample_df):
        """Test stratified train/test splitting."""
        train_df, test_df = train_test_split(
            sample_df, test_size=0.2, random_state=42, stratify_by="category"
        )
        
        # Check that the proportions are correct
        assert len(train_df) == 80
        assert len(test_df) == 20
        
        # Check that all rows are accounted for
        assert len(train_df) + len(test_df) == len(sample_df)
        
        # Check that there's no overlap between train and test
        assert set(train_df.index).isdisjoint(set(test_df.index))
        
        # Check that the stratification was maintained
        train_category_counts = train_df["category"].value_counts(normalize=True)
        test_category_counts = test_df["category"].value_counts(normalize=True)
        
        # The distributions should be roughly the same
        for category in ["A", "B", "C", "D"]:
            assert abs(train_category_counts[category] - test_category_counts[category]) < 0.1
    
    def test_time_series_split(self, sample_df):
        """Test time series splitting."""
        train_df, test_df = time_series_split(sample_df, "date", test_size=0.2)
        
        # Check that the proportions are correct
        assert len(train_df) == 80
        assert len(test_df) == 20
        
        # Check that all rows are accounted for
        assert len(train_df) + len(test_df) == len(sample_df)
        
        # Check that there's no overlap between train and test
        assert set(train_df.index).isdisjoint(set(test_df.index))
        
        # Check that the split is based on time
        assert train_df["date"].max() < test_df["date"].min()
    
    def test_time_series_split_with_gap(self, sample_df):
        """Test time series splitting with a gap."""
        train_df, test_df = time_series_split(
            sample_df, "date", test_size=0.2, gap="5D"
        )
        
        # Check that all rows are accounted for (some may be excluded due to the gap)
        assert len(train_df) + len(test_df) <= len(sample_df)
        
        # Check that there's no overlap between train and test
        assert set(train_df.index).isdisjoint(set(test_df.index))
        
        # Check that the split is based on time
        assert train_df["date"].max() < test_df["date"].min()
        
        # Check that there's a gap of at least 5 days
        gap_days = (test_df["date"].min() - train_df["date"].max()).days
        assert gap_days >= 5
    
    def test_stratified_time_series_split(self, sample_df):
        """Test stratified time series splitting."""
        # Add a group column for testing
        sample_df["group"] = ["X", "Y", "Z"] * 33 + ["X"]
        
        train_df, test_df = stratified_time_series_split(
            sample_df, "date", "group", test_size=0.2, random_state=42
        )
        
        # Check that the proportions are roughly correct
        assert abs(len(train_df) - 80) <= 3
        assert abs(len(test_df) - 20) <= 3
        
        # Check that all rows are accounted for
        assert len(train_df) + len(test_df) == len(sample_df)
        
        # Check that there's no overlap between train and test
        assert set(train_df.index).isdisjoint(set(test_df.index))
        
        # Check that all groups are represented in both train and test
        assert set(train_df["group"]) == set(test_df["group"])
        
        # Check that for each group, the test set contains the later dates
        for group in ["X", "Y", "Z"]:
            group_train = train_df[train_df["group"] == group]
            group_test = test_df[test_df["group"] == group]
            
            if not group_train.empty and not group_test.empty:
                assert group_train["date"].max() <= group_test["date"].max()