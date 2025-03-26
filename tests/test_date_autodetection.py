"""
Tests for date auto-detection and date similarity in flag_similar_records function.

This module tests the date auto-detection feature and date similarity calculations
added to the flag_similar_records function.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from freamon.deduplication.flag_duplicates import (
    flag_similar_records,
)


class TestDateAutoDetection:
    """Test class for date auto-detection in flag_similar_records."""
    
    @pytest.fixture
    def sample_df_with_dates(self):
        """Create a sample dataframe with various date formats."""
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i*30) for i in range(10)]
        
        # Create records with strongly similar attributes
        # but different dates (within and outside threshold)
        return pd.DataFrame({
            "id": list(range(1, 6)),
            "name": ["John Smith"] * 5,
            "amount": [100.0] * 5,
            "date_native": [
                dates[0],                        # Base date
                dates[0] + timedelta(days=5),    # Within threshold (5 days)
                dates[0] + timedelta(days=15),   # Within threshold (15 days)
                dates[0] + timedelta(days=45),   # Outside threshold (45 days)
                dates[3]                         # Different date
            ],
            "date_string": [d.strftime('%Y-%m-%d') for d in [
                dates[0],                        # Base date
                dates[0] + timedelta(days=5),    # Within threshold (5 days)
                dates[0] + timedelta(days=15),   # Within threshold (15 days)
                dates[0] + timedelta(days=45),   # Outside threshold (45 days)
                dates[3]                         # Different date
            ]],
            "date_timestamp": [(d - datetime(1970, 1, 1)).total_seconds() for d in [
                dates[0],                        # Base date
                dates[0] + timedelta(days=5),    # Within threshold (5 days)
                dates[0] + timedelta(days=15),   # Within threshold (15 days)
                dates[0] + timedelta(days=45),   # Outside threshold (45 days)
                dates[3]                         # Different date
            ]]
        })
    
    @pytest.fixture
    def mixed_date_df(self):
        """Create a dataframe with mixed date formats."""
        base_date = datetime(2023, 1, 15)
        return pd.DataFrame({
            "id": range(1, 7),
            "name": ["Person A"] * 6,
            "amount": [100.0] * 6,
            "date_column": [
                base_date,                       # Native datetime
                base_date.strftime('%Y-%m-%d'),  # ISO format
                base_date.strftime('%m/%d/%Y'),  # MM/DD/YYYY
                base_date.strftime('%d/%m/%Y'),  # DD/MM/YYYY
                base_date.toordinal() - 693594,  # Excel date (2023-01-15)
                int((base_date - datetime(1970, 1, 1)).total_seconds()),  # Unix timestamp
            ]
        })
    
    def test_date_autodetection_with_native_dates(self, sample_df_with_dates):
        """Test auto-detection of native datetime columns."""
        result = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_native"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=20,  # Detect dates within 20 days
            flag_column="is_similar"
        )
        
        # Records 1 and 2 should be similar to record 0 (within threshold)
        # Records 3 and 4 should not be similar (outside threshold)
        assert not result.loc[0, "is_similar"]  # First record is not marked similar
        assert result.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result.loc[3, "is_similar"]  # 45 days difference, should not be similar
        assert not result.loc[4, "is_similar"]  # Different date, should not be similar
    
    def test_date_autodetection_with_string_dates(self, sample_df_with_dates):
        """Test auto-detection of date strings."""
        result = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_string"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=20,  # Detect dates within 20 days
            flag_column="is_similar"
        )
        
        # Records 1 and 2 should be similar to record 0 (within threshold)
        # Records 3 and 4 should not be similar (outside threshold)
        assert not result.loc[0, "is_similar"]  # First record is not marked similar
        assert result.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result.loc[3, "is_similar"]  # 45 days difference, should not be similar
        assert not result.loc[4, "is_similar"]  # Different date, should not be similar
    
    def test_date_autodetection_with_timestamps(self, sample_df_with_dates):
        """Test auto-detection of Unix timestamps."""
        result = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_timestamp"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=20,  # Detect dates within 20 days
            flag_column="is_similar"
        )
        
        # Records 1 and 2 should be similar to record 0 (within threshold)
        # Records 3 and 4 should not be similar (outside threshold)
        assert not result.loc[0, "is_similar"]  # First record is not marked similar
        assert result.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result.loc[3, "is_similar"]  # 45 days difference, should not be similar
        assert not result.loc[4, "is_similar"]  # Different date, should not be similar
    
    def test_mixed_date_formats_detection(self, mixed_date_df):
        """Test detection of mixed date formats in a single column."""
        # Create a copy with an additional record that's slightly different in date
        modified_df = mixed_date_df.copy()
        new_row = modified_df.iloc[0].copy()
        new_row["id"] = 7
        new_row["date_column"] = datetime(2023, 1, 18)  # 3 days difference
        modified_df = pd.concat([modified_df, pd.DataFrame([new_row])], ignore_index=True)
        
        result = flag_similar_records(
            modified_df,
            columns=["name", "amount", "date_column"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=5,  # Smaller threshold (5 days)
            flag_column="is_similar"
        )
        
        # The new record (index 6) should be similar to the records with similar dates
        assert result.loc[6, "is_similar"]
    
    def test_explicitly_specified_date_columns(self, sample_df_with_dates):
        """Test with explicitly specified date columns."""
        result = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_native"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=False,  # Turn off auto-detection
            date_columns=["date_native"],  # Explicitly specify
            date_threshold_days=20,  # Detect dates within 20 days
            flag_column="is_similar"
        )
        
        # Verify explicitly specified date column was used
        assert not result.loc[0, "is_similar"]  # First record is not marked similar
        assert result.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result.loc[3, "is_similar"]  # 45 days difference, should not be similar
        assert not result.loc[4, "is_similar"]  # Different date, should not be similar
    
    def test_date_similarity_methods(self, sample_df_with_dates):
        """Test different date similarity methods."""
        # Create result with similarity column to check similarity values
        # Sample with smaller date difference (5 days) for clearer results
        test_df = sample_df_with_dates.iloc[[0, 1]].copy()  # 0 and 5 days
        
        # Test linear method
        result_linear = flag_similar_records(
            test_df,
            columns=["name", "amount", "date_native"],
            threshold=0.70,
            auto_detect_dates=True,
            date_threshold_days=30,
            date_similarity_method="linear",
            flag_column="is_similar",
            similarity_column="similarity"
        )
        
        # Test exponential method
        result_exp = flag_similar_records(
            test_df,
            columns=["name", "amount", "date_native"],
            threshold=0.70,
            auto_detect_dates=True,
            date_threshold_days=30,
            date_similarity_method="exponential",
            flag_column="is_similar",
            similarity_column="similarity"
        )
        
        # Test threshold method
        result_threshold = flag_similar_records(
            test_df,
            columns=["name", "amount", "date_native"],
            threshold=0.70,
            auto_detect_dates=True,
            date_threshold_days=30,
            date_similarity_method="threshold",
            flag_column="is_similar",
            similarity_column="similarity"
        )
        
        # All methods should detect row 1 as similar to row 0
        assert result_linear.loc[1, "is_similar"]
        assert result_exp.loc[1, "is_similar"]
        assert result_threshold.loc[1, "is_similar"]
        
        # Linear similarity should be less than 1.0 but greater than 0.8 (5 days out of 30)
        assert result_linear.loc[1, "similarity"] < 1.0
        assert result_linear.loc[1, "similarity"] > 0.8
        
        # Exponential similarity should be less than linear similarity
        assert result_exp.loc[1, "similarity"] < result_linear.loc[1, "similarity"]
        
        # Threshold method should be 1.0 (binary)
        assert result_threshold.loc[1, "similarity"] == 1.0
    
    def test_date_threshold_behavior(self, sample_df_with_dates):
        """Test different date threshold values."""
        # Test with 10-day threshold - should detect dates within 10 days
        result_strict = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_native"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=10,  # Strict threshold (10 days)
            flag_column="is_similar"
        )
        
        # Test with 20-day threshold - should detect dates within 20 days
        result_lenient = flag_similar_records(
            sample_df_with_dates,
            columns=["name", "amount", "date_native"],
            threshold=0.85,  # Higher threshold to ensure date is significant
            auto_detect_dates=True,
            date_threshold_days=20,  # Lenient threshold (20 days)
            flag_column="is_similar"
        )
        
        # Strict threshold (10 days)
        assert not result_strict.loc[0, "is_similar"]  # First record is not marked similar
        assert result_strict.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert not result_strict.loc[2, "is_similar"]  # 15 days difference, should not be similar
        assert not result_strict.loc[3, "is_similar"]  # 45 days difference, should not be similar
        
        # Lenient threshold (20 days)
        assert not result_lenient.loc[0, "is_similar"]  # First record is not marked similar
        assert result_lenient.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result_lenient.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result_lenient.loc[3, "is_similar"]  # 45 days difference, should not be similar
    
    def test_compatibility_with_existing_parameters(self, sample_df_with_dates):
        """Test compatibility with existing parameters like weights and text_columns."""
        # Add an email column
        df = sample_df_with_dates.copy()
        df["email"] = [f"user{i}@example.com" for i in range(len(df))]
        
        # Use text columns, weights, and date parameters together
        result = flag_similar_records(
            df,
            columns=["name", "amount", "date_native", "email"],
            weights={"name": 0.4, "amount": 0.2, "date_native": 0.3, "email": 0.1},
            threshold=0.80,
            auto_detect_dates=True,
            date_threshold_days=20,
            text_columns=["name", "email"],
            text_threshold=0.8,
            flag_column="is_similar"
        )
        
        # Same behavior as before with combined parameters
        assert not result.loc[0, "is_similar"]  # First record is not marked similar
        assert result.loc[1, "is_similar"]      # 5 days difference, should be similar
        assert result.loc[2, "is_similar"]      # 15 days difference, should be similar
        assert not result.loc[3, "is_similar"]  # 45 days difference, should not be similar
    
    def test_with_polars_dataframe(self, sample_df_with_dates):
        """Test compatibility with polars DataFrames."""
        try:
            import polars as pl
            
            # Convert pandas df to polars
            polars_df = pl.from_pandas(sample_df_with_dates)
            
            # Test with polars df
            result = flag_similar_records(
                polars_df,
                columns=["name", "amount", "date_native"],
                threshold=0.85,  # Higher threshold to ensure date is significant
                auto_detect_dates=True,
                date_threshold_days=20,  # Detect dates within 20 days
                flag_column="is_similar"
            )
            
            # Convert result back to pandas for assertion
            result_pd = result.to_pandas()
            
            # Verify date column was detected
            assert not result_pd.loc[0, "is_similar"]  # First record is not marked similar
            assert result_pd.loc[1, "is_similar"]      # 5 days difference, should be similar
            assert result_pd.loc[2, "is_similar"]      # 15 days difference, should be similar
            assert not result_pd.loc[3, "is_similar"]  # 45 days difference, should not be similar
            
        except ImportError:
            pytest.skip("Polars not installed, skipping polars dataframe test")


if __name__ == "__main__":
    pytest.main()