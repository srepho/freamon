"""
Tests for the utils.encoders module.
"""
import numpy as np
import pandas as pd
import pytest

from freamon.utils.encoders import (
    OneHotEncoderWrapper,
    OrdinalEncoderWrapper,
    TargetEncoderWrapper,
)


class TestOneHotEncoderWrapper:
    """Test class for OneHotEncoderWrapper."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "C", "A", "B"],
            "binary": ["X", "Y", "X", "Y", "X"],
        })
    
    def test_fit_transform(self, sample_df):
        """Test that fit_transform correctly encodes categorical columns."""
        encoder = OneHotEncoderWrapper(drop="first")
        result = encoder.fit_transform(sample_df)
        
        # Check that the result has the expected columns
        assert "numeric" in result.columns
        assert "categorical_B" in result.columns
        assert "categorical_C" in result.columns
        assert "binary_Y" in result.columns
        
        # Check that 'categorical_A' is dropped
        assert "categorical_A" not in result.columns
        
        # Check that the binary column is correctly encoded
        assert result["binary_Y"].tolist() == [0, 1, 0, 1, 0]
    
    def test_columns_param(self, sample_df):
        """Test that the columns parameter correctly limits encoding."""
        encoder = OneHotEncoderWrapper(columns=["categorical"])
        result = encoder.fit_transform(sample_df)
        
        # Check that only 'categorical' is encoded
        assert "binary" in result.columns
        assert "categorical_A" in result.columns
        assert "categorical_B" in result.columns
        assert "categorical_C" in result.columns
        
        # Check that 'binary' is not encoded
        assert "binary_X" not in result.columns
        assert "binary_Y" not in result.columns


class TestOrdinalEncoderWrapper:
    """Test class for OrdinalEncoderWrapper."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "C", "A", "B"],
            "binary": ["X", "Y", "X", "Y", "X"],
        })
    
    def test_fit_transform(self, sample_df):
        """Test that fit_transform correctly encodes categorical columns."""
        encoder = OrdinalEncoderWrapper()
        result = encoder.fit_transform(sample_df)
        
        # Check that the columns are preserved
        assert set(result.columns) == set(sample_df.columns)
        
        # Check that categorical values are encoded consistently
        assert result["categorical"][0] == result["categorical"][3]  # both 'A'
        assert result["categorical"][1] == result["categorical"][4]  # both 'B'
        
        # Check that binary values are encoded
        assert result["binary"][0] == result["binary"][2]  # both 'X'
        assert result["binary"][1] == result["binary"][3]  # both 'Y'
    
    def test_unknown_values(self):
        """Test handling of unknown values."""
        train_df = pd.DataFrame({
            "cat": ["A", "B", "C"]
        })
        test_df = pd.DataFrame({
            "cat": ["A", "D", "B"]
        })
        
        encoder = OrdinalEncoderWrapper(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train_df)
        result = encoder.transform(test_df)
        
        # Check that 'D' is encoded as -1
        assert result["cat"][1] == -1


class TestTargetEncoderWrapper:
    """Test class for TargetEncoderWrapper."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "C", "A", "B"],
            "binary": ["X", "Y", "X", "Y", "X"],
            "target": [0, 1, 0, 1, 0],
        })
    
    def test_fit_transform(self, sample_df):
        """Test that fit_transform correctly encodes categorical columns."""
        encoder = TargetEncoderWrapper(smoothing=1.0)
        result = encoder.fit_transform(sample_df, "target")
        
        # Check that the columns are preserved
        assert set(result.columns) == set(sample_df.columns)
        
        # Check that values for the same category are encoded the same
        assert result["categorical"][0] == result["categorical"][3]  # both 'A'
        assert result["categorical"][1] == result["categorical"][4]  # both 'B'
        
        # Check that encoding reflects target associations
        # 'A' is associated with [0, 1], avg = 0.5
        # 'B' is associated with [1, 0], avg = 0.5
        # 'C' is associated with [0], avg = 0
        # With smoothing, these will be adjusted
        assert result["categorical"][2] < result["categorical"][0]  # 'C' < 'A'
    
    def test_unknown_values(self):
        """Test handling of unknown values."""
        train_df = pd.DataFrame({
            "cat": ["A", "B", "C", "A", "B"],
            "target": [0, 1, 0, 1, 0],
        })
        test_df = pd.DataFrame({
            "cat": ["A", "D", "B"],
            "target": [0, 0, 0],  # not used for transform
        })
        
        encoder = TargetEncoderWrapper(handle_unknown="value")
        encoder.fit(train_df, "target")
        result = encoder.transform(test_df)
        
        # Check that 'D' is encoded as the global mean
        global_mean = train_df["target"].mean()
        assert result["cat"][1] == global_mean
        
        # Check that 'A' and 'B' are encoded correctly
        assert result["cat"][0] != global_mean
        assert result["cat"][2] != global_mean