"""
Tests for the utils.text_utils module.
"""
import pandas as pd
import pytest

from freamon.utils.text_utils import TextProcessor


class TestTextProcessor:
    """Test class for TextProcessor."""
    
    @pytest.fixture
    def sample_text(self):
        """Create sample text for testing."""
        return "This is a sample text with numbers like 123 and punctuation marks!"
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "text": [
                "This is a sample text with numbers like 123!",
                "Another example text with different words.",
                "More text for testing with some repeated words words.",
            ],
        })
    
    def test_preprocess_text_basic(self, sample_text):
        """Test basic text preprocessing without spaCy."""
        processor = TextProcessor(use_spacy=False)
        
        # Test lowercase
        result = processor.preprocess_text(sample_text, lowercase=True)
        assert result.lower() == result
        
        # Test remove punctuation
        result = processor.preprocess_text(sample_text, remove_punctuation=True)
        assert "!" not in result
        
        # Test remove numbers
        result = processor.preprocess_text(sample_text, remove_numbers=True)
        assert "123" not in result
        
        # Test all options
        result = processor.preprocess_text(
            sample_text,
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True,
        )
        assert result.lower() == result
        assert "!" not in result
        assert "123" not in result
    
    def test_process_dataframe_column(self, sample_df):
        """Test processing a text column in a dataframe."""
        processor = TextProcessor(use_spacy=False)
        
        # Test basic preprocessing
        result = processor.process_dataframe_column(
            sample_df,
            "text",
            lowercase=True,
            remove_punctuation=True,
        )
        
        # Check that punctuation is removed
        assert "!" not in result["text"][0]
        
        # Check that text is lowercased
        assert result["text"][0] == result["text"][0].lower()
        
        # Test with output to a new column
        result = processor.process_dataframe_column(
            sample_df,
            "text",
            result_column="processed_text",
            lowercase=True,
        )
        
        # Check that a new column is created
        assert "processed_text" in result.columns
        
        # Check that the original column is preserved
        assert result["text"][0] == sample_df["text"][0]
    
    def test_create_bow_features(self, sample_df):
        """Test creating bag-of-words features."""
        processor = TextProcessor(use_spacy=False)
        
        # Create BOW features
        bow_df = processor.create_bow_features(
            sample_df,
            "text",
            max_features=10,
            binary=False,
        )
        
        # Check that the result has the expected columns
        assert all(col.startswith("bow_") for col in bow_df.columns)
        
        # Check that the values are counts
        assert (bow_df >= 0).all().all()
        assert (bow_df.astype(int) == bow_df).all().all()
        
        # Test with binary=True
        bow_binary = processor.create_bow_features(
            sample_df,
            "text",
            max_features=10,
            binary=True,
        )
        
        # Check that the values are binary
        assert set(bow_binary.values.flatten()) <= {0.0, 1.0}
    
    def test_create_tfidf_features(self, sample_df):
        """Test creating TF-IDF features."""
        processor = TextProcessor(use_spacy=False)
        
        # Create TF-IDF features
        tfidf_df = processor.create_tfidf_features(
            sample_df,
            "text",
            max_features=10,
        )
        
        # Check that the result has the expected columns
        assert all(col.startswith("tfidf_") for col in tfidf_df.columns)
        
        # Check that the values are non-negative
        assert (tfidf_df >= 0).all().all()
        
        # Values should be continuous (not just integers)
        assert not (tfidf_df.astype(int) == tfidf_df).all().all()