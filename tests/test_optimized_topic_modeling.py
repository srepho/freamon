"""
Tests for the optimized topic modeling functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from freamon.utils.text_utils import create_topic_model_optimized

class TestOptimizedTopicModeling:
    """Test cases for the optimized topic modeling functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample text data for testing."""
        texts = [
            "This is a document about machine learning and artificial intelligence",
            "Machine learning models can process text data efficiently",
            "Deep learning is a subset of machine learning",
            "Natural language processing helps computers understand text",
            "This is a duplicate document about machine learning and artificial intelligence",
            "Text mining and NLP are related fields"
        ]
        
        categories = ["tech", "tech", "tech", "nlp", "tech", "nlp"]
        
        return pd.DataFrame({
            "text": texts,
            "category": categories
        })
    
    def test_basic_functionality(self, sample_data):
        """Test that the basic functionality works correctly."""
        # Run with minimal options
        result = create_topic_model_optimized(
            sample_data,
            text_column="text",
            n_topics=2,
            method="nmf",
            max_docs=None,
            preprocessing_options={"enabled": True},
            deduplication_options={"enabled": True},
            return_full_data=True
        )
        
        # Check that the result contains the expected keys
        assert "topic_model" in result
        assert "document_topics" in result
        assert "topics" in result
        assert "processing_info" in result
        
        # Check that the correct number of topics was created
        assert len(result["topics"]) == 2
        
        # Check that duplicate was removed
        assert result["processing_info"]["duplicates_removed"] == 1
        
        # Check document-topic distributions shape
        doc_topics = result["document_topics"]
        assert doc_topics.shape[0] == len(sample_data)  # All documents
        assert doc_topics.shape[1] == 2  # 2 topics
    
    def test_no_preprocessing(self, sample_data):
        """Test with preprocessing disabled."""
        result = create_topic_model_optimized(
            sample_data,
            text_column="text",
            n_topics=2,
            preprocessing_options={"enabled": False},
            deduplication_options={"enabled": False}
        )
        
        # Check that preprocessing was disabled
        assert result["processing_info"]["preprocessing_enabled"] is False
        
        # Check that duplicates were not removed
        assert result["processing_info"]["duplicates_removed"] == 0
    
    def test_fuzzy_deduplication(self, sample_data):
        """Test fuzzy deduplication."""
        # Add a document with slight variation
        df = sample_data.copy()
        df.loc[len(df)] = {
            "text": "This is a document about machine learning & artificial intelligence",
            "category": "tech"
        }
        
        result = create_topic_model_optimized(
            df,
            text_column="text",
            n_topics=2,
            deduplication_options={
                "enabled": True,
                "method": "fuzzy",
                "similarity_threshold": 0.8,
                "similarity_method": "cosine"
            }
        )
        
        # Check that fuzzy deduplication was used
        assert result["processing_info"]["deduplication_method"] == "fuzzy"
        
        # Should have found the near-duplicate
        assert result["processing_info"]["duplicates_removed"] >= 1
    
    def test_return_original_mapping(self, sample_data):
        """Test returning the original mapping."""
        result = create_topic_model_optimized(
            sample_data,
            text_column="text",
            n_topics=2,
            deduplication_options={"enabled": True},
            return_original_mapping=True
        )
        
        # Check that the deduplication map was returned
        assert "deduplication_map" in result
        
        # Map should not be empty
        assert len(result["deduplication_map"]) > 0
        
        # Check that the mapping has the correct structure
        for key, indices in result["deduplication_map"].items():
            assert isinstance(key, (int, np.integer))
            assert isinstance(indices, list)
            assert key in indices  # The key should be in its own indices
    
    @patch("multiprocessing.cpu_count", return_value=4)
    def test_multiprocessing(self, mock_cpu_count, sample_data):
        """Test multiprocessing configuration."""
        # Create larger dataset to trigger multiprocessing
        large_df = pd.DataFrame({
            "text": ["Document " + str(i) for i in range(15000)],
            "category": ["cat" + str(i % 5) for i in range(15000)]
        })
        
        with patch("multiprocessing.Pool") as mock_pool:
            # Mock the Pool.imap method to return a simple iterator
            mock_instance = mock_pool.return_value.__enter__.return_value
            mock_instance.imap.return_value = iter([[f"processed_{i}"] for i in range(10)])
            
            result = create_topic_model_optimized(
                large_df,
                text_column="text",
                n_topics=2,
                max_docs=100,  # Limit to avoid long test
                use_multiprocessing=True
            )
            
            # Check that multiprocessing was configured
            assert mock_pool.called
            
            # Should show in processing info
            assert result["processing_info"]["multiprocessing_enabled"] is True
    
    def test_sampling(self, sample_data):
        """Test sampling for large datasets."""
        # Create larger dataset to trigger sampling
        large_df = pd.DataFrame({
            "text": ["Document " + str(i) for i in range(1000)],
            "category": ["cat" + str(i % 5) for i in range(1000)]
        })
        
        # Set max_docs to a small number to force sampling
        result = create_topic_model_optimized(
            large_df,
            text_column="text",
            n_topics=2,
            max_docs=100
        )
        
        # Check that sampling was used
        assert result["processing_info"]["sampled"] is True
        assert result["processing_info"]["sample_size"] == 100
        
        # Document topics should still cover all documents
        assert result["document_topics"].shape[0] == len(large_df)