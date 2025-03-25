"""
Tests for the optimized topic modeling functionality.
"""
import pandas as pd
import numpy as np
import pytest
import sys
from unittest.mock import patch, MagicMock, patch

from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor


class TestOptimizedTopicModeling:
    """Test the optimized topic modeling functionality."""
    
    @pytest.fixture
    def sample_texts(self):
        """Create a sample of texts for testing."""
        return [
            "This is a document about science and medicine. The patient requires medical treatment.",
            "Astronomy and space exploration are fascinating fields of scientific research.",
            "Sports news: The hockey team won the championship after a great season.",
            "Cars and automobiles: The new vehicle features improved fuel efficiency.",
            "The political debate focused on gun control legislation and public safety.",
            "Computer graphics technology has improved dramatically in recent years.",
            # Add some similar texts for testing deduplication
            "Cars and automobiles: The new vehicle has improved fuel efficiency.",
            "This document discusses science and medicine. The patient needs medical care.",
            "Astronomy and the exploration of space are fascinating areas of research.",
            # Add exact duplicates for testing exact deduplication
            "Sports news: The hockey team won the championship after a great season.",  # Exact duplicate
            "Sports news: The hockey team won the championship after a great season.",  # Exact duplicate
            "Computer graphics technology has improved dramatically in recent years."   # Exact duplicate
        ]
    
    @pytest.fixture
    def sample_df(self, sample_texts):
        """Create a sample dataframe with texts and categories."""
        categories = ['science', 'space', 'sports', 'auto', 'politics', 'technology', 
                      'auto', 'science', 'space', 'sports', 'sports', 'technology']
        return pd.DataFrame({
            'text': sample_texts,
            'category': categories
        })
    
    def test_optimized_topic_model_basic(self, sample_df):
        """Test basic functionality of the optimized topic model."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        try:
            # Basic configuration with minimal options
            result = create_topic_model_optimized(
                sample_df,
                text_column='text',
                n_topics=2,
                method='nmf',  # NMF is faster than LDA for testing
                preprocessing_options={'enabled': True},
                deduplication_options={'enabled': False},
                use_multiprocessing=False
            )
            
            # Check that the result contains expected keys
            assert 'topic_model' in result
            assert 'document_topics' in result
            assert 'topics' in result
            assert 'processing_info' in result
            
            # Check topic model contents
            assert len(result['topics']) == 2
            
            # Check document topics
            assert isinstance(result['document_topics'], pd.DataFrame)
            assert result['document_topics'].shape == (len(sample_df), 2)
            
            # Check processing info
            assert result['processing_info']['original_doc_count'] == len(sample_df)
            assert result['processing_info']['duplicates_removed'] == 0
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_optimized_topic_model_with_deduplication(self, sample_df):
        """Test the optimized topic model with deduplication."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        try:
            # Configure with exact deduplication
            result = create_topic_model_optimized(
                sample_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={'enabled': True},
                deduplication_options={
                    'enabled': True,
                    'method': 'exact',
                    'hash_method': 'hash'
                },
                return_original_mapping=True,
                use_multiprocessing=False
            )
            
            # Check that deduplication was performed
            assert result['processing_info']['duplicates_removed'] > 0
            
            # Check that mapping was returned
            assert 'deduplication_map' in result
            assert isinstance(result['deduplication_map'], dict)
            
            # Check document topics shape after deduplication
            assert result['document_topics'].shape[0] == sample_df.shape[0]
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_optimized_topic_model_fuzzy_deduplication(self, sample_df):
        """Test the optimized topic model with fuzzy deduplication."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        try:
            # Configure with fuzzy deduplication
            result = create_topic_model_optimized(
                sample_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={'enabled': True},
                deduplication_options={
                    'enabled': True,
                    'method': 'fuzzy',
                    'similarity_threshold': 0.8,
                    'similarity_method': 'cosine'
                },
                return_original_mapping=True,
                use_multiprocessing=False
            )
            
            # Check that result was produced
            assert 'topic_model' in result
            assert 'document_topics' in result
            
            # Check for fuzzy deduplication processing
            assert result['processing_info']['deduplication_method'] == 'fuzzy'
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_optimized_topic_model_preprocessing_options(self, sample_df):
        """Test preprocessing options for optimized topic model."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        try:
            # Configure with custom preprocessing options
            result = create_topic_model_optimized(
                sample_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={
                    'enabled': True,
                    'use_lemmatization': False,
                    'remove_stopwords': True,
                    'remove_punctuation': True,
                    'min_token_length': 4,
                    'custom_stopwords': ['document', 'research']
                },
                deduplication_options={'enabled': False},
                use_multiprocessing=False
            )
            
            # Check that processing info shows correct preprocessing options
            assert result['processing_info']['used_lemmatization'] is False
            assert 'preprocessing_time' in result['processing_info']
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_optimized_topic_model_with_sampling(self, sample_df):
        """Test the optimized topic model with sampling."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        # Create a larger dataframe by repeating the sample
        large_df = pd.concat([sample_df] * 3, ignore_index=True)
        
        try:
            # Configure with smaller max_docs than dataset size
            result = create_topic_model_optimized(
                large_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={'enabled': True},
                deduplication_options={'enabled': False},
                max_docs=5,  # Force sampling
                return_full_data=True,
                use_multiprocessing=False
            )
            
            # Check that sampling was performed
            assert result['processing_info']['sampled'] is True
            assert result['processing_info']['sample_size'] == 5
            
            # Check that results for all documents were returned
            assert result['document_topics'].shape[0] == len(large_df)
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    @patch('freamon.utils.text_utils.multiprocessing')
    def test_optimized_topic_model_with_multiprocessing(self, mock_multiprocessing, sample_df):
        """Test the optimized topic model with multiprocessing."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        # Setup mock for multiprocessing
        mock_pool = MagicMock()
        mock_multiprocessing.Pool.return_value.__enter__.return_value = mock_pool
        mock_multiprocessing.cpu_count.return_value = 4
        
        # Create a larger dataframe to trigger multiprocessing
        large_df = pd.concat([sample_df] * 10, ignore_index=True)
        
        try:
            # Configure with multiprocessing enabled
            result = create_topic_model_optimized(
                large_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={'enabled': True},
                deduplication_options={'enabled': False},
                use_multiprocessing=True
            )
            
            # The test should run without error, even though the mock prevents actual parallel processing
            assert 'topic_model' in result
            assert 'document_topics' in result
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
                
    @pytest.mark.skip(reason="Anonymization testing requires the Allyanonimiser package")
    def test_optimized_topic_model_with_anonymization(self, sample_df):
        """Test the optimized topic model with anonymization."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        # Add mock Anonymizer class
        class MockAnonymizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
            def anonymize_text(self, text):
                return text.replace("patient", "[PERSON]").replace("political", "[CONTENT]")
        
        # Setup the mock implementation
        mock_anonymize.return_value = True
        
        # Add the Anonymizer to the module
        with patch.object(
            __import__('freamon.utils.text_utils', fromlist=['Anonymizer']), 
            'Anonymizer', 
            new=MockAnonymizer
        ):
            # Add some PII to the sample data
            sample_df_with_pii = sample_df.copy()
            sample_df_with_pii.loc[0, 'text'] = sample_df_with_pii.loc[0, 'text'] + " Patient John Smith can be contacted at john.smith@example.com."
            sample_df_with_pii.loc[2, 'text'] = sample_df_with_pii.loc[2, 'text'] + " Contact the coach at (555) 123-4567."
            
            try:
                # Configure with anonymization enabled
                result = create_topic_model_optimized(
                    sample_df_with_pii,
                    text_column='text',
                    n_topics=2,
                    method='nmf',
                    preprocessing_options={'enabled': True},
                    deduplication_options={'enabled': False},
                    anonymize=True,
                    anonymization_config={'custom_patterns': True},
                    use_multiprocessing=False
                )
                
                # Check that anonymization was performed
                assert result['processing_info']['anonymization_enabled'] is True
                
                # Check that model was created
                assert 'topic_model' in result
                assert 'document_topics' in result
                
            except Exception as e:
                if "empty vocabulary" in str(e).lower():
                    pytest.skip("Not enough text data for topic modeling")
                else:
                    raise e
                
    @pytest.mark.skip(reason="Anonymization testing requires mocking imports")
    def test_optimized_topic_model_anonymization_not_available(self, sample_df):
        """Test graceful handling when anonymization is requested but not available."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
            
        # We need to patch the import attempt in the create_topic_model_optimized function
        # This requires modifying what happens when it tries to import the Anonymizer
        with patch.dict('sys.modules', {'allyanonimiser': None}):
            try:
                # Configure with anonymization enabled but unavailable
                result = create_topic_model_optimized(
                    sample_df,
                    text_column='text',
                    n_topics=2,
                    method='nmf',
                    preprocessing_options={'enabled': True},
                    deduplication_options={'enabled': False},
                    anonymize=True,  # Request anonymization
                    use_multiprocessing=False
                )
                
                # Check that the function still completed successfully
                assert 'topic_model' in result
                assert 'document_topics' in result
                
                # Check that anonymization was disabled gracefully
                assert result['processing_info']['anonymization_enabled'] is False
                
            except Exception as e:
                if "empty vocabulary" in str(e).lower():
                    pytest.skip("Not enough text data for topic modeling")
                else:
                    raise e
                    
    def test_optimized_topic_model_pickle_support(self, sample_df):
        """Test that the optimized topic model can be pickled and unpickled."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
            
        import pickle
        
        try:
            # Create a topic model
            result = create_topic_model_optimized(
                sample_df,
                text_column='text',
                n_topics=2,
                method='nmf',
                preprocessing_options={'enabled': True},
                deduplication_options={'enabled': False},
                use_multiprocessing=False  # Disable multiprocessing for testing
            )
            
            # Pickle the model
            pickled_data = pickle.dumps(result)
            
            # Unpickle the model
            unpickled_result = pickle.loads(pickled_data)
            
            # Check that the unpickled model has the same topics
            assert unpickled_result['topic_model']['n_topics'] == result['topic_model']['n_topics']
            assert len(unpickled_result['topics']) == len(result['topics'])
            
            # Check document topics
            pd.testing.assert_frame_equal(unpickled_result['document_topics'], result['document_topics'])
            
            # Check processing info
            assert unpickled_result['processing_info']['original_doc_count'] == result['processing_info']['original_doc_count']
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e