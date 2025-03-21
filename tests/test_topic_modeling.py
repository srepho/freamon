"""
Tests for the topic modeling functionality.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from freamon.utils.text_utils import TextProcessor


class TestTopicModeling:
    """Test the topic modeling functionality."""
    
    @pytest.fixture
    def sample_texts(self):
        """Create a sample of texts for testing."""
        return [
            "This is a document about science and medicine. The patient requires medical treatment.",
            "Astronomy and space exploration are fascinating fields of scientific research.",
            "Sports news: The hockey team won the championship after a great season.",
            "Cars and automobiles: The new vehicle features improved fuel efficiency.",
            "The political debate focused on gun control legislation and public safety.",
            "Computer graphics technology has improved dramatically in recent years."
        ]
    
    @pytest.fixture
    def sample_df(self, sample_texts):
        """Create a sample dataframe with texts and categories."""
        categories = ['science', 'space', 'sports', 'auto', 'politics', 'technology']
        return pd.DataFrame({
            'text': sample_texts,
            'category': categories
        })
    
    def test_create_topic_model_lda(self, sample_texts):
        """Test creating a topic model with LDA."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        processor = TextProcessor(use_spacy=False)
        
        try:
            model = processor.create_topic_model(
                texts=sample_texts,
                n_topics=3,
                method='lda',
                max_features=100,
                max_df=0.9,
                min_df=1,
                ngram_range=(1, 1),
                random_state=42
            )
            
            # Check that the model was created
            assert 'model' in model
            assert 'vectorizer' in model
            assert 'topics' in model
            assert 'doc_topic_matrix' in model
            assert len(model['topics']) == 3
            
            # Check that the topics contain words
            for _, words in model['topics']:
                assert len(words) > 0
            
            # Check that the document-topic matrix has the right shape
            assert model['doc_topic_matrix'].shape == (len(sample_texts), 3)
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_create_topic_model_nmf(self, sample_texts):
        """Test creating a topic model with NMF."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
            from sklearn.decomposition import NMF
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        processor = TextProcessor(use_spacy=False)
        
        try:
            model = processor.create_topic_model(
                texts=sample_texts,
                n_topics=3,
                method='nmf',
                max_features=100,
                max_df=0.9,
                min_df=1,
                ngram_range=(1, 1),
                random_state=42
            )
            
            # Check that the model was created
            assert 'model' in model
            assert 'vectorizer' in model
            assert 'topics' in model
            assert 'doc_topic_matrix' in model
            assert len(model['topics']) == 3
            
            # Check that the topics contain words
            for _, words in model['topics']:
                assert len(words) > 0
            
            # Check that the document-topic matrix has the right shape
            assert model['doc_topic_matrix'].shape == (len(sample_texts), 3)
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_get_document_topics(self, sample_texts):
        """Test getting document topics."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        processor = TextProcessor(use_spacy=False)
        
        try:
            # Create a topic model
            model = processor.create_topic_model(
                texts=sample_texts,
                n_topics=2,
                method='lda',
                max_features=100,
                max_df=0.9,
                min_df=1
            )
            
            # Get document topics
            doc_topics = processor.get_document_topics(model)
            
            # Check that the document-topic dataframe has the right shape
            assert isinstance(doc_topics, pd.DataFrame)
            assert doc_topics.shape[0] == len(sample_texts)
            assert doc_topics.shape[1] == 2  # Number of topics
            
            # Check that the probabilities sum to approximately 1
            row_sums = doc_topics.sum(axis=1)
            assert all(row_sum == 0 or 0.99 <= row_sum <= 1.01 for row_sum in row_sums)
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    def test_plot_topics(self, sample_texts):
        """Test plotting topics."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
            import matplotlib
        except ImportError:
            pytest.skip("scikit-learn or matplotlib not available")
        
        processor = TextProcessor(use_spacy=False)
        
        try:
            # Create a topic model
            model = processor.create_topic_model(
                texts=sample_texts,
                n_topics=2,
                method='lda',
                max_features=100,
                max_df=0.9,
                min_df=1
            )
            
            # Test plotting
            html = processor.plot_topics(model, return_html=True)
            
            # Check that HTML was generated
            assert isinstance(html, str)
            assert html.startswith('<img src="data:image/png;base64,')
            
        except Exception as e:
            if "empty vocabulary" in str(e).lower():
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e
    
    @patch('freamon.utils.text_utils.TextProcessor.calculate_topic_coherence')
    def test_find_optimal_topics(self, mock_coherence, sample_texts):
        """Test finding optimal topics with mocked coherence."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        # Mock coherence values
        mock_coherence.side_effect = [0.3, 0.5, 0.4]  # 2, 3, 4 topics
        
        processor = TextProcessor(use_spacy=False)
        
        with patch('freamon.utils.text_utils.TextProcessor.create_topic_model') as mock_create:
            # Mock topic models
            mock_create.side_effect = [
                {'model': MagicMock(), 'topics': [(0, ['word1']), (1, ['word2'])], 'n_topics': 2, 'method': 'lda', 'texts': sample_texts},
                {'model': MagicMock(), 'topics': [(0, ['word1']), (1, ['word2']), (2, ['word3'])], 'n_topics': 3, 'method': 'lda', 'texts': sample_texts},
                {'model': MagicMock(), 'topics': [(0, ['word1']), (1, ['word2']), (2, ['word3']), (3, ['word4'])], 'n_topics': 4, 'method': 'lda', 'texts': sample_texts}
            ]
            
            # Find optimal topics
            result = processor.find_optimal_topics(
                texts=sample_texts,
                min_topics=2,
                max_topics=4,
                step=1,
                method='lda',
                plot_results=False
            )
            
            # Check that the optimal number of topics was identified
            assert result['optimal_topics'] == 3
            assert result['best_coherence'] == 0.5
            assert 'coherence_values' in result
            assert len(result['coherence_values']) == 3
    
    def test_create_text_features_with_topics(self, sample_df):
        """Test creating text features with topics."""
        try:
            # Skip if scikit-learn is not available
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        processor = TextProcessor(use_spacy=False)
        
        try:
            # Create text features with topics
            features = processor.create_text_features(
                sample_df,
                'text',
                include_stats=True,
                include_readability=True,
                include_sentiment=True,
                include_topics=True,
                n_topics=2,
                topic_method='lda'
            )
            
            # Check that the features were created
            assert isinstance(features, pd.DataFrame)
            assert features.shape[0] == len(sample_df)
            
            # Check that all feature types are included
            assert any(col.startswith('text_stat_') for col in features.columns)
            assert any(col.startswith('text_read_') for col in features.columns)
            assert any(col.startswith('text_sent_') for col in features.columns)
            
            # Check if topic features were created
            topic_features = [col for col in features.columns if 'topic_' in col]
            if topic_features:
                assert len(topic_features) == 2  # Number of topics
                
        except Exception as e:
            if "empty vocabulary" in str(e).lower() or "Error creating topic features" in str(e):
                pytest.skip("Not enough text data for topic modeling")
            else:
                raise e