"""
Tests for automatic topic number detection in text_utils.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor


class TestAutoTopicDetection:
    """Test class for automatic topic number detection."""
    
    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for testing."""
        return [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Deep learning is a subset of machine learning in artificial intelligence that has networks.",
            "Natural language processing is a field of artificial intelligence related to interactions.",
            "Computer vision is an interdisciplinary field that deals with how computers can gain understanding.",
            "AI ethics is a system of moral principles and techniques intended to inform AI development.",
            "Reinforcement learning is an area of machine learning concerned with how software agents.",
            "Neural networks are computing systems vaguely inspired by the biological neural networks.",
            "Supervised learning is the machine learning task of learning a function that maps input to output.",
            "Unsupervised learning is a type of machine learning that looks for patterns in data set.",
            "Semi-supervised learning is an approach to machine learning that combines labeled and unlabeled data."
        ]
    
    @pytest.fixture
    def sample_df(self, sample_texts):
        """Create a sample dataframe with texts for testing."""
        return pd.DataFrame({
            "text": sample_texts,
            "id": list(range(len(sample_texts)))
        })
    
    def test_auto_topic_detection_default(self, sample_df):
        """Test auto topic detection with default parameters."""
        # Run with automatic topic detection
        result = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(2, 5),  # Small range for testing
            preprocessing_options={'enabled': True, 'remove_stopwords': True}
        )
        
        # Check that topic selection info is in the result
        assert 'topic_selection' in result
        assert 'best_n_topics' in result['topic_selection']
        assert 'coherence_scores' in result['topic_selection']
        assert 'method' in result['topic_selection']
        assert result['topic_selection']['method'] == 'coherence'
        
        # Check that best_n_topics is within the specified range
        assert 2 <= result['topic_selection']['best_n_topics'] <= 5
        
        # Check that processing_info contains auto topic detection information
        assert 'auto_topic_detection' in result['processing_info']
        assert result['processing_info']['auto_topic_detection'] is True
        assert 'best_n_topics' in result['processing_info']
        assert result['processing_info']['best_n_topics'] == result['topic_selection']['best_n_topics']
    
    def test_auto_topic_detection_lda(self, sample_df):
        """Test auto topic detection with LDA."""
        # Run with automatic topic detection using LDA
        result = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='lda',
            auto_topics_range=(2, 4),  # Small range for testing
            preprocessing_options={'enabled': True, 'remove_stopwords': True}
        )
        
        # Check that topic selection info is in the result
        assert 'topic_selection' in result
        assert 'best_n_topics' in result['topic_selection']
        assert 2 <= result['topic_selection']['best_n_topics'] <= 4
        
        # Verify that the correct model was created
        assert result['topic_model']['method'] == 'lda'
        assert result['topic_model']['n_topics'] == result['topic_selection']['best_n_topics']
        
    def test_stability_method(self, sample_df):
        """Test auto topic detection using the stability method."""
        # Run with automatic topic detection using stability method
        result = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(2, 4),
            auto_topics_method='stability',
            preprocessing_options={'enabled': True, 'remove_stopwords': True}
        )
        
        # Check that topic selection info includes stability scores
        assert 'topic_selection' in result
        assert 'stability_scores' in result['topic_selection']
        assert len(result['topic_selection']['stability_scores']) == 3  # For topics 2, 3, 4
        
        # The method should be reported correctly
        assert result['topic_selection']['method'] == 'stability'
        assert result['processing_info']['topic_selection_method'] == 'stability'
        
        # Validate best_n_topics is in range
        assert 2 <= result['topic_selection']['best_n_topics'] <= 4
    
    def test_auto_topic_detection_with_fixed_topics(self, sample_df):
        """Test that fixed n_topics bypasses auto detection."""
        # Run with fixed topic number
        result = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics=3,  # Fixed number of topics
            method='nmf'
        )
        
        # Check that topic_selection is not in the result
        assert 'topic_selection' not in result
        
        # Check that processing_info indicates auto detection was not used
        assert 'auto_topic_detection' in result['processing_info']
        assert result['processing_info']['auto_topic_detection'] is False
        
        # Verify that the correct number of topics was used
        assert result['topic_model']['n_topics'] == 3
    
    @patch('freamon.utils.text_utils.TextProcessor.create_topic_model')
    def test_auto_topic_selection_mechanism(self, mock_create_topic, sample_df):
        """Test the mechanism for selecting the optimal number of topics."""
        # Create mock models with different coherence scores
        coherence_values = {
            2: 0.1,
            3: 0.3,  # Best score
            4: 0.2,
            5: 0.15
        }
        
        # Setup the mock to return models with predetermined coherence scores
        def side_effect(texts, n_topics, **kwargs):
            # Create a topic term matrix with some controlled variation
            # Topics should be more distinct for n_topics=3 (our desired outcome)
            topic_term_matrix = np.zeros((n_topics, 10))
            for i in range(n_topics):
                # Assign values to create artificially distinct topics for n_topics=3
                if n_topics == 3:
                    # Create very distinct topics
                    start_idx = i * 3
                    topic_term_matrix[i, start_idx:start_idx+3] = np.random.rand(3) + 0.5
                else:
                    # Create less distinct topics with more overlap
                    topic_term_matrix[i] = np.random.rand(10) * 0.3
                    topic_term_matrix[i, i % 10] = 0.8  # Ensure some distinctiveness
            
            model = {
                'coherence_score': coherence_values.get(n_topics, 0),
                'topics': [(i, [f"word{j}" for j in range(5)]) for i in range(n_topics)],
                'method': kwargs.get('method', 'nmf'),
                'n_topics': n_topics,
                'model': MagicMock(),
                'vectorizer': MagicMock(),
                'topic_term_matrix': topic_term_matrix,
                'doc_topic_matrix': np.zeros((len(sample_df), n_topics)),
                'feature_names': [f"word{j}" for j in range(10)]
            }
            return model
        
        mock_create_topic.side_effect = side_effect
        
        # Run auto topic detection with coherence method
        result_coherence = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(2, 5),
            auto_topics_method='coherence'
        )
        
        # Verify that the best model was selected based on coherence (n_topics=3)
        assert result_coherence['topic_selection']['best_n_topics'] == 3
        assert result_coherence['topic_model']['n_topics'] == 3
        
        # Verify that coherence scores were captured correctly
        expected_scores = [coherence_values[t] for t in range(2, 6)]
        assert result_coherence['topic_selection']['coherence_scores'] == expected_scores
        
        # Run auto topic detection with stability method
        result_stability = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(2, 5),
            auto_topics_method='stability'
        )
        
        # Verify that stability scores were captured
        assert 'stability_scores' in result_stability['topic_selection']
        assert len(result_stability['topic_selection']['stability_scores']) == 4  # For topics 2-5
        
        # Optimal number should still be 3 for our mock data
        assert result_stability['topic_selection']['best_n_topics'] == 3
    
    def test_auto_topic_detection_with_extreme_ranges(self, sample_df):
        """Test auto topic detection with extreme ranges."""
        # Run with a very narrow range
        result_narrow = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(3, 3),  # Only one possibility
            preprocessing_options={'enabled': True}
        )
        
        # When there's only one option, it must be selected
        assert result_narrow['topic_selection']['best_n_topics'] == 3
        assert len(result_narrow['topic_selection']['coherence_scores']) == 1
        
        # Run with a range of 2 options
        result_small = create_topic_model_optimized(
            sample_df,
            'text',
            n_topics='auto',
            method='nmf',
            auto_topics_range=(2, 3),  # Two possibilities
            preprocessing_options={'enabled': True}
        )
        
        # Check that selection was made between 2 options
        assert 2 <= result_small['topic_selection']['best_n_topics'] <= 3
        assert len(result_small['topic_selection']['coherence_scores']) == 2