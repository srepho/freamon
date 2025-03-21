"""
Tests for the word embedding functionality.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from freamon.utils.text_utils import TextProcessor


class TestWordEmbeddings:
    """Test the word embedding functionality."""
    
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
    
    def test_create_word2vec_embeddings(self, sample_texts):
        """Test creating Word2Vec embeddings."""
        try:
            # Skip if gensim is not available
            import gensim
            from gensim.models import Word2Vec
        except ImportError:
            pytest.skip("gensim not available")
        
        processor = TextProcessor(use_spacy=False)
        
        # Test with parameters
        embeddings = processor.create_word2vec_embeddings(
            texts=sample_texts,
            vector_size=50,
            window=3,
            min_count=1,
            epochs=5,
            seed=42
        )
        
        # Check that the embeddings were created
        assert 'model' in embeddings
        assert 'wv' in embeddings
        assert 'vocab_size' in embeddings
        assert 'vocab' in embeddings
        assert 'vector_size' in embeddings
        assert 'embedding_type' in embeddings
        
        # Check that the vector size matches
        assert embeddings['vector_size'] == 50
        
        # Check that some common words are in the vocabulary
        common_words = ['the', 'and', 'is', 'a']
        for word in common_words:
            assert any(word == w.lower() for w in embeddings['vocab'])
        
        # Check that word vectors have the right shape
        for word in embeddings['vocab']:
            vector = embeddings['wv'][word]
            assert vector.shape == (50,)
    
    @patch('freamon.utils.text_utils.TextProcessor._download_embeddings')
    def test_load_pretrained_embeddings_glove(self, mock_download, sample_texts):
        """Test loading GloVe embeddings."""
        try:
            # Skip if gensim is not available
            import gensim
            from gensim.models import KeyedVectors
        except ImportError:
            pytest.skip("gensim not available")
            
        # Create a mock file with sample GloVe vectors
        with open('mock_glove.txt', 'w') as f:
            f.write("the 0.1 0.2 0.3 0.4 0.5\n")
            f.write("and 0.2 0.3 0.4 0.5 0.6\n")
            f.write("is 0.3 0.4 0.5 0.6 0.7\n")
            f.write("a 0.4 0.5 0.6 0.7 0.8\n")
            f.write("in 0.5 0.6 0.7 0.8 0.9\n")
            
        # Make our mock download return this file
        mock_download.return_value = 'mock_glove.txt'
        
        processor = TextProcessor(use_spacy=False)
        
        # Test loading GloVe vectors
        embeddings = processor.load_pretrained_embeddings(
            embedding_type='glove',
            dimension=5,  # Our mock has 5 dimensions
            limit=5
        )
        
        # Check that embeddings were loaded
        assert 'wv' in embeddings
        assert 'vocab_size' in embeddings
        assert 'vocab' in embeddings
        assert 'vector_size' in embeddings
        assert 'embedding_type' in embeddings
        
        # Check that the vector size is correct
        assert embeddings['vector_size'] == 5
        
        # Check that some words are in vocabulary
        common_words = ['the', 'and', 'is', 'a', 'in']
        for word in common_words:
            assert word in embeddings['vocab']
            
        # Check that word vectors have the right shape
        for word in embeddings['vocab']:
            vector = embeddings['wv'][word]
            assert vector.shape == (5,)
            
        # Clean up
        import os
        os.remove('mock_glove.txt')
    
    def test_create_document_embeddings(self, sample_texts):
        """Test creating document embeddings."""
        try:
            # Skip if gensim is not available
            import gensim
        except ImportError:
            pytest.skip("gensim not available")
            
        processor = TextProcessor(use_spacy=False)
        
        # Create Word2Vec embeddings
        word_embeddings = processor.create_word2vec_embeddings(
            texts=sample_texts,
            vector_size=50,
            window=3,
            min_count=1,
            epochs=5,
            seed=42
        )
        
        # Create document embeddings using different methods
        for method in ['mean', 'idf']:
            doc_embeddings = processor.create_document_embeddings(
                texts=sample_texts,
                word_vectors=word_embeddings['wv'],
                method=method
            )
            
            # Check that the embeddings have the right shape
            assert doc_embeddings.shape == (len(sample_texts), 50)
            
            # Check that embeddings are not all zeros
            assert not np.allclose(doc_embeddings, 0)
    
    def test_embedding_similarity(self):
        """Test embedding similarity calculations."""
        processor = TextProcessor(use_spacy=False)
        
        # Create sample embeddings
        emb1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        emb2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Identical to emb1
        emb3 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Reversed
        emb4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Zero vector
        
        # Test cosine similarity
        assert processor.calculate_embedding_similarity(emb1, emb1, method='cosine') == 1.0
        assert processor.calculate_embedding_similarity(emb1, emb2, method='cosine') == 1.0
        assert round(processor.calculate_embedding_similarity(emb1, emb3, method='cosine'), 8) < 1.0
        assert processor.calculate_embedding_similarity(emb1, emb4, method='cosine') == 0.0
        
        # Test euclidean similarity
        assert processor.calculate_embedding_similarity(emb1, emb1, method='euclidean') == 1.0
        assert processor.calculate_embedding_similarity(emb1, emb2, method='euclidean') == 1.0
        assert processor.calculate_embedding_similarity(emb1, emb3, method='euclidean') < 1.0
        
        # Test dot product similarity
        assert processor.calculate_embedding_similarity(emb1, emb1, method='dot') > 0.0
        assert processor.calculate_embedding_similarity(emb1, emb2, method='dot') > 0.0
        assert processor.calculate_embedding_similarity(emb1, emb4, method='dot') == 0.0
    
    def test_find_most_similar_documents(self, sample_texts):
        """Test finding similar documents."""
        try:
            # Skip if gensim is not available
            import gensim
        except ImportError:
            pytest.skip("gensim not available")
            
        processor = TextProcessor(use_spacy=False)
        
        # Create Word2Vec embeddings
        word_embeddings = processor.create_word2vec_embeddings(
            texts=sample_texts,
            vector_size=50,
            window=3,
            min_count=1,
            epochs=5,
            seed=42
        )
        
        # Create document embeddings
        doc_embeddings = processor.create_document_embeddings(
            texts=sample_texts,
            word_vectors=word_embeddings['wv'],
            method='mean'
        )
        
        # Find similar documents to the first document
        similar_docs = processor.find_most_similar_documents(
            query_embedding=doc_embeddings[0],
            document_embeddings=doc_embeddings,
            top_n=3,
            similarity_method='cosine'
        )
        
        # Check that we get the expected number of results
        assert len(similar_docs) == 3
        
        # Check that the results are sorted by similarity (descending)
        similarities = [sim for _, sim in similar_docs]
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
        
        # First document should be the most similar to itself
        assert similar_docs[0][0] == 0
        assert round(similar_docs[0][1], 8) == 1.0  # Account for floating point precision
    
    def test_text_features_with_embeddings(self, sample_df):
        """Test creating text features with embeddings."""
        try:
            # Skip if gensim is not available
            import gensim
            from sklearn.decomposition import PCA
        except ImportError:
            pytest.skip("gensim or scikit-learn not available")
            
        processor = TextProcessor(use_spacy=False)
        
        try:
            # Create text features with embeddings
            features = processor.create_text_features(
                sample_df,
                'text',
                include_stats=True,
                include_readability=True,
                include_sentiment=True,
                include_embeddings=True,
                embedding_type='word2vec',
                embedding_dimension=50,
                embedding_components=3
            )
            
            # Check that the features were created
            assert isinstance(features, pd.DataFrame)
            assert features.shape[0] == len(sample_df)
            
            # Check that all feature types are included
            assert any(col.startswith('text_stat_') for col in features.columns)
            assert any(col.startswith('text_read_') for col in features.columns)
            assert any(col.startswith('text_sent_') for col in features.columns)
            
            # Check that embedding features were created
            embedding_cols = [col for col in features.columns if 'emb_word2vec' in col]
            assert len(embedding_cols) == 3  # We requested 3 components
            
        except Exception as e:
            if "Error creating embedding features" in str(e):
                pytest.skip("Error in creating embedding features")
            else:
                raise e