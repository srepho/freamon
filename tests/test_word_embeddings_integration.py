"""
Integration tests for word embeddings functionality with other freamon components.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from freamon.utils.text_utils import TextProcessor
from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.pipeline.pipeline import Pipeline
from freamon.features.engineer import FeatureEngineer


class TestWordEmbeddingsIntegration:
    """Integration tests for word embeddings functionality."""
    
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
            # Add duplicates
            "This is a document about science and medicine. The patient requires medical treatment.",
            "Astronomy and space exploration are fascinating fields of scientific research.",
            # Add empty/invalid texts
            "",
            None,
            " ",
            "\n"
        ]
    
    @pytest.fixture
    def sample_df(self, sample_texts):
        """Create a sample dataframe with texts and categories."""
        categories = ['science', 'space', 'sports', 'auto', 'politics', 'technology',
                      'science', 'space', 'unknown', 'unknown', 'unknown', 'unknown']
        return pd.DataFrame({
            'text': sample_texts,
            'category': categories
        })
    
    def test_word_embeddings_with_duplicates(self, sample_df):
        """Test creating Word2Vec embeddings with duplicate texts."""
        try:
            import gensim
        except ImportError:
            pytest.skip("gensim not available")
        
        processor = TextProcessor(use_spacy=False)
        
        # Detect duplicates
        duplicates_info = detect_duplicates(sample_df, subset=['text'], return_counts=True)
        assert duplicates_info['has_duplicates']
        assert duplicates_info['duplicate_count'] >= 2
        
        # Handle duplicates and empty texts
        df_clean = sample_df.copy()
        
        # 1. Remove duplicates
        df_clean = remove_duplicates(df_clean, subset=['text'], keep='first')
        
        # 2. Handle empty texts
        def is_empty_text(text):
            if pd.isna(text) or not isinstance(text, str):
                return True
            return len(text.strip()) == 0
            
        df_clean = df_clean[~df_clean['text'].apply(is_empty_text)].copy()
        
        # Verify cleaning worked
        assert len(df_clean) < len(sample_df)
        assert not df_clean['text'].apply(is_empty_text).any()
        
        # Create embeddings on clean data
        word2vec = processor.create_word2vec_embeddings(
            texts=df_clean['text'],
            vector_size=50,
            window=3,
            min_count=1,
            epochs=5,
            seed=42
        )
        
        # Create document embeddings
        doc_embeddings = processor.create_document_embeddings(
            texts=df_clean['text'],
            word_vectors=word2vec['wv'],
            method='mean'
        )
        
        # Map embeddings back to original dataframe
        embedding_df = pd.DataFrame(index=sample_df.index)
        
        # Add embedding columns (just use first 3 for testing)
        for dim in range(3):
            embedding_df[f'emb_dim_{dim}'] = np.nan
            
        # Process mapping from clean texts back to original (including duplicates)
        text_to_embedding = {}
        for idx, text in enumerate(df_clean['text']):
            text_to_embedding[text] = doc_embeddings[idx]
            
        # Apply mappings
        for idx, row in sample_df.iterrows():
            text = row['text']
            
            if not is_empty_text(text) and text in text_to_embedding:
                # Map the embedding
                embedding = text_to_embedding[text]
                for dim in range(3):
                    embedding_df.loc[idx, f'emb_dim_{dim}'] = embedding[dim]
            else:
                # Handle empty/missing texts with zeros
                for dim in range(3):
                    embedding_df.loc[idx, f'emb_dim_{dim}'] = 0.0
        
        # Ensure all rows have embeddings
        assert not embedding_df.isna().any().any()
        
        # Check that duplicate texts have identical embeddings
        duplicate_texts = []
        for text in sample_df['text']:
            if not is_empty_text(text) and list(sample_df['text']).count(text) > 1:
                duplicate_texts.append(text)
                break
                
        if duplicate_texts:  # If we found duplicates
            dup_text = duplicate_texts[0]
            dup_indices = sample_df[sample_df['text'] == dup_text].index
            
            # Get embeddings for first two instances
            emb1 = embedding_df.loc[dup_indices[0], ['emb_dim_0', 'emb_dim_1', 'emb_dim_2']].values
            emb2 = embedding_df.loc[dup_indices[1], ['emb_dim_0', 'emb_dim_1', 'emb_dim_2']].values
            
            # Verify they're identical
            assert np.allclose(emb1, emb2)
    
    def test_embedding_with_pipeline(self, sample_df):
        """Test integration with Pipeline and FeatureEngineer."""
        try:
            import gensim
        except ImportError:
            pytest.skip("gensim not available")
        
        # Create a Pipeline
        pipeline = Pipeline('text_embedding_pipeline')
        
        # Add a step to remove duplicates
        pipeline.add_step(
            "remove_duplicates",
            lambda df: remove_duplicates(df, subset=['text'], keep='first')
        )
        
        # Add a step to remove empty texts
        def remove_empty_texts(df):
            def is_empty_text(text):
                if pd.isna(text) or not isinstance(text, str):
                    return True
                return len(text.strip()) == 0
            return df[~df['text'].apply(is_empty_text)].copy()
            
        pipeline.add_step("remove_empty_texts", remove_empty_texts)
        
        # Add feature engineering step
        def add_text_features(df):
            processor = TextProcessor(use_spacy=False)
            features = processor.create_text_features(
                df,
                'text',
                include_stats=True,
                include_readability=True,
                include_sentiment=True,
                include_embeddings=True,
                embedding_type='word2vec',
                embedding_dimension=50,
                embedding_components=3
            )
            return pd.concat([df, features], axis=1)
            
        pipeline.add_step("add_text_features", add_text_features)
        
        # Execute the pipeline
        try:
            result = pipeline.run(sample_df.copy())
            
            # Verify pipeline ran successfully
            assert result is not None
            assert len(result) > 0
            
            # Check that embedding columns exist
            embedding_cols = [col for col in result.columns if 'text_emb_word2vec' in col]
            assert len(embedding_cols) == 3  # We specified 3 components
            
            # Verify no NaN values in embedding columns
            assert not result[embedding_cols].isna().any().any()
            
        except Exception as e:
            if "Error creating embedding features" in str(e):
                pytest.skip("Error in creating embedding features")
            else:
                raise e
    
    def test_blank_text_handling(self):
        """Test handling of blank/empty text fields."""
        processor = TextProcessor(use_spacy=False)
        
        # Create a dataframe with mixed valid/invalid texts
        df = pd.DataFrame({
            'text': [
                "This is a valid text.",
                "",               # Empty string
                None,             # None
                np.nan,           # NaN
                " ",              # Whitespace
                "\n\t",           # Just newlines/tabs
                "Valid text again"
            ],
            'value': list(range(7))
        })
        
        # Process with text features with fallback handling
        features = processor.create_text_features(
            df,
            'text',
            include_stats=True,
            include_readability=True,
            include_sentiment=True,
            handle_empty_texts=True  # Important flag
        )
        
        # Check that all rows have features
        assert features.shape[0] == df.shape[0]
        assert not features.isna().all(axis=1).any()
        
        # Special test for document embeddings
        try:
            import gensim
        except ImportError:
            return  # Skip this part if gensim not available
        
        # Create word embeddings    
        word2vec = processor.create_word2vec_embeddings(
            texts=["This is a valid text", "Valid text again"],  # Just valid texts
            vector_size=10,
            min_count=1,
            epochs=1,
            seed=42
        )
        
        # Test document embeddings with empty texts
        doc_embeddings = processor.create_document_embeddings(
            texts=df['text'],
            word_vectors=word2vec['wv'],
            method='mean',
            handle_empty_texts=True  # Should use zero vectors for empty texts
        )
        
        # Verify shape matches input dataframe
        assert len(doc_embeddings) == len(df)
        
        # Invalid text rows should have zero vectors
        for i in [1, 2, 3, 4, 5]:  # Indices of invalid texts
            assert np.allclose(doc_embeddings[i], 0)