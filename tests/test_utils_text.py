"""
Tests for the utils.text_utils module.
"""
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import column, data_frames

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
    
    @given(
        st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll'),  # Include letters only
                    whitelist_characters=' '            # Include space
                ),
                min_size=5,  # Ensure texts are long enough
                max_size=100
            ).filter(lambda x: len(x.strip()) > 3),  # Ensure non-empty text after stripping
            min_size=2,      # Need at least 2 documents
            max_size=10
        )
    )
    def test_text_processing_properties(self, text_samples):
        """Test text processing properties using hypothesis."""
        # Create a dataframe from the generated text samples
        df = pd.DataFrame({'text': text_samples})
        
        # Skip empty strings (which can be generated if we only get spaces)
        if df.text.str.strip().str.len().eq(0).any():
            return
            
        processor = TextProcessor(use_spacy=False)
        
        # Property 1: Lowercase processing should make all text lowercase
        lowercased = processor.process_dataframe_column(df, 'text', lowercase=True)
        assert (lowercased['text'] == lowercased['text'].str.lower()).all()
        
        # Property 2: Removing punctuation should remove all punctuation
        no_punct = processor.process_dataframe_column(df, 'text', remove_punctuation=True)
        assert not no_punct['text'].str.contains('[!.,?]').any()
        
        # Property 3: Processing should preserve row count
        processed = processor.process_dataframe_column(
            df, 'text', lowercase=True, remove_punctuation=True, remove_numbers=True
        )
        assert len(processed) == len(df)
        
        # Property 4: Creating BOW features should generate a DataFrame with rows matching the input
        if len(df) > 1:  # CountVectorizer needs at least two documents
            bow_df = processor.create_bow_features(df, 'text', max_features=5)
            assert len(bow_df) == len(df)
            assert all(col.startswith('bow_') for col in bow_df.columns)
            
    def test_extract_text_statistics(self, sample_text):
        """Test text statistics extraction."""
        processor = TextProcessor(use_spacy=False)
        
        # Test with a regular text
        stats = processor.extract_text_statistics(sample_text)
        
        # Check that all expected statistics are present
        assert 'word_count' in stats
        assert 'char_count' in stats
        assert 'avg_word_length' in stats
        assert 'unique_word_ratio' in stats
        
        # Check basic validations
        assert stats['word_count'] > 0
        assert stats['char_count'] > 0
        assert stats['avg_word_length'] > 0
        assert 0 <= stats['unique_word_ratio'] <= 1
        assert 0 <= stats['uppercase_ratio'] <= 1
        assert 0 <= stats['digit_ratio'] <= 1
        assert 0 <= stats['punctuation_ratio'] <= 1
        
        # Test with empty string
        empty_stats = processor.extract_text_statistics("")
        assert empty_stats['word_count'] == 0
        assert empty_stats['char_count'] == 0
        
    def test_calculate_readability(self, sample_text):
        """Test readability metrics calculation."""
        processor = TextProcessor(use_spacy=False)
        
        # Test with a regular text
        metrics = processor.calculate_readability(sample_text)
        
        # Check that all expected metrics are present
        assert 'flesch_reading_ease' in metrics
        assert 'flesch_kincaid_grade' in metrics
        assert 'coleman_liau_index' in metrics
        assert 'automated_readability_index' in metrics
        
        # Check value ranges
        assert 0 <= metrics['flesch_reading_ease'] <= 100
        assert metrics['flesch_kincaid_grade'] >= 0
        assert metrics['coleman_liau_index'] >= 0
        assert metrics['automated_readability_index'] >= 0
        
        # Test with empty string
        empty_metrics = processor.calculate_readability("")
        assert empty_metrics['flesch_reading_ease'] == 0
        assert empty_metrics['flesch_kincaid_grade'] == 0
    
    def test_extract_keywords_rake(self, sample_df):
        """Test keyword extraction using RAKE."""
        processor = TextProcessor(use_spacy=False)
        
        # Test with normal text
        for text in sample_df['text']:
            keywords = processor.extract_keywords_rake(text, max_keywords=5)
            
            # Check that keywords were extracted
            assert isinstance(keywords, list)
            
            # Each keyword should be a tuple of (phrase, score)
            for keyword in keywords:
                assert isinstance(keyword, tuple)
                assert len(keyword) == 2
                assert isinstance(keyword[0], str)
                assert isinstance(keyword[1], (int, float))
        
        # Test with empty text
        empty_keywords = processor.extract_keywords_rake("")
        assert empty_keywords == []
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        processor = TextProcessor(use_spacy=False)
        
        # Test with positive text
        positive_text = "This is great! I love it. The product is excellent and amazing."
        positive_sentiment = processor.analyze_sentiment(positive_text)
        
        # Check sentiment metrics
        assert positive_sentiment['sentiment_score'] > 0
        assert positive_sentiment['positive_ratio'] > 0
        assert 0 <= positive_sentiment['positive_ratio'] <= 1
        assert 0 <= positive_sentiment['negative_ratio'] <= 1
        assert 0 <= positive_sentiment['neutral_ratio'] <= 1
        
        # Test with negative text
        negative_text = "This is terrible! I hate it. The product is awful and disappointing."
        negative_sentiment = processor.analyze_sentiment(negative_text)
        
        # Check sentiment metrics
        assert negative_sentiment['sentiment_score'] < 0
        assert negative_sentiment['negative_ratio'] > 0
        
        # Test with neutral text
        neutral_text = "This is a product. It has features. The color is blue."
        neutral_sentiment = processor.analyze_sentiment(neutral_text)
        
        # Check sentiment metrics - should be mostly neutral
        assert neutral_sentiment['neutral_ratio'] > 0.5
        
        # Test with empty text
        empty_sentiment = processor.analyze_sentiment("")
        assert empty_sentiment['sentiment_score'] == 0.0
        assert empty_sentiment['neutral_ratio'] == 1.0
    
    def test_document_similarity(self):
        """Test document similarity calculation."""
        processor = TextProcessor(use_spacy=False)
        
        # Test identical documents
        doc1 = "This is a test document about similarity."
        identical_similarity = processor.calculate_document_similarity(doc1, doc1)
        assert round(identical_similarity, 8) == 1.0
        
        # Test similar documents
        doc2 = "This is a test document about documents."
        similar_similarity = processor.calculate_document_similarity(doc1, doc2)
        assert 0 < similar_similarity < 1.0
        
        # Test very different documents
        doc3 = "Completely unrelated content with no overlap whatsoever."
        different_similarity = processor.calculate_document_similarity(doc1, doc3)
        assert different_similarity < similar_similarity
        
        # Test different similarity methods
        jaccard_similarity = processor.calculate_document_similarity(doc1, doc2, method='jaccard')
        assert 0 <= jaccard_similarity <= 1.0
        
        overlap_similarity = processor.calculate_document_similarity(doc1, doc2, method='overlap')
        assert 0 <= overlap_similarity <= 1.0
        
        # Test empty documents
        empty_similarity = processor.calculate_document_similarity("", "")
        assert empty_similarity == 0.0
    
    def test_create_text_features(self, sample_df):
        """Test creating combined text features."""
        processor = TextProcessor(use_spacy=False)
        
        # Test with all features enabled
        features_df = processor.create_text_features(
            sample_df,
            'text',
            include_stats=True,
            include_readability=True,
            include_sentiment=True,
            prefix='test_'
        )
        
        # Check that features were created
        assert len(features_df) == len(sample_df)
        assert any(col.startswith('test_stat_') for col in features_df.columns)
        assert any(col.startswith('test_read_') for col in features_df.columns)
        assert any(col.startswith('test_sent_') for col in features_df.columns)
        
        # Test with specific features disabled
        stats_only = processor.create_text_features(
            sample_df,
            'text',
            include_stats=True,
            include_readability=False,
            include_sentiment=False
        )
        
        # Check that only stats features were created
        assert len(stats_only) == len(sample_df)
        assert any(col.startswith('text_stat_') for col in stats_only.columns)
        assert not any(col.startswith('text_read_') for col in stats_only.columns)
        assert not any(col.startswith('text_sent_') for col in stats_only.columns)