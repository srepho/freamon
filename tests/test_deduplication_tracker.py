"""
Tests for the advanced DeduplicationTracker class.
"""
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
import os

from freamon.deduplication.exact_deduplication import hash_deduplication, ngram_fingerprint_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts


class TestDeduplicationTracker:
    """Test class for advanced DeduplicationTracker functionality."""
    
    @pytest.fixture
    def deduplication_tracker_class(self):
        """Import the DeduplicationTracker class from the example."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        from advanced_deduplication_tracking import DeduplicationTracker
        return DeduplicationTracker
    
    @pytest.fixture
    def sample_text_df(self):
        """Create a sample text dataframe with duplicates for testing."""
        # Base texts
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "Python is a programming language that lets you work quickly and integrate systems more effectively.",
            "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data.",
            "Natural language processing is a field of artificial intelligence that helps computers understand human language."
        ]
        
        # Generate variations
        np.random.seed(42)
        texts = []
        
        # Add exact duplicates
        for text in base_texts:
            texts.extend([text] * 3)
        
        # Add variations
        for text in base_texts:
            texts.append(text.replace("the", "a"))
            texts.append(text.lower())
            texts.append(text + " Additional text.")
        
        # Add some more random texts
        words = "the quick brown fox jumps over lazy dog machine learning field study gives computers ability learn without explicitly programmed python programming language lets work quickly integrate systems effectively data science interdisciplinary uses scientific methods extract knowledge natural processing artificial intelligence helps understand human".split()
        
        for _ in range(20):
            text_length = np.random.randint(5, 15)
            random_text = " ".join(np.random.choice(words, text_length))
            texts.append(random_text)
        
        # Create DataFrame
        sentiments = np.random.choice(['positive', 'negative', 'neutral'], len(texts))
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        return df
    
    def test_initialize(self, deduplication_tracker_class, sample_text_df):
        """Test initializing the tracker with a dataframe."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Check basic properties
        assert tracker.original_size == len(sample_text_df)
        assert tracker.current_size == len(sample_text_df)
        assert len(tracker.original_to_current) == len(sample_text_df)
        assert len(tracker.current_to_original) == len(sample_text_df)
        
        # Check mappings are identity
        for i in sample_text_df.index:
            assert tracker.original_to_current[i] == i
            assert tracker.current_to_original[i] == i
    
    def test_update_mapping(self, deduplication_tracker_class, sample_text_df):
        """Test updating tracker after deduplication."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Perform hash-based deduplication
        kept_indices = hash_deduplication(sample_text_df['text'], return_indices=True)
        
        # Update tracking
        tracker.update_mapping(kept_indices, "Hash-based")
        
        # Check deduplication step is recorded
        assert len(tracker.dedup_steps) == 1
        assert tracker.dedup_steps[0]['method'] == "Hash-based"
        assert tracker.dedup_steps[0]['original_size'] == len(sample_text_df)
        assert tracker.dedup_steps[0]['new_size'] == len(kept_indices)
        
        # Check mappings
        assert tracker.current_size == len(kept_indices)
        assert len(tracker.original_to_current) == len(kept_indices)
        
        # Check specific mappings
        for i, orig_idx in enumerate(kept_indices):
            assert tracker.current_to_original[i] == orig_idx
            assert tracker.original_to_current[orig_idx] == i
    
    def test_multi_step_deduplication(self, deduplication_tracker_class, sample_text_df):
        """Test tracking through multiple deduplication steps."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Step 1: Hash-based deduplication
        kept_indices1 = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices1, "Hash-based")
        df1 = sample_text_df.iloc[kept_indices1].reset_index(drop=True)
        
        # Step 2: N-gram fingerprint deduplication
        kept_indices2 = ngram_fingerprint_deduplication(df1['text'], n=3, return_indices=True)
        tracker.update_mapping(kept_indices2, "N-gram Fingerprint")
        df2 = df1.iloc[kept_indices2].reset_index(drop=True)
        
        # Check deduplication steps
        assert len(tracker.dedup_steps) == 2
        assert tracker.dedup_steps[0]['method'] == "Hash-based"
        assert tracker.dedup_steps[1]['method'] == "N-gram Fingerprint"
        
        # Check final mapping size
        assert tracker.current_size == len(df2)
        
        # Verify mapping chain works
        # For each final index, check that it maps correctly to original
        for curr_idx in range(len(df2)):
            orig_idx = tracker.current_to_original[curr_idx]
            assert df2.iloc[curr_idx]['text'] == sample_text_df.iloc[orig_idx]['text']
    
    def test_store_similarity_info(self, deduplication_tracker_class, sample_text_df):
        """Test storing similarity information."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Create a sample similarity dict
        similarity_dict = {
            0: [1, 2, 3],
            4: [5, 6],
            7: [8]
        }
        
        # Store similarity info
        tracker.store_similarity_info(similarity_dict)
        
        # Check stored correctly
        assert tracker.similarity_map == similarity_dict
    
    def test_store_clusters(self, deduplication_tracker_class, sample_text_df):
        """Test storing clustering information."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Create a sample cluster dict
        cluster_dict = {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8]
        }
        
        # Store clusters
        tracker.store_clusters(cluster_dict)
        
        # Check stored correctly
        assert tracker.clusters == cluster_dict
    
    def test_map_to_original(self, deduplication_tracker_class, sample_text_df):
        """Test mapping current indices to original."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Perform deduplication
        kept_indices = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices, "Hash-based")
        
        # Test with list of indices
        curr_indices = list(range(5))
        orig_indices = tracker.map_to_original(curr_indices)
        
        # Check mapping
        for i, orig_idx in enumerate(orig_indices):
            assert orig_idx == tracker.current_to_original[i]
        
        # Test with pandas Index
        df = pd.DataFrame({'col': range(5)}, index=range(5))
        orig_index = tracker.map_to_original(df.index)
        
        # Check index mapping
        for i, orig_idx in enumerate(orig_index):
            assert orig_idx == tracker.current_to_original[i]
    
    def test_map_to_current(self, deduplication_tracker_class, sample_text_df):
        """Test mapping original indices to current."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Perform deduplication
        kept_indices = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices, "Hash-based")
        
        # Test with list of kept indices
        orig_indices = kept_indices[:5]
        curr_indices = tracker.map_to_current(orig_indices)
        
        # Check mapping
        for i, curr_idx in enumerate(curr_indices):
            assert tracker.original_to_current[orig_indices[i]] == curr_idx
        
        # Test with pandas Index
        orig_index = pd.Index(kept_indices[:5])
        curr_index = tracker.map_to_current(orig_index)
        
        # Check index mapping
        for i, orig_idx in enumerate(orig_index):
            assert curr_index[i] == tracker.original_to_current[orig_idx]
        
        # Test with non-kept indices (should return None)
        non_kept = list(set(sample_text_df.index) - set(kept_indices))[:5]
        result = tracker.map_to_current(non_kept)
        assert all(x is None for x in result)
    
    def test_create_full_result_df(self, deduplication_tracker_class, sample_text_df):
        """Test creating full result dataframe with original indices."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Perform deduplication
        kept_indices = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices, "Hash-based")
        df1 = sample_text_df.iloc[kept_indices].reset_index(drop=True)
        
        # Create a result dataframe
        result_df = pd.DataFrame({
            'prediction': np.random.choice(['pos', 'neg'], size=len(df1)),
            'score': np.random.random(size=len(df1))
        }, index=df1.index)
        
        # Create full result
        full_result = tracker.create_full_result_df(
            result_df, 
            sample_text_df, 
            fill_value={'prediction': 'unknown', 'score': 0.0},
            include_duplicate_flag=True
        )
        
        # Check full result
        assert len(full_result) == len(sample_text_df)
        assert 'is_duplicate' in full_result.columns
        
        # Check mapped values
        for curr_idx, row in result_df.iterrows():
            orig_idx = tracker.current_to_original[curr_idx]
            assert full_result.loc[orig_idx, 'prediction'] == row['prediction']
            assert full_result.loc[orig_idx, 'score'] == row['score']
            assert full_result.loc[orig_idx, 'is_duplicate'] == False
        
        # Check non-kept indices have fill values and are marked as duplicates
        non_kept = set(sample_text_df.index) - set(kept_indices)
        for idx in non_kept:
            assert full_result.loc[idx, 'prediction'] == 'unknown'
            assert full_result.loc[idx, 'score'] == 0.0
            assert full_result.loc[idx, 'is_duplicate'] == True
    
    def test_plot_deduplication_steps(self, deduplication_tracker_class, sample_text_df):
        """Test plotting deduplication steps."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Step 1: Hash-based
        kept_indices1 = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices1, "Hash-based")
        df1 = sample_text_df.iloc[kept_indices1].reset_index(drop=True)
        
        # Step 2: N-gram
        kept_indices2 = ngram_fingerprint_deduplication(df1['text'], n=3, return_indices=True)
        tracker.update_mapping(kept_indices2, "N-gram")
        
        # Plot
        fig = tracker.plot_deduplication_steps()
        
        # Basic checks
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        # Clean up
        plt.close(fig)
    
    def test_plot_cluster_distribution(self, deduplication_tracker_class, sample_text_df):
        """Test plotting cluster distribution."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Create clusters
        cluster_dict = {
            0: [0, 1, 2, 3],
            1: [4, 5],
            2: [6, 7, 8, 9, 10]
        }
        tracker.store_clusters(cluster_dict)
        
        # Plot
        fig = tracker.plot_cluster_distribution()
        
        # Basic checks
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        # Clean up
        plt.close(fig)
    
    def test_generate_tracking_report(self, deduplication_tracker_class, sample_text_df, tmpdir):
        """Test generating tracking report."""
        tracker = deduplication_tracker_class().initialize(sample_text_df)
        
        # Perform deduplication
        kept_indices = hash_deduplication(sample_text_df['text'], return_indices=True)
        tracker.update_mapping(kept_indices, "Hash-based")
        df1 = sample_text_df.iloc[kept_indices].reset_index(drop=True)
        
        # Create output path
        output_file = os.path.join(tmpdir, "tracking_report.html")
        
        # Generate report
        report_path = tracker.generate_tracking_report(sample_text_df, df1, output_file)
        
        # Check report was generated
        assert os.path.exists(report_path)
        
        # Basic content check
        with open(report_path, 'r') as f:
            content = f.read()
            assert "Deduplication Tracking Report" in content
            assert "Deduplication Summary" in content
            assert "Deduplication Steps" in content