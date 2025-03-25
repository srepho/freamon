"""
Tests for the IndexTracker class for deduplication tracking.
"""
import pandas as pd
import numpy as np
import pytest

from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.deduplication.exact_deduplication import hash_deduplication


class TestIndexTracker:
    """Test class for IndexTracker functionality."""
    
    @pytest.fixture
    def index_tracker_class(self):
        """Import the IndexTracker class from the example."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        from deduplication_tracking_example import IndexTracker
        return IndexTracker
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with duplicates for testing."""
        np.random.seed(42)
        
        # Generate original data
        n_samples = 100
        n_duplicates = 20
        
        X = np.random.randn(n_samples - n_duplicates, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Add duplicates
        duplicate_indices = np.random.choice(range(len(X)), n_duplicates, replace=True)
        X_duplicates = X[duplicate_indices]
        y_duplicates = y[duplicate_indices]
        
        X = np.vstack([X, X_duplicates])
        y = np.hstack([y, y_duplicates])
        
        # Create dataframe
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])
        df['target'] = y
        
        return df

    def test_initialize_from_df(self, index_tracker_class, sample_df):
        """Test initializing IndexTracker from a dataframe."""
        tracker = index_tracker_class().initialize_from_df(sample_df)
        
        # Check mappings are initialized
        assert len(tracker.original_to_current) == len(sample_df)
        assert len(tracker.current_to_original) == len(sample_df)
        
        # Check initial mappings are identity mappings
        for i in sample_df.index:
            assert tracker.original_to_current[i] == i
            assert tracker.current_to_original[i] == i
    
    def test_update_from_kept_indices(self, index_tracker_class, sample_df):
        """Test updating tracker after deduplication."""
        # Initialize tracker
        tracker = index_tracker_class().initialize_from_df(sample_df)
        
        # Remove duplicates
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        
        # Update tracker
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Check mappings are updated correctly
        assert len(tracker.original_to_current) == len(kept_indices)
        assert len(tracker.current_to_original) == len(kept_indices)
        
        # Check each kept index is mapped correctly
        for new_idx, orig_idx in zip(deduped_df.index, kept_indices):
            assert tracker.original_to_current[orig_idx] == new_idx
            assert tracker.current_to_original[new_idx] == orig_idx
    
    def test_map_to_original(self, index_tracker_class, sample_df):
        """Test mapping current indices to original indices."""
        # Initialize tracker and deduplicate
        tracker = index_tracker_class().initialize_from_df(sample_df)
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Create test indices
        test_indices = deduped_df.index[:5].tolist()
        
        # Map to original
        original_indices = tracker.map_to_original(test_indices)
        
        # Check mapping is correct
        assert len(original_indices) == len(test_indices)
        for i, orig_idx in enumerate(original_indices):
            assert orig_idx == tracker.current_to_original[test_indices[i]]
    
    def test_map_to_current(self, index_tracker_class, sample_df):
        """Test mapping original indices to current indices."""
        # Initialize tracker and deduplicate
        tracker = index_tracker_class().initialize_from_df(sample_df)
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Create test indices
        test_indices = kept_indices[:5]
        
        # Map to current
        current_indices = tracker.map_to_current(test_indices)
        
        # Check mapping is correct
        assert len(current_indices) == len(test_indices)
        for i, curr_idx in enumerate(current_indices):
            assert curr_idx == tracker.original_to_current[test_indices[i]]
    
    def test_map_series_to_original(self, index_tracker_class, sample_df):
        """Test mapping a series with current indices to original indices."""
        # Initialize tracker and deduplicate
        tracker = index_tracker_class().initialize_from_df(sample_df)
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Create a test series with current indices
        series = pd.Series(np.random.randn(5), index=deduped_df.index[:5])
        
        # Map to original
        mapped_series = tracker.map_series_to_original(series)
        
        # Check mapping is correct
        assert len(mapped_series) == len(series)
        for curr_idx, value in series.items():
            orig_idx = tracker.current_to_original[curr_idx]
            assert mapped_series[orig_idx] == value
    
    def test_create_full_result_df(self, index_tracker_class, sample_df):
        """Test creating a full result dataframe with all original indices."""
        # Initialize tracker
        tracker = index_tracker_class().initialize_from_df(sample_df)
        
        # Deduplicate
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Create result dataframe on deduplicated data - use a smaller subset
        subset_indices = deduped_df.index[:5].tolist()  # Use just first 5 indices
        result_df = pd.DataFrame({
            'prediction': np.random.randint(0, 2, size=len(subset_indices))
        }, index=subset_indices)
        
        # Create full result
        full_result = tracker.create_full_result_df(result_df, sample_df, fill_value=None)
        
        # Check full result
        assert len(full_result) == len(sample_df)
        
        # Check values are mapped correctly
        for orig_idx in kept_indices:
            curr_idx = tracker.original_to_current[orig_idx]
            if curr_idx in result_df.index:
                assert full_result.loc[orig_idx, 'prediction'] == result_df.loc[curr_idx, 'prediction']
                
        # Get indices that should have values
        mapped_indices = [tracker.current_to_original[idx] for idx in subset_indices]
        
        # Check non-mapped indices have NaN values
        non_mapped_indices = set(sample_df.index) - set(mapped_indices)
        for idx in non_mapped_indices:
            assert pd.isna(full_result.loc[idx, 'prediction'])
    
    def test_create_full_result_df_with_fill(self, index_tracker_class, sample_df):
        """Test creating a full result dataframe with fill values for missing data."""
        # Initialize tracker
        tracker = index_tracker_class().initialize_from_df(sample_df)
        
        # Deduplicate
        deduped_df = remove_duplicates(sample_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Create result dataframe on deduplicated data - use a smaller subset
        subset_indices = deduped_df.index[:5].tolist()  # Use just first 5 indices
        result_df = pd.DataFrame({
            'prediction': np.random.randint(0, 2, size=len(subset_indices)),
            'probability': np.random.random(size=len(subset_indices))
        }, index=subset_indices)
        
        # Create full result with fill values
        fill_value = {'prediction': -1, 'probability': 0.0}
        full_result = tracker.create_full_result_df(result_df, sample_df, fill_value=fill_value)
        
        # Check full result
        assert len(full_result) == len(sample_df)
        
        # Get indices that should have values
        mapped_indices = [tracker.current_to_original[idx] for idx in subset_indices]
        
        # Check values are mapped correctly for mapped indices
        for orig_idx in mapped_indices:
            # Get corresponding current index
            curr_idx = tracker.original_to_current[orig_idx]
            # Verify values match
            assert full_result.loc[orig_idx, 'prediction'] == result_df.loc[curr_idx, 'prediction']
            assert full_result.loc[orig_idx, 'probability'] == result_df.loc[curr_idx, 'probability']
        
        # Check non-mapped indices have fill values
        non_mapped_indices = set(sample_df.index) - set(mapped_indices)
        for idx in non_mapped_indices:
            assert full_result.loc[idx, 'prediction'] == -1
            assert full_result.loc[idx, 'probability'] == 0.0
    
    def test_multi_step_tracking(self, index_tracker_class, sample_df):
        """Test tracking indices through multiple deduplication steps."""
        # Initialize tracker
        tracker = index_tracker_class().initialize_from_df(sample_df)
        
        # First deduplication step (hash based)
        feature_cols = [f'feature_{i}' for i in range(3)]
        feature_str = sample_df[feature_cols].astype(str).agg(' '.join, axis=1)
        kept_indices1 = hash_deduplication(feature_str, return_indices=True)
        deduped_df1 = sample_df.iloc[kept_indices1].reset_index(drop=True)
        
        # Update tracker after first step
        tracker.update_from_kept_indices(kept_indices1, deduped_df1)
        
        # Second deduplication step (remove additional rows)
        # Take just a few rows to avoid potential issues
        if len(deduped_df1) >= 6:
            selected_indices = [0, 2, 4]  # Take a few specific indices
        else:
            selected_indices = list(range(min(3, len(deduped_df1))))
            
        deduped_df2 = deduped_df1.iloc[selected_indices].reset_index(drop=True)
        
        # Get original indices for these rows
        selected_original_indices = [tracker.current_to_original[idx] for idx in selected_indices]
        
        # Update tracker after second step
        tracker.update_from_kept_indices(selected_original_indices, deduped_df2)
        
        # Create result dataframe with a small subset
        result_df = pd.DataFrame({
            'prediction': np.random.randint(0, 2, size=len(deduped_df2))
        }, index=deduped_df2.index)
        
        # Map back to original
        full_result = tracker.create_full_result_df(result_df, sample_df, fill_value={'prediction': -1})
        
        # Check mapping is correct
        assert len(full_result) == len(sample_df)
        
        # Get the final mapping to verify predictions
        final_mapping = {}
        for orig_idx, curr_idx in tracker.original_to_current.items():
            if curr_idx is not None and curr_idx in result_df.index:
                final_mapping[orig_idx] = result_df.loc[curr_idx, 'prediction']
        
        # Check mapped predictions
        for orig_idx, expected_pred in final_mapping.items():
            assert full_result.loc[orig_idx, 'prediction'] == expected_pred
            
        # Check unmapped have default value
        unmapped_indices = set(sample_df.index) - set(final_mapping.keys())
        for idx in unmapped_indices:
            assert full_result.loc[idx, 'prediction'] == -1