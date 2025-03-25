#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating how to track data through deduplication and map results back.

This example shows:
1. Creating an IndexTracker to maintain mappings between original and deduplicated data
2. Performing deduplication while preserving mapping information
3. Running a machine learning task on deduplicated data
4. Mapping results back to the original dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.utils.dataframe_utils import convert_dataframe


class IndexTracker:
    """Tracks indices through transformations like deduplication."""
    
    def __init__(self, original_indices=None):
        """Initialize the index tracker.
        
        Args:
            original_indices: Initial indices to track. If None, created when needed.
        """
        self.original_to_current = {}
        self.current_to_original = {}
        self._original_indices = original_indices
    
    def initialize_from_df(self, df):
        """Initialize tracking from a dataframe."""
        self._original_indices = df.index.tolist()
        self.update_mapping(self._original_indices, self._original_indices)
        return self
    
    def update_mapping(self, original_indices, new_indices):
        """Update the mapping after a transformation.
        
        Args:
            original_indices: Indices before transformation
            new_indices: Corresponding indices after transformation
        """
        # Update mappings
        for orig, new in zip(original_indices, new_indices):
            self.original_to_current[orig] = new
            self.current_to_original[new] = orig
        return self
    
    def update_from_kept_indices(self, kept_indices, df=None):
        """Update mapping based on which indices were kept.
        
        Args:
            kept_indices: List of indices that were kept
            df: Optional dataframe with new indices
        """
        if self._original_indices is None and df is not None:
            self.initialize_from_df(df)
            return self
            
        # Get new dataframe indices if provided
        new_indices = list(range(len(kept_indices))) if df is None else df.index.tolist()
        
        # Update mappings
        self.original_to_current = {}
        self.current_to_original = {}
        for new_idx, orig_idx in zip(new_indices, kept_indices):
            self.original_to_current[orig_idx] = new_idx
            self.current_to_original[new_idx] = orig_idx
        return self
    
    def map_to_original(self, current_indices):
        """Map current indices back to original indices."""
        if isinstance(current_indices, pd.Series):
            return current_indices.index.map(lambda x: self.current_to_original.get(x))
        return [self.current_to_original.get(idx) for idx in current_indices]
    
    def map_to_current(self, original_indices):
        """Map original indices to current indices."""
        if isinstance(original_indices, pd.Series):
            return original_indices.index.map(lambda x: self.original_to_current.get(x))
        return [self.original_to_current.get(idx) for idx in original_indices]
    
    def map_series_to_original(self, series):
        """Map a series with current indices to original indices."""
        mapped_index = self.map_to_original(series.index)
        return pd.Series(series.values, index=mapped_index)
    
    def create_full_result_df(self, result_df, original_df, fill_value=None):
        """Create a result dataframe with all original indices, filling in gaps.
        
        Args:
            result_df: Dataframe with results (using current indices)
            original_df: Original dataframe before deduplication
            fill_value: Value to use for missing results (default: None)
            
        Returns:
            DataFrame with results mapped back to all original indices
        """
        # Create a new dataframe with the original index
        full_result = pd.DataFrame(index=original_df.index)
        
        # Initialize with columns from result_df
        for col in result_df.columns:
            full_result[col] = pd.NA
        
        # Map each result back to its original index
        for curr_idx, row in result_df.iterrows():
            orig_idx = self.current_to_original.get(curr_idx)
            if orig_idx is not None:
                for col in result_df.columns:
                    full_result.loc[orig_idx, col] = row[col]
            
        # Fill missing values
        if fill_value is not None:
            if isinstance(fill_value, dict):
                for col, val in fill_value.items():
                    if col in full_result.columns:
                        full_result[col] = full_result[col].fillna(val)
            else:
                full_result = full_result.fillna(fill_value)
            
        return full_result


def main():
    """Run the example."""
    # Create a dataset with duplicates
    print("Creating sample dataset with duplicates...")
    np.random.seed(42)
    
    # Generate original data
    n_samples = 1000
    n_duplicates = 200
    
    X = np.random.randn(n_samples - n_duplicates, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Add duplicates
    duplicate_indices = np.random.choice(range(len(X)), n_duplicates, replace=True)
    X_duplicates = X[duplicate_indices]
    y_duplicates = y[duplicate_indices]
    
    X = np.vstack([X, X_duplicates])
    y = np.hstack([y, y_duplicates])
    
    # Create dataframe
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    print(f"Original dataset shape: {df.shape}")
    
    # Detect duplicates
    print("\nDetecting duplicates...")
    dup_stats = detect_duplicates(df.drop('target', axis=1))
    print(f"Found {dup_stats['duplicate_count']} duplicate rows")
    
    # Initialize index tracker
    tracker = IndexTracker().initialize_from_df(df)
    
    # Remove duplicates and update tracker
    print("\nRemoving duplicates...")
    deduped_df = remove_duplicates(df, keep='first')
    
    # Get the indices that were kept
    kept_indices = deduped_df.index.tolist()
    tracker.update_from_kept_indices(kept_indices, deduped_df)
    
    print(f"Deduplicated dataset shape: {deduped_df.shape}")
    
    # Perform machine learning task on deduplicated data
    print("\nRunning machine learning on deduplicated data...")
    X_deduped = deduped_df.drop('target', axis=1)
    y_deduped = deduped_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_deduped, y_deduped, test_size=0.3, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pd.Series(clf.predict(X_test), index=X_test.index)
    print("Classification report on deduplicated test data:")
    print(classification_report(y_test, y_pred))
    
    # Create a dataframe with test results
    test_results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    
    # Map results back to original dataset
    print("\nMapping results back to original dataset...")
    full_results = tracker.create_full_result_df(
        test_results, df, fill_value={'actual': None, 'predicted': None}
    )
    
    # Count how many original records have predictions
    has_prediction = full_results['predicted'].notna().sum()
    print(f"Original records with predictions: {has_prediction} out of {len(full_results)}")
    
    # Show examples of duplicate records and their predictions
    print("\nDuplicate records example:")
    # Find duplicated values in original features
    feature_cols = [f'feature_{i}' for i in range(5)]
    dupe_mask = df.duplicated(subset=feature_cols, keep=False)
    
    # Get some example duplicate groups
    example_dupes = df[dupe_mask].iloc[:10]
    for idx, row in example_dupes.iterrows():
        mapped_idx = tracker.original_to_current.get(idx)
        prediction = None
        if mapped_idx in y_pred.index:
            prediction = y_pred[mapped_idx]
        print(f"Original index: {idx}, Mapped index: {mapped_idx}, Prediction: {prediction}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()