"""
Tests for the deduplication and reporting examples.
"""
import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
from unittest.mock import patch

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.deduplication import hash_deduplication, ngram_fingerprint_deduplication


class TestDeduplicationExamples(unittest.TestCase):
    """Tests for the deduplication functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a small test dataframe with duplicates
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 1, 2],
            'text': ['hello world', 'test text', 'another text', 'final text', 'hello world', 'test text'],
            'numeric': [1.5, 2.5, 3.5, 4.5, 1.5, 2.5],
            'category': ['A', 'B', 'C', 'D', 'A', 'B']
        })
    
    def test_detect_duplicates(self):
        """Test the duplicate detection function."""
        # Test with all columns
        result = detect_duplicates(self.df)
        self.assertTrue(result['has_duplicates'])
        self.assertEqual(result['duplicate_count'], 2)
        self.assertEqual(result['duplicate_percentage'], 2 / 6 * 100)
        
        # Test with subset of columns
        result = detect_duplicates(self.df, subset=['text'])
        self.assertTrue(result['has_duplicates'])
        self.assertEqual(result['duplicate_count'], 2)
        
        # Test with return_counts
        result = detect_duplicates(self.df, return_counts=True)
        self.assertTrue('value_counts' in result)
    
    def test_remove_duplicates(self):
        """Test the duplicate removal function."""
        # Test with all columns
        df_unique = remove_duplicates(self.df)
        self.assertEqual(len(df_unique), 4)
        
        # Test with subset
        df_unique = remove_duplicates(self.df, subset=['category'])
        self.assertEqual(len(df_unique), 4)
        
        # Test with keep='last'
        df_unique = remove_duplicates(self.df, keep='last')
        self.assertEqual(len(df_unique), 4)
    
    def test_hash_deduplication(self):
        """Test hash-based text deduplication."""
        texts = self.df['text']
        
        # Test basic functionality
        unique_indices = hash_deduplication(texts)
        self.assertEqual(len(unique_indices), 4)
        
        # Test with keep='last'
        unique_indices = hash_deduplication(texts, keep='last')
        self.assertEqual(len(unique_indices), 4)
        self.assertTrue(5 in unique_indices)  # The last duplicate should be kept
        
        # Test with keep='longest'
        unique_indices = hash_deduplication(
            ['short', 'longer text', 'short'], 
            keep='longest'
        )
        self.assertEqual(len(unique_indices), 2)
        self.assertTrue(1 in unique_indices)  # The longest duplicate should be kept
    
    def test_ngram_fingerprint_deduplication(self):
        """Test n-gram fingerprint deduplication."""
        texts = self.df['text']
        
        # Test basic functionality
        unique_indices = ngram_fingerprint_deduplication(texts)
        self.assertEqual(len(unique_indices), 4)
        
        # Test with threshold < 1.0
        modified_texts = [
            'hello world',
            'test text',
            'another text',
            'final text',
            'hello world!',  # Slightly modified duplicate
            'test texting'   # Slightly modified duplicate
        ]
        
        unique_indices = ngram_fingerprint_deduplication(
            modified_texts, 
            threshold=0.7,
            n=2
        )
        
        # Should find the near-duplicates
        self.assertTrue(len(unique_indices) < 6)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plotting_functions(self, mock_savefig):
        """Test the plotting functions in the examples."""
        # Import the example module
        try:
            from examples.deduplication_reporting_example import plot_deduplication_comparison
            
            # Create a mock results dictionary
            results = {
                'method1': {'time': 0.1, 'percentage': 10.0, 'removed': 100},
                'method2': {'time': 0.2, 'percentage': 20.0, 'removed': 200}
            }
            
            # Test the plotting function
            plot_deduplication_comparison(results)
            
            # Check that savefig was called
            mock_savefig.assert_called_once()
            
        except ImportError:
            # Skip test if the example module is not available
            self.skipTest("Example module not available")


if __name__ == '__main__':
    unittest.main()