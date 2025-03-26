"""
Test that the flag_similar_records function handles empty weights dictionaries correctly.
"""

import pytest
import pandas as pd

from freamon.deduplication.flag_duplicates import flag_similar_records


def test_flag_similar_records_empty_weights():
    """Test that flag_similar_records handles empty weights dictionaries."""
    # Create a simple test dataframe
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["John Smith", "Jane Doe", "Bob Wilson", "John Smith", "Jane Doe"],
        "email": ["john@example.com", "jane@example.com", "bob@example.com", 
                "john.smith@example.com", "jane.doe@example.com"],
        "amount": [100.0, 200.0, 300.0, 100.0, 200.0]
    })
    
    # Test with empty weights dictionary
    result = flag_similar_records(
        df,
        columns=["name", "email", "amount"],
        weights={},  # Empty weights dictionary
        threshold=0.7,
        method="composite"
    )
    
    # Verify that the function completed without errors
    assert "is_similar" in result.columns
    
    # The function should assign equal weights to all columns
    # So rows with the same name and amount should be flagged
    # First occurrence should not be flagged
    assert not result.loc[0, "is_similar"]
    assert not result.loc[1, "is_similar"]
    # Duplicates should be flagged
    assert result.loc[3, "is_similar"]
    assert result.loc[4, "is_similar"]
    
    # Test with explicit None weights
    result = flag_similar_records(
        df,
        columns=["name", "email", "amount"],
        weights=None,  # None weights
        threshold=0.7,
        method="composite"
    )
    
    # Verify that the function completed without errors
    assert "is_similar" in result.columns
    
    # First occurrence should not be flagged
    assert not result.loc[0, "is_similar"]
    assert not result.loc[1, "is_similar"]
    # Duplicates should be flagged
    assert result.loc[3, "is_similar"]
    assert result.loc[4, "is_similar"]


if __name__ == "__main__":
    test_flag_similar_records_empty_weights()
    print("All tests passed!")