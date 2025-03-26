"""
Example demonstrating the robust handling of empty weights dictionaries in flag_similar_records.

This script shows how the function automatically handles cases where:
1. No weights are provided (weights=None)
2. Empty weights dictionary is provided (weights={})
3. All columns have zero weights (weights={"col1": 0, "col2": 0})

In all these cases, the function gracefully falls back to using equal weights.
"""

import pandas as pd
import numpy as np
from freamon.deduplication.flag_duplicates import flag_similar_records


def create_example_data(n_samples=100, n_duplicates=20):
    """Create example data with some duplicates."""
    # Create original data
    np.random.seed(42)
    names = ["John Smith", "Jane Doe", "Bob Wilson", "Mary Johnson", 
             "David Brown", "Sarah Miller", "Michael Davis", "Emma Wilson"]
    
    df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "name": np.random.choice(names, n_samples),
        "age": np.random.randint(18, 65, n_samples),
        "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n_samples),
        "income": np.random.randint(30000, 120000, n_samples)
    })
    
    # Add some exact duplicates (with slight variations)
    for i in range(n_duplicates):
        original_idx = np.random.randint(0, n_samples)
        row = df.iloc[original_idx].copy()
        # Add random variation to income
        if i % 2 == 0:
            row["income"] += np.random.randint(-5000, 5000)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df


def main():
    """Run the example."""
    print("Freamon Empty Weights Example")
    print("=" * 40)
    
    # Create example data
    df = create_example_data(100, 20)
    print(f"Created dataset with {len(df)} rows")
    
    # Columns to use for similarity calculation
    columns = ["name", "age", "city", "income"]
    
    print("\nComparing different weight scenarios:\n")
    
    # Scenario 1: No weights provided (None)
    print("Scenario 1: No weights provided (weights=None)")
    result1 = flag_similar_records(
        df,
        columns=columns,
        weights=None,
        threshold=0.8,
        method="composite",
        similarity_column="similarity_score1"
    )
    print(f"Found {result1['is_similar'].sum()} similar records")
    
    # Scenario 2: Empty weights dictionary
    print("\nScenario 2: Empty weights dictionary (weights={})")
    result2 = flag_similar_records(
        df,
        columns=columns,
        weights={},
        threshold=0.8,
        method="composite",
        similarity_column="similarity_score2"
    )
    print(f"Found {result2['is_similar'].sum()} similar records")
    
    # Scenario 3: All zero weights
    print("\nScenario 3: All zero weights (weights={col: 0 for col in columns})")
    result3 = flag_similar_records(
        df,
        columns=columns,
        weights={col: 0.0 for col in columns},
        threshold=0.8,
        method="composite",
        similarity_column="similarity_score3"
    )
    print(f"Found {result3['is_similar'].sum()} similar records")
    
    # Check if results are the same
    are_same = (
        result1["is_similar"].equals(result2["is_similar"]) and 
        result1["is_similar"].equals(result3["is_similar"])
    )
    
    print("\nResults Summary:")
    print(f"All scenarios produced the same results: {are_same}")
    
    # Sample of results
    sample = pd.concat([
        result1["similarity_score1"],
        result2["similarity_score2"],
        result3["similarity_score3"]
    ], axis=1)
    
    print("\nSample of similarity scores:")
    print(sample.head(5))
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()