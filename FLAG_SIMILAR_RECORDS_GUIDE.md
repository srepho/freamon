# Guide to Using `flag_similar_records` Function

The `flag_similar_records` function in Freamon is a powerful tool for identifying similar records in a dataset. This guide explains how to use it effectively with various options and settings.

## Overview

`flag_similar_records` analyzes multiple columns of a DataFrame to find records that are similar to each other based on a combination of criteria. It's useful for:

- Finding near-duplicate records
- Grouping similar entries
- Data cleaning and deduplication
- Entity resolution

## Basic Usage

The most basic usage compares records across specified columns with default settings:

```python
import pandas as pd
from freamon.deduplication.flag_duplicates import flag_similar_records

# Create a sample dataset
df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["John Smith", "Jane Doe", "Jon Smith", "Mary Jones", "John Smith"],
    "email": ["john@example.com", "jane@example.com", "jon@example.com", 
              "mary@example.com", "johnsmith@example.com"],
    "phone": ["555-1234", "555-5678", "555-9012", "555-3456", "555-1234"]
})

# Flag similar records based on name, email, and phone
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    threshold=0.7
)

print(result[["id", "name", "is_similar"]])
```

## Weights and Similarity

You can customize how much each column contributes to the similarity calculation using weights:

```python
# Weight name and phone higher than email
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    weights={"name": 0.5, "email": 0.2, "phone": 0.3},
    threshold=0.7
)
```

### Handling Empty Weights (New in v0.3.42)

The function now properly handles empty weights dictionaries:

```python
# With empty weights dictionary - automatically assigns equal weights
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    weights={},  # Empty dictionary - will use equal weights
    threshold=0.7
)

# With None weights - also assigns equal weights
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    weights=None,  # None weights - will use equal weights
    threshold=0.7
)
```

## Similarity Methods

The function offers several methods for calculating similarity:

```python
# 1. Composite (default) - weighted combination of column similarities
result_composite = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    method="composite",
    threshold=0.7
)

# 2. Exact Subset - match if a subset of columns match exactly
result_exact = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    method="exact_subset",
    threshold=0.6  # Match if 60% of weighted columns match exactly
)

# 3. Fuzzy Subset - match if a subset of columns match with high similarity
result_fuzzy = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    method="fuzzy_subset",
    threshold=0.5  # Match if 50% of weighted columns are highly similar
)
```

## Additional Outputs

You can request additional information about the similar records:

```python
# Add similarity scores
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    threshold=0.7,
    similarity_column="similarity_score"  # Add similarity scores
)

# Add group IDs to identify clusters of similar records
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    threshold=0.7,
    group_column="similar_group"  # Add group IDs
)

# Add both similarity scores and group IDs
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    threshold=0.7,
    similarity_column="similarity_score",
    group_column="similar_group"
)
```

## Legacy Parameter Support

The function supports legacy parameter names for backward compatibility:

```python
# Legacy parameters
result = flag_similar_records(
    df,
    columns=["name", "email", "phone"],
    threshold=0.7,
    add_similarity_score=True,           # Legacy for similarity_column
    add_group_id=True,                   # Legacy for group_column
    group_id_column="similar_group",     # Legacy for group_column
    duplicate_flag_column="is_duplicate" # Legacy for flag_column
)
```

## Large Dataset Optimization

For large datasets, use these parameters to optimize performance and memory usage:

```python
# For datasets with 20k+ rows
result = flag_similar_records(
    df_large,
    columns=["name", "email", "phone"],
    threshold=0.7,
    chunk_size=2000,  # Process 2000 rows at a time
    max_comparisons=1000000  # Limit total comparisons
)
```

## Complete Example

Here's a complete example showing various features:

```python
import pandas as pd
import numpy as np
from freamon.deduplication.flag_duplicates import flag_similar_records

# Create a dataset with duplicates and similar records
def create_example_data(n_samples=100, n_duplicates=20):
    """Create example data with similar records."""
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
    
    # Add some similar records with variations
    for i in range(n_duplicates):
        original_idx = np.random.randint(0, n_samples)
        row = df.iloc[original_idx].copy()
        
        # Add random variations
        if i % 3 == 0:
            # Slightly change name (e.g., add middle initial)
            name_parts = row["name"].split()
            if len(name_parts) == 2:
                row["name"] = f"{name_parts[0]} {chr(65 + i % 26)}. {name_parts[1]}"
        elif i % 3 == 1:
            # Slightly change age
            row["age"] += np.random.randint(-2, 3)
        else:
            # Slightly change income
            row["income"] += np.random.randint(-5000, 5000)
            
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df

# Create sample data
data = create_example_data(100, 20)
print(f"Dataset shape: {data.shape}")

# 1. Basic usage with default weights (equal weights)
print("\n1. Basic usage with default weights:")
result1 = flag_similar_records(
    data,
    columns=["name", "age", "city", "income"],
    threshold=0.8
)
print(f"Found {result1['is_similar'].sum()} similar records")

# 2. With custom weights
print("\n2. With custom weights:")
result2 = flag_similar_records(
    data,
    columns=["name", "age", "city", "income"],
    weights={"name": 0.5, "age": 0.2, "city": 0.2, "income": 0.1},
    threshold=0.8,
    similarity_column="similarity"
)
print(f"Found {result2['is_similar'].sum()} similar records")

# 3. With empty weights dictionary (v0.3.42+ feature)
print("\n3. With empty weights dictionary:")
result3 = flag_similar_records(
    data,
    columns=["name", "age", "city", "income"],
    weights={},  # Empty weights - will use equal weights
    threshold=0.8
)
print(f"Found {result3['is_similar'].sum()} similar records")

# 4. With exact subset method
print("\n4. With exact subset method:")
result4 = flag_similar_records(
    data,
    columns=["name", "age", "city", "income"],
    method="exact_subset",
    threshold=0.5,  # Match if 50% of columns match exactly
    group_column="group_id"
)
print(f"Found {result4['is_similar'].sum()} similar records")
print(f"Found {result4['group_id'].max()} groups of similar records")

# 5. Show some examples of similar records found
print("\n5. Examples of similar records found:")
# Get pairs of similar records from the same group
if result4['is_similar'].sum() > 0:
    # Get a group ID with at least 2 members
    groups_with_members = result4[result4['is_similar']]['group_id'].value_counts()
    groups_with_members = groups_with_members[groups_with_members > 1]
    
    if not groups_with_members.empty:
        example_group = groups_with_members.index[0]
        example_records = result4[result4['group_id'] == example_group]
        
        print(f"\nExample Group {example_group}:")
        print(example_records[['id', 'name', 'age', 'city', 'income', 'is_similar']])
    else:
        print("No groups with multiple members found.")
else:
    print("No similar records found.")

print("\nExample completed successfully!")
```

## Performance Tips

For optimal performance with different dataset sizes:

| Dataset Size | Recommended Settings                                        |
|--------------|-------------------------------------------------------------|
| < 20k rows   | Default settings                                            |
| 20k-100k     | `chunk_size=2000`                                          |
| 100k-500k    | `chunk_size=1000`, `max_comparisons=5000000`               |
| > 500k       | `chunk_size=500`, `max_comparisons=1000000`, `n_jobs=4`    |

## Function Signature

For reference, here's the complete function signature:

```python
def flag_similar_records(
    df: Any,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_similar',
    inplace: bool = False,
    group_column: Optional[str] = None,
    similarity_column: Optional[str] = None,
    max_comparisons: Optional[int] = None,
    chunk_size: Optional[int] = None,
    n_jobs: int = 1,
    use_polars: bool = False,
    add_similarity_score: bool = False,
    add_group_id: bool = False,
    group_id_column: Optional[str] = None,
    duplicate_flag_column: Optional[str] = None,
) -> Any:
    # ...
```