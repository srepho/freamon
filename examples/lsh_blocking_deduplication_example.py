"""
Example demonstrating the LSH and blocking enhancements for flag_similar_records.

This example shows how to use the blocking and LSH features to significantly
speed up deduplication of large datasets.
"""

import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from freamon.deduplication.flag_duplicates import flag_similar_records


def create_example_data(n_samples=1000, n_duplicates=200):
    """Create example data with duplicates and near-duplicates."""
    np.random.seed(42)
    
    # Create original data
    names = ["John Smith", "Jane Doe", "Bob Wilson", "Mary Johnson", 
             "David Brown", "Sarah Miller", "Michael Davis", "Emma Wilson"]
    
    df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "name": np.random.choice(names, n_samples),
        "age": np.random.randint(18, 65, n_samples),
        "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n_samples),
        "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], n_samples),
        "income": np.random.randint(30000, 120000, n_samples)
    })
    
    # Generate emails based on names (for realistic data)
    df["email"] = df["name"].apply(lambda x: x.lower().replace(" ", ".") + "@example.com")
    
    # Add some similar records with variations
    similar_records = []
    for i in range(n_duplicates):
        original_idx = np.random.randint(0, n_samples)
        row = df.iloc[original_idx].copy()
        
        # Introduce variations
        case = i % 4
        
        if case == 0:
            # Name variation (e.g., typo or formal vs informal)
            name_parts = row["name"].split()
            if len(name_parts) == 2:
                if np.random.random() < 0.5:
                    # Introduce typo
                    if len(name_parts[0]) > 3:
                        pos = np.random.randint(1, len(name_parts[0]) - 1)
                        name_parts[0] = name_parts[0][:pos] + name_parts[0][pos+1:]
                    row["name"] = " ".join(name_parts)
                else:
                    # Different format (add middle initial)
                    row["name"] = f"{name_parts[0]} {chr(65 + i % 26)}. {name_parts[1]}"
            row["email"] = row["name"].lower().replace(" ", ".").replace(".", "") + "@example.com"
            
        elif case == 1:
            # Age variation (off by a small amount)
            row["age"] += np.random.randint(-2, 3)
            
        elif case == 2:
            # Income variation (off by a larger amount but still similar)
            row["income"] += np.random.randint(-5000, 5000)
            
        elif case == 3:
            # Email variation but same core identity
            email = row["email"]
            if "@" in email:
                username, domain = email.split("@")
                if "." in username:
                    row["email"] = email.replace(".", "_")
                else:
                    row["email"] = username + str(np.random.randint(1, 100)) + "@" + domain
        
        similar_records.append(row)
    
    # Combine original data and similar records
    result = pd.concat([df] + [pd.DataFrame([record]) for record in similar_records], ignore_index=True)
    
    return result


def run_benchmark(df, test_name, **kwargs):
    """Run flag_similar_records with given parameters and measure performance."""
    print(f"\nRunning test: {test_name}")
    print("Parameters:", ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != 'columns'))
    
    start_time = time()
    result = flag_similar_records(
        df,
        columns=["name", "email", "age", "income"],
        **kwargs
    )
    end_time = time()
    
    runtime = end_time - start_time
    similar_count = result["is_similar"].sum()
    
    print(f"Found {similar_count} similar records in {runtime:.2f} seconds")
    
    return {
        "test_name": test_name,
        "runtime": runtime,
        "similar_count": similar_count,
        "parameters": kwargs
    }


def main():
    """Run benchmark tests for different approaches."""
    print("LSH and Blocking Deduplication Example")
    print("======================================")
    
    # Create test data
    n_samples = 3000
    n_duplicates = 500
    print(f"Creating dataset with {n_samples} original records and {n_duplicates} similar records...")
    df = create_example_data(n_samples, n_duplicates)
    print(f"Total dataset size: {len(df)} records")
    
    # Run benchmarks
    results = []
    
    # Standard approach (baseline)
    results.append(run_benchmark(
        df, 
        "Standard approach",
        threshold=0.8
    ))
    
    # Using blocking
    results.append(run_benchmark(
        df, 
        "Blocking by state",
        threshold=0.8,
        blocking_columns=["state"]
    ))
    
    # Using LSH
    results.append(run_benchmark(
        df, 
        "LSH (MinHash)",
        threshold=0.8,
        use_lsh=True,
        lsh_method="minhash",
        lsh_threshold=0.7
    ))
    
    # Combined approach
    results.append(run_benchmark(
        df, 
        "Combined (blocking + LSH)",
        threshold=0.8,
        blocking_columns=["state"],
        use_lsh=True,
        lsh_method="minhash",
        lsh_threshold=0.7
    ))
    
    # Using max_comparisons
    results.append(run_benchmark(
        df, 
        "Limited comparisons",
        threshold=0.8,
        max_comparisons=10000
    ))
    
    # Display summary
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Test':<25} {'Runtime (s)':<15} {'Similar Records':<20} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_time = results[0]["runtime"]
    for result in results:
        speedup = baseline_time / result["runtime"] if result["runtime"] > 0 else float('inf')
        print(f"{result['test_name']:<25} {result['runtime']:<15.2f} {result['similar_count']:<20} {speedup:<10.2f}x")
    
    # Plot benchmark results
    plt.figure(figsize=(10, 6))
    plt.bar([r["test_name"] for r in results], [r["runtime"] for r in results])
    plt.ylabel("Runtime (seconds)")
    plt.title("Deduplication Performance Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("deduplication_benchmark.png")
    print("\nBenchmark plot saved as deduplication_benchmark.png")
    
    # Show a sample of duplicate records found
    print("\nSample of similar records found:")
    result_df = flag_similar_records(
        df,
        columns=["name", "email", "age", "income"],
        threshold=0.8,
        blocking_columns=["state"],
        use_lsh=True,
        similarity_column="similarity_score",
        group_column="group_id"
    )
    
    # Get some examples of duplicate groups
    if result_df["is_similar"].sum() > 0:
        # Find groups with multiple records
        groups = result_df[result_df["group_id"] > 0]["group_id"].unique()
        if len(groups) > 0:
            # Show a few example groups
            for group_id in groups[:3]:  # Show up to 3 groups
                group_records = result_df[result_df["group_id"] == group_id]
                print(f"\nGroup {group_id}:")
                print(group_records[["name", "email", "age", "income", "state", "is_similar", "similarity_score"]])


if __name__ == "__main__":
    main()