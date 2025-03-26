"""
Example demonstrating how LSH and blocking would work with flag_similar_records.

This is a proof of concept implementation that shows how these features
would integrate with the existing function.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
import datasketch
from collections import defaultdict

def flag_similar_records_with_lsh_blocking(
    df: pd.DataFrame,
    columns: List[str],
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    blocking_columns: Optional[List[str]] = None,
    use_lsh: bool = False,
    lsh_method: str = 'minhash',
    lsh_threshold: float = 0.7,
    num_permutations: int = 128,
    bands: int = 20,
    rows: int = 4
) -> pd.DataFrame:
    """
    Enhanced version of flag_similar_records that uses blocking and LSH for improved performance.
    
    This is a proof of concept implementation that shows the potential performance gains.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process
    columns : List[str]
        Columns to consider when calculating similarity
    weights : Optional[Dict[str, float]], default=None
        Dictionary mapping column names to their weights in similarity calculation
    threshold : float, default=0.8
        Similarity threshold above which records are considered similar
    blocking_columns : Optional[List[str]], default=None
        Columns to use for blocking (similar records must have the same value)
    use_lsh : bool, default=False
        Whether to use Locality Sensitive Hashing for faster similarity search
    lsh_method : str, default='minhash'
        LSH method to use: 'minhash' for text data, 'random_projection' for numerical data
    lsh_threshold : float, default=0.7
        LSH similarity threshold (can be lower than main threshold as LSH is a pre-filter)
    num_permutations : int, default=128
        Number of permutations for MinHash LSH
    bands : int, default=20
        Number of bands for LSH banding technique
    rows : int, default=4
        Number of rows per band for LSH banding technique
    
    Returns
    -------
    pd.DataFrame
        DataFrame with similar records flagged
    """
    start_time = time.time()
    result_df = df.copy()
    n_rows = len(df)
    
    # Initialize similarity flags
    result_df['is_similar'] = False
    result_df['similarity_score'] = 0.0
    result_df['group_id'] = 0
    
    # Normalize weights if provided
    if weights is None or not weights:
        weights = {col: 1.0 / len(columns) for col in columns}
    else:
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {col: weight / total_weight for col, weight in weights.items()}
        else:
            weights = {col: 1.0 / len(columns) for col in columns}
    
    # Ensure all columns have weights
    for col in columns:
        if col not in weights:
            weights[col] = 0.0
    
    print(f"Dataset has {n_rows} rows, which would require {n_rows * (n_rows - 1) // 2:,} comparisons without optimization.")
    
    # Step 1: Apply blocking if specified
    if blocking_columns:
        blocks = defaultdict(list)
        for idx, row in df.iterrows():
            # Create blocking key based on blocking columns
            block_key = tuple(str(row[col]) for col in blocking_columns)
            blocks[block_key].append(idx)
        
        print(f"Created {len(blocks)} blocks using columns: {blocking_columns}")
        
        # Count potential comparisons after blocking
        potential_comparisons = sum(len(block) * (len(block) - 1) // 2 for block in blocks.values())
        print(f"Blocking reduced comparisons to {potential_comparisons:,} ({potential_comparisons/(n_rows * (n_rows - 1) // 2)*100:.2f}% of original)")
        
        # We'll use these blocks for comparisons
        comparison_pairs = []
        for block_indices in blocks.values():
            if len(block_indices) > 1:  # Only blocks with at least 2 records
                for i, idx1 in enumerate(block_indices):
                    for idx2 in block_indices[i+1:]:
                        comparison_pairs.append((idx1, idx2))
    else:
        # Without blocking, we would compare all pairs (but we'll use LSH instead if specified)
        comparison_pairs = None
    
    # Step 2: Apply LSH if specified
    if use_lsh:
        if lsh_method == 'minhash':
            # Create MinHash LSH for text data
            print(f"Using MinHash LSH with {num_permutations} permutations, {bands} bands, {rows} rows per band")
            print(f"This configuration approximates a similarity threshold of ~{1 - (1 - (1/bands) ** rows) ** (1/rows):.2f}")
            
            # Initialize MinHash objects for each record
            minhashes = {}
            for idx, row in df.iterrows():
                m = datasketch.MinHash(num_perm=num_permutations)
                # Combine all column values for hashing (weighted by importance)
                for col in columns:
                    value = str(row[col])
                    weight_factor = int(10 * weights.get(col, 1.0)) + 1  # Convert weight to repetition factor
                    for _ in range(weight_factor):
                        for token in value.lower().split():
                            m.update(token.encode('utf8'))
                minhashes[idx] = m
            
            # Create LSH index
            lsh = datasketch.MinHashLSH(threshold=lsh_threshold, num_perm=num_permutations)
            for idx, minhash in minhashes.items():
                lsh.insert(str(idx), minhash)
            
            # Query LSH for similar pairs
            lsh_pairs = set()
            for idx, minhash in minhashes.items():
                similar_indices = lsh.query(minhash)
                for similar_idx in similar_indices:
                    if similar_idx != str(idx):
                        pair = tuple(sorted([idx, int(similar_idx)]))
                        lsh_pairs.add(pair)
            
            print(f"LSH identified {len(lsh_pairs):,} potential similar pairs")
            
            # If we also have blocking, take the intersection of LSH pairs and blocking pairs
            if comparison_pairs is not None:
                blocking_pairs_set = set(tuple(sorted(pair)) for pair in comparison_pairs)
                combined_pairs = lsh_pairs.intersection(blocking_pairs_set)
                print(f"Combined blocking and LSH: {len(combined_pairs):,} pairs to compare")
                comparison_pairs = list(combined_pairs)
            else:
                comparison_pairs = list(lsh_pairs)
                
        elif lsh_method == 'random_projection':
            # Implementation for numerical data would go here
            # Using random projections to hash similar numerical vectors
            raise NotImplementedError("Random projection LSH not implemented in this example")
        else:
            raise ValueError(f"Unknown LSH method: {lsh_method}")
    
    # If we don't have any filtering methods or they didn't yield results, fall back to all pairs
    if comparison_pairs is None:
        print("No optimization specified, falling back to comparing all pairs")
        comparison_pairs = [(i, j) for i in range(n_rows) for j in range(i+1, n_rows)]
    
    # Step 3: Compute similarities for the filtered pairs
    similar_pairs = []
    group_map = {}  # To track connected components
    next_group_id = 1
    
    print(f"Computing similarities for {len(comparison_pairs):,} pairs")
    for idx1, idx2 in comparison_pairs:
        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]
        
        # Calculate weighted similarity across columns
        similarity = 0.0
        for col in columns:
            val1 = row1[col]
            val2 = row2[col]
            
            # Skip if either value is missing
            if pd.isna(val1) or pd.isna(val2):
                continue
            
            # Calculate column similarity based on type
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    col_sim = 1.0 if val1 == val2 else 0.0
                else:
                    col_sim = 1.0 - min(1.0, abs(val1 - val2) / max_val)
            
            elif isinstance(val1, str) and isinstance(val2, str):
                # Text similarity using Levenshtein distance
                import Levenshtein
                col_sim = 1.0 - Levenshtein.distance(val1, val2) / max(len(val1), len(val2), 1)
            
            else:
                # Other types - exact match only
                col_sim = 1.0 if val1 == val2 else 0.0
            
            # Add weighted similarity
            similarity += weights.get(col, 0.0) * col_sim
        
        # If similarity is above threshold, mark records as similar
        if similarity >= threshold:
            similar_pairs.append((idx1, idx2, similarity))
            
            # Update the group mapping for connected components
            group1 = group_map.get(idx1)
            group2 = group_map.get(idx2)
            
            if group1 is None and group2 is None:
                # Create a new group with both records
                group_map[idx1] = next_group_id
                group_map[idx2] = next_group_id
                next_group_id += 1
            elif group1 is None:
                # Add idx1 to idx2's group
                group_map[idx1] = group2
            elif group2 is None:
                # Add idx2 to idx1's group
                group_map[idx2] = group1
            elif group1 != group2:
                # Merge the two groups
                for idx, group in list(group_map.items()):
                    if group == group2:
                        group_map[idx] = group1
    
    # Update the result dataframe with similarity flags and groups
    for idx1, idx2, similarity in similar_pairs:
        # Flag the second record (arbitrary choice)
        result_df.loc[idx2, 'is_similar'] = True
        result_df.loc[idx2, 'similarity_score'] = similarity
    
    # Assign group IDs
    for idx, group in group_map.items():
        result_df.loc[idx, 'group_id'] = group
    
    print(f"Found {len(similar_pairs)} similar pairs forming {next_group_id - 1} groups")
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Create a test dataset with some duplicates
    np.random.seed(42)
    
    # Generate a dataset of people
    n_records = 10000
    
    # Create realistic-looking data with some patterns
    names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", 
            "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White"]
    first_names = ["James", "John", "Robert", "Michael", "William", "David", "Richard", 
                  "Joseph", "Thomas", "Charles", "Mary", "Patricia", "Jennifer", "Linda", 
                  "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
    states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    
    # Generate base data
    data = []
    for i in range(n_records):
        first_name = np.random.choice(first_names)
        last_name = np.random.choice(names)
        state = np.random.choice(states)
        age = np.random.randint(18, 90)
        
        # Generate address with some structure
        street_num = np.random.randint(1, 9999)
        streets = ["Main St", "Oak Ave", "Maple Dr", "Washington Blvd", "Park Rd", "Lake Ave"]
        street = np.random.choice(streets)
        city_prefix = ["North", "South", "East", "West", "New", "Old", "Lake", "River", "Royal"]
        city_suffix = ["town", "ville", "burg", "field", "wood", "ford", "port", "land", "view", "city"]
        city = f"{np.random.choice(city_prefix)} {np.random.choice(city_suffix)}".strip()
        
        # Generate email based on name
        email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        email = f"{first_name.lower()}.{last_name.lower()}@{np.random.choice(email_domains)}"
        
        # Generate phone with same area code for same state
        area_codes = {"CA": ["213", "310", "415"], "NY": ["212", "315", "518"], 
                     "TX": ["214", "512", "713"], "FL": ["305", "407", "813"],
                     "IL": ["217", "312", "618"]}
        if state in area_codes:
            area_code = np.random.choice(area_codes[state])
        else:
            area_code = str(np.random.randint(200, 999))
        phone = f"{area_code}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        
        # Generate zip code with first digits correlated with state
        state_zip_prefixes = {"CA": ["9"], "NY": ["1"], "TX": ["7"], "FL": ["3"], 
                             "IL": ["6"], "PA": ["1"], "OH": ["4"], "GA": ["3"]}
        if state in state_zip_prefixes:
            zip_prefix = np.random.choice(state_zip_prefixes[state])
        else:
            zip_prefix = str(np.random.randint(0, 9))
        
        zipcode = f"{zip_prefix}{np.random.randint(1000, 9999):04d}"
        
        data.append({
            "first_name": first_name,
            "last_name": last_name,
            "age": age,
            "address": f"{street_num} {street}",
            "city": city,
            "state": state,
            "zipcode": zipcode,
            "email": email,
            "phone": phone
        })
    
    # Add duplicates and near-duplicates
    n_duplicates = 1000
    for i in range(n_duplicates):
        # Pick a random record to duplicate with variations
        orig_idx = np.random.randint(0, len(data))
        duplicate = data[orig_idx].copy()
        
        # Introduce variations by case
        case = np.random.randint(0, 5)
        
        if case == 0:
            # Name typo
            if np.random.random() < 0.5:
                # First name typo
                name = duplicate["first_name"]
                if len(name) > 3:
                    pos = np.random.randint(1, len(name)-1)
                    duplicate["first_name"] = name[:pos] + name[pos+1:]  # Delete a character
            else:
                # Last name typo
                name = duplicate["last_name"]
                if len(name) > 3:
                    pos = np.random.randint(1, len(name)-1)
                    duplicate["last_name"] = name[:pos] + name[pos+1:]  # Delete a character
        
        elif case == 1:
            # Address variation
            address = duplicate["address"]
            if "Street" in address:
                duplicate["address"] = address.replace("Street", "St")
            elif "St" in address:
                duplicate["address"] = address.replace("St", "Street")
            elif "Avenue" in address:
                duplicate["address"] = address.replace("Avenue", "Ave")
            elif "Ave" in address:
                duplicate["address"] = address.replace("Ave", "Avenue")
        
        elif case == 2:
            # Phone number formatting change
            phone = duplicate["phone"]
            parts = phone.split("-")
            if len(parts) == 3:
                duplicate["phone"] = f"({parts[0]}) {parts[1]}-{parts[2]}"
        
        elif case == 3:
            # Email variation
            email = duplicate["email"]
            if "." in email and "@" in email:
                username = email.split("@")[0]
                if "." in username:
                    duplicate["email"] = email.replace(".", "_")
                else:
                    parts = username.split("_")
                    if len(parts) == 2:
                        duplicate["email"] = email.replace("_", ".")
        
        elif case == 4:
            # Age variation
            duplicate["age"] += np.random.randint(-2, 3)  # Age within ±2 years
        
        data.append(duplicate)
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} records ({n_records} original + {n_duplicates} variations)")
    
    # Test standard approach (commented out as it's too slow for large datasets)
    #print("\nRunning standard comparison (all pairs):")
    #result_standard = flag_similar_records_with_lsh_blocking(
    #    df, 
    #    columns=["first_name", "last_name", "email", "phone", "address"],
    #    threshold=0.8
    #)
    #print(f"Standard approach found {result_standard['is_similar'].sum()} similar records")
    
    # Test with blocking
    print("\nRunning with blocking:")
    result_blocking = flag_similar_records_with_lsh_blocking(
        df, 
        columns=["first_name", "last_name", "email", "phone", "address"],
        threshold=0.8,
        blocking_columns=["state", "zipcode"]
    )
    print(f"Blocking approach found {result_blocking['is_similar'].sum()} similar records")
    
    # Test with LSH
    print("\nRunning with LSH:")
    result_lsh = flag_similar_records_with_lsh_blocking(
        df, 
        columns=["first_name", "last_name", "email", "phone", "address"],
        threshold=0.8,
        use_lsh=True,
        lsh_method='minhash',
        lsh_threshold=0.7
    )
    print(f"LSH approach found {result_lsh['is_similar'].sum()} similar records")
    
    # Test with both blocking and LSH
    print("\nRunning with both blocking and LSH:")
    result_combined = flag_similar_records_with_lsh_blocking(
        df, 
        columns=["first_name", "last_name", "email", "phone", "address"],
        threshold=0.8,
        blocking_columns=["state"],
        use_lsh=True,
        lsh_method='minhash',
        lsh_threshold=0.7
    )
    print(f"Combined approach found {result_combined['is_similar'].sum()} similar records")
    
    # Print a few examples of similar records found
    if result_combined['is_similar'].sum() > 0:
        similar = result_combined[result_combined['is_similar']]
        print("\nExamples of similar records found:")
        for group_id in similar['group_id'].unique()[:3]:  # Show up to 3 groups
            group = result_combined[result_combined['group_id'] == group_id]
            print(f"\nGroup {group_id}:")
            print(group[['first_name', 'last_name', 'address', 'state', 'is_similar', 'similarity_score']])