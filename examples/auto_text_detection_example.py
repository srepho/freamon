"""
Example of using the automatic text column detection in flag_similar_records.

This example demonstrates how to use the automatic text column detection functionality
in the flag_similar_records function from the freamon.deduplication module.
"""

import pandas as pd
import numpy as np
from freamon.deduplication.flag_duplicates import flag_similar_records
try:
    from freamon.utils.datatype_detector import DataTypeDetector
    HAS_DATATYPE_DETECTOR = True
except ImportError:
    HAS_DATATYPE_DETECTOR = False

# Create a sample dataset with mixed column types
def create_sample_dataset(n_records=100, seed=42):
    """Create a sample dataset with mixed column types including text."""
    np.random.seed(seed)
    
    # Generate some sample data
    ids = [f"ID{i:03d}" for i in range(n_records)]
    
    # Short categorical strings
    categories = ["apple", "banana", "cherry", "date", "fig", "grape"]
    short_text = np.random.choice(categories, n_records)
    
    # Names - mix of short strings
    first_names = ["John", "Jane", "Alice", "Bob", "Carol", "David", "Emma", "Frank"]
    last_names = ["Smith", "Doe", "Brown", "Wilson", "Taylor", "Johnson", "Miller", "Davis"]
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_records)]
    
    # Generate emails based on names
    emails = [f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com" for name in names]
    
    # Numeric data
    amounts = np.random.uniform(100, 1000, n_records).round(2)
    
    # Boolean data
    is_active = np.random.choice([True, False], n_records)
    
    # Dates
    start_date = pd.Timestamp('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_records)]
    
    # Long text descriptions
    description_templates = [
        "This product is {quality} for {usage}. It features {feature} and is priced {price}.",
        "A {quality} option for {usage}. Comes with {feature} and is {price}.",
        "Ideal for {usage}, this {quality} product includes {feature}. It is {price}.",
        "This {quality} item is perfect for {usage}. With {feature}, it represents {price}.",
        "Designed for {usage}, this product offers {quality} performance with {feature}. It is {price}."
    ]
    
    qualities = ["excellent", "good", "average", "high-quality", "premium", "standard"]
    usages = ["home use", "professional use", "office environments", "outdoor activities", "daily tasks"]
    features = ["advanced technology", "durable construction", "ergonomic design", "compact size", "versatile functionality"]
    prices = ["competitively priced", "a great value", "affordably priced", "premium priced", "budget-friendly"]
    
    descriptions = []
    for _ in range(n_records):
        template = np.random.choice(description_templates)
        desc = template.format(
            quality=np.random.choice(qualities),
            usage=np.random.choice(usages),
            feature=np.random.choice(features),
            price=np.random.choice(prices)
        )
        descriptions.append(desc)
    
    # Notes - another text field
    note_templates = [
        "Customer reported {satisfaction}. Follow up in {timeframe}.",
        "Product received in {condition}. Customer {satisfaction}.",
        "{timeframe} delivery. Customer noted {condition}.",
        "No issues reported. {satisfaction}.",
        "{condition} upon arrival. {satisfaction}. Check back in {timeframe}."
    ]
    
    satisfactions = ["high satisfaction", "satisfaction", "some concerns", "dissatisfaction", "mixed feedback"]
    conditions = ["excellent condition", "good condition", "acceptable condition", "damaged condition", "perfect condition"]
    timeframes = ["one week", "two weeks", "30 days", "quarterly", "next order"]
    
    notes = []
    for _ in range(n_records):
        template = np.random.choice(note_templates)
        note = template.format(
            satisfaction=np.random.choice(satisfactions),
            condition=np.random.choice(conditions),
            timeframe=np.random.choice(timeframes)
        )
        notes.append(note)
    
    # Create the DataFrame
    df = pd.DataFrame({
        "id": ids,
        "category": short_text,
        "name": names,
        "email": emails,
        "amount": amounts,
        "description": descriptions,
        "notes": notes,
        "date": dates,
        "is_active": is_active
    })
    
    # Add a few similar records to demonstrate the similarity detection
    # Duplicate with slight variations in the description
    similar_records = []
    for i in range(5):
        # Pick a random record to duplicate
        idx = np.random.randint(0, n_records)
        duplicate = df.iloc[idx].copy()
        
        # Modify the ID and make slight changes to the text fields
        duplicate["id"] = f"DUPE{i:03d}"
        
        # Modify description slightly
        words = duplicate["description"].split()
        if len(words) > 5:
            # Change a few words in the middle
            for j in range(2, 5):
                if np.random.random() < 0.3 and j < len(words):
                    words[j] = np.random.choice(["excellent", "great", "good", "decent", "fantastic"])
        duplicate["description"] = " ".join(words)
        
        # Modify notes slightly
        words = duplicate["notes"].split()
        if len(words) > 3:
            # Change a word or two
            for j in range(1, 3):
                if np.random.random() < 0.3 and j < len(words):
                    words[j] = np.random.choice(["promptly", "quickly", "slowly", "eventually", "finally"])
        duplicate["notes"] = " ".join(words)
        
        similar_records.append(duplicate)
    
    # Add the similar records to the DataFrame
    df = pd.concat([df, pd.DataFrame(similar_records)], ignore_index=True)
    
    return df

def main():
    # Create a sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset(n_records=100)
    print(f"Dataset created with {len(df)} records")
    
    print("\nSample record:")
    print(df.iloc[0])
    
    # Detect datatypes if available
    if HAS_DATATYPE_DETECTOR:
        print("\nDetecting datatypes using DataTypeDetector...")
        detector = DataTypeDetector()
        detector.detect_types(df)
        print("Detected datatypes:")
        for col, dtype in detector.semantic_types.items():
            print(f"  {col}: {dtype}")
    
    # 1. Basic usage with automatic text column detection
    print("\n1. Running flag_similar_records with automatic text column detection...")
    result = flag_similar_records(
        df,
        threshold=0.7,
        auto_detect_columns=True,
        flag_column="is_similar"
    )
    
    # Show results
    similar_count = result["is_similar"].sum()
    print(f"Found {similar_count} similar records")
    
    if similar_count > 0:
        print("\nSimilar records:")
        print(result[result["is_similar"]].head())
    
    # 2. Using with explicit text columns
    print("\n2. Running with explicit text columns...")
    result = flag_similar_records(
        df,
        threshold=0.7,
        auto_detect_columns=True,
        text_columns=["description", "notes"],
        text_method="tfidf",
        text_threshold=0.6,
        flag_column="is_similar"
    )
    
    # Show results
    similar_count = result["is_similar"].sum()
    print(f"Found {similar_count} similar records with explicit text columns")
    
    # 3. Using with text weight boost
    print("\n3. Running with text weight boost...")
    result = flag_similar_records(
        df,
        threshold=0.7,
        auto_detect_columns=True,
        text_weight_boost=2.0,
        flag_column="is_similar"
    )
    
    # Show results
    similar_count = result["is_similar"].sum()
    print(f"Found {similar_count} similar records with text weight boost")
    
    # 4. Compare different text methods
    print("\n4. Comparing different text similarity methods...")
    
    methods = ["fuzzy", "tfidf", "ngram", "lsh"]
    for method in methods:
        try:
            result = flag_similar_records(
                df,
                threshold=0.7,
                auto_detect_columns=True,
                text_method=method,
                flag_column="is_similar"
            )
            
            similar_count = result["is_similar"].sum()
            print(f"  {method}: Found {similar_count} similar records")
        except Exception as e:
            print(f"  {method}: Error - {str(e)}")
    
    # 5. Using with similarity column
    print("\n5. Running with similarity column...")
    result = flag_similar_records(
        df,
        threshold=0.7,
        auto_detect_columns=True,
        flag_column="is_similar",
        similarity_column="similarity_score"
    )
    
    # Show results
    similar_count = result["is_similar"].sum()
    print(f"Found {similar_count} similar records")
    
    if similar_count > 0:
        print("\nSimilar records with similarity scores:")
        print(result[result["is_similar"]][["id", "name", "similarity_score"]].head())
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()