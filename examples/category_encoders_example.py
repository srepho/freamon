"""
Example of using category_encoders in freamon.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from freamon.utils.encoders import (
    BinaryEncoderWrapper,
    HashingEncoderWrapper,
    WOEEncoderWrapper,
)

# Create a sample dataset
def create_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Create categorical features with different cardinalities
    cat_low = np.random.choice(['A', 'B', 'C'], n_samples)
    cat_medium = np.random.choice([f'val_{i}' for i in range(10)], n_samples)
    cat_high = np.random.choice([f'val_{i}' for i in range(100)], n_samples)
    
    # Create numerical features
    num1 = np.random.normal(0, 1, n_samples)
    num2 = np.random.normal(0, 1, n_samples)
    
    # Create a target variable
    y = ((cat_low == 'A') & (num1 > 0)) | ((cat_medium == 'val_1') & (num2 > 0))
    y = y.astype(int)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'cat_low': cat_low,
        'cat_medium': cat_medium,
        'cat_high': cat_high,
        'num1': num1,
        'num2': num2,
        'target': y
    })
    
    return df

def run_binary_encoder_example():
    print("\n===== Binary Encoder Example =====")
    
    # Create data
    df = create_sample_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and fit a binary encoder
    encoder = BinaryEncoderWrapper(columns=['cat_low', 'cat_medium', 'cat_high'])
    
    # Transform train and test data
    train_encoded = encoder.fit_transform(train_df)
    test_encoded = encoder.transform(test_df)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Encoded columns: {train_encoded.columns.tolist()}")
    print(f"Number of columns after encoding: {len(train_encoded.columns)}")
    
    # Train a model
    X_train = train_encoded.drop(columns=['target'])
    y_train = train_encoded['target']
    X_test = test_encoded.drop(columns=['target'])
    y_test = test_encoded['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy with Binary Encoder: {accuracy:.4f}")

def run_hashing_encoder_example():
    print("\n===== Hashing Encoder Example =====")
    
    # Create data
    df = create_sample_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and fit a hashing encoder
    encoder = HashingEncoderWrapper(
        columns=['cat_low', 'cat_medium', 'cat_high'],
        n_components=10
    )
    
    # Transform train and test data
    train_encoded = encoder.fit_transform(train_df)
    test_encoded = encoder.transform(test_df)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Encoded columns: {train_encoded.columns.tolist()}")
    print(f"Number of columns after encoding: {len(train_encoded.columns)}")
    
    # Train a model
    X_train = train_encoded.drop(columns=['target'])
    y_train = train_encoded['target']
    X_test = test_encoded.drop(columns=['target'])
    y_test = test_encoded['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy with Hashing Encoder: {accuracy:.4f}")
    
def run_woe_encoder_example():
    print("\n===== WOE Encoder Example =====")
    
    # Create data
    df = create_sample_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and fit a WOE encoder
    encoder = WOEEncoderWrapper(columns=['cat_low', 'cat_medium', 'cat_high'])
    
    # Transform train and test data
    train_encoded = encoder.fit_transform(train_df, 'target')
    test_encoded = encoder.transform(test_df)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Encoded columns: {train_encoded.columns.tolist()}")
    
    # View encoded values
    print("\nWOE encoded values for low cardinality feature:")
    print(train_encoded['cat_low'].head())
    
    # Train a model
    X_train = train_encoded.drop(columns=['target'])
    y_train = train_encoded['target']
    X_test = test_encoded.drop(columns=['target'])
    y_test = test_encoded['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy with WOE Encoder: {accuracy:.4f}")

if __name__ == "__main__":
    run_binary_encoder_example()
    run_hashing_encoder_example()
    run_woe_encoder_example()