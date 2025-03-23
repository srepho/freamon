"""
Example demonstrating the text analytics capabilities in freamon,
including train/test splitting for TF-IDF features.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from freamon.utils.text_utils import TextProcessor

# Load a subset of the 20 newsgroups dataset
categories = ['sci.med', 'sci.space', 'rec.autos', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Create a DataFrame with the text data
df = pd.DataFrame({
    'text': newsgroups.data[:100],  # Using first 100 documents for brevity
    'category': [newsgroups.target_names[target] for target in newsgroups.target[:100]]
})

# Initialize the TextProcessor
processor = TextProcessor(use_spacy=False)

print("="*80)
print("Basic Text Processing Example")
print("="*80)

# Process the first document
sample_text = df['text'][0]
print(f"Original text length: {len(sample_text)} characters")
processed_text = processor.preprocess_text(
    sample_text,
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True
)
print(f"Processed text length: {len(processed_text)} characters")
print("\nFirst 200 characters of processed text:")
print(processed_text[:200] + "...\n")

# Calculate text statistics
print("="*80)
print("Text Statistics Example")
print("="*80)
stats = processor.extract_text_statistics(sample_text)
print("Text statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.3f}")

# Calculate readability metrics
print("\nReadability metrics:")
readability = processor.calculate_readability(sample_text)
for key, value in readability.items():
    print(f"  {key}: {value:.2f}")

# Extract keywords with RAKE
print("\nTop keywords (RAKE algorithm):")
keywords = processor.extract_keywords_rake(sample_text, max_keywords=5)
for keyword, score in keywords:
    print(f"  {keyword}: {score:.2f}")

# Sentiment analysis
print("\nSentiment analysis:")
sentiment = processor.analyze_sentiment(sample_text)
for key, value in sentiment.items():
    print(f"  {key}: {value:.3f}")

# Create text features for all documents
print("="*80)
print("Text Feature Engineering Example")
print("="*80)
features_df = processor.create_text_features(
    df,
    'text',
    include_stats=True,
    include_readability=True,
    include_sentiment=True
)

print(f"Created {len(features_df.columns)} text features")
print("First 5 feature names:")
for col in sorted(features_df.columns)[:5]:
    print(f"  {col}")

# Calculate document similarities
print("="*80)
print("Document Similarity Example")
print("="*80)

# Take one document from each category
category_samples = {}
for category in categories:
    category_samples[category] = df[df['category'] == category]['text'].iloc[0]

print("Document similarity between categories (cosine similarity):\n")
for cat1 in categories:
    for cat2 in categories:
        if cat1 <= cat2:  # Only print each pair once
            similarity = processor.calculate_document_similarity(
                category_samples[cat1],
                category_samples[cat2],
                method='cosine'
            )
            print(f"  {cat1} vs {cat2}: {similarity:.3f}")

# Example: Use text features for analysis
print("="*80)
print("Using Text Features For Analysis")
print("="*80)

# Add category to features DataFrame
features_with_category = features_df.copy()
features_with_category['category'] = df['category']

# Calculate average sentiment by category
sentiment_by_category = features_with_category.groupby('category')['text_sent_sentiment_score'].mean()
print("Average sentiment score by category:")
for category, score in sentiment_by_category.items():
    print(f"  {category}: {score:.4f}")

# Calculate average readability by category
readability_by_category = features_with_category.groupby('category')['text_read_flesch_reading_ease'].mean()
print("\nAverage Flesch Reading Ease by category:")
for category, score in readability_by_category.items():
    print(f"  {category}: {score:.2f}")

# Train/test split for TF-IDF features example
print("="*80)
print("Train/Test Split for TF-IDF Features")
print("="*80)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# Create and fit TF-IDF features on training data
print("\nCreating TF-IDF features on training data...")
train_tfidf = processor.create_tfidf_features(
    train_df,
    'text',
    max_features=20
    # Default behavior is to fit=True
)
print(f"Training TF-IDF shape: {train_tfidf.shape}")
print(f"First 5 TF-IDF feature names: {list(train_tfidf.columns[:5])}")

# Apply the same transformation to test data
print("\nApplying TF-IDF transformation to test data...")
test_tfidf = processor.transform_tfidf_features(
    test_df,
    'text'
)
print(f"Test TF-IDF shape: {test_tfidf.shape}")

# Verify that training and test features have the same columns
print(f"\nSame feature names in train and test: {list(train_tfidf.columns) == list(test_tfidf.columns)}")

print("\nExample complete!")