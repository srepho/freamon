# Text Analytics

Freamon provides lightweight text analytics capabilities through the `TextProcessor` class. These capabilities are designed to work without heavy dependencies while still providing valuable features for text analysis and feature engineering.

## Basic Text Processing

The `TextProcessor` provides basic text preprocessing capabilities:

```python
from freamon.utils.text_utils import TextProcessor

# Initialize with or without spaCy (default is without)
processor = TextProcessor(use_spacy=False)

# Preprocess text
text = "Here's some Text with MIXED case, punctuation, and numbers like 123!"
processed = processor.preprocess_text(
    text,
    lowercase=True,         # Convert to lowercase
    remove_punctuation=True,  # Remove punctuation
    remove_numbers=False,     # Keep numbers
    remove_stopwords=False,   # Only available with spaCy
    lemmatize=False          # Only available with spaCy
)
print(processed)  # "heres some text with mixed case punctuation and numbers like 123"
```

## Text Statistics and Readability

Extract statistical features and readability metrics from text:

```python
# Get basic text statistics
stats = processor.extract_text_statistics(text)
print(stats)
# {
#   'word_count': 12,
#   'char_count': 63,
#   'avg_word_length': 4.33,
#   'unique_word_ratio': 0.92,
#   'uppercase_ratio': 0.13,
#   'digit_ratio': 0.05,
#   'punctuation_ratio': 0.14
# }

# Calculate readability metrics
readability = processor.calculate_readability(text)
print(readability)
# {
#   'flesch_reading_ease': 82.3,        # Higher = easier to read
#   'flesch_kincaid_grade': 4.2,        # US grade level
#   'coleman_liau_index': 7.8,          # US grade level
#   'automated_readability_index': 5.7   # US grade level
# }
```

## Keyword Extraction with RAKE

Extract important keywords and phrases using the RAKE (Rapid Automatic Keyword Extraction) algorithm:

```python
# Extract keywords using RAKE
keywords = processor.extract_keywords_rake(
    text,
    max_keywords=5,            # Maximum number of keywords to return
    min_phrase_length=1,       # Minimum words in a phrase
    max_phrase_length=3,       # Maximum words in a phrase
    min_keyword_frequency=1    # Minimum keyword frequency
)

print(keywords)
# [('mixed case', 4.5), ('text', 1.5), ('numbers', 1.0), ('heres', 1.0), ('punctuation', 1.0)]
```

## Sentiment Analysis

Perform lexicon-based sentiment analysis:

```python
# Analyze sentiment
sentiment = processor.analyze_sentiment(text)
print(sentiment)
# {
#   'sentiment_score': 0.2,       # Overall sentiment (-2 to +2)
#   'positive_ratio': 0.1,        # Ratio of positive words
#   'negative_ratio': 0.0,        # Ratio of negative words
#   'neutral_ratio': 0.9,         # Ratio of neutral words
#   'sentiment_variance': 0.04    # Variance in sentiment
# }

# Use a custom lexicon
custom_lexicon = {
    'good': 1.0, 'great': 1.5, 'excellent': 2.0,
    'bad': -1.0, 'terrible': -2.0, 'awful': -1.8
}
sentiment = processor.analyze_sentiment(text, lexicon=custom_lexicon)
```

## Document Similarity

Calculate similarity between documents using various metrics:

```python
doc1 = "This is a document about machine learning."
doc2 = "This text discusses artificial intelligence and machine learning."

# Calculate cosine similarity
similarity = processor.calculate_document_similarity(
    doc1, 
    doc2,
    method='cosine',           # Options: 'cosine', 'jaccard', 'overlap'
    lowercase=True,
    remove_punctuation=True
)
print(f"Similarity: {similarity:.2f}")  # Similarity: 0.63
```

## Text Feature Engineering

Create a comprehensive set of text features for machine learning:

```python
import pandas as pd

# Sample dataframe with text
df = pd.DataFrame({
    'id': [1, 2, 3],
    'text': [
        "This is a positive review. The product is excellent!",
        "I had a terrible experience with this product.",
        "The product works as described. It's average."
    ]
})

# Create text features
features_df = processor.create_text_features(
    df,
    'text',
    include_stats=True,        # Include statistical features
    include_readability=True,  # Include readability metrics
    include_sentiment=True,    # Include sentiment features
    prefix='text_'             # Prefix for column names
)

print(features_df.columns)
# ['text_stat_word_count', 'text_stat_char_count', ...
#  'text_read_flesch_reading_ease', 'text_read_flesch_kincaid_grade', ...
#  'text_sent_sentiment_score', 'text_sent_positive_ratio', ...]
```

## Vectorization for Machine Learning

Create bag-of-words or TF-IDF features for machine learning models:

```python
# Create bag-of-words features
bow_df = processor.create_bow_features(
    df,
    'text',
    max_features=100,         # Maximum number of features
    min_df=1,                 # Minimum document frequency
    max_df=0.9,               # Maximum document frequency (as a proportion)
    ngram_range=(1, 2),       # Include unigrams and bigrams
    binary=False,             # Use counts (False) or binary (True)
    prefix='bow_'             # Prefix for feature names
)

# Create TF-IDF features
tfidf_df = processor.create_tfidf_features(
    df,
    'text',
    max_features=100,         # Maximum number of features
    min_df=1,                 # Minimum document frequency
    max_df=0.9,               # Maximum document frequency
    ngram_range=(1, 2),       # Include unigrams and bigrams
    prefix='tfidf_'           # Prefix for feature names
)
```

## Optional spaCy Integration

The `TextProcessor` can use spaCy for more advanced NLP features:

```python
# Initialize with spaCy
processor = TextProcessor(use_spacy=True, spacy_model='en_core_web_sm')

# Extract named entities
entities = processor.extract_entities("Apple is looking at buying U.K. startup for $1 billion")
print(entities)
# {
#   'ORG': ['Apple'],
#   'GPE': ['U.K.'],
#   'MONEY': ['$1 billion']
# }

# Use spaCy features in preprocessing
processed = processor.preprocess_text(
    text,
    remove_stopwords=True,    # Remove common words like "the", "is"
    lemmatize=True            # Convert words to base form (e.g., "running" -> "run")
)
```

## Complete Example

For a complete example demonstrating all text analytics capabilities, see:

```python
# examples/text_analytics_example.py
```