"""
Example demonstrating how to run topic modeling on a pandas DataFrame and
add the dominant topic as a new column to the DataFrame.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from freamon.utils.text_utils import TextProcessor

# Helper function to get dominant topic for each document
def get_dominant_topic(doc_topic_df):
    """
    Identifies the dominant topic for each document based on highest probability.
    
    Parameters:
    -----------
    doc_topic_df : pandas.DataFrame
        DataFrame containing document-topic distributions
        
    Returns:
    --------
    pandas.Series
        Series containing the dominant topic (column name) for each document
    """
    # For each row, find the column with the highest value
    return doc_topic_df.idxmax(axis=1)

if __name__ == '__main__':
    print("=== DataFrame Topic Modeling Example ===")
    
    # Create example DataFrame with text data
    print("Loading example dataset...")
    
    # Option 1: Use 20 newsgroups dataset
    categories = ['sci.med', 'sci.space', 'rec.autos', 'comp.graphics']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Create DataFrame with text data and original category
    df = pd.DataFrame({
        'text': newsgroups.data,
        'original_category': [newsgroups.target_names[target] for target in newsgroups.target]
    })
    
    # Add document IDs for easier tracking
    df['document_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # For demonstration, use a subset
    print(f"Using {len(df)} documents from 20 newsgroups dataset")
    
    # Initialize the text processor with spaCy for better preprocessing
    processor = TextProcessor(use_spacy=True)
    
    # Preprocess the text data
    print("Preprocessing texts...")
    
    # Process in batches for better progress reporting
    batch_size = 100
    all_processed_texts = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:min(i+batch_size, len(df))]
        processed_batch = [
            processor.preprocess_text(
                text, 
                remove_stopwords=True,
                remove_punctuation=True,
                lemmatize=True
            ) for text in batch['text']
        ]
        all_processed_texts.extend(processed_batch)
        
        # Report progress
        progress = min(100, (i + len(batch)) * 100 // len(df))
        print(f"  Progress: {progress}%", end='\r')
    
    print("\nPreprocessing complete")
    
    # Create the topic model
    print("\nCreating topic model...")
    n_topics = 4  # Match the number of categories in our dataset
    
    # Run the topic model
    topic_model = processor.create_topic_model(
        texts=all_processed_texts,
        n_topics=n_topics,
        method='nmf',  # NMF is typically faster than LDA
        max_features=1000,
        max_df=0.7,
        min_df=3,
        ngram_range=(1, 2),
        random_state=42
    )
    
    # Get document-topic distribution
    print("Getting document-topic distributions...")
    doc_topics = processor.get_document_topics(topic_model)
    
    # Set index to match the DataFrame
    doc_topics.index = df.index
    
    # Get the dominant topic for each document
    print("Identifying dominant topic for each document...")
    df['dominant_topic'] = get_dominant_topic(doc_topics)
    
    # Add document-topic probabilities as columns to the DataFrame
    print("Adding all topic probabilities to the DataFrame...")
    for col in doc_topics.columns:
        df[f"prob_{col}"] = doc_topics[col]
    
    # Print topics for reference
    print("\nTopic model topics:")
    for topic_idx, words in topic_model['topics']:
        print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
    
    # Print sample of the augmented DataFrame
    print("\nDataFrame with topic information (first 5 rows):")
    print(df[['document_id', 'original_category', 'dominant_topic']].head(5))
    
    # Visualization: Compare original categories to assigned topics
    print("\nCreating category-topic distribution visualization...")
    category_topic_counts = df.groupby(['original_category', 'dominant_topic']).size().unstack(fill_value=0)
    
    # Convert to percentages
    category_percentages = category_topic_counts.div(category_topic_counts.sum(axis=1), axis=0) * 100
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    category_percentages.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    plt.title('Distribution of Topics by Original Category')
    plt.xlabel('Original Category')
    plt.ylabel('Percentage of Documents')
    plt.legend(title='Assigned Topic')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('dataframe_topic_distribution.png')
    print("Visualization saved to dataframe_topic_distribution.png")
    
    # Advanced usage: Save topic model for later use
    print("\nSaving topic model information to CSV files...")
    
    # Save the document-topic distribution
    doc_topics.to_csv('document_topic_distribution.csv')
    
    # Save the topics as a DataFrame
    topics_df = pd.DataFrame({
        'topic_id': [f"Topic {idx+1}" for idx, _ in topic_model['topics']],
        'top_words': [', '.join(words[:15]) for _, words in topic_model['topics']]
    })
    topics_df.to_csv('topic_keywords.csv', index=False)
    
    # Save the augmented DataFrame
    df.to_csv('documents_with_topics.csv', index=False)
    
    print("\nExample complete! Files saved:")
    print("- document_topic_distribution.csv: Topic probabilities for each document")
    print("- topic_keywords.csv: Top words for each topic")
    print("- documents_with_topics.csv: Original DataFrame with topic information")
    print("- dataframe_topic_distribution.png: Visualization of topic distribution by category")