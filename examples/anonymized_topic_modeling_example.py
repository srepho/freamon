#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the enhanced topic modeling with anonymization.

This example shows:
1. Using the optimized topic modeling with anonymization flag
2. Processing data containing PII (personally identifiable information)
3. Comparing topic models with and without anonymization
4. Analyzing the impact of anonymization on topic distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Import the optimized topic modeling function
from freamon.utils.text_utils import create_topic_model_optimized, TextProcessor

# Check if Allyanonimiser is available
def check_anonymizer_availability():
    """Check if Allyanonimiser is available and provide a mock if not."""
    try:
        from allyanonimiser import Anonymizer
        return True
    except ImportError:
        print("Allyanonimiser not found. Creating mock implementation for demonstration.")
        
        # Create mock classes
        class MockAnonymizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
            def anonymize_text(self, text):
                """Simple mock anonymization."""
                if not text:
                    return ""
                # Replace email patterns
                if "@" in text:
                    text = text.replace("john.smith@example.com", "[EMAIL]")
                    text = text.replace("jane.doe@example.com", "[EMAIL]")
                    text = text.replace("example.com", "[EMAIL_DOMAIN]")
                    text = text.replace("company.org", "[EMAIL_DOMAIN]")
                
                # Replace generic email pattern with regex
                import re
                text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
                
                # Replace phone patterns
                text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
                
                # Replace names (common names in the dataset)
                common_names = ["John", "Paul", "George", "Ringo", "David", "Michael", 
                               "Richard", "Robert", "William", "James", "Thomas", "Mark"]
                for name in common_names:
                    text = re.sub(r'\b' + name + r'\b', '[NAME]', text)
                
                # Replace addresses (basic pattern)
                text = re.sub(r'\d+\s+[A-Za-z]+\s+St[.]?', '[ADDRESS]', text)
                text = re.sub(r'\d+\s+[A-Za-z]+\s+Ave[.]?', '[ADDRESS]', text)
                
                return text
        
        # Patch the imported modules
        import sys
        import types
        
        module = types.ModuleType("allyanonimiser")
        module.Anonymizer = MockAnonymizer
        sys.modules["allyanonimiser"] = module
        
        return False


def generate_dataset_with_pii():
    """Generate a dataset with added PII content for demonstration."""
    # Load base 20 newsgroups dataset
    print("Loading and enhancing 20 newsgroups dataset with PII...")
    categories = ['sci.med', 'sci.space', 'rec.autos', 'talk.politics.guns']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers'),  # Keep headers to have emails
        random_state=42
    )
    
    # Create base dataframe
    df = pd.DataFrame({
        'text': newsgroups.data,
        'category': [newsgroups.target_names[target] for target in newsgroups.target]
    })
    
    # Add synthetic PII to some documents
    np.random.seed(42)
    pii_templates = [
        "Please contact me at {name}@{domain}.com or call {phone}.",
        "My name is {name} and I live at {address}.",
        "For more information, email {name}.{surname}@{domain}.com.",
        "Contact {name} at {phone} regarding the {topic} discussion.",
        "The {topic} expert, {name} {surname}, can be reached at {address}."
    ]
    
    domains = ["example", "gmail", "company", "provider", "service"]
    names = ["John", "Jane", "Michael", "Sarah", "David", "Jennifer", "Robert", "Maria"]
    surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]
    addresses = [
        "123 Main St, Boston, MA 02108",
        "456 Oak Ave, San Francisco, CA 94102",
        "789 Pine Rd, Chicago, IL 60601",
        "321 Maple Dr, Austin, TX 78701",
        "654 Elm Blvd, Seattle, WA 98101"
    ]
    phone_patterns = [
        "555-123-{0}",
        "(555) 123-{0}",
        "555.123.{0}"
    ]
    
    # Select random documents to inject PII
    pii_indices = np.random.choice(len(df), size=int(len(df) * 0.3), replace=False)
    
    for idx in pii_indices:
        # Select random PII template
        template = np.random.choice(pii_templates)
        
        # Fill in the template
        name = np.random.choice(names)
        surname = np.random.choice(surnames)
        domain = np.random.choice(domains)
        address = np.random.choice(addresses)
        phone_suffix = str(np.random.randint(1000, 9999))
        phone = np.random.choice(phone_patterns).format(phone_suffix)
        topic = df.loc[idx, 'category']
        
        # Format the PII text
        pii_text = template.format(
            name=name,
            surname=surname,
            domain=domain,
            address=address,
            phone=phone,
            topic=topic
        )
        
        # Add PII to the beginning or end of the document
        if np.random.random() > 0.5:
            df.loc[idx, 'text'] = pii_text + "\n\n" + df.loc[idx, 'text']
        else:
            df.loc[idx, 'text'] = df.loc[idx, 'text'] + "\n\n" + pii_text
    
    # Mark documents with injected PII
    df['has_injected_pii'] = False
    df.loc[pii_indices, 'has_injected_pii'] = True
    
    print(f"Enhanced {len(pii_indices)} documents with synthetic PII")
    return df


def main():
    """Run the example."""
    print("======== Anonymized Topic Modeling Example ========")
    
    # Check for Allyanonimiser
    using_real_anonymizer = check_anonymizer_availability()
    print(f"Using {('real' if using_real_anonymizer else 'mock')} Allyanonimiser implementation.\n")
    
    # Generate dataset with PII
    df = generate_dataset_with_pii()
    print(f"Generated dataset with {len(df)} documents")
    
    # Configure options for topic modeling
    preprocessing_options = {
        'enabled': True,
        'use_lemmatization': True,
        'remove_stopwords': True,
        'remove_punctuation': True,
        'min_token_length': 3,
        'custom_stopwords': []
    }
    
    deduplication_options = {
        'enabled': True,
        'method': 'exact',
        'hash_method': 'hash'
    }
    
    anonymization_config = {
        # Configuration options for Allyanonimiser (empty for defaults)
    }
    
    # Run topic modeling WITHOUT anonymization
    print("\n=== Running topic modeling WITHOUT anonymization ===")
    result_standard = create_topic_model_optimized(
        df, 
        text_column='text',
        n_topics=4,  # One per category
        method='nmf',
        preprocessing_options=preprocessing_options,
        deduplication_options=deduplication_options,
        max_docs=None,  # Use all documents
        anonymize=False,
        return_full_data=True
    )
    
    # Run topic modeling WITH anonymization
    print("\n=== Running topic modeling WITH anonymization ===")
    result_anonymized = create_topic_model_optimized(
        df, 
        text_column='text',
        n_topics=4,  # One per category
        method='nmf',
        preprocessing_options=preprocessing_options,
        deduplication_options=deduplication_options,
        max_docs=None,  # Use all documents
        anonymize=True,
        anonymization_config=anonymization_config,
        return_full_data=True
    )
    
    # Print topics from both models
    print("\n=== Topic Comparison ===")
    print("\nTop 10 words for each topic WITHOUT anonymization:")
    for topic_idx, words in result_standard['topics']:
        print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
    
    print("\nTop 10 words for each topic WITH anonymization:")
    for topic_idx, words in result_anonymized['topics']:
        print(f"Topic {topic_idx + 1}: {', '.join(words[:10])}")
    
    # Compare topic distributions
    print("\n=== Topic Distribution Comparison ===")
    
    # Add categories to the topic distributions
    standard_topics = result_standard['document_topics'].copy()
    anonymized_topics = result_anonymized['document_topics'].copy()
    
    standard_topics['category'] = df.loc[standard_topics.index, 'category'].values
    anonymized_topics['category'] = df.loc[anonymized_topics.index, 'category'].values
    
    # Get documents with injected PII
    pii_docs = df[df['has_injected_pii']]
    standard_pii_topics = standard_topics.loc[pii_docs.index]
    anonymized_pii_topics = anonymized_topics.loc[pii_docs.index]
    
    # Calculate average topic distribution by category
    standard_category_dist = standard_topics.groupby('category').mean()
    anonymized_category_dist = anonymized_topics.groupby('category').mean()
    
    # Compare topics for documents with PII
    pii_standard_dist = standard_pii_topics.drop(columns=['category']).mean()
    pii_anonymized_dist = anonymized_pii_topics.drop(columns=['category']).mean()
    
    print("\nAverage topic distribution by category WITHOUT anonymization:")
    print(standard_category_dist)
    
    print("\nAverage topic distribution by category WITH anonymization:")
    print(anonymized_category_dist)
    
    # Create visualization to compare topic distributions
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot category-topic distributions for standard model
    axes[0].imshow(standard_category_dist.values, cmap='viridis', aspect='auto')
    axes[0].set_title('Topic Distribution by Category (Without Anonymization)')
    axes[0].set_xticks(range(len(standard_category_dist.columns)))
    axes[0].set_xticklabels(standard_category_dist.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(standard_category_dist.index)))
    axes[0].set_yticklabels(standard_category_dist.index)
    
    # Plot category-topic distributions for anonymized model
    im = axes[1].imshow(anonymized_category_dist.values, cmap='viridis', aspect='auto')
    axes[1].set_title('Topic Distribution by Category (With Anonymization)')
    axes[1].set_xticks(range(len(anonymized_category_dist.columns)))
    axes[1].set_xticklabels(anonymized_category_dist.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(anonymized_category_dist.index)))
    axes[1].set_yticklabels(anonymized_category_dist.index)
    
    # Add colorbar
    plt.colorbar(im, ax=axes.ravel().tolist(), label='Topic Probability')
    
    plt.tight_layout()
    plt.savefig('anonymized_topic_comparison.png')
    print("\nSaved topic distribution comparison to anonymized_topic_comparison.png")
    
    # Compare impact of anonymization on PII documents vs. non-PII documents
    print("\n=== Impact of Anonymization on PII Documents ===")
    
    # Calculate average change in topic distribution for PII and non-PII docs
    pii_indices = df[df['has_injected_pii']].index
    non_pii_indices = df[~df['has_injected_pii']].index
    
    # Function to calculate absolute difference between topic distributions
    def calc_topic_diff(idx):
        standard_dist = standard_topics.loc[idx].drop('category')
        anonymized_dist = anonymized_topics.loc[idx].drop('category')
        return np.mean(np.abs(standard_dist.values - anonymized_dist.values))
    
    # Calculate average difference for PII and non-PII docs
    pii_diffs = [calc_topic_diff(idx) for idx in pii_indices]
    non_pii_diffs = [calc_topic_diff(idx) for idx in non_pii_indices]
    
    print(f"Average topic distribution change for documents WITH PII: {np.mean(pii_diffs):.4f}")
    print(f"Average topic distribution change for documents WITHOUT PII: {np.mean(non_pii_diffs):.4f}")
    print(f"Anonymization impact ratio (PII vs. non-PII): {np.mean(pii_diffs)/np.mean(non_pii_diffs):.2f}x")
    
    # Create visualization of the impact
    plt.figure(figsize=(10, 6))
    plt.boxplot([pii_diffs, non_pii_diffs], labels=['Documents with PII', 'Documents without PII'])
    plt.title('Impact of Anonymization on Topic Distribution')
    plt.ylabel('Average Absolute Change in Topic Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('anonymization_impact.png')
    print("Saved anonymization impact visualization to anonymization_impact.png")
    
    # Show example of anonymized text
    print("\n=== Example of Anonymized Text ===")
    # Find a document with PII
    pii_doc_idx = pii_indices[0]
    original_text = df.loc[pii_doc_idx, 'text']
    
    # Get anonymized version by running anonymizer directly
    try:
        from allyanonimiser import Anonymizer
        anonymizer = Anonymizer(**(anonymization_config or {}))
        anonymized_text = anonymizer.anonymize_text(original_text)
    except:
        # Use our mock implementation if needed
        class MockAnonymizer:
            def anonymize_text(self, text):
                import re
                text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
                text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
                common_names = ["John", "Jane", "Michael", "Sarah", "David", "Jennifer", "Robert", "Maria"]
                for name in common_names:
                    text = re.sub(r'\b' + name + r'\b', '[NAME]', text)
                return text
        
        anonymizer = MockAnonymizer()
        anonymized_text = anonymizer.anonymize_text(original_text)
    
    # Print limited portions of the text
    max_len = 500  # Limit output length
    print("Original text (excerpt):")
    print(original_text[:max_len] + ("..." if len(original_text) > max_len else ""))
    print("\nAnonymized text (excerpt):")
    print(anonymized_text[:max_len] + ("..." if len(anonymized_text) > max_len else ""))
    
    print("\nExample complete!")
    return {
        'df': df,
        'standard_result': result_standard,
        'anonymized_result': result_anonymized,
        'standard_topics': standard_topics,
        'anonymized_topics': anonymized_topics
    }


if __name__ == "__main__":
    main()