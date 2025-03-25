#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating anonymization integrated with deduplication tracking.

This example shows:
1. Creating a pipeline that includes anonymization and deduplication steps
2. Tracking data through the pipeline while maintaining original indices
3. Running a machine learning task on deduplicated and anonymized data
4. Mapping results back to the original dataset
5. Evaluating the impact on model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts
from freamon.pipeline.pipeline import Pipeline
from freamon.utils.text_utils import TextProcessor

# Import from examples since these components aren't included in the main package
import sys
import os
sys.path.append(os.path.dirname(__file__))
from pipeline_with_deduplication_tracking import (
    IndexTrackingPipeline,
    SimilarityDeduplicationStep,
    HashDeduplicationStep
)

# Import the Allyanonimiser bridge components
from freamon.integration.allyanonimiser_bridge import (
    AnonymizationStep,
    EnhancedTextPreprocessingStep,
    PIIDetectionStep
)


def generate_pii_dataset(n_samples=300, n_duplicates=50, n_near_duplicates=30):
    """Generate a sample dataset with PII for demonstration."""
    np.random.seed(42)
    
    # Base texts with PII
    base_texts = [
        "Hello, my name is John Smith and my email is john.smith@example.com.",
        "Please contact Jane Doe at jane.doe@example.com or call 123-456-7890.",
        "Our company CEO, Sarah Johnson, approved the new initiative.",
        "Send your documents to Michael Brown at 123 Main Street, New York, NY 10001.",
        "The meeting is scheduled with David Wilson at his office on February 5th.",
        "Please transfer $5,000 to account #12345678 for the project expenses.",
        "Jennifer Davis from HR will explain the new healthcare policies.",
        "Driver's license number DL-123456789 needs to be verified.",
        "The contractor, Robert Thompson, provided his SSN: 123-45-6789 for payment.",
        "Call George Martinez on his cell at (987) 654-3210 for urgent matters."
    ]
    
    # Regular texts without PII
    regular_texts = [
        "The quarterly report showed significant growth in all departments.",
        "Project timeline has been extended by two weeks due to scope changes.",
        "The new product launch is scheduled for next month.",
        "Customer satisfaction ratings have improved by 15% this quarter.",
        "All team members should review the updated guidelines.",
        "The machine learning model showed 92% accuracy in tests.",
        "Budget allocation for next year will be decided next week.",
        "The software update includes several bug fixes and performance improvements.",
        "Market analysis suggests favorable conditions for expansion.",
        "Quarterly goals were met across all divisions."
    ]
    
    # Generate variations of base texts with PII
    texts = []
    for text in base_texts:
        texts.append(text)
        # Add variations
        if "email" in text:
            texts.append(text.replace("example.com", "company.org"))
        if "call" in text:
            texts.append(text.replace("call", "phone").replace("123-456-7890", "123.456.7890"))
        if "name is" in text:
            texts.append(text.replace("name is", "name's"))
    
    # Add regular texts
    for text in regular_texts:
        texts.append(text)
        # Add variations
        texts.append(text.replace("the", "a").replace("is", "was"))
    
    # Add more random texts
    for _ in range(n_samples - len(texts) - n_duplicates - n_near_duplicates):
        if np.random.random() < 0.3:
            # Random PII text
            pii_template = np.random.choice(base_texts)
            texts.append(pii_template)
        else:
            # Random regular text
            regular_template = np.random.choice(regular_texts)
            texts.append(regular_template)
    
    # Generate categories
    categories = []
    for text in texts:
        if any(term in text.lower() for term in ["email", "@", "call", "phone"]):
            categories.append("contact")
        elif any(term in text.lower() for term in ["name", "john", "jane", "ceo", "hr"]):
            categories.append("person")
        elif any(term in text.lower() for term in ["report", "project", "product", "customer"]):
            categories.append("business")
        else:
            categories.append("other")
    
    # Add exact duplicates
    duplicate_indices = np.random.choice(range(len(texts)), n_duplicates)
    duplicate_texts = [texts[i] for i in duplicate_indices]
    duplicate_categories = [categories[i] for i in duplicate_indices]
    
    # Add near duplicates (with small modifications)
    near_duplicate_indices = np.random.choice(range(len(texts)), n_near_duplicates)
    near_duplicate_texts = []
    near_duplicate_categories = []
    
    for i in near_duplicate_indices:
        text = texts[i]
        # Make small modifications
        if len(text) > 20:
            # Add a filler word or change punctuation
            words = text.split()
            pos = np.random.randint(1, len(words))
            words.insert(pos, np.random.choice(["basically", "essentially", "actually", "normally"]))
            modified_text = " ".join(words)
        else:
            modified_text = text + np.random.choice([" indeed", " of course", " naturally"])
            
        near_duplicate_texts.append(modified_text)
        near_duplicate_categories.append(categories[i])
    
    # Combine everything
    all_texts = texts + duplicate_texts + near_duplicate_texts
    all_categories = categories + duplicate_categories + near_duplicate_categories
    
    # Create DataFrame
    return pd.DataFrame({
        'text': all_texts,
        'category': all_categories
    })


def check_anonymizer_availability():
    """Check if Allyanonimiser is available and provide a mock if not."""
    try:
        from allyanonimiser import Anonymizer, Analyzer
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
                
                # Replace phone patterns
                text = text.replace("123-456-7890", "[PHONE]")
                text = text.replace("123.456.7890", "[PHONE]")
                text = text.replace("(987) 654-3210", "[PHONE]")
                
                # Replace names
                text = text.replace("John Smith", "[NAME]")
                text = text.replace("Jane Doe", "[NAME]")
                text = text.replace("Sarah Johnson", "[NAME]")
                text = text.replace("Michael Brown", "[NAME]")
                text = text.replace("David Wilson", "[NAME]")
                text = text.replace("Jennifer Davis", "[NAME]")
                text = text.replace("Robert Thompson", "[NAME]")
                text = text.replace("George Martinez", "[NAME]")
                
                # Replace addresses
                text = text.replace("123 Main Street, New York, NY 10001", "[ADDRESS]")
                
                # Replace financial information
                text = text.replace("$5,000", "[MONEY]")
                text = text.replace("account #12345678", "[ACCOUNT]")
                
                # Replace IDs
                text = text.replace("DL-123456789", "[ID]")
                text = text.replace("123-45-6789", "[SSN]")
                
                return text
        
        class MockAnalyzer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
            def analyze_text(self, text):
                """Simple mock analysis."""
                findings = []
                has_pii = False
                
                if "@" in text:
                    has_pii = True
                    findings.append({"entity_type": "EMAIL"})
                
                if any(pattern in text for pattern in ["123-456-7890", "123.456.7890", "(987) 654-3210"]):
                    has_pii = True
                    findings.append({"entity_type": "PHONE"})
                
                if any(name in text for name in ["John Smith", "Jane Doe", "Sarah Johnson", 
                                                 "Michael Brown", "David Wilson", "Jennifer Davis", 
                                                 "Robert Thompson", "George Martinez"]):
                    has_pii = True
                    findings.append({"entity_type": "PERSON"})
                
                if "123 Main Street, New York, NY 10001" in text:
                    has_pii = True
                    findings.append({"entity_type": "ADDRESS"})
                
                if any(pattern in text for pattern in ["$5,000", "account #12345678"]):
                    has_pii = True
                    findings.append({"entity_type": "FINANCIAL"})
                
                if "123-45-6789" in text:
                    has_pii = True
                    findings.append({"entity_type": "SSN"})
                
                if "DL-123456789" in text:
                    has_pii = True
                    findings.append({"entity_type": "ID"})
                
                return {"has_pii": has_pii, "findings": findings}
        
        # Patch the imported modules
        import sys
        import types
        
        module = types.ModuleType("allyanonimiser")
        module.Anonymizer = MockAnonymizer
        module.Analyzer = MockAnalyzer
        sys.modules["allyanonimiser"] = module
        
        # Update the bridge module to use our mocks
        from freamon.integration import allyanonimiser_bridge
        allyanonimiser_bridge.Anonymizer = MockAnonymizer
        allyanonimiser_bridge.Analyzer = MockAnalyzer
        
        return False


def main():
    """Run the example."""
    print("Checking for Allyanonimiser...")
    using_real_anonymizer = check_anonymizer_availability()
    print(f"Using {('real' if using_real_anonymizer else 'mock')} Allyanonimiser implementation.\n")
    
    print("Generating sample data with PII and duplicates...")
    df = generate_pii_dataset(n_samples=300, n_duplicates=50, n_near_duplicates=30)
    print(f"Original dataset shape: {df.shape}")
    
    # Store original data
    original_df = df.copy()
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")
    
    # Create pipeline steps
    pipeline_steps = [
        # PII Detection - identify records containing PII
        PIIDetectionStep(
            text_column='text',
            name='pii_detection'
        ),
        
        # Enhanced text preprocessing with anonymization
        EnhancedTextPreprocessingStep(
            text_column='text',
            anonymize=True,
            name='enhanced_preprocessing'
        ),
        
        # Deduplication steps
        HashDeduplicationStep(
            text_column='processed_text',
            name='hash_deduplication'
        ),
        
        SimilarityDeduplicationStep(
            text_column='processed_text',
            method='cosine',
            threshold=0.8,
            name='similarity_deduplication'
        )
    ]
    
    # Create tracking pipeline
    pipeline = IndexTrackingPipeline(steps=pipeline_steps, name="anonymization_deduplication_pipeline")
    
    # Run pipeline on training data
    print("\nRunning pipeline on training data...")
    processed_train = pipeline.fit_transform(train_df)
    print(f"Processed training dataset shape: {processed_train.shape}")
    
    # Check how many records have PII
    print(f"Records with PII in original training set: {train_df.shape[0]}")
    print(f"Records with PII in processed training set: {processed_train['pii_has_pii'].sum()}")
    
    # Run a simple machine learning task
    print("\nTraining classification model on anonymized data...")
    
    # Extract features using bag of words on processed text
    vectorizer = CountVectorizer(max_features=100)
    X_train = vectorizer.fit_transform(processed_train['processed_text'])
    y_train = processed_train['category']
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Process test data through the same pipeline
    print("\nProcessing test data...")
    test_pipeline = IndexTrackingPipeline(steps=pipeline_steps[:3])  # Only use PII detection, preprocessing and hash dedup
    processed_test = test_pipeline.fit_transform(test_df)
    print(f"Processed test dataset shape: {processed_test.shape}")
    
    # Make predictions
    X_test = vectorizer.transform(processed_test['processed_text'])
    y_test = processed_test['category']
    y_pred = model.predict(X_test)
    
    # Evaluate on processed test data
    print("\nEvaluation on processed test data:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report on anonymized test data:")
    print(classification_report(y_test, y_pred))
    
    # Create results dataframe for mapping back
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'has_pii': processed_test['pii_has_pii'],
        'anonymized_text': processed_test['processed_text']
    }, index=processed_test.index)
    
    # Map results back to original test data
    print("\nMapping results back to original test data...")
    full_results = test_pipeline.create_full_result_df(
        'hash_deduplication',
        results_df,
        fill_value={'predicted': 'unknown', 'has_pii': False, 'anonymized_text': None}
    )
    
    # Print comparison statistics
    print("\nComparison of original vs. processed test results:")
    
    # Test size info
    test_size = len(test_df)
    processed_test_size = len(processed_test)
    reduction = test_size - processed_test_size
    reduction_percent = round(100 * reduction / test_size, 2)
    
    # Analyze what was removed vs what was kept
    records_with_pii = full_results['has_pii'].sum()
    percent_with_pii = round(100 * records_with_pii / len(full_results), 2)
    
    # Accuracy on non-unknown predictions
    mapped_accuracy = accuracy_score(
        full_results[full_results['predicted'] != 'unknown']['actual'],
        full_results[full_results['predicted'] != 'unknown']['predicted']
    )
    
    print(f"Original test size: {test_size}")
    print(f"Processed test size: {processed_test_size}")
    print(f"Reduction: {reduction} records ({reduction_percent}%)")
    print(f"Records with PII: {records_with_pii} ({percent_with_pii}%)")
    print(f"Original accuracy: {accuracy:.4f}")
    print(f"Mapped accuracy: {mapped_accuracy:.4f}")
    
    # Example of anonymized texts
    print("\nExamples of anonymized texts:")
    anonymization_examples = full_results[full_results['has_pii']].iloc[:5]
    for idx, row in anonymization_examples.iterrows():
        print(f"Original: {test_df.loc[idx, 'text']}")
        print(f"Anonymized: {row['anonymized_text']}")
        print(f"Category: {row['actual']}, Prediction: {row['predicted']}")
        print("---")
    
    print("\nExample complete!")
    return {
        "original_df": original_df,
        "train_df": train_df,
        "test_df": test_df,
        "processed_train": processed_train,
        "processed_test": processed_test,
        "full_results": full_results,
        "pipeline": pipeline,
        "test_pipeline": test_pipeline,
        "model": model
    }


if __name__ == "__main__":
    main()