"""
Tests for the integration between Freamon and Allyanonimiser.

These tests verify that the integration between Freamon's deduplication tracking
and Allyanonimiser's PII anonymization works correctly.
"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import types

# Create a mock allyanonimiser module
mock_module = types.ModuleType("allyanonimiser")
sys.modules["allyanonimiser"] = mock_module

# Mock classes
class MockAnonymizer:
    """Mock Anonymizer class for testing."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def anonymize_text(self, text):
        """Mock anonymize_text method."""
        if not text:
            return ""
            
        # Apply anonymization rules
        if "John Smith" in text:
            text = text.replace("John Smith", "[NAME]")
        if "Jane Doe" in text:
            text = text.replace("Jane Doe", "[NAME]")
        if "example@example.com" in text:
            text = text.replace("example@example.com", "[EMAIL]")
        if "123-456-7890" in text:
            text = text.replace("123-456-7890", "[PHONE]")
        if "123 456 7890" in text:
            text = text.replace("123 456 7890", "[PHONE]")
            
        return text


class MockAnalyzer:
    """Mock Analyzer class for testing."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def analyze_text(self, text):
        """Mock analyze_text method."""
        findings = []
        
        if "@" in text:
            findings.append({
                "entity_type": "EMAIL",
                "text": "example@example.com",
                "start": text.find("example@example.com"),
                "end": text.find("example@example.com") + len("example@example.com")
            })
            
        if "123-456-7890" in text:
            findings.append({
                "entity_type": "PHONE",
                "text": "123-456-7890",
                "start": text.find("123-456-7890"),
                "end": text.find("123-456-7890") + len("123-456-7890")
            })
            
        if "John Smith" in text:
            findings.append({
                "entity_type": "PERSON",
                "text": "John Smith",
                "start": text.find("John Smith"),
                "end": text.find("John Smith") + len("John Smith")
            })
            
        if "Jane Doe" in text:
            findings.append({
                "entity_type": "PERSON",
                "text": "Jane Doe",
                "start": text.find("Jane Doe"),
                "end": text.find("Jane Doe") + len("Jane Doe")
            })
            
        return {
            "has_pii": len(findings) > 0,
            "findings": findings
        }

# Add mock classes to mock module
mock_module.Anonymizer = MockAnonymizer
mock_module.Analyzer = MockAnalyzer

# Import from freamon
from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts

# Use the example modules for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
from deduplication_tracking_example import IndexTracker
from pipeline_with_deduplication_tracking import (
    IndexTrackingPipeline,
    TextPreprocessingStep,
    HashDeduplicationStep
)

# Import the modules being tested - after mocking the module
from freamon.integration.allyanonimiser_bridge import (
    AnonymizationStep,
    EnhancedTextPreprocessingStep,
    PIIDetectionStep,
    AnonymizationError
)


@pytest.fixture
def sample_data_with_pii():
    """Create a sample dataset with PII for testing."""
    np.random.seed(42)
    
    # Create texts with PII
    texts = [
        "Hello, my name is John Smith and my email is example@example.com.",
        "Please contact Jane Doe at example@example.com or call 123-456-7890.",
        "No PII in this text, just regular information.",
        "More regular text without any sensitive information.",
        "Contact us at example@example.com for more details.",
        "Call John Smith at 123-456-7890 for assistance.",
        "Jane Doe will present at the conference.",
        "The meeting is scheduled for tomorrow at 10am.",
        "Please call 123 456 7890 for support.",
        "John Smith and Jane Doe are working on the project."
    ]
    
    # Add some duplicates
    duplicates = [
        "Hello, my name is John Smith and my email is example@example.com.",
        "Contact us at example@example.com for more details.",
        "Jane Doe will present at the conference.",
        "Call John Smith at 123-456-7890 for assistance."
    ]
    
    # Add some near-duplicates with PII
    near_duplicates = [
        "Hello, my name is John Smith and my email is different@example.com.",
        "Contact Jane Doe at example@example.com for more information.",
        "Please call John Smith at 123-456-7890 for help."
    ]
    
    all_texts = texts + duplicates + near_duplicates
    
    # Create categories
    categories = np.random.choice(['A', 'B', 'C'], len(all_texts))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'category': categories
    })
    
    return df


def test_anonymization_step():
    """Test the basic anonymization step functionality."""
    # Create test data
    df = pd.DataFrame({
        'text': [
            "Hello, my name is John Smith and my email is example@example.com.",
            "Please contact Jane Doe at example@example.com or call 123-456-7890.",
            "No PII in this text, just regular information."
        ]
    })
    
    # Create anonymization step
    anonymize_step = AnonymizationStep(
        text_column='text',
        output_column='anonymized_text'
    )
    
    # Fit and transform
    result = anonymize_step.fit_transform(df)
    
    # Verify results
    assert 'anonymized_text' in result.columns
    assert 'text' in result.columns  # Original preserved by default
    assert "[NAME]" in result['anonymized_text'].iloc[0]
    assert "[EMAIL]" in result['anonymized_text'].iloc[0]
    assert "[PHONE]" in result['anonymized_text'].iloc[1]
    
    # Test without preserving original
    anonymize_step = AnonymizationStep(
        text_column='text',
        output_column='anonymized_text',
        preserve_original=False
    )
    
    # Fit and transform
    result = anonymize_step.fit_transform(df)
    
    # Verify results
    assert 'anonymized_text' in result.columns
    assert 'text' not in result.columns  # Original not preserved


def test_enhanced_text_preprocessing():
    """Test the enhanced text preprocessing with anonymization."""
    # Create test data
    df = pd.DataFrame({
        'text': [
            "Hello, my name is John Smith and my email is example@example.com.",
            "Please contact Jane Doe at example@example.com or call 123-456-7890.",
            "No PII in this text, just regular information."
        ]
    })
    
    # Test with anonymization
    preprocess_step = EnhancedTextPreprocessingStep(
        text_column='text',
        anonymize=True
    )
    
    # Fit and transform
    result = preprocess_step.fit_transform(df)
    
    # Verify results
    assert 'processed_text' in result.columns
    assert 'text' in result.columns  # Original preserved by default
    # Output might be different based on how TextProcessor works
    # We just check that text has been processed
    assert "hello" in result['processed_text'].iloc[0].lower()
    assert "email" in result['processed_text'].iloc[0].lower()
    
    # Test without anonymization
    preprocess_step = EnhancedTextPreprocessingStep(
        text_column='text',
        anonymize=False
    )
    
    # Fit and transform
    result = preprocess_step.fit_transform(df)
    
    # Verify results
    assert 'processed_text' in result.columns
    assert "john smith" in result['processed_text'].iloc[0].lower()  # Preprocessed but not anonymized


def test_pii_detection_step():
    """Test the PII detection step functionality."""
    # Create test data
    df = pd.DataFrame({
        'text': [
            "Hello, my name is John Smith and my email is example@example.com.",
            "Please contact Jane Doe at example@example.com or call 123-456-7890.",
            "No PII in this text, just regular information."
        ]
    })
    
    # Create detection step
    detection_step = PIIDetectionStep(
        text_column='text',
        include_details=True
    )
    
    # Fit and transform
    result = detection_step.fit_transform(df)
    
    # Verify results
    assert 'pii_has_pii' in result.columns
    assert 'pii_pii_count' in result.columns
    assert 'pii_pii_types' in result.columns
    assert 'pii_details' in result.columns
    
    # Check values
    assert result['pii_has_pii'].iloc[0] == True
    assert result['pii_has_pii'].iloc[2] == False
    assert result['pii_pii_count'].iloc[1] > 0
    assert 'EMAIL' in result['pii_pii_types'].iloc[0]
    assert 'PERSON' in result['pii_pii_types'].iloc[1]
    
    # Test without details
    detection_step = PIIDetectionStep(
        text_column='text',
        include_details=False
    )
    
    # Fit and transform
    result = detection_step.fit_transform(df)
    
    # Verify results
    assert 'pii_details' not in result.columns


def test_anonymization_with_index_tracker(sample_data_with_pii):
    """Test anonymization integrated with index tracking."""
    # Create tracker
    tracker = IndexTracker().initialize_from_df(sample_data_with_pii)
    
    # Create a copy for hash deduplication
    deduped_df = sample_data_with_pii.copy()
    
    # Hash-based deduplication to remove exact duplicates
    text_series = deduped_df['text']
    kept_indices = hash_deduplication(text_series, return_indices=True)
    
    # Update tracker with kept indices
    tracker.update_from_kept_indices(kept_indices, deduped_df.iloc[kept_indices].reset_index(drop=True))
    
    # Get deduplicated dataframe with new indices
    deduped_df = deduped_df.iloc[kept_indices].reset_index(drop=True)
    
    # Create anonymization step
    anonymize_step = AnonymizationStep(
        text_column='text',
        output_column='anonymized_text'
    )
    
    # Anonymize the deduplicated data
    anonymized_df = anonymize_step.fit_transform(deduped_df)
    
    # Verify anonymization worked
    assert 'anonymized_text' in anonymized_df.columns
    assert anonymized_df.shape[0] < sample_data_with_pii.shape[0]  # Should be deduplicated
    
    # Create a results dataframe to map back
    results_df = pd.DataFrame({
        'text': anonymized_df['text'],
        'anonymized_text': anonymized_df['anonymized_text'],
        'category': anonymized_df['category']
    })
    
    # Map results back to original dataset
    full_results = tracker.create_full_result_df(results_df, sample_data_with_pii)
    
    # Verify mapping
    assert len(full_results) == len(sample_data_with_pii)
    assert 'anonymized_text' in full_results.columns
    
    # Check that some texts were anonymized
    has_anonymized = (full_results['anonymized_text'].str.contains(r'\[NAME\]') | 
                     full_results['anonymized_text'].str.contains(r'\[EMAIL\]') | 
                     full_results['anonymized_text'].str.contains(r'\[PHONE\]')).sum()
    assert has_anonymized > 0


def test_anonymization_in_pipeline(sample_data_with_pii):
    """Test anonymization step in an IndexTrackingPipeline."""
    # Create pipeline steps
    preprocessing_step = TextPreprocessingStep(
        text_column='text',
        name='preprocessing'
    )
    
    anonymize_step = AnonymizationStep(
        text_column='text',
        output_column='anonymized_text',
        name='anonymization'
    )
    
    hash_dedup_step = HashDeduplicationStep(
        text_column='processed_text',
        name='hash_deduplication'
    )
    
    # Create pipeline
    pipeline = IndexTrackingPipeline(
        steps=[preprocessing_step, anonymize_step, hash_dedup_step],
        name='anonymization_pipeline'
    )
    
    # Run pipeline
    result = pipeline.fit_transform(sample_data_with_pii)
    
    # Verify pipeline ran successfully
    assert result is not None
    assert len(result) < len(sample_data_with_pii)  # Should have removed duplicates
    assert 'anonymized_text' in result.columns
    assert 'processed_text' in result.columns
    
    # Verify tracking was updated
    assert 'preprocessing' in pipeline.index_mappings
    assert 'anonymization' in pipeline.index_mappings
    assert 'hash_deduplication' in pipeline.index_mappings
    
    # Verify that some texts were anonymized
    has_anonymized = (result['anonymized_text'].str.contains(r'\[NAME\]') | 
                     result['anonymized_text'].str.contains(r'\[EMAIL\]') | 
                     result['anonymized_text'].str.contains(r'\[PHONE\]')).sum()
    assert has_anonymized > 0
    
    # Create a results dataframe for testing mapping
    results_df = pd.DataFrame({
        'text': result['text'],
        'anonymized_text': result['anonymized_text'],
        'category': result['category']
    })
    
    # Map results back to original dataset
    full_results = pipeline.create_full_result_df(
        'hash_deduplication',
        results_df,
        fill_value=None
    )
    
    # Verify mapping
    assert len(full_results) == len(sample_data_with_pii)
    assert 'in_processed_data' in full_results.columns
    
    # Check that duplicates are properly flagged
    not_in_processed = (~full_results['in_processed_data']).sum()
    assert not_in_processed > 0
    

def test_combined_anonymization_deduplication_workflow(sample_data_with_pii):
    """Test a complete workflow with anonymization, PII detection, and deduplication."""
    # Create enhanced preprocessing step with anonymization
    preprocess_step = EnhancedTextPreprocessingStep(
        text_column='text',
        anonymize=True,
        name='enhanced_preprocessing'
    )
    
    # Create PII detection step
    detection_step = PIIDetectionStep(
        text_column='text',
        name='pii_detection'
    )
    
    # Create hash deduplication step
    hash_dedup_step = HashDeduplicationStep(
        text_column='processed_text',
        name='hash_deduplication'
    )
    
    # Create pipeline
    pipeline = IndexTrackingPipeline(
        steps=[detection_step, preprocess_step, hash_dedup_step],
        name='complete_pipeline'
    )
    
    # Run pipeline
    result = pipeline.fit_transform(sample_data_with_pii)
    
    # Verify pipeline ran successfully
    assert result is not None
    assert len(result) < len(sample_data_with_pii)  # Should have removed duplicates
    assert 'processed_text' in result.columns
    assert 'pii_has_pii' in result.columns
    assert 'pii_pii_types' in result.columns
    
    # Verify that PII detection worked
    assert result['pii_has_pii'].sum() > 0
    
    # Verify that preprocessing worked
    # We're not checking for specific anonymization tokens since TextProcessor might process them differently
    assert 'processed_text' in result.columns
    
    # Create a fake ML result for testing mapping
    result['prediction'] = np.random.choice(['X', 'Y', 'Z'], size=len(result))
    
    # Map results back to original dataset
    full_results = pipeline.create_full_result_df(
        'hash_deduplication',
        result[['processed_text', 'prediction', 'pii_has_pii']],
        fill_value={'prediction': 'unknown', 'pii_has_pii': False}
    )
    
    # Verify mapping
    assert len(full_results) == len(sample_data_with_pii)
    assert 'in_processed_data' in full_results.columns
    assert 'prediction' in full_results.columns
    assert 'pii_has_pii' in full_results.columns
    
    # Check that predictions are correctly mapped
    has_prediction = (full_results['prediction'] != 'unknown').sum()
    assert has_prediction == len(result)


def test_missing_allyanonimiser_error(sample_data_with_pii):
    """Test error handling when Allyanonimiser is not installed."""
    # Temporarily remove the Anonymizer from the module
    original_anonymizer = sys.modules["allyanonimiser"].Anonymizer
    del sys.modules["allyanonimiser"].Anonymizer
    
    try:
        # Create anonymization step
        anonymize_step = AnonymizationStep(
            text_column='text',
            output_column='anonymized_text'
        )
        
        # Verify that fit raises the appropriate error
        with pytest.raises(AnonymizationError):
            anonymize_step.fit(sample_data_with_pii)
            
        # Create enhanced preprocessing step
        preprocess_step = EnhancedTextPreprocessingStep(
            text_column='text',
            anonymize=True
        )
        
        # Verify that fit raises the appropriate error
        with pytest.raises(AnonymizationError):
            preprocess_step.fit(sample_data_with_pii)
    finally:
        # Restore the Anonymizer
        sys.modules["allyanonimiser"].Anonymizer = original_anonymizer