# Anonymization Integration with Allyanonimiser

Freamon provides integration with the Allyanonimiser library for anonymizing personally identifiable information (PII) in text data while maintaining the ability to track indices through deduplication pipelines.

## Overview

The integration allows you to:

1. Detect PII in text data
2. Anonymize PII in text data
3. Integrate anonymization into deduplication pipelines
4. Track original indices through anonymization and deduplication
5. Map results back to the original dataset

## Components

The integration is provided through the following components:

- `AnonymizationStep`: A pipeline step that anonymizes PII in text data
- `EnhancedTextPreprocessingStep`: A pipeline step that combines text preprocessing and anonymization
- `PIIDetectionStep`: A pipeline step that detects PII in text data without anonymizing it

## Requirements

To use the integration, you need to install the Allyanonimiser library:

```bash
pip install allyanonimiser
```

## Usage Examples

### Basic Anonymization

```python
from freamon.integration.allyanonimiser_bridge import AnonymizationStep

# Create anonymization step
anonymize_step = AnonymizationStep(
    text_column='text',
    output_column='anonymized_text'
)

# Apply to dataframe
result = anonymize_step.fit_transform(df)
```

### Enhanced Text Preprocessing with Anonymization

```python
from freamon.integration.allyanonimiser_bridge import EnhancedTextPreprocessingStep

# Create enhanced preprocessing step with anonymization
preprocess_step = EnhancedTextPreprocessingStep(
    text_column='text',
    anonymize=True,
    preprocessing_options={
        'lowercase': True,
        'remove_punctuation': True
    }
)

# Apply to dataframe
result = preprocess_step.fit_transform(df)
```

### PII Detection

```python
from freamon.integration.allyanonimiser_bridge import PIIDetectionStep

# Create PII detection step
detection_step = PIIDetectionStep(
    text_column='text',
    include_details=True
)

# Apply to dataframe
result = detection_step.fit_transform(df)

# Check which records have PII
pii_records = result[result['pii_has_pii'] == True]
```

### Integration with Deduplication Pipeline

```python
from freamon.integration.allyanonimiser_bridge import AnonymizationStep
from pipeline_with_deduplication_tracking import (
    IndexTrackingPipeline,
    TextPreprocessingStep,
    HashDeduplicationStep
)

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
result = pipeline.fit_transform(df)

# Map results back to original dataset
full_results = pipeline.create_full_result_df(
    'hash_deduplication',
    result[['text', 'anonymized_text', 'category']],
    fill_value=None
)
```

## Complete Example

A complete example showing how to use the integration with a machine learning workflow is available in the examples directory:

```python
from examples.anonymization_deduplication_example import main

results = main()
```

This example demonstrates:

1. Creating a pipeline that includes anonymization and deduplication steps
2. Tracking data through the pipeline while maintaining original indices
3. Running a machine learning task on deduplicated and anonymized data
4. Mapping results back to the original dataset
5. Evaluating the impact on model performance

## Customizing Anonymization

You can customize the behavior of the Allyanonimiser components by passing configuration options:

```python
from freamon.integration.allyanonimiser_bridge import AnonymizationStep

# Create anonymization step with custom configuration
anonymize_step = AnonymizationStep(
    text_column='text',
    output_column='anonymized_text',
    anonymization_config={
        'keep_original_casing': True,
        'pattern_groups': ['financial', 'au_pii', 'general_pii']
    }
)
```

Refer to the Allyanonimiser documentation for more details on available configuration options.

## Using with Index Tracking

All the integration components are designed to work with Freamon's index tracking system, which allows you to maintain the relationship between original and processed data through multiple transformation steps. This is particularly important when combining anonymization with deduplication, as both operations can change the dataset size and structure.

```python
# First anonymize
anonymize_step = AnonymizationStep(
    text_column='text',
    output_column='anonymized_text'
)
anonymized_df = anonymize_step.fit_transform(df)

# Track indices through deduplication
tracker = IndexTracker().initialize_from_df(anonymized_df)
text_series = anonymized_df['anonymized_text']
kept_indices = hash_deduplication(text_series, return_indices=True)
tracker.update_from_kept_indices(kept_indices, anonymized_df.iloc[kept_indices].reset_index(drop=True))

# Get deduplicated dataframe
deduped_df = anonymized_df.iloc[kept_indices].reset_index(drop=True)

# Map results back to original
results_df = pd.DataFrame({'result': model.predict(deduped_df)}, index=deduped_df.index)
full_results = tracker.create_full_result_df(
    results_df, 
    anonymized_df,
    fill_value={'result': 'unknown'}
)
```