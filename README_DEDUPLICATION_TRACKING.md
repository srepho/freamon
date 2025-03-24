# Freamon Deduplication Tracking

This document provides an overview of how to use Freamon's deduplication tracking functionality to map results between deduplication and other machine learning tasks.

## Overview

When performing deduplication in a data pipeline, it's essential to maintain the relationship between the original dataset and the deduplicated dataset. This allows you to:

1. Map model predictions back to the original dataset
2. Track which records were identified as duplicates
3. Perform analysis on both original and deduplicated data
4. Apply transformations to deduplicated data and propagate results to the original data

## Examples

Freamon provides several examples demonstrating different approaches to deduplication tracking:

### Basic Deduplication Tracking

[`deduplication_tracking_example.py`](examples/deduplication_tracking_example.py) demonstrates:
- Creating a simple `IndexTracker` class to maintain mappings
- Performing deduplication while preserving index mappings
- Running a machine learning task on deduplicated data
- Mapping predictions back to the original dataset

```python
from examples.deduplication_tracking_example import IndexTracker

# Initialize tracker
tracker = IndexTracker().initialize_from_df(df)

# Remove duplicates and update tracker
deduped_df = remove_duplicates(df, keep='first')
kept_indices = deduped_df.index.tolist()
tracker.update_from_kept_indices(kept_indices, deduped_df)

# Map results back to original dataset
full_results = tracker.create_full_result_df(
    test_results, df, fill_value={'predicted': None}
)
```

### Advanced Deduplication Tracking

[`advanced_deduplication_tracking.py`](examples/advanced_deduplication_tracking.py) demonstrates:
- Using advanced text deduplication techniques (hash, n-gram, similarity)
- Tracking indices through a multi-step deduplication process
- Visualizing the deduplication process with charts
- Generating HTML reports showing the tracking information

```python
from examples.advanced_deduplication_tracking import DeduplicationTracker

# Initialize tracker
tracker = DeduplicationTracker().initialize(df)

# Perform hash-based deduplication and update tracking
kept_indices = hash_deduplication(text_series, return_indices=True)
tracker.update_mapping(kept_indices, "Hash-based")

# Perform n-gram fingerprint deduplication and update tracking
kept_indices = ngram_fingerprint_deduplication(text_series, n=3, return_indices=True)
tracker.update_mapping(kept_indices, "N-gram Fingerprint")

# Generate a tracking report
report_path = tracker.generate_tracking_report(
    original_df, deduplicated_df, "deduplication_tracking_report.html"
)
```

### Pipeline Integration with Deduplication Tracking

[`pipeline_with_deduplication_tracking.py`](examples/pipeline_with_deduplication_tracking.py) demonstrates:
- Creating a pipeline that includes deduplication steps
- Tracking data through the pipeline while maintaining original indices
- Running a machine learning task on deduplicated data
- Mapping results back to the original dataset
- Evaluating the impact of deduplication on model performance

```python
from examples.pipeline_with_deduplication_tracking import IndexTrackingPipeline, HashDeduplicationStep

# Create pipeline with deduplication steps
pipeline = IndexTrackingPipeline(steps=[
    TextPreprocessingStep(text_column='text'),
    HashDeduplicationStep(text_column='processed_text'),
    SimilarityDeduplicationStep(text_column='processed_text'),
    ModelTrainingStep()
])

# Run pipeline with tracking
processed_data = pipeline.fit_transform(train_df)

# Map results back to original indices
mapped_results = pipeline.create_full_result_df(
    'model_training', results_df, fill_value={'predicted': 'unknown'}
)
```

## Implementation Details

### IndexTracker Class

The `IndexTracker` class maintains bidirectional mappings between original and current indices:

```python
class IndexTracker:
    def __init__(self):
        self.original_to_current = {}  # Maps original indices to current indices
        self.current_to_original = {}  # Maps current indices to original indices
    
    def update_mapping(self, original_indices, new_indices):
        # Update mappings after a transformation
        
    def map_to_original(self, current_indices):
        # Map current indices back to original indices
        
    def map_to_current(self, original_indices):
        # Map original indices to current indices
        
    def create_full_result_df(self, result_df, original_df, fill_value=None):
        # Create a result dataframe with all original indices
```

### DeduplicationTracker Class

The `DeduplicationTracker` class extends this functionality with:

1. Multiple deduplication step tracking
2. Visualization capabilities
3. Similarity and cluster tracking
4. HTML report generation

### IndexTrackingPipeline Class

The `IndexTrackingPipeline` class integrates index tracking into Freamon's pipeline:

1. Tracks indices through each pipeline step
2. Updates mappings when data size changes (e.g., after deduplication)
3. Provides methods to map between different pipeline steps
4. Supports creating full result dataframes with all original indices

## Best Practices

1. **Initialize tracking early**: Create and initialize your tracker with the original dataframe before any transformations
2. **Update after each transformation**: Update index mappings after any step that changes the dataset size
3. **Preserve original data**: Keep a copy of the original dataframe for reference and final mapping
4. **Use descriptive step names**: When using a pipeline, use descriptive step names to make mapping easier
5. **Fill missing values appropriately**: When mapping results back, choose appropriate fill values for records that were deduplicated

## Advanced Usage

### Multi-step Tracking

Track data through multiple transformations:

```python
tracker = IndexTracker().initialize_from_df(original_df)

# First transformation
df1 = transform_1(original_df)
tracker.update_from_kept_indices(df1.index)

# Second transformation
df2 = transform_2(df1)
tracker.update_from_kept_indices(df2.index)

# Map from df2 back to original
original_indices = tracker.map_to_original(df2.index)
```

### Clustering and Similarity Analysis

Track cluster membership and similarity relationships:

```python
tracker = DeduplicationTracker().initialize(df)

# Find similar texts
similarity_dict = find_similar_texts(texts, method='cosine')
tracker.store_similarity_info(similarity_dict)

# Cluster texts
clusters = cluster_texts_hierarchical(texts)
tracker.store_clusters(clusters)

# Generate visualization
tracker.plot_cluster_distribution()
```

### Custom Pipeline Steps

Create custom pipeline steps that update index tracking:

```python
class CustomDeduplicationStep(PreprocessingStep):
    def transform(self, df):
        # Perform deduplication
        result_df = deduplicate(df)
        
        # Update tracking in the pipeline
        if hasattr(self.pipeline, 'update_mapping'):
            self.pipeline.update_mapping(self.name, result_df.index)
            
        return result_df
```

## Conclusion

Freamon's deduplication tracking functionality makes it easy to integrate deduplication into your data science workflow while maintaining the ability to map results back to your original data. This ensures you can perform analysis on deduplicated data for efficiency, while still being able to apply those insights to all of your original data.

For more details, see the example implementations and API documentation.