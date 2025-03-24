# Deduplication Tracker Development Guide

This document provides guidance for developers working on extending the deduplication tracking functionality in Freamon.

## Architecture Overview

The deduplication tracking system consists of three main components:

1. **IndexTracker** - Basic tracking for maintaining mappings between original and current indices
2. **DeduplicationTracker** - Enhanced tracking with visualization and reporting capabilities
3. **IndexTrackingPipeline** - Pipeline integration for preserving mappings through transformations

## Development Guidelines

### General Principles

1. **Index Preservation** - All tracking classes must maintain bidirectional mappings between original and current indices
2. **Performance First** - Tracking should add minimal overhead, especially for large datasets
3. **Consistent API** - Follow existing patterns for method naming and parameter conventions
4. **Progressive Enhancement** - Basic functionality should work without dependencies, advanced features can require additional libraries
5. **Comprehensive Testing** - All components need thorough unit and integration tests

### Testing Strategy

For new tracking components, implement tests at multiple levels:

1. **Unit Tests**
   - Test individual mapping functions with controlled inputs
   - Verify correct behavior with edge cases (empty datasets, all duplicates, etc.)
   - Test serialization/deserialization of mapping information

2. **Component Tests**
   - Test visualization functions with known input data
   - Verify HTML report generation with different configurations
   - Test integration between tracking components

3. **Integration Tests**
   - Test full workflows from data loading through deduplication to prediction
   - Verify correct mapping of results back to original indices
   - Test performance with various dataset sizes

4. **Regression Tests**
   - Create benchmark datasets for consistent performance testing
   - Track memory usage and execution time
   - Ensure backward compatibility with existing APIs

### Development Tasks

#### Phase 1: Core Functionality

- [ ] **Enhanced IndexTracker**
  - Add support for multi-step transformations
  - Implement serialization/deserialization
  - Add memory-efficient storage for large mappings

- [ ] **Pipeline Integration**
  - Create base `TrackingPipelineStep` class
  - Add automatic tracking to existing pipeline steps
  - Implement visualization of data flow

- [ ] **Test Suite**
  - Create comprehensive unit tests
  - Add integration tests with real-world datasets
  - Implement performance benchmarks

#### Phase 2: Visualization and Reporting

- [ ] **Performance Visualization**
  - Add impact visualization for ML metrics
  - Create duplicate cluster visualization
  - Implement interactive dashboards

- [ ] **Enhanced Reporting**
  - Add deduplication metrics to HTML reports
  - Create standalone deduplication reports
  - Implement exportable visualizations

- [ ] **Documentation**
  - Create architectural diagrams
  - Add comprehensive API documentation
  - Develop tutorial notebooks

#### Phase 3: Advanced Features

- [ ] **Smart Deduplication**
  - Implement method recommendation
  - Add adaptive thresholding
  - Create confidence scores

- [ ] **Scalability Enhancements**
  - Add distributed processing support
  - Implement chunk-based processing
  - Add progress monitoring

## Code Examples

### Basic Tracking Template

```python
def process_with_tracking(df, transformation_fn):
    """
    Process a dataframe with a transformation while tracking indices.
    
    Args:
        df: Input dataframe
        transformation_fn: Function that transforms the dataframe
        
    Returns:
        Tuple of (transformed_df, tracker)
    """
    # Initialize tracker
    tracker = IndexTracker().initialize_from_df(df)
    
    # Apply transformation
    result_df = transformation_fn(df)
    
    # Update tracking
    tracker.update_from_kept_indices(result_df.index, result_df)
    
    return result_df, tracker
```

### Pipeline Step Template

```python
class TrackingTransformStep(PipelineStep):
    """Pipeline step that tracks indices through transformations."""
    
    def __init__(self, transform_fn, name=None):
        super().__init__(name=name or "tracking_transform")
        self.transform_fn = transform_fn
        self._tracker = None
        
    def fit(self, df):
        """Fit the transformation (if needed)."""
        return self
        
    def transform(self, df):
        """Transform the data while tracking indices."""
        # Initialize tracker if not already done
        if self._tracker is None:
            self._tracker = IndexTracker().initialize_from_df(df)
            
        # Apply transformation
        result_df = self.transform_fn(df)
        
        # Update tracking
        self._tracker.update_from_kept_indices(result_df.index, result_df)
        
        # Store tracking in pipeline if available
        if hasattr(self.pipeline, 'update_mapping') and self.name:
            self.pipeline.update_mapping(self.name, result_df.index)
            
        return result_df
        
    def get_tracker(self):
        """Get the current tracker."""
        return self._tracker
```

## Contributing

To contribute to the deduplication tracking functionality:

1. Review the development tasks and choose an item to work on
2. Create a feature branch from `main`
3. Implement your changes following the guidelines above
4. Add comprehensive tests
5. Submit a pull request with detailed description of changes

Please refer to CLAUDE.md and the main README for general contribution guidelines.