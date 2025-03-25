"""
Tests for the IndexTrackingPipeline class.
"""
import pandas as pd
import numpy as np
import pytest

from freamon.pipeline.steps import (
    PreprocessingStep,
    ModelTrainingStep,
    TransformationStep
)


class TestIndexTrackingPipeline:
    """Test class for IndexTrackingPipeline functionality."""
    
    @pytest.fixture
    def pipeline_classes(self):
        """Import the pipeline classes from the example."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        from pipeline_with_deduplication_tracking import (
            IndexTrackingPipeline,
            HashDeduplicationStep,
            SimilarityDeduplicationStep,
            TextPreprocessingStep
        )
        return {
            'IndexTrackingPipeline': IndexTrackingPipeline,
            'HashDeduplicationStep': HashDeduplicationStep,
            'SimilarityDeduplicationStep': SimilarityDeduplicationStep,
            'TextPreprocessingStep': TextPreprocessingStep
        }
    
    @pytest.fixture
    def sample_text_df(self):
        """Create a sample text dataframe for pipeline testing."""
        np.random.seed(42)
        
        # Create categories and templates
        categories = ['business', 'technology', 'health']
        templates = {
            'business': [
                "The company announced quarterly earnings today.",
                "Investors are concerned about market trends.",
                "The stock market showed signs of recovery."
            ],
            'technology': [
                "The new smartphone features cutting-edge technology.",
                "Developers are excited about the new programming language.",
                "The startup secured funding for their AI project."
            ],
            'health': [
                "Researchers discovered a new treatment for the disease.",
                "The study shows a correlation between diet and health.",
                "Doctors recommend regular exercise for well-being."
            ]
        }
        
        # Generate data
        texts = []
        labels = []
        
        # Generate original data
        for _ in range(80):
            category = np.random.choice(categories)
            template = np.random.choice(templates[category])
            texts.append(template)
            labels.append(category)
        
        # Add duplicates
        for _ in range(20):
            idx = np.random.randint(0, len(texts))
            texts.append(texts[idx])
            labels.append(labels[idx])
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'category': labels
        })
        
        return df

    def test_initialize_tracking(self, pipeline_classes, sample_text_df):
        """Test initializing tracking with a dataframe."""
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        
        # Create pipeline
        pipeline = IndexTrackingPipeline()
        pipeline.initialize_tracking(sample_text_df)
        
        # Check tracking is initialized
        assert pipeline.original_indices is not None
        assert len(pipeline.original_indices) == len(sample_text_df)
        assert 'original' in pipeline.index_mappings
        assert len(pipeline.index_mappings['original']['current_to_original']) == len(sample_text_df)
        
        # Check initial mappings are identity
        for i in sample_text_df.index:
            assert pipeline.index_mappings['original']['current_to_original'][i] == i
            assert pipeline.index_mappings['original']['original_to_current'][i] == i
    
    def test_update_mapping(self, pipeline_classes, sample_text_df):
        """Test updating mappings after a transformation."""
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        
        # Create pipeline
        pipeline = IndexTrackingPipeline()
        pipeline.initialize_tracking(sample_text_df)
        
        # Create a subset of indices to keep
        kept_indices = sample_text_df.index[::2].tolist()  # Keep every other row
        
        # Update mapping
        pipeline.update_mapping('test_step', kept_indices)
        
        # Check mapping is updated
        assert 'test_step' in pipeline.index_mappings
        mapping = pipeline.index_mappings['test_step']
        
        # Check size of mappings
        assert len(mapping['current_to_original']) == len(kept_indices)
        assert len(mapping['original_to_current']) == len(kept_indices)
        
        # Check mapping values
        for new_idx, orig_idx in enumerate(kept_indices):
            assert mapping['current_to_original'][new_idx] == orig_idx
            assert mapping['original_to_current'][orig_idx] == new_idx
    
    def test_map_to_original(self, pipeline_classes, sample_text_df):
        """Test mapping indices from a step back to original indices."""
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        
        # Create pipeline
        pipeline = IndexTrackingPipeline()
        pipeline.initialize_tracking(sample_text_df)
        
        # Create and apply step
        kept_indices = sample_text_df.index[::2].tolist()  # Keep every other row
        pipeline.update_mapping('test_step', kept_indices)
        
        # Map back to original
        new_indices = [0, 1, 2, 3, 4]
        original_indices = pipeline.map_to_original('test_step', new_indices)
        
        # Check mapping
        for i, new_idx in enumerate(new_indices):
            expected_orig = pipeline.index_mappings['test_step']['current_to_original'][new_idx]
            assert original_indices[i] == expected_orig
    
    def test_map_to_step(self, pipeline_classes, sample_text_df):
        """Test mapping indices between different pipeline steps."""
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        
        # Create pipeline
        pipeline = IndexTrackingPipeline()
        pipeline.initialize_tracking(sample_text_df)
        
        # Create first step
        kept_indices1 = sample_text_df.index[::2].tolist()  # Keep every other row
        pipeline.update_mapping('step1', kept_indices1)
        
        # Create second step
        kept_indices2 = [0, 2, 4, 6, 8]  # Keep some rows from first step
        step2_original_indices = [pipeline.index_mappings['step1']['current_to_original'][i] for i in kept_indices2]
        pipeline.update_mapping('step2', step2_original_indices)
        
        # Map from step1 to step2
        step1_indices = [0, 2, 4]
        step2_indices = pipeline.map_to_step('step1', 'step2', step1_indices)
        
        # Check mapping
        for i, step1_idx in enumerate(step1_indices):
            # Map to original
            orig_idx = pipeline.index_mappings['step1']['current_to_original'][step1_idx]
            # Map to step2
            expected_step2_idx = pipeline.index_mappings['step2']['original_to_current'].get(orig_idx)
            assert step2_indices[i] == expected_step2_idx
    
    def test_create_full_result_df(self, pipeline_classes, sample_text_df):
        """Test creating a full result dataframe with all original indices."""
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        
        # Create pipeline
        pipeline = IndexTrackingPipeline()
        pipeline.initialize_tracking(sample_text_df)
        
        # Create step
        kept_indices = sample_text_df.index[::2].tolist()  # Keep every other row
        pipeline.update_mapping('test_step', kept_indices)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'prediction': ['A', 'B', 'C'] * (len(kept_indices) // 3 + 1),
            'score': np.random.random(len(kept_indices))
        }, index=range(len(kept_indices)))
        
        # Create full result
        full_result = pipeline.create_full_result_df('test_step', result_df, fill_value={'prediction': 'unknown', 'score': 0.0})
        
        # Check full result
        assert len(full_result) == len(sample_text_df)
        assert 'in_processed_data' in full_result.columns
        
        # Check values for kept indices
        for i, orig_idx in enumerate(kept_indices):
            assert full_result.loc[orig_idx, 'prediction'] == result_df.loc[i, 'prediction']
            assert full_result.loc[orig_idx, 'score'] == result_df.loc[i, 'score']
            assert full_result.loc[orig_idx, 'in_processed_data'] == True
        
        # Check values for non-kept indices
        non_kept = set(sample_text_df.index) - set(kept_indices)
        for idx in non_kept:
            assert full_result.loc[idx, 'prediction'] == 'unknown'
            assert full_result.loc[idx, 'score'] == 0.0
            assert full_result.loc[idx, 'in_processed_data'] == False
    
    def test_pipeline_with_deduplication_steps(self, pipeline_classes, sample_text_df):
        """Test running a pipeline with deduplication steps and tracking."""
        # Get classes
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        TextPreprocessingStep = pipeline_classes['TextPreprocessingStep']
        HashDeduplicationStep = pipeline_classes['HashDeduplicationStep']
        
        # Create pipeline steps
        preprocessing_step = TextPreprocessingStep(text_column='text', name='preprocessing')
        dedup_step = HashDeduplicationStep(text_column='processed_text', name='deduplication')
        transform_step = TransformationStep(
            transform_fn=lambda df: pd.get_dummies(df['category'], prefix='category'),
            name='feature_extraction'
        )
        
        # Create pipeline
        pipeline = IndexTrackingPipeline(
            steps=[preprocessing_step, dedup_step, transform_step],
            name='test_pipeline'
        )
        
        # Run pipeline
        result = pipeline.fit_transform(sample_text_df)
        
        # Check pipeline ran successfully
        assert result is not None
        assert len(result) < len(sample_text_df)  # Should have removed duplicates
        
        # Check tracking was updated
        assert 'preprocessing' in pipeline.index_mappings
        assert 'deduplication' in pipeline.index_mappings
        assert 'feature_extraction' in pipeline.index_mappings
        
        # Verify deduplication mappings are correct
        dedup_mapping = pipeline.index_mappings['deduplication']
        assert len(dedup_mapping['current_to_original']) == len(result)
        
        # Create a result dataframe
        mock_result = pd.DataFrame({'value': range(len(result))}, index=result.index)
        
        # Map back to original
        full_result = pipeline.create_full_result_df('feature_extraction', mock_result)
        
        # Check mapping
        assert len(full_result) == len(sample_text_df)
        assert full_result['in_processed_data'].sum() == len(result)
    
    def test_fit_transform_with_stop_after(self, pipeline_classes, sample_text_df):
        """Test stopping pipeline after a specific step."""
        # Get classes
        IndexTrackingPipeline = pipeline_classes['IndexTrackingPipeline']
        TextPreprocessingStep = pipeline_classes['TextPreprocessingStep']
        HashDeduplicationStep = pipeline_classes['HashDeduplicationStep']
        
        # Create pipeline steps
        preprocessing_step = TextPreprocessingStep(text_column='text', name='preprocessing')
        dedup_step = HashDeduplicationStep(text_column='processed_text', name='deduplication')
        transform_step = TransformationStep(
            transform_fn=lambda df: pd.get_dummies(df['category'], prefix='category'),
            name='feature_extraction'
        )
        
        # Create pipeline
        pipeline = IndexTrackingPipeline(
            steps=[preprocessing_step, dedup_step, transform_step],
            name='test_pipeline'
        )
        
        # Run pipeline with stop_after
        result = pipeline.fit_transform(sample_text_df, stop_after='preprocessing')
        
        # Check pipeline stopped after preprocessing
        assert 'processed_text' in result.columns
        assert len(result) == len(sample_text_df)  # No deduplication yet
        
        # Check tracking
        assert 'preprocessing' in pipeline.index_mappings
        assert 'deduplication' not in pipeline.index_mappings
        assert 'feature_extraction' not in pipeline.index_mappings
    
    def test_custom_deduplication_steps(self, pipeline_classes, sample_text_df):
        """Test custom deduplication steps in the pipeline."""
        # Get classes
        HashDeduplicationStep = pipeline_classes['HashDeduplicationStep']
        SimilarityDeduplicationStep = pipeline_classes['SimilarityDeduplicationStep']
        
        # Create test dataframe with duplicates
        df = sample_text_df.copy()
        
        # Test HashDeduplicationStep
        hash_step = HashDeduplicationStep(text_column='text', name='hash_dedup')
        hash_step.fit(df)
        hash_result = hash_step.transform(df)
        
        # Check hash deduplication worked
        assert len(hash_result) < len(df)
        
        # Test SimilarityDeduplicationStep
        sim_step = SimilarityDeduplicationStep(
            text_column='text',
            method='cosine',
            threshold=0.8,
            name='sim_dedup'
        )
        sim_step.fit(df)
        sim_result = sim_step.transform(df)
        
        # Check similarity deduplication worked
        assert len(sim_result) < len(df)