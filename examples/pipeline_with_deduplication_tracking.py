#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating a complete pipeline with deduplication and result tracking.

This example shows:
1. Creating a pipeline that includes deduplication steps
2. Tracking data through the pipeline while maintaining original indices
3. Running a machine learning task on deduplicated data
4. Mapping results back to the original dataset
5. Evaluating the impact of deduplication on model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts
from freamon.pipeline.pipeline import Pipeline
from freamon.pipeline.steps import (
    PreprocessingStep,
    ModelTrainingStep,
    TransformationStep,
    DataSplittingStep,
    PredictionStep,
    EvaluationStep
)
from freamon.utils.text_utils import TextProcessor
from freamon.modeling.factory import create_model


class IndexTrackingPipeline(Pipeline):
    """Pipeline extension that tracks indices through transformations."""
    
    def __init__(self, steps=None, name=None):
        """Initialize the pipeline with index tracking."""
        super().__init__(steps=steps)
        self.name = name  # Store name separately
        self.original_indices = None
        self.index_mappings = {}  # Maps step name to index mapping dict
        self._original_df = None
    
    def initialize_tracking(self, df):
        """Initialize tracking with the original dataframe."""
        self.original_indices = df.index.tolist()
        self._original_df = df.copy()
        self.index_mappings['original'] = {
            'current_to_original': {i: i for i in df.index},
            'original_to_current': {i: i for i in df.index}
        }
        return self
    
    def update_mapping(self, step_name, current_indices):
        """Update index mapping after a step transformation."""
        if step_name not in self.index_mappings:
            prev_step = list(self.index_mappings.keys())[-1]
            prev_mapping = self.index_mappings[prev_step]
        else:
            prev_mapping = self.index_mappings[step_name]
        
        # Create new mappings
        current_to_original = {}
        original_to_current = {}
        
        # Map each current index to its original
        for new_idx, curr_idx in enumerate(current_indices):
            # Look up the original index through previous mapping
            orig_idx = prev_mapping['current_to_original'].get(curr_idx, curr_idx)
            
            # Update mappings
            current_to_original[new_idx] = orig_idx
            original_to_current[orig_idx] = new_idx
        
        # Store new mappings
        self.index_mappings[step_name] = {
            'current_to_original': current_to_original,
            'original_to_current': original_to_current
        }
        
        return self
    
    def map_to_original(self, step_name, indices):
        """Map indices from a specific step back to original indices."""
        if step_name not in self.index_mappings:
            raise ValueError(f"Step '{step_name}' not found in index mappings")
            
        mapping = self.index_mappings[step_name]['current_to_original']
        
        if isinstance(indices, pd.Index):
            return indices.map(lambda x: mapping.get(x, x))
        return [mapping.get(idx, idx) for idx in indices]
    
    def map_to_step(self, from_step, to_step, indices):
        """Map indices from one step to another."""
        if from_step not in self.index_mappings or to_step not in self.index_mappings:
            raise ValueError(f"Steps '{from_step}' or '{to_step}' not found in index mappings")
            
        # First map to original
        from_mapping = self.index_mappings[from_step]['current_to_original']
        to_mapping = self.index_mappings[to_step]['original_to_current']
        
        # Map through original
        if isinstance(indices, pd.Index):
            orig_indices = indices.map(lambda x: from_mapping.get(x, x))
            return orig_indices.map(lambda x: to_mapping.get(x, None))
        
        result = []
        for idx in indices:
            orig_idx = from_mapping.get(idx, idx)
            step_idx = to_mapping.get(orig_idx, None)
            result.append(step_idx)
        
        return result
    
    def create_full_result_df(self, step_name, result_df, fill_value=None):
        """Create a result dataframe with all original indices, filling in gaps."""
        # Create a new dataframe with the original index
        full_result = pd.DataFrame(index=self.original_indices)
        
        # Initialize with columns from result_df
        for col in result_df.columns:
            full_result[col] = pd.NA
        
        # Get mapping for the step
        mapping = self.index_mappings[step_name]['current_to_original']
        
        # Map each result back to its original index
        for curr_idx, row in result_df.iterrows():
            orig_idx = mapping.get(curr_idx)
            if orig_idx is not None and orig_idx in full_result.index:
                for col in result_df.columns:
                    full_result.loc[orig_idx, col] = row[col]
            
        # Add flag indicating if record is in the current step
        orig_to_current = self.index_mappings[step_name]['original_to_current']
        full_result['in_processed_data'] = full_result.index.map(
            lambda x: x in orig_to_current
        )
            
        # Fill missing values
        if fill_value is not None:
            if isinstance(fill_value, dict):
                for col, val in fill_value.items():
                    if col in full_result.columns:
                        full_result[col] = full_result[col].fillna(val)
            else:
                full_result = full_result.fillna(fill_value)
            
        return full_result
    
    def fit_transform(self, df, stop_after=None):
        """Override to add index tracking."""
        # Initialize tracking with original dataframe
        self.initialize_tracking(df)
        
        # Run the pipeline with tracking
        result = df.copy()
        
        for i, step in enumerate(self.steps):
            if stop_after and step.name == stop_after:
                break
                
            print(f"Running step: {step.name}")
            
            # Transform the data
            result = step.fit_transform(result)
            
            # If the step changes the data size, update mapping
            if len(result) != len(df):
                self.update_mapping(step.name, result.index)
            else:
                # Copy previous mapping
                prev_step = list(self.index_mappings.keys())[-1]
                self.index_mappings[step.name] = self.index_mappings[prev_step]
            
            df = result
        
        return result
    
    def predict(self, df, step_name=None):
        """Make predictions with the pipeline, with optional step tracking."""
        if 'model_training' not in self.get_step_names():
            raise ValueError("Pipeline does not have a model training step")
            
        # Get the model
        model_step = self.get_step('model_training')
        model = model_step.get_model()
        
        # Check if there's a prediction step
        if 'prediction' in self.get_step_names():
            pred_step = self.get_step('prediction')
            predictions = pred_step.transform(df)
        else:
            # Make predictions directly
            predictions = model.predict(df)
        
        # Update mapping if step_name is provided
        if step_name:
            self.update_mapping(step_name, df.index)
        
        return predictions
    
    def evaluate_with_tracking(self, X_test, y_test, step_name=None):
        """Evaluate the model and track results."""
        predictions = self.predict(X_test, step_name)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        }, index=X_test.index)
        
        # Map to original indices and return
        return self.create_full_result_df(
            step_name or list(self.index_mappings.keys())[-1],
            results_df
        )


# Define custom pipeline steps for deduplication
class HashDeduplicationStep(PreprocessingStep):
    """Pipeline step to perform hash-based deduplication."""
    
    def __init__(self, text_column, name="hash_deduplication"):
        """Initialize the step."""
        super().__init__(name=name)
        self.text_column = text_column
        
    def fit(self, X, y=None, **kwargs):
        """Fit the deduplication step (no-op)."""
        return self
        
    def transform(self, X, **kwargs):
        """Apply hash-based deduplication."""
        text_series = X[self.text_column]
        kept_indices = hash_deduplication(text_series, return_indices=True)
        
        return X.iloc[kept_indices].reset_index(drop=True)


class SimilarityDeduplicationStep(PreprocessingStep):
    """Pipeline step to perform similarity-based deduplication."""
    
    def __init__(self, text_column, method='cosine', threshold=0.8, name="similarity_deduplication"):
        """Initialize the step."""
        super().__init__(name=name)
        self.text_column = text_column
        self.method = method
        self.threshold = threshold
        
    def fit(self, X, y=None, **kwargs):
        """Fit the deduplication step (no-op)."""
        return self
        
    def transform(self, X, **kwargs):
        """Apply similarity-based deduplication."""
        text_series = X[self.text_column]
        kept_indices = deduplicate_texts(
            text_series,
            method=self.method,
            threshold=self.threshold,
            return_indices=True
        )
        
        return X.iloc[kept_indices].reset_index(drop=True)


class TextPreprocessingStep(PreprocessingStep):
    """Pipeline step to preprocess text data."""
    
    def __init__(self, text_column, output_column=None, name="text_preprocessing"):
        """Initialize the step."""
        super().__init__(name=name)
        self.text_column = text_column
        self.output_column = output_column or f"processed_{text_column}"
        self.text_processor = TextProcessor()
        
    def fit(self, X, y=None, **kwargs):
        """Fit the preprocessing step (no-op)."""
        return self
        
    def transform(self, X, **kwargs):
        """Apply text preprocessing."""
        df = X.copy()
        df[self.output_column] = df[self.text_column].apply(
            lambda x: self.text_processor.preprocess_text(x)
        )
        return df


def generate_sample_data(n_samples=1000, n_duplicates=200):
    """Generate sample text data with duplicates for classification."""
    # Create categories
    categories = ['business', 'technology', 'health', 'entertainment', 'sports']
    
    # Template texts for each category
    templates = {
        'business': [
            "The company announced their quarterly earnings today.",
            "Investors are concerned about the market trends.",
            "The stock market showed signs of recovery after recent losses.",
            "The CEO announced a new strategic direction for the company.",
            "Economic indicators suggest a potential recession next year."
        ],
        'technology': [
            "The new smartphone features cutting-edge technology.",
            "Developers are excited about the new programming language.",
            "The startup secured funding for their AI project.",
            "The latest software update includes security fixes.",
            "Tech giants are facing increased scrutiny from regulators."
        ],
        'health': [
            "Researchers discovered a new treatment for the disease.",
            "The study shows a correlation between diet and health outcomes.",
            "Doctors recommend regular exercise for overall well-being.",
            "The hospital announced a new wing dedicated to research.",
            "The pandemic has changed how healthcare is delivered."
        ],
        'entertainment': [
            "The movie broke box office records on opening weekend.",
            "The actor won an award for their performance in the film.",
            "The new streaming service acquired rights to popular shows.",
            "Concert tickets sold out within minutes of being released.",
            "Critics praised the director's latest work."
        ],
        'sports': [
            "The team won the championship after a close game.",
            "The player announced their retirement after a successful career.",
            "The coach implemented a new strategy for the upcoming season.",
            "The tournament attracted record viewership this year.",
            "Athletes are preparing for the upcoming international competition."
        ]
    }
    
    # Generate base data
    np.random.seed(42)
    
    texts = []
    labels = []
    
    for _ in range(n_samples - n_duplicates):
        category = np.random.choice(categories)
        template = np.random.choice(templates[category])
        
        # Add some variation
        if np.random.random() < 0.3:
            template = template.replace("the", "a")
        if np.random.random() < 0.3:
            template = template.replace("is", "was")
        if np.random.random() < 0.2:
            template += " This is additional context."
            
        texts.append(template)
        labels.append(category)
    
    # Add duplicates
    for _ in range(n_duplicates):
        idx = np.random.randint(0, len(texts))
        texts.append(texts[idx])
        labels.append(labels[idx])
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'category': labels
    })
    
    return df


def main():
    """Run the example."""
    print("Generating sample data with duplicates...")
    df = generate_sample_data(n_samples=1000, n_duplicates=300)
    print(f"Original dataset shape: {df.shape}")
    
    # Store the original data
    original_df = df.copy()
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {train_df.shape}, Test set: {test_df.shape}")
    
    # Create pipeline steps
    pipeline_steps = [
        # Preprocessing
        TextPreprocessingStep(
            text_column='text',
            output_column='processed_text',
            name='text_preprocessing'
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
        ),
        
        # Feature extraction
        TransformationStep(
            transform_fn=lambda df: pd.get_dummies(df['category'], prefix='category'),
            name='feature_extraction'
        ),
        
        # Data splitting
        DataSplittingStep(
            target_column='category',
            test_size=0.2,
            random_state=42,
            name='data_splitting'
        ),
        
        # Model training
        ModelTrainingStep(
            model_type='random_forest',
            model_params={
                'n_estimators': 100,
                'random_state': 42
            },
            name='model_training'
        ),
        
        # Prediction
        PredictionStep(
            name='prediction'
        ),
        
        # Evaluation
        EvaluationStep(
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            name='evaluation'
        )
    ]
    
    # Create tracking pipeline
    pipeline = IndexTrackingPipeline(steps=pipeline_steps, name="deduplication_pipeline")
    
    # Run pipeline on training data
    print("\nRunning pipeline on training data...")
    processed_train = pipeline.fit_transform(train_df)
    
    # Evaluate on test data - first with original indices
    print("\nEvaluating on original test data...")
    original_results = pipeline.evaluate_with_tracking(
        test_df, 
        test_df['category'],
        step_name='original_test'
    )
    
    # Now run test data through deduplication
    print("\nRunning test data through deduplication...")
    dedup_test_pipeline = IndexTrackingPipeline(
        steps=[
            pipeline_steps[0],  # text preprocessing
            pipeline_steps[1],  # hash deduplication
            pipeline_steps[2],  # similarity deduplication
        ],
        name="test_deduplication_pipeline"
    )
    
    deduplicated_test = dedup_test_pipeline.fit_transform(test_df)
    print(f"Original test size: {len(test_df)}, Deduplicated test size: {len(deduplicated_test)}")
    
    # Evaluate on deduplicated test data
    print("\nEvaluating on deduplicated test data...")
    model = pipeline.get_step('model_training').get_model()
    deduplicated_predictions = model.predict(deduplicated_test)
    
    # Create a results dataframe
    deduplicated_results = pd.DataFrame({
        'actual': deduplicated_test['category'],
        'predicted': deduplicated_predictions,
        'in_deduplicated_data': True
    }, index=deduplicated_test.index)
    
    # Map back to original test indices
    mapped_results = dedup_test_pipeline.create_full_result_df(
        'similarity_deduplication',
        deduplicated_results,
        fill_value={'predicted': 'unknown', 'in_deduplicated_data': False}
    )
    
    # Print comparison statistics
    print("\nComparison of original vs. deduplicated test results:")
    
    # For original test data
    original_accuracy = accuracy_score(
        original_results[original_results['predicted'] != 'unknown']['actual'],
        original_results[original_results['predicted'] != 'unknown']['predicted']
    )
    
    # For deduplicated test data
    dedup_accuracy = accuracy_score(
        mapped_results[mapped_results['predicted'] != 'unknown']['actual'],
        mapped_results[mapped_results['predicted'] != 'unknown']['predicted']
    )
    
    # Calculate reduction
    test_size = len(test_df)
    dedup_test_size = len(deduplicated_test)
    reduction = test_size - dedup_test_size
    reduction_percent = round(100 * reduction / test_size, 2)
    
    # Calculate duplicate impact
    duplicate_total = mapped_results[~mapped_results['in_deduplicated_data']].shape[0]
    duplicate_percent = round(100 * duplicate_total / len(mapped_results), 2)
    
    print(f"Original test size: {test_size}")
    print(f"Deduplicated test size: {dedup_test_size}")
    print(f"Reduction: {reduction} records ({reduction_percent}%)")
    print(f"Duplicate records: {duplicate_total} ({duplicate_percent}%)")
    print(f"Original accuracy: {original_accuracy:.4f}")
    print(f"Deduplicated accuracy: {dedup_accuracy:.4f}")
    
    # Show classification report
    print("\nClassification report on deduplicated test data:")
    print(classification_report(
        deduplicated_test['category'],
        deduplicated_predictions
    ))
    
    # Examples of duplicate predictions
    print("\nExamples of predictions for duplicates:")
    duplicates = mapped_results[~mapped_results['in_deduplicated_data']].head(5)
    for idx, row in duplicates.iterrows():
        # Find the original text
        text = test_df.loc[idx, 'text']
        orig_idx = idx
        curr_idx = dedup_test_pipeline.index_mappings['similarity_deduplication']['original_to_current'].get(orig_idx)
        
        # If this record was removed during deduplication, find what it was similar to
        if curr_idx is None:
            similar_texts = find_similar_texts_in_df(test_df, text, 'text', threshold=0.8)
            similar_indices = [i for i in similar_texts if i != idx]
            
            if similar_indices:
                similar_idx = similar_indices[0]
                similar_text = test_df.loc[similar_idx, 'text']
                similar_curr_idx = dedup_test_pipeline.index_mappings['similarity_deduplication']['original_to_current'].get(similar_idx)
                
                print(f"Original: [{orig_idx}] '{text}'")
                print(f"Similar:  [{similar_idx}] '{similar_text}'")
                print(f"Category: {row['actual']}, Prediction: {mapped_results.loc[similar_idx, 'predicted'] if similar_curr_idx is not None else 'unknown'}")
                print("---")
    
    print("\nExample complete!")
    return {
        "original_df": original_df,
        "train_df": train_df,
        "test_df": test_df,
        "deduplicated_test": deduplicated_test,
        "original_results": original_results,
        "mapped_results": mapped_results,
        "pipeline": pipeline,
        "dedup_test_pipeline": dedup_test_pipeline
    }


def find_similar_texts_in_df(df, text, column, threshold=0.8):
    """Helper function to find similar texts in a dataframe."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create a temporary dataframe with the new text
    temp_df = df.copy()
    
    # Add a temporary row with proper handling of dataframe structure
    temp_row = pd.Series([''] * len(df.columns), index=df.columns)
    temp_row[column] = text
    if 'category' in df.columns:
        temp_row['category'] = ''  # Handle category column which is required
    
    temp_df.loc[-1] = temp_row
    
    # Vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(temp_df[column])
    
    # Calculate similarities
    query_vector = vectors[-1]
    similarities = cosine_similarity(query_vector, vectors[:-1])[0]
    
    # Get indices of similar texts
    similar_indices = [idx for idx, sim in zip(df.index, similarities) if sim >= threshold]
    
    return similar_indices


if __name__ == "__main__":
    results = main()