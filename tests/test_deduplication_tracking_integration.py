"""
Integration tests for the deduplication tracking system.
"""
import pandas as pd
import numpy as np
import pytest
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.deduplication.exact_deduplication import hash_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts


class TestDeduplicationTrackingIntegration:
    """Integration test class for deduplication tracking functionality."""
    
    @pytest.fixture
    def tracking_classes(self):
        """Import all tracking classes from the examples."""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        
        from deduplication_tracking_example import IndexTracker
        from advanced_deduplication_tracking import DeduplicationTracker
        from pipeline_with_deduplication_tracking import (
            IndexTrackingPipeline,
            HashDeduplicationStep,
            SimilarityDeduplicationStep,
            TextPreprocessingStep
        )
        
        return {
            'IndexTracker': IndexTracker,
            'DeduplicationTracker': DeduplicationTracker,
            'IndexTrackingPipeline': IndexTrackingPipeline,
            'HashDeduplicationStep': HashDeduplicationStep,
            'SimilarityDeduplicationStep': SimilarityDeduplicationStep,
            'TextPreprocessingStep': TextPreprocessingStep
        }
    
    @pytest.fixture
    def text_classification_df(self):
        """Create a dataset for text classification with duplicates."""
        np.random.seed(42)
        
        # Categories and template texts
        categories = ['business', 'technology', 'health']
        templates = {
            'business': [
                "The company announced quarterly earnings today which exceeded expectations.",
                "Investors are concerned about market trends due to economic uncertainty.",
                "The stock market showed signs of recovery after recent volatility."
            ],
            'technology': [
                "The new smartphone features cutting-edge technology and improved battery life.",
                "Developers are excited about the new programming language which promises efficiency.",
                "The startup secured funding for their artificial intelligence project."
            ],
            'health': [
                "Researchers discovered a promising new treatment for the disease after clinical trials.",
                "The study shows a correlation between diet and health outcomes over time.",
                "Doctors recommend regular exercise for overall well-being and longevity."
            ]
        }
        
        # Generate data
        texts = []
        labels = []
        ids = []
        
        id_counter = 1
        
        # Original texts
        for _ in range(150):
            category = np.random.choice(categories)
            template = np.random.choice(templates[category])
            
            # Sometimes add variations
            if np.random.random() < 0.3:
                template = template.replace("the", "a")
            if np.random.random() < 0.2:
                template += " This trend is expected to continue."
                
            texts.append(template)
            labels.append(category)
            ids.append(f"ID_{id_counter}")
            id_counter += 1
        
        # Add exact duplicates
        for _ in range(30):
            idx = np.random.randint(0, len(texts))
            texts.append(texts[idx])
            labels.append(labels[idx])
            ids.append(f"ID_{id_counter}")
            id_counter += 1
        
        # Add near-duplicates with small variations
        for _ in range(20):
            idx = np.random.randint(0, len(texts) - 30)  # Only use original texts
            text = texts[idx]
            
            # Create variation
            if len(text) > 20:
                # Change a word or add punctuation
                words = text.split()
                if len(words) > 5:
                    pos = np.random.randint(1, len(words) - 1)
                    if words[pos].lower() in ['the', 'a', 'and', 'or', 'to', 'for']:
                        words[pos] = 'some'
                    elif words[pos].lower() in ['market', 'health', 'company', 'technology']:
                        words[pos] = words[pos] + ','
                    modified_text = ' '.join(words)
                else:
                    modified_text = text.replace('.', '!')
            else:
                modified_text = text + " Modified."
                
            texts.append(modified_text)
            labels.append(labels[idx])  # Same label as original
            ids.append(f"ID_{id_counter}")
            id_counter += 1
        
        # Create DataFrame
        df = pd.DataFrame({
            'id': ids,
            'text': texts,
            'category': labels
        })
        
        return df
    
    def test_basic_ml_workflow_with_tracking(self, tracking_classes, text_classification_df):
        """Test a basic ML workflow with deduplication and result tracking."""
        IndexTracker = tracking_classes['IndexTracker']
        
        # Initialize tracker
        tracker = IndexTracker().initialize_from_df(text_classification_df)
        
        # Detect duplicates - use just text column for duplicate detection
        dup_stats = detect_duplicates(text_classification_df.drop('category', axis=1), subset=['text'])
        assert dup_stats['has_duplicates'] == True
        assert dup_stats['duplicate_count'] > 0
        
        # Remove duplicates
        deduped_df = remove_duplicates(text_classification_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # Verify tracker mappings are updated
        assert len(tracker.original_to_current) == len(kept_indices)
        assert len(tracker.current_to_original) == len(kept_indices)
        
        # Prepare ML task - simple bag of words for demonstration
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=100)
        X = vectorizer.fit_transform(deduped_df['text'])
        y = deduped_df['category']
        
        # Split the dataframe instead of the sparse matrix
        df_train, df_test = train_test_split(
            deduped_df, test_size=0.3, random_state=42
        )
        
        # Extract features for train and test sets
        X_train = vectorizer.transform(df_train['text'])
        y_train = df_train['category']
        X_test = vectorizer.transform(df_test['text'])
        y_test = df_test['category']
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred
        }, index=df_test.index)
        
        # Map results back to original dataset
        full_results = tracker.create_full_result_df(
            results_df, 
            text_classification_df,
            fill_value={'actual': 'unknown', 'predicted': 'unknown'}
        )
        
        # Verify mapping worked correctly
        assert len(full_results) == len(text_classification_df)
        
        # Check that predictions are correctly mapped back
        has_prediction = (full_results['predicted'] != 'unknown').sum()
        original_test_size = len(df_test)
        assert has_prediction == original_test_size
        
        # Verify specific mappings
        for curr_idx in df_test.index:
            orig_idx = tracker.current_to_original[curr_idx]
            assert full_results.loc[orig_idx, 'predicted'] == results_df.loc[curr_idx, 'predicted']
    
    def test_advanced_tracking_with_visualization(self, tracking_classes, text_classification_df, tmpdir):
        """Test advanced tracking with multi-step deduplication and visualization."""
        DeduplicationTracker = tracking_classes['DeduplicationTracker']
        
        # Initialize tracker
        tracker = DeduplicationTracker().initialize(text_classification_df)
        
        # Store original dataframe
        original_df = text_classification_df.copy()
        
        # Step 1: Preprocess texts
        from freamon.utils.text_utils import TextProcessor
        text_processor = TextProcessor()
        df = text_classification_df.copy()
        df['processed_text'] = df['text'].apply(text_processor.preprocess_text)
        
        # Step 2: Hash-based deduplication
        text_series = df['processed_text']
        kept_indices = hash_deduplication(text_series, return_indices=True)
        
        # Update tracking
        tracker.update_mapping(kept_indices, "Hash-based")
        
        # Filter dataframe
        df = df.iloc[kept_indices].reset_index(drop=True)
        
        # Step 3: Similarity-based deduplication
        text_series = df['processed_text']
        similarity_dict = deduplicate_texts(
            text_series,
            method='cosine',
            threshold=0.9,
            return_indices=True,
            return_similarity_dict=True
        )
        
        # Store similarity info if available
        if isinstance(similarity_dict, tuple):
            kept_indices, sim_dict = similarity_dict
            tracker.store_similarity_info(sim_dict)
        else:
            kept_indices = similarity_dict
        
        # Update tracking
        tracker.update_mapping(kept_indices, "Similarity-based")
        
        # Filter dataframe
        df = df.iloc[kept_indices].reset_index(drop=True)
        
        # Verify tracking steps
        assert len(tracker.dedup_steps) == 2
        assert tracker.dedup_steps[0]['method'] == "Hash-based"
        assert tracker.dedup_steps[1]['method'] == "Similarity-based"
        
        # Create a test ML model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        # Extract features
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(df['processed_text'])
        y = df['category']
        
        # Create a train-test split for the dataframe
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        
        # Create corresponding training and test feature matrices
        X_train = vectorizer.transform(df_train['processed_text'])
        y_train = df_train['category']
        X_test = vectorizer.transform(df_test['processed_text'])
        y_test = df_test['category']
        
        # Train the model
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        
        # Get predictions
        y_pred = clf.predict(X_test)
        
        # Create results dataframe with the proper index
        results_df = pd.DataFrame({
            'actual': y_test.tolist(),
            'predicted': y_pred.tolist()
        }, index=df_test.index)
        
        # Map back to original dataset
        full_results = tracker.create_full_result_df(
            results_df,
            original_df,
            fill_value={'actual': 'unknown', 'predicted': 'unknown'},
            include_duplicate_flag=True
        )
        
        # Verify mapping
        assert len(full_results) == len(original_df)
        assert 'is_duplicate' in full_results.columns
        
        # Check duplicate flagging
        assert full_results['is_duplicate'].sum() > 0
        
        # Test plotting
        fig = tracker.plot_deduplication_steps()
        
        # Plot to file for inspection
        plot_path = os.path.join(tmpdir, "dedup_steps.png")
        fig.savefig(plot_path)
        assert os.path.exists(plot_path)
        
        # Generate HTML report
        report_path = os.path.join(tmpdir, "tracking_report.html")
        tracker.generate_tracking_report(original_df, df, report_path)
        assert os.path.exists(report_path)
    
    def test_pipeline_integration(self, tracking_classes, text_classification_df):
        """Test the integration of tracking with a pipeline."""
        # Get classes
        IndexTrackingPipeline = tracking_classes['IndexTrackingPipeline']
        HashDeduplicationStep = tracking_classes['HashDeduplicationStep']
        TextPreprocessingStep = tracking_classes['TextPreprocessingStep']
        
        # Create pipeline steps
        preprocessing_step = TextPreprocessingStep(
            text_column='text',
            name='preprocessing'
        )
        
        dedup_step = HashDeduplicationStep(
            text_column='processed_text',
            name='deduplication'
        )
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        class FeatureExtractionStep:
            def __init__(self, text_column, name='feature_extraction'):
                self.text_column = text_column
                self.name = name
                self.vectorizer = TfidfVectorizer(max_features=100)
                
            def fit(self, df):
                self.vectorizer.fit(df[self.text_column])
                return self
                
            def transform(self, df):
                X = self.vectorizer.transform(df[self.text_column])
                # Convert to dataframe and preserve index
                feature_df = pd.DataFrame(
                    X.toarray(),
                    columns=[f'feature_{i}' for i in range(X.shape[1])],
                    index=df.index
                )
                return pd.concat([df, feature_df], axis=1)
                
            def fit_transform(self, df):
                self.fit(df)
                return self.transform(df)
        
        feature_step = FeatureExtractionStep(
            text_column='processed_text',
            name='feature_extraction'
        )
        
        # Create pipeline
        pipeline = IndexTrackingPipeline(
            steps=[preprocessing_step, dedup_step, feature_step],
            name='test_pipeline'
        )
        
        # Run pipeline
        result = pipeline.fit_transform(text_classification_df)
        
        # Check pipeline ran successfully
        assert result is not None
        assert len(result) < len(text_classification_df)  # Should have removed duplicates
        assert result.shape[1] > text_classification_df.shape[1]  # Should have added features
        
        # Verify tracking was updated
        assert 'preprocessing' in pipeline.index_mappings
        assert 'deduplication' in pipeline.index_mappings
        assert 'feature_extraction' in pipeline.index_mappings
        
        # Train a model on the transformed data
        from sklearn.ensemble import RandomForestClassifier
        
        # Prepare ML data
        feature_cols = [col for col in result.columns if col.startswith('feature_')]
        X = result[feature_cols]
        y = result['category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        }, index=X_test.index)
        
        # Map results back to original dataset
        full_results = pipeline.create_full_result_df(
            'feature_extraction',
            results_df,
            fill_value={'predicted': 'unknown'}
        )
        
        # Verify mapping
        assert len(full_results) == len(text_classification_df)
        assert 'in_processed_data' in full_results.columns
        
        # Check that predictions are correctly mapped back
        has_prediction = (full_results['predicted'] != 'unknown').sum()
        assert has_prediction == len(X_test)
        
        # Verify that duplicates are flagged
        not_in_processed = (~full_results['in_processed_data']).sum()
        assert not_in_processed > 0  # Should have some duplicates
    
    def test_end_to_end_workflow(self, tracking_classes, text_classification_df):
        """Test a complete end-to-end workflow with all trackers."""
        # Get classes
        IndexTracker = tracking_classes['IndexTracker']
        DeduplicationTracker = tracking_classes['DeduplicationTracker']
        IndexTrackingPipeline = tracking_classes['IndexTrackingPipeline']
        TextPreprocessingStep = tracking_classes['TextPreprocessingStep']
        
        # 1. First use simple IndexTracker for basic deduplication
        basic_tracker = IndexTracker().initialize_from_df(text_classification_df)
        
        # Remove exact duplicates
        deduped_df = remove_duplicates(text_classification_df, keep='first')
        kept_indices = deduped_df.index.tolist()
        basic_tracker.update_from_kept_indices(kept_indices, deduped_df)
        
        # 2. Use DeduplicationTracker for more detailed tracking
        advanced_tracker = DeduplicationTracker().initialize(deduped_df)
        
        # Preprocess texts
        from freamon.utils.text_utils import TextProcessor
        text_processor = TextProcessor()
        df = deduped_df.copy()
        df['processed_text'] = df['text'].apply(text_processor.preprocess_text)
        
        # Perform similarity deduplication
        text_series = df['processed_text']
        kept_indices = deduplicate_texts(
            text_series,
            method='cosine',
            threshold=0.9,
            return_indices=True
        )
        
        # Update tracking
        advanced_tracker.update_mapping(kept_indices, "Similarity-based")
        
        # Filter dataframe
        df = df.iloc[kept_indices].reset_index(drop=True)
        
        # 3. Split into train/test
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
        
        # 4. Use pipeline for feature extraction and model training
        # Create preprocessing pipeline
        preprocessing_pipeline = IndexTrackingPipeline(
            steps=[
                TextPreprocessingStep(
                    text_column='text',
                    output_column='processed_text',
                    name='preprocessing'
                )
            ],
            name='preprocessing_pipeline'
        )
        
        # Initialize pipeline tracking with training data
        preprocessing_pipeline.initialize_tracking(train_df)
        
        # Process training data
        processed_train = preprocessing_pipeline.fit_transform(train_df)
        
        # 5. Train a model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        # Extract features
        vectorizer = TfidfVectorizer(max_features=100)
        X_train = vectorizer.fit_transform(processed_train['processed_text'])
        y_train = processed_train['category']
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # 6. Process test data
        processed_test = preprocessing_pipeline.transform(test_df)
        
        # Extract features
        X_test = vectorizer.transform(processed_test['processed_text'])
        y_test = processed_test['category']
        
        # Make predictions
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        }, index=processed_test.index)
        
        # 7. Map results back using both trackers
        # First map through advanced tracker
        intermediate_results = advanced_tracker.create_full_result_df(
            results_df, 
            deduped_df,
            fill_value={'predicted': 'unknown'},
            include_duplicate_flag=True
        )
        
        # Then map through basic tracker
        final_results = basic_tracker.create_full_result_df(
            intermediate_results,
            text_classification_df,
            fill_value={'predicted': 'unknown'}
        )
        
        # 8. Verify final results
        assert len(final_results) == len(text_classification_df)
        
        # Count how many records have predictions
        has_prediction = (final_results['predicted'] != 'unknown').sum()
        assert has_prediction > 0
        
        # Check predictions preserved through mapping chain
        orig_preds = set(results_df['predicted'])
        final_preds = set(final_results.loc[final_results['predicted'] != 'unknown', 'predicted'])
        assert orig_preds == final_preds