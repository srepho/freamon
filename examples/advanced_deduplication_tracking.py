#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced example of tracking data through text deduplication and mapping ML results back.

This example demonstrates:
1. Using advanced text deduplication techniques (hash, n-gram, similarity)
2. Tracking indices through a multi-step pipeline including deduplication
3. Performing text analysis and classification on deduplicated data
4. Mapping predictions and clusters back to the original dataset
5. Visualizing the relationship between original and deduplicated data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans

# Import freamon modules
from freamon.deduplication.exact_deduplication import hash_deduplication, ngram_fingerprint_deduplication
from freamon.deduplication.fuzzy_deduplication import deduplicate_texts, find_similar_texts
from freamon.utils.text_utils import TextProcessor
from freamon.eda.report import generate_html_report


class DeduplicationTracker:
    """Advanced tracker for deduplication operations with visualization."""
    
    def __init__(self):
        """Initialize the tracker."""
        self.original_size = 0
        self.current_size = 0
        self.dedup_steps = []
        self.original_to_current = {}
        self.current_to_original = {}
        self.clusters = {}
        self.similarity_map = {}
        
    def initialize(self, df):
        """Initialize tracking with a dataframe."""
        self.original_size = len(df)
        self.current_size = len(df)
        self.original_to_current = {i: i for i in df.index}
        self.current_to_original = {i: i for i in df.index}
        return self
    
    def update_mapping(self, kept_indices, method_name, df=None):
        """Update mapping after a deduplication step."""
        # Store the deduplication step
        reduction = self.current_size - len(kept_indices)
        self.dedup_steps.append({
            'method': method_name,
            'original_size': self.original_size,
            'previous_size': self.current_size,
            'new_size': len(kept_indices),
            'reduction': reduction,
            'reduction_percent': round(100 * reduction / self.current_size, 2) if self.current_size > 0 else 0
        })
        
        # Update size
        self.current_size = len(kept_indices)
        
        # Update mappings
        new_current_to_original = {}
        new_original_to_current = {}
        
        # Assign new indices if a dataframe is provided, otherwise use kept_indices directly
        if df is not None and len(df) == len(kept_indices):
            new_indices = df.index
            for new_idx, kept_idx in zip(new_indices, kept_indices):
                # Map through existing mappings to maintain chain
                orig_idx = self.current_to_original.get(kept_idx, kept_idx)
                new_current_to_original[new_idx] = orig_idx
                new_original_to_current[orig_idx] = new_idx
        else:
            for i, kept_idx in enumerate(kept_indices):
                # Map through existing mappings to maintain chain
                orig_idx = self.current_to_original.get(kept_idx, kept_idx)
                new_current_to_original[i] = orig_idx
                new_original_to_current[orig_idx] = i
        
        self.current_to_original = new_current_to_original
        self.original_to_current = new_original_to_current
        
        return self
    
    def store_similarity_info(self, similarity_dict):
        """Store similarity information between texts."""
        self.similarity_map = similarity_dict
        return self
    
    def store_clusters(self, cluster_dict):
        """Store clustering information."""
        self.clusters = cluster_dict
        return self
    
    def map_to_original(self, indices):
        """Map current indices to original indices."""
        if isinstance(indices, pd.Index):
            return indices.map(lambda x: self.current_to_original.get(x, x))
        return [self.current_to_original.get(idx, idx) for idx in indices]
    
    def map_to_current(self, indices):
        """Map original indices to current indices."""
        if isinstance(indices, pd.Index):
            return indices.map(lambda x: self.original_to_current.get(x, None))
        return [self.original_to_current.get(idx, None) for idx in indices]
    
    def map_series_to_original(self, series):
        """Map a series with current indices to original indices."""
        mapped_index = self.map_to_original(series.index)
        return pd.Series(series.values, index=mapped_index)
    
    def create_full_result_df(self, result_df, original_df, fill_value=None, include_duplicate_flag=True):
        """Create a result dataframe with all original indices, filling in gaps.
        
        Args:
            result_df: Dataframe with results (using current indices)
            original_df: Original dataframe before deduplication
            fill_value: Value to use for missing results (default: None)
            include_duplicate_flag: Add column indicating if record is a duplicate
            
        Returns:
            DataFrame with results mapped back to all original indices
        """
        # Create a new dataframe with the original index
        full_result = pd.DataFrame(index=original_df.index)
        
        # Initialize with columns from result_df
        for col in result_df.columns:
            full_result[col] = pd.NA
        
        # Map each result back to its original index
        for curr_idx, row in result_df.iterrows():
            orig_idx = self.current_to_original.get(curr_idx)
            if orig_idx is not None:
                for col in result_df.columns:
                    full_result.loc[orig_idx, col] = row[col]
        
        # Add flag indicating if record is a duplicate
        if include_duplicate_flag:
            full_result['is_duplicate'] = ~full_result.index.isin(self.current_to_original.values())
            
        # Fill missing values
        if fill_value is not None:
            if isinstance(fill_value, dict):
                for col, val in fill_value.items():
                    if col in full_result.columns:
                        full_result[col] = full_result[col].fillna(val)
                        # Handle downcasting warning
                        full_result[col] = full_result[col].infer_objects(copy=False)
            else:
                full_result = full_result.fillna(fill_value)
                # Handle downcasting warning
                full_result = full_result.infer_objects(copy=False)
            
        return full_result
    
    def plot_deduplication_steps(self, figsize=(12, 6)):
        """Plot the deduplication steps and their impact."""
        if not self.dedup_steps:
            print("No deduplication steps recorded")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot dataset size reduction
        steps = [0] + list(range(1, len(self.dedup_steps) + 1))
        sizes = [self.original_size] + [step['new_size'] for step in self.dedup_steps]
        
        ax1.plot(steps, sizes, 'o-', markersize=8)
        ax1.set_title('Dataset Size Reduction')
        ax1.set_xlabel('Deduplication Step')
        ax1.set_ylabel('Dataset Size')
        ax1.grid(True, alpha=0.3)
        
        # Annotate points with reduction percentage
        for i, step in enumerate(self.dedup_steps, 1):
            ax1.annotate(f"{step['reduction_percent']}%",
                        (i, step['new_size']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        # Plot method comparison
        methods = [step['method'] for step in self.dedup_steps]
        reductions = [step['reduction'] for step in self.dedup_steps]
        
        ax2.bar(methods, reductions)
        ax2.set_title('Records Removed by Method')
        ax2.set_xlabel('Deduplication Method')
        ax2.set_ylabel('Records Removed')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_distribution(self, figsize=(10, 6)):
        """Plot the distribution of cluster sizes."""
        if not self.clusters:
            print("No clustering information available")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Count items in each cluster
        cluster_sizes = {cluster_id: len(indices) for cluster_id, indices in self.clusters.items()}
        
        # Convert to dataframe for easier plotting
        cluster_df = pd.DataFrame.from_dict(cluster_sizes, orient='index', columns=['size'])
        cluster_df = cluster_df.sort_values('size', ascending=False)
        
        # Plot
        ax.bar(cluster_df.index.astype(str), cluster_df['size'])
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Records')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.xticks(rotation=90)
        plt.tight_layout()
        return fig
    
    def generate_tracking_report(self, original_df, deduplicated_df, output_path):
        """Generate an HTML report showing deduplication tracking info."""
        # Create report dataframes
        step_df = pd.DataFrame(self.dedup_steps)
        
        # Create mapping summary
        mapping_data = []
        for orig_idx, curr_idx in self.original_to_current.items():
            mapping_data.append({
                'original_index': orig_idx,
                'current_index': curr_idx,
                'is_in_final_dataset': curr_idx is not None
            })
        mapping_df = pd.DataFrame(mapping_data)
        
        # Create similarity pairs dataframe if available
        similarity_data = []
        for idx, similar_indices in self.similarity_map.items():
            for sim_idx in similar_indices:
                similarity_data.append({
                    'text_index': idx,
                    'similar_text_index': sim_idx,
                    'retained': idx in self.current_to_original.keys(),
                    'similar_retained': sim_idx in self.current_to_original.keys()
                })
        similarity_df = pd.DataFrame(similarity_data) if similarity_data else None
        
        # Create summary stats
        stats_df = pd.DataFrame([{
            'original_size': self.original_size,
            'final_size': self.current_size,
            'total_reduction': self.original_size - self.current_size,
            'reduction_percent': round(100 * (self.original_size - self.current_size) / self.original_size, 2),
            'total_deduplication_steps': len(self.dedup_steps)
        }])
        
        # Generate plots
        self.plot_deduplication_steps()
        plt.savefig('deduplication_steps.png')
        
        if self.clusters:
            self.plot_cluster_distribution()
            plt.savefig('cluster_distribution.png')
        
        # Create a simplified HTML report for testing purposes
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deduplication Tracking Report</title>
        </head>
        <body>
            <h1>Deduplication Tracking Report</h1>
            
            <h2>Deduplication Summary</h2>
            <p>Original size: {self.original_size}</p>
            <p>Final size: {self.current_size}</p>
            <p>Reduction: {self.original_size - self.current_size} records ({stats_df['reduction_percent'].iloc[0]}%)</p>
            
            <h2>Deduplication Steps</h2>
            <table border="1">
                <tr>
                    <th>Method</th>
                    <th>Previous Size</th>
                    <th>New Size</th>
                    <th>Reduction</th>
                    <th>Reduction %</th>
                </tr>
        """
        
        for step in self.dedup_steps:
            html_content += f"""
                <tr>
                    <td>{step['method']}</td>
                    <td>{step['previous_size']}</td>
                    <td>{step['new_size']}</td>
                    <td>{step['reduction']}</td>
                    <td>{step['reduction_percent']}%</td>
                </tr>
            """
            
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <img src="deduplication_steps.png" alt="Deduplication Steps">
        """
        
        if self.clusters:
            html_content += """
            <img src="cluster_distribution.png" alt="Cluster Distribution">
            """
            
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML to the output file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path


def generate_sample_text_data(n_samples=1000, n_duplicates=200, n_near_duplicates=100):
    """Generate sample text data with exact and near duplicates."""
    # Base texts
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Python is a programming language that lets you work quickly and integrate systems more effectively.",
        "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data.",
        "Natural language processing is a field of artificial intelligence that helps computers understand human language."
    ]
    
    # Generate variations
    variations = []
    for text in base_texts:
        # Exact copies
        variations.extend([text] * 10)
        
        # Minor variations
        variations.append(text.replace("the", "a"))
        variations.append(text.replace("is", "was"))
        variations.append(text + " This is additional text.")
        variations.append(text.lower())
        variations.append(text.upper())
    
    # Generate random texts
    np.random.seed(42)
    words = "the quick brown fox jumps over lazy dog machine learning field study gives computers ability learn without explicitly programmed python programming language lets work quickly integrate systems effectively data science interdisciplinary uses scientific methods extract knowledge natural processing artificial intelligence helps understand human".split()
    
    random_texts = []
    for _ in range(n_samples - len(variations) - n_duplicates - n_near_duplicates):
        text_length = np.random.randint(5, 15)
        random_text = " ".join(np.random.choice(words, text_length))
        random_texts.append(random_text)
    
    # Combine all texts
    all_texts = variations + random_texts
    
    # Add duplicates of random texts
    duplicate_indices = np.random.choice(range(len(all_texts)), n_duplicates, replace=True)
    duplicates = [all_texts[i] for i in duplicate_indices]
    
    # Add near-duplicates with small edits
    near_duplicate_indices = np.random.choice(range(len(all_texts)), n_near_duplicates, replace=True)
    near_duplicates = []
    for i in near_duplicate_indices:
        text = all_texts[i]
        # Make small modifications
        if len(text) > 10:
            modified_text = text[:len(text)//2] + " " + text[len(text)//2:]
            near_duplicates.append(modified_text)
        else:
            near_duplicates.append(text + " extra")
    
    # Final combined list
    final_texts = all_texts + duplicates + near_duplicates
    
    # Generate random sentiment labels
    sentiments = np.random.choice(['positive', 'negative', 'neutral'], len(final_texts))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': final_texts,
        'sentiment': sentiments
    })
    
    return df


def main():
    """Run the example."""
    print("Generating sample text data with duplicates...")
    df = generate_sample_text_data(n_samples=800, n_duplicates=150, n_near_duplicates=50)
    print(f"Original dataset shape: {df.shape}")
    
    # Initialize tracker
    tracker = DeduplicationTracker().initialize(df)
    
    # Store original dataframe
    original_df = df.copy()
    
    # Step 1: Preprocess texts
    print("\nPreprocessing texts...")
    text_processor = TextProcessor()
    df['processed_text'] = df['text'].apply(lambda x: text_processor.preprocess_text(x))
    
    # Step 2: Perform hash-based deduplication
    print("\nPerforming hash-based deduplication...")
    text_series = df['processed_text']
    kept_indices = hash_deduplication(text_series, return_indices=True)
    
    # Update tracking
    tracker.update_mapping(kept_indices, "Hash-based")
    
    # Filter dataframe
    df = df.iloc[kept_indices].reset_index(drop=True)
    print(f"After hash deduplication: {df.shape}")
    
    # Step 3: Perform n-gram fingerprint deduplication
    print("\nPerforming n-gram fingerprint deduplication...")
    text_series = df['processed_text']
    kept_indices = ngram_fingerprint_deduplication(text_series, n=3, return_indices=True)
    
    # Update tracking
    tracker.update_mapping(kept_indices, "N-gram Fingerprint")
    
    # Filter dataframe
    df = df.iloc[kept_indices].reset_index(drop=True)
    print(f"After n-gram deduplication: {df.shape}")
    
    # Step 4: Find similar texts
    print("\nFinding similar texts...")
    text_series = df['processed_text']
    similarity_dict = find_similar_texts(
        text_series, 
        method='cosine',
        threshold=0.8,
        return_similarity_dict=True
    )
    
    # Store similarity info
    tracker.store_similarity_info(similarity_dict)
    
    # Step 5: Perform similarity-based deduplication
    print("\nPerforming similarity-based deduplication...")
    kept_indices = deduplicate_texts(
        text_series,
        method='cosine',
        threshold=0.8,
        return_indices=True
    )
    
    # Update tracking
    tracker.update_mapping(kept_indices, "Similarity-based")
    
    # Filter dataframe
    df = df.iloc[kept_indices].reset_index(drop=True)
    print(f"After similarity deduplication: {df.shape}")
    
    # Step 6: Perform machine learning on deduplicated data
    print("\nRunning machine learning on deduplicated data...")
    
    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Get test indices for later mapping
    test_indices = df.iloc[X_test.nonzero()[0]].index
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predictions
    y_pred = clf.predict(X_test)
    
    # Create results dataframe with current indices
    results_df = pd.DataFrame({
        'predicted_sentiment': y_pred
    }, index=test_indices)
    
    print("Classification report on deduplicated test data:")
    print(classification_report(y_test, y_pred))
    
    # Step 7: Perform clustering on deduplicated data
    print("\nPerforming clustering on deduplicated data...")
    
    # Use k-means for clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Add to results dataframe
    df['cluster'] = clusters
    
    # Create cluster dictionary for tracker
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(df.index[idx])
    
    # Store cluster info in tracker
    tracker.store_clusters(cluster_dict)
    
    # Step 8: Map results back to original dataset
    print("\nMapping results back to original dataset...")
    
    # Map predictions back
    full_results = tracker.create_full_result_df(
        df[['sentiment', 'cluster']].join(results_df, how='left'),
        original_df,
        fill_value={'predicted_sentiment': 'unknown', 'cluster': -1}
    )
    
    # Create a summary of how many records were mapped back
    predicted_count = full_results['predicted_sentiment'].notna().sum()
    predicted_percent = round(100 * predicted_count / len(full_results), 2)
    
    duplicate_count = full_results['is_duplicate'].sum()
    duplicate_percent = round(100 * duplicate_count / len(full_results), 2)
    
    print(f"Original records with predictions: {predicted_count} ({predicted_percent}%)")
    print(f"Duplicate records: {duplicate_count} ({duplicate_percent}%)")
    
    # Step 9: Generate tracking report
    print("\nGenerating tracking report...")
    report_path = tracker.generate_tracking_report(
        original_df, 
        df, 
        "deduplication_tracking_report.html"
    )
    
    print(f"\nExample complete! Report generated at: {report_path}")
    
    # Return tracker and dataframes for further exploration
    return {
        "tracker": tracker,
        "original_df": original_df,
        "deduplicated_df": df,
        "full_results": full_results
    }


if __name__ == "__main__":
    main()