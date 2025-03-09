"""
Advanced ShapIQ usage example demonstrating more sophisticated interaction detection 
and feature engineering techniques with higher-order interactions.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import Freamon components
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
from freamon.explainability import ShapIQExplainer
from freamon.modeling import ModelTrainer
from freamon.eda.explainability_report import generate_interaction_report

# Set random seed for reproducibility
np.random.seed(42)

class AdvancedShapIQAnalysis:
    """
    A class demonstrating advanced ShapIQ interaction detection and
    feature engineering techniques for improved model performance.
    """
    
    def __init__(self, dataset_type='regression'):
        """Initialize with either regression or classification dataset."""
        self.dataset_type = dataset_type
        
        # Load appropriate dataset
        if dataset_type == 'regression':
            california = fetch_california_housing()
            self.X = pd.DataFrame(california.data, columns=california.feature_names)
            self.y = pd.Series(california.target, name="target")
            self.model_type = RandomForestRegressor
            self.model_params = {'n_estimators': 100, 'random_state': 42}
            self.metric = 'r2'
        else:
            cancer = load_breast_cancer()
            self.X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
            self.y = pd.Series(cancer.target, name="target")
            self.model_type = RandomForestClassifier
            self.model_params = {'n_estimators': 100, 'random_state': 42}
            self.metric = 'accuracy'
            
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Store models and results
        self.base_model = None
        self.enhanced_model = None
        self.base_score = None
        self.enhanced_score = None
        self.interaction_strengths = {}
        
    def train_base_model(self):
        """Train the baseline model without feature engineering."""
        print(f"\nTraining baseline {self.dataset_type} model...")
        
        # Create and train base model
        self.base_model = self.model_type(**self.model_params)
        self.base_model.fit(self.X_train, self.y_train)
        
        # Evaluate base model
        self.base_score = self.base_model.score(self.X_test, self.y_test)
        print(f"Base model {self.metric}: {self.base_score:.4f}")
        return self.base_model
    
    def detect_all_interactions(self, 
                              threshold: float = 0.01, 
                              max_interactions: int = 20):
        """
        Comprehensive interaction detection with detailed analysis.
        
        This method goes beyond basic interaction detection by:
        1. Analyzing stability of interactions across data subsets
        2. Checking interactions across multiple thresholds
        3. Supporting both pairwise and higher-order interactions
        """
        print("\nPerforming comprehensive interaction detection...")
        
        if self.base_model is None:
            self.train_base_model()
            
        # Initialize ShapIQ feature engineer
        shapiq_engineer = ShapIQFeatureEngineer(
            model=self.base_model,
            X=self.X_train,
            y=self.y_train,
            threshold=threshold,
            max_interactions=max_interactions,
            max_order=2  # Pairwise interactions
        )
        
        # Detect standard interactions
        interactions = shapiq_engineer.detect_interactions()
        
        # Store interaction strengths
        self.interaction_strengths = shapiq_engineer.interaction_strengths
        
        # Create an interaction stability check
        print("\nAnalyzing interaction stability across data samples...")
        bootstrap_samples = 5
        stable_interactions = self._check_interaction_stability(
            interactions, bootstrap_samples)
        
        # Check threshold sensitivity
        print("\nAnalyzing sensitivity to threshold changes...")
        threshold_range = [0.005, 0.01, 0.02, 0.05]
        threshold_interactions = self._check_threshold_sensitivity(threshold_range)
        
        # Summarize findings
        print(f"\nDetected {len(interactions)} significant interactions")
        print(f"Of these, {len(stable_interactions)} are stable across data samples")
        
        # Print top interactions
        top_k = min(5, len(interactions))
        print(f"\nTop {top_k} strongest interactions:")
        for idx, (feature1, feature2) in enumerate(interactions[:top_k]):
            strength = shapiq_engineer.interaction_strengths[(feature1, feature2)]
            stability = "✓" if (feature1, feature2) in stable_interactions else "✗"
            print(f"{idx+1}. {feature1} × {feature2}: {strength:.4f} (Stable: {stability})")
        
        # Return both the interaction list and the ShapIQ engineer for further use
        return interactions, shapiq_engineer
    
    def _check_interaction_stability(self, 
                                    interactions: List[Tuple[str, str]], 
                                    n_samples: int = 5,
                                    sample_size: float = 0.7) -> List[Tuple[str, str]]:
        """Check if interactions are stable across random data subsamples."""
        interaction_counts = {interaction: 0 for interaction in interactions}
        sample_interactions = []
        
        # Check interactions across multiple data samples
        for i in range(n_samples):
            # Create random subsample
            indices = np.random.choice(
                len(self.X_train), 
                size=int(len(self.X_train) * sample_size), 
                replace=False
            )
            X_sample = self.X_train.iloc[indices]
            y_sample = self.y_train.iloc[indices]
            
            # Train a model on this sample
            sample_model = self.model_type(**self.model_params)
            sample_model.fit(X_sample, y_sample)
            
            # Detect interactions on this sample
            engineer = ShapIQFeatureEngineer(
                model=sample_model,
                X=X_sample,
                y=y_sample,
                threshold=0.01,
                max_interactions=len(interactions) * 2  # Wider net to catch possible interactions
            )
            
            sample_result = engineer.detect_interactions()
            sample_interactions.append(sample_result)
            
            # Count occurrences
            for interaction in interactions:
                if interaction in sample_result:
                    interaction_counts[interaction] += 1
                    
        # Consider stable if appears in at least 60% of samples
        stability_threshold = n_samples * 0.6
        stable_interactions = [
            interaction for interaction, count in interaction_counts.items()
            if count >= stability_threshold
        ]
        
        return stable_interactions
    
    def _check_threshold_sensitivity(self, thresholds: List[float]) -> Dict[float, List[Tuple[str, str]]]:
        """Check how interaction detection changes with different thresholds."""
        threshold_results = {}
        
        for threshold in thresholds:
            engineer = ShapIQFeatureEngineer(
                model=self.base_model,
                X=self.X_train,
                y=self.y_train,
                threshold=threshold,
                max_interactions=50  # Allow more interactions at lower thresholds
            )
            interactions = engineer.detect_interactions()
            threshold_results[threshold] = interactions
            print(f"  Threshold {threshold:.3f}: {len(interactions)} interactions")
            
        return threshold_results
    
    def create_enhanced_features(self, 
                                interactions: List[Tuple[str, str]],
                                engineer: ShapIQFeatureEngineer,
                                operations: List[str] = ['multiply', 'ratio']):
        """Create enhanced feature set using detected interactions."""
        print("\nCreating enhanced feature set...")
        
        # Create interaction features
        X_train_enhanced, _ = engineer.pipeline(operations=operations)
        X_test_enhanced = engineer.create_features(self.X_test)
        
        # Print feature information
        n_original = self.X_train.shape[1]
        n_new = X_train_enhanced.shape[1] - n_original
        print(f"Original features: {n_original}")
        print(f"New interaction features: {n_new}")
        print(f"Total features: {X_train_enhanced.shape[1]}")
        
        # Find most important features
        self._analyze_feature_importance(X_train_enhanced, engineer)
        
        return X_train_enhanced, X_test_enhanced
    
    def _analyze_feature_importance(self, X_enhanced, engineer):
        """Analyze feature importance after enhancement."""
        # Train a model to get feature importance
        temp_model = self.model_type(**self.model_params)
        temp_model.fit(X_enhanced, self.y_train)
        
        # Get feature importance
        if hasattr(temp_model, 'feature_importances_'):
            importance = pd.Series(
                temp_model.feature_importances_, 
                index=X_enhanced.columns
            ).sort_values(ascending=False)
            
            # Identify interaction features
            interaction_features = [col for col in X_enhanced.columns 
                                   if col.startswith('shapiq_')]
            
            # Calculate importance of interaction features
            if interaction_features:
                interaction_importance = importance[interaction_features]
                total_importance = importance.sum()
                interaction_importance_sum = interaction_importance.sum()
                
                print(f"\nInteraction features contribute {interaction_importance_sum/total_importance:.1%} of total importance")
                
                # Print top interaction features
                top_k = min(5, len(interaction_features))
                print(f"\nTop {top_k} most important interaction features:")
                for idx, (feature, imp) in enumerate(interaction_importance.head(top_k).items()):
                    print(f"{idx+1}. {feature}: {imp:.4f} ({imp/total_importance:.1%} of total)")
    
    def train_enhanced_model(self, X_train_enhanced, X_test_enhanced):
        """Train model with enhanced features and evaluate improvement."""
        print("\nTraining enhanced model with interaction features...")
        
        # Create and train enhanced model
        self.enhanced_model = self.model_type(**self.model_params)
        self.enhanced_model.fit(X_train_enhanced, self.y_train)
        
        # Evaluate enhanced model
        self.enhanced_score = self.enhanced_model.score(X_test_enhanced, self.y_test)
        print(f"Enhanced model {self.metric}: {self.enhanced_score:.4f}")
        
        # Calculate improvement
        improvement = (self.enhanced_score - self.base_score)
        relative_improvement = improvement / max(abs(self.base_score), 1e-10) * 100
        print(f"Absolute improvement: {improvement:.4f}")
        print(f"Relative improvement: {relative_improvement:.2f}%")
        
        return self.enhanced_model
    
    def visualize_interactions(self):
        """Create visualizations for detected interactions."""
        print("\nGenerating interaction visualizations...")
        
        if not self.interaction_strengths:
            print("No interactions detected to visualize.")
            return
            
        # Create a network graph of interactions
        self._plot_interaction_network()
        
        # Generate a heatmap of feature interactions
        self._plot_interaction_heatmap()
        
        # Generate HTML report
        report_path = f"shapiq_{self.dataset_type}_report.html"
        generate_interaction_report(
            model=self.base_model,
            X=self.X_train,
            y=self.y_train,
            output_path=report_path,
            threshold=0.01,
            max_interactions=20,
            title=f"ShapIQ Analysis - {self.dataset_type.capitalize()} Dataset"
        )
        print(f"Interactive HTML report saved to {report_path}")
        
    def _plot_interaction_network(self):
        """Create a network graph of feature interactions."""
        try:
            import networkx as nx
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes (features)
            all_features = set()
            for f1, f2 in self.interaction_strengths.keys():
                all_features.add(f1)
                all_features.add(f2)
                
            for feature in all_features:
                G.add_node(feature)
                
            # Add edges (interactions)
            for (f1, f2), strength in self.interaction_strengths.items():
                G.add_edge(f1, f2, weight=strength)
                
            # Plot
            plt.figure(figsize=(12, 10))
            
            # Calculate positions using force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            # Get edge weights for line thickness
            edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
            nx.draw_networkx_labels(G, pos, font_size=10)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')
            
            # Add a title
            plt.title(f"Feature Interaction Network - {self.dataset_type.capitalize()} Dataset", size=15)
            plt.axis('off')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f"shapiq_{self.dataset_type}_network.png", dpi=300)
            print(f"Network graph saved to shapiq_{self.dataset_type}_network.png")
            
        except ImportError:
            print("NetworkX package required for network visualization.")
    
    def _plot_interaction_heatmap(self):
        """Create a heatmap of feature interactions."""
        # Get unique features
        all_features = set()
        for f1, f2 in self.interaction_strengths.keys():
            all_features.add(f1)
            all_features.add(f2)
        
        feature_list = sorted(list(all_features))
        n_features = len(feature_list)
        
        # Create interaction matrix
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Fill matrix
        for i, f1 in enumerate(feature_list):
            for j, f2 in enumerate(feature_list):
                if i == j:
                    continue  # Skip diagonal
                    
                if (f1, f2) in self.interaction_strengths:
                    interaction_matrix[i, j] = self.interaction_strengths[(f1, f2)]
                elif (f2, f1) in self.interaction_strengths:
                    interaction_matrix[i, j] = self.interaction_strengths[(f2, f1)]
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        plt.imshow(interaction_matrix, cmap='viridis')
        
        # Add labels
        plt.xticks(range(n_features), feature_list, rotation=90)
        plt.yticks(range(n_features), feature_list)
        
        # Add colorbar and title
        cbar = plt.colorbar()
        cbar.set_label('Interaction Strength')
        plt.title(f"Feature Interaction Heatmap - {self.dataset_type.capitalize()} Dataset", size=15)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"shapiq_{self.dataset_type}_heatmap.png", dpi=300)
        print(f"Interaction heatmap saved to shapiq_{self.dataset_type}_heatmap.png")
    
    def run_full_pipeline(self):
        """Run the complete ShapIQ analysis pipeline."""
        print(f"\n=== Running Complete ShapIQ Analysis Pipeline - {self.dataset_type.capitalize()} Dataset ===")
        
        # Step 1: Train base model
        self.train_base_model()
        
        # Step 2: Detect interactions
        interactions, engineer = self.detect_all_interactions()
        
        # Step 3: Create enhanced features
        X_train_enhanced, X_test_enhanced = self.create_enhanced_features(
            interactions, engineer, operations=['multiply', 'ratio', 'add']
        )
        
        # Step 4: Train enhanced model
        self.train_enhanced_model(X_train_enhanced, X_test_enhanced)
        
        # Step 5: Visualize interactions
        self.visualize_interactions()
        
        print("\n=== ShapIQ Analysis Pipeline Complete ===")
        return {
            'base_score': self.base_score,
            'enhanced_score': self.enhanced_score,
            'improvement': self.enhanced_score - self.base_score,
            'interaction_count': len(interactions),
            'feature_count': X_train_enhanced.shape[1]
        }

def main():
    """Run ShapIQ analysis on regression and classification datasets."""
    # Run regression example
    print("\n\n" + "="*80)
    print("REGRESSION EXAMPLE - CALIFORNIA HOUSING DATASET")
    print("="*80)
    regression_analysis = AdvancedShapIQAnalysis(dataset_type='regression')
    regression_results = regression_analysis.run_full_pipeline()
    
    # Run classification example
    print("\n\n" + "="*80)
    print("CLASSIFICATION EXAMPLE - BREAST CANCER DATASET")
    print("="*80)
    classification_analysis = AdvancedShapIQAnalysis(dataset_type='classification')
    classification_results = classification_analysis.run_full_pipeline()
    
    # Summarize results
    print("\n\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print("Regression (California Housing):")
    print(f"- Base score: {regression_results['base_score']:.4f}")
    print(f"- Enhanced score: {regression_results['enhanced_score']:.4f}")
    print(f"- Improvement: {regression_results['improvement']:.4f}")
    print(f"- Significant interactions: {regression_results['interaction_count']}")
    
    print("\nClassification (Breast Cancer):")
    print(f"- Base score: {classification_results['base_score']:.4f}")
    print(f"- Enhanced score: {classification_results['enhanced_score']:.4f}")
    print(f"- Improvement: {classification_results['improvement']:.4f}")
    print(f"- Significant interactions: {classification_results['interaction_count']}")

if __name__ == "__main__":
    main()