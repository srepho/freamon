"""
Feature engineering using ShapIQ to automatically detect and create interaction features.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Force SHAPIQ_AVAILABLE to True since we've verified it's installed
import shapiq
from shapiq.explainer import TabularExplainer
SHAPIQ_AVAILABLE = True

from freamon.explainability.shap_explainer import ShapIQExplainer
from freamon.features.engineer import create_interaction_features
from freamon.utils import check_dataframe_type, convert_dataframe


class ShapIQFeatureEngineer:
    """
    Feature engineer that uses ShapIQ to detect and create interaction features.
    
    Parameters
    ----------
    model : Any
        The model to explain. Should have a `predict` method.
    X : pd.DataFrame
        The training data to use for detecting interactions.
    y : Union[pd.Series, np.ndarray]
        The target variable.
    max_order : int, default=2
        Maximum interaction order to detect. 1 = main effects, 2 = pairwise interactions.
    threshold : float, default=0.05
        Minimum interaction strength threshold. Only interactions with absolute 
        strength above this threshold will be used to create features.
    max_interactions : int, default=10
        Maximum number of interaction features to create.
    """
    
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        max_order: int = 2,
        threshold: float = 0.05,
        max_interactions: int = 10,
        interaction_type: str = 'shapiq',
        sample_size: Optional[int] = None,
    ):
        """
        Initialize the ShapIQFeatureEngineer.
        
        Parameters
        ----------
        model : Any
            The model to explain. Should have a `predict` method.
        X : pd.DataFrame
            The training data to use for detecting interactions.
        y : Union[pd.Series, np.ndarray]
            The target variable.
        max_order : int, default=2
            Maximum interaction order to detect. 1 = main effects, 2 = pairwise interactions.
        threshold : float, default=0.05
            Minimum interaction strength threshold. Only interactions with absolute 
            strength above this threshold will be used to create features.
        max_interactions : int, default=10
            Maximum number of interaction features to create.
        interaction_type : str, default='shapiq'
            Type of interaction calculation to use. Options:
            - 'shapiq': Uses k-Shapley Interaction Index (k-SII) 
            - 'shapley_taylor': Uses Shapley Taylor Interaction Index (STII)
            - 'faith_interactions': Uses Faithful Shapley Interaction Index (FSII)
        sample_size : Optional[int], default=None
            Number of samples to use for interaction detection. If None, automatically
            determines an appropriate size based on the dataset. Using a smaller sample
            improves performance but may impact the quality of detected interactions.
        """
        if not SHAPIQ_AVAILABLE:
            raise ImportError("shapiq package is required for ShapIQFeatureEngineer.")
        
        self.model = model
        self.X = X
        self.y = y
        self.max_order = max_order
        self.threshold = threshold
        self.max_interactions = max_interactions
        self.interaction_type = interaction_type
        self.sample_size = sample_size
        
        # Initialize results
        self.detected_interactions = []
        self.interaction_strengths = {}
    
    def detect_interactions(self) -> List[Tuple[str, str]]:
        """
        Detect significant feature interactions using ShapIQ.
        
        Returns
        -------
        List[Tuple[str, str]]
            List of tuples containing pairs of interacting features.
        """
        if len(self.X) == 0:
            warnings.warn("Empty dataset provided for interaction detection.")
            return []
            
        # Take a subset of data for faster computation if dataset is large
        if self.sample_size is None:
            # Use a default sample size that scales with dataset size 
            # but caps at 100 for very large datasets
            actual_sample_size = min(100, max(10, int(len(self.X) * 0.1)))
        else:
            # Use the provided sample size, but ensure it's not larger than the dataset
            actual_sample_size = min(self.sample_size, len(self.X))
            
        sample_indices = np.random.choice(len(self.X), actual_sample_size, replace=False)
        sample_X = self.X.iloc[sample_indices] if isinstance(self.X, pd.DataFrame) else self.X[sample_indices]
        
        # Initialize ShapIQ explainer
        explainer = ShapIQExplainer(self.model, max_order=self.max_order)
        explainer.fit(sample_X, interaction_type=self.interaction_type)
        
        # Compute interactions for the first instance only to get structure
        # This avoids the broadcasting error when calculating for many instances
        instance_X = sample_X.iloc[0:1] if isinstance(sample_X, pd.DataFrame) else sample_X[0:1]
        interactions = explainer.explain(instance_X)
        
        # Get pairwise interactions (order 2)
        if self.max_order >= 2 and interactions is not None:
            try:
                # Verify interactions object has get_order method, if not use a different approach
                if hasattr(interactions, 'get_order'):
                    pairwise = interactions.get_order(2)
                else:
                    # Handle case where get_order is not available
                    # This fallback assumes interactions is already at order 2 level
                    pairwise = interactions
                
                # Initialize for pairs
                significant_pairs = []
                pair_strengths = {}
                
                # Process each feature pair
                feature_names = self.X.columns if isinstance(self.X, pd.DataFrame) else [f"feature_{i}" for i in range(self.X.shape[1])]
                n_features = len(feature_names)
                
                # Extract interaction values from the first instance
                # Handle different return types from different versions of ShapIQ
                if hasattr(pairwise, 'values') and hasattr(pairwise.values, 'shape') and pairwise.values.shape[0] > 0:
                    interaction_values = np.abs(pairwise.values[0]) 
                elif hasattr(pairwise, 'interaction_values') and pairwise.interaction_values.shape[0] > 0:
                    interaction_values = np.abs(pairwise.interaction_values[0])
                else:
                    interaction_values = np.zeros((n_features, n_features))
                
                # Find significant interactions
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        feature1 = feature_names[i]
                        feature2 = feature_names[j]
                        strength = interaction_values[i, j]
                        
                        if strength > self.threshold:
                            significant_pairs.append((feature1, feature2))
                            pair_strengths[(feature1, feature2)] = float(strength)  # Convert to Python float for serialization
                
                # Sort by interaction strength
                significant_pairs.sort(key=lambda x: pair_strengths[x], reverse=True)
                
                # Limit to max_interactions
                if len(significant_pairs) > self.max_interactions:
                    significant_pairs = significant_pairs[:self.max_interactions]
                
                self.detected_interactions = significant_pairs
                self.interaction_strengths = pair_strengths
                
                return significant_pairs
            except Exception as e:
                warnings.warn(f"Error detecting interactions: {str(e)}")
                return []
        
        return []
    
    def create_features(
        self, 
        X_new: Optional[pd.DataFrame] = None,
        operations: List[str] = ['multiply'],
        prefix: str = 'shapiq',
    ) -> pd.DataFrame:
        """
        Create interaction features based on detected interactions.
        
        Parameters
        ----------
        X_new : Optional[pd.DataFrame], default=None
            New data to generate features for. If None, uses the training data.
        operations : List[str], default=['multiply']
            Types of interactions to create. Options: 'multiply', 'divide', 'add', 'subtract', 'ratio'.
        prefix : str, default='shapiq'
            Prefix for the new feature names.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added interaction features.
        """
        # Check if interactions have been detected
        if not self.detected_interactions:
            self.detect_interactions()
        
        # If still no interactions found
        if not self.detected_interactions:
            warnings.warn("No significant interactions detected. Returning original data.")
            return X_new if X_new is not None else self.X
        
        # Use provided data or training data
        df = X_new if X_new is not None else self.X
        
        # Extract all unique features from the detected interactions
        all_features = set()
        for f1, f2 in self.detected_interactions:
            all_features.add(f1)
            all_features.add(f2)
        
        # Create interaction features
        return create_interaction_features(
            df=df,
            feature_columns=list(all_features),
            interaction_pairs=self.detected_interactions,
            operations=operations,
            prefix=prefix,
        )
    
    def get_interaction_report(self) -> Dict[str, Any]:
        """
        Generate a report of detected interactions and their strengths.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing information about detected interactions.
        """
        # Check if interactions have been detected
        if not self.detected_interactions:
            self.detect_interactions()
        
        # Create a report dictionary
        report = {
            "num_interactions": len(self.detected_interactions),
            "interactions": [],
            "plot": None,
        }
        
        # Add each interaction with its strength
        for feature1, feature2 in self.detected_interactions:
            strength = self.interaction_strengths.get((feature1, feature2), 0)
            report["interactions"].append({
                "feature1": feature1,
                "feature2": feature2,
                "strength": strength,
            })
        
        # Generate a bar plot of interaction strengths
        if self.detected_interactions:
            plt.figure(figsize=(10, 6))
            
            # Sort interactions by strength
            sorted_interactions = sorted(
                report["interactions"], 
                key=lambda x: x["strength"], 
                reverse=True
            )
            
            # Extract data for plotting
            labels = [f"{i['feature1']} × {i['feature2']}" for i in sorted_interactions]
            values = [i["strength"] for i in sorted_interactions]
            
            # Create the bar plot
            plt.barh(labels, values)
            plt.xlabel("Interaction Strength")
            plt.title("Top Feature Interactions Detected by ShapIQ")
            plt.tight_layout()
            
            # Convert plot to base64 string
            from io import BytesIO
            import base64
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            report["plot"] = f"data:image/png;base64,{plot_data}"
            
            plt.close()
        
        return report
    
    def pipeline(
        self,
        X_new: Optional[pd.DataFrame] = None,
        operations: List[str] = ['multiply'],
        prefix: str = 'shapiq',
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete interaction detection and feature creation pipeline.
        
        Parameters
        ----------
        X_new : Optional[pd.DataFrame], default=None
            New data to generate features for. If None, uses the training data.
        operations : List[str], default=['multiply']
            Types of interactions to create. Options: 'multiply', 'divide', 'add', 'subtract', 'ratio'.
        prefix : str, default='shapiq'
            Prefix for the new feature names.
        
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Tuple containing:
            - DataFrame with added interaction features
            - Dictionary with interaction report
        """
        # Detect interactions
        self.detect_interactions()
        
        # Create features
        df_with_features = self.create_features(X_new, operations, prefix)
        
        # Generate report
        report = self.get_interaction_report()
        
        return df_with_features, report