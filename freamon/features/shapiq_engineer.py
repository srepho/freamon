"""
Feature engineering using ShapIQ to automatically detect and create interaction features.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shapiq
    from shapiq.explainer import KernelExplainer
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False
    warnings.warn("shapiq package is not installed. ShapIQFeatureEngineer will not be available.")

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
    ):
        """Initialize the ShapIQFeatureEngineer."""
        if not SHAPIQ_AVAILABLE:
            raise ImportError("shapiq package is required for ShapIQFeatureEngineer.")
        
        self.model = model
        self.X = X
        self.y = y
        self.max_order = max_order
        self.threshold = threshold
        self.max_interactions = max_interactions
        
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
        # Initialize ShapIQ explainer
        explainer = ShapIQExplainer(self.model, max_order=self.max_order)
        explainer.fit(self.X)
        
        # Compute interactions
        interactions = explainer.explain(self.X)
        
        # Get pairwise interactions (order 2)
        if self.max_order >= 2:
            pairwise = interactions.get_order(2)
            
            # Compute average absolute interaction strength
            avg_interactions = np.abs(pairwise.values).mean(axis=0)
            
            # Get feature pairs exceeding the threshold
            significant_pairs = []
            pair_strengths = {}
            
            for i in range(len(self.X.columns)):
                for j in range(i+1, len(self.X.columns)):
                    feature1 = self.X.columns[i]
                    feature2 = self.X.columns[j]
                    strength = avg_interactions[i, j]
                    
                    if strength > self.threshold:
                        significant_pairs.append((feature1, feature2))
                        pair_strengths[(feature1, feature2)] = strength
            
            # Sort by interaction strength
            significant_pairs.sort(key=lambda x: pair_strengths[x], reverse=True)
            
            # Limit to max_interactions
            if len(significant_pairs) > self.max_interactions:
                significant_pairs = significant_pairs[:self.max_interactions]
            
            self.detected_interactions = significant_pairs
            self.interaction_strengths = pair_strengths
            
            return significant_pairs
        
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