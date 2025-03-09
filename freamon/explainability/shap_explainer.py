"""
SHAP and ShapIQ based explainers for model interpretability.

This module provides classes for explaining model predictions using SHAP values
and ShapIQ interaction values, with special support for LightGBM models and
cohort-based analysis.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from sklearn.cluster import KMeans

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap package is not installed. ShapExplainer will not be available.")

try:
    import shapiq
    from shapiq.explainer import KernelExplainer as ShapIQKernelExplainer
    from shapiq.interactions import InteractionValues
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False
    warnings.warn("shapiq package is not installed. ShapIQExplainer will not be available.")

# Check if LightGBM is available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("lightgbm package is not installed. LightGBM-specific features will not be available.")


class ShapExplainer:
    """
    Wrapper for SHAP explainers to provide model interpretability.
    
    Parameters
    ----------
    model : Any
        The model to explain. Should have a `predict` method.
    model_type : str, default='tree'
        The type of model. Options: 'tree', 'linear', 'kernel'.
    """
    
    def __init__(self, model: Any, model_type: str = 'tree'):
        """Initialize the SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("shap package is required for ShapExplainer.")
        
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.is_fitted = False
        
        # Set a mapping of model_type to SHAP explainer
        self.explainer_mapping = {
            'tree': shap.TreeExplainer,
            'linear': shap.LinearExplainer,
            'kernel': shap.KernelExplainer,
        }
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'ShapExplainer':
        """
        Fit the SHAP explainer to the data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The data to use for the explainer.
        
        Returns
        -------
        ShapExplainer
            The fitted explainer.
        """
        if self.model_type not in self.explainer_mapping:
            raise ValueError(f"Unknown model_type: {self.model_type}. "
                            f"Available options: {list(self.explainer_mapping.keys())}")
        
        # Convert pandas DataFrame to numpy array if necessary
        X_data = X
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        
        # Create the explainer based on model_type
        if self.model_type == 'kernel':
            # Kernel explainer requires a function that returns numpy arrays
            def model_predict(X):
                return self.model.predict(X)
            self.explainer = self.explainer_mapping[self.model_type](model_predict, X_data)
        else:
            self.explainer = self.explainer_mapping[self.model_type](self.model, X_data)
        
        self.is_fitted = True
        return self
    
    def explain(self, X: Union[pd.DataFrame, np.ndarray]) -> Any:
        """
        Generate SHAP values to explain predictions.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The data to explain predictions for.
        
        Returns
        -------
        Any
            SHAP values object.
        """
        if not self.is_fitted:
            raise ValueError("Explainer is not fitted. Call fit() first.")
        
        # Convert pandas DataFrame to numpy array if necessary
        X_data = X
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        
        # Generate SHAP values
        shap_values = self.explainer.shap_values(X_data)
        
        # If the output is a list (for multi-class models), convert to a more usable format
        if isinstance(shap_values, list) and isinstance(X, pd.DataFrame):
            result = []
            for class_idx, class_shap_values in enumerate(shap_values):
                df = pd.DataFrame(class_shap_values, columns=X.columns, index=X.index)
                df['_class'] = class_idx
                result.append(df)
            return pd.concat(result)
        
        # For binary classification or regression with pandas input
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        
        return shap_values
    
    def summary_plot(self, shap_values: Any, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """
        Generate a summary plot of SHAP values.
        
        Parameters
        ----------
        shap_values : Any
            SHAP values from the explain method.
        X : Union[pd.DataFrame, np.ndarray]
            The data used to generate the SHAP values.
        **kwargs : Dict
            Additional arguments to pass to shap.summary_plot.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap package is required for summary_plot.")
        
        # Convert back to numpy if we converted to pandas in explain()
        if isinstance(shap_values, pd.DataFrame):
            class_values = shap_values['_class'].unique() if '_class' in shap_values.columns else [0]
            if '_class' in shap_values.columns:
                shap_values_list = []
                for class_idx in class_values:
                    class_df = shap_values[shap_values['_class'] == class_idx].drop('_class', axis=1)
                    shap_values_list.append(class_df.values)
                shap_values = shap_values_list
            else:
                shap_values = shap_values.values
        
        # Create the summary plot
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        shap.summary_plot(shap_values, X, feature_names=feature_names, **kwargs)


class ShapIQExplainer:
    """
    Wrapper for ShapIQ explainers to provide model interpretability with interactions.
    
    Parameters
    ----------
    model : Any
        The model to explain. Should have a `predict` method.
    max_order : int, default=2
        Maximum interaction order to compute. 1 = main effects, 2 = pairwise interactions, etc.
    """
    
    def __init__(self, model: Any, max_order: int = 2):
        """Initialize the ShapIQ explainer."""
        if not SHAPIQ_AVAILABLE:
            raise ImportError("shapiq package is required for ShapIQExplainer.")
        
        self.model = model
        self.max_order = max_order
        self.explainer = None
        self.interactions = None
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, interaction_type: str = 'shapley_taylor') -> 'ShapIQExplainer':
        """
        Fit the ShapIQ explainer to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to use for the explainer.
        interaction_type : str, default='shapley_taylor'
            The type of interaction values to compute.
            Options: 'shapley_taylor', 'faith_interactions', 'shapiq'
        
        Returns
        -------
        ShapIQExplainer
            The fitted explainer.
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create a prediction function that returns numpy arrays
        def model_predict(X):
            return self.model.predict(X)
        
        # Create the ShapIQ explainer
        self.explainer = ShapIQKernelExplainer(model_predict, X.values)
        
        # Set the interaction type
        self.interaction_type = interaction_type
        
        self.is_fitted = True
        return self
    
    def explain(self, X: pd.DataFrame) -> Any:
        """
        Generate ShapIQ interaction values to explain predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to explain predictions for.
        
        Returns
        -------
        Any
            ShapIQ interaction values object.
        """
        if not self.is_fitted:
            raise ValueError("Explainer is not fitted. Call fit() first.")
        
        # Generate interaction values based on the selected type
        if self.interaction_type == 'shapley_taylor':
            self.interactions = self.explainer.stx(X.values, max_order=self.max_order)
        elif self.interaction_type == 'faith_interactions':
            self.interactions = self.explainer.faith(X.values, max_order=self.max_order)
        elif self.interaction_type == 'shapiq':
            self.interactions = self.explainer.shapiq(X.values, max_order=self.max_order)
        else:
            raise ValueError(f"Unknown interaction_type: {self.interaction_type}. "
                           f"Available options: 'shapley_taylor', 'faith_interactions', 'shapiq'")
        
        return self.interactions
    
    def plot_main_effects(self, instance_idx: int = 0, top_k: int = 10, **kwargs) -> None:
        """
        Plot the main effects (first-order interactions) for a specific instance.
        
        Parameters
        ----------
        instance_idx : int, default=0
            The index of the instance to explain.
        top_k : int, default=10
            The number of top features to show.
        **kwargs : Dict
            Additional arguments to pass to the plotting function.
        """
        if self.interactions is None:
            raise ValueError("No interaction values available. Call explain() first.")
        
        # Extract main effects (order 1)
        main_effects = self.interactions.get_order(1)
        
        # Get values for the specified instance
        instance_values = main_effects.values[instance_idx]
        
        # Create a DataFrame with feature names
        effect_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Effect': instance_values
        })
        
        # Sort by absolute value
        effect_df['Abs'] = effect_df['Effect'].abs()
        effect_df = effect_df.sort_values('Abs', ascending=False).head(top_k)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(effect_df['Feature'], effect_df['Effect'])
        plt.xlabel('SHAP Value')
        plt.title(f'Top {top_k} Main Effects for Instance {instance_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_interaction_effects(self, instance_idx: int = 0, top_k: int = 10, **kwargs) -> None:
        """
        Plot the pairwise interaction effects for a specific instance.
        
        Parameters
        ----------
        instance_idx : int, default=0
            The index of the instance to explain.
        top_k : int, default=10
            The number of top interactions to show.
        **kwargs : Dict
            Additional arguments to pass to the plotting function.
        """
        if self.interactions is None:
            raise ValueError("No interaction values available. Call explain() first.")
        
        if self.max_order < 2:
            raise ValueError("Pairwise interactions were not computed. Set max_order >= 2.")
        
        # Extract pairwise interactions (order 2)
        pairwise = self.interactions.get_order(2)
        
        # Get values for the specified instance
        instance_values = pairwise.values[instance_idx]
        
        # Create tuples of feature pairs
        pairs = []
        pair_values = []
        
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                pair_name = f"{self.feature_names[i]} × {self.feature_names[j]}"
                pair_value = instance_values[i, j]
                pairs.append(pair_name)
                pair_values.append(pair_value)
        
        # Create a DataFrame
        interaction_df = pd.DataFrame({
            'Interaction': pairs,
            'Effect': pair_values
        })
        
        # Sort by absolute value
        interaction_df['Abs'] = interaction_df['Effect'].abs()
        interaction_df = interaction_df.sort_values('Abs', ascending=False).head(top_k)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.barh(interaction_df['Interaction'], interaction_df['Effect'])
        plt.xlabel('Interaction Strength')
        plt.title(f'Top {top_k} Pairwise Interactions for Instance {instance_idx}')
        plt.tight_layout()
        plt.show()


class LightGBMShapExplainer:
    """
    Advanced SHAP explainer optimized for LightGBM models with cohort analysis.
    
    This explainer provides LightGBM-specific optimizations and adds the ability
    to analyze explanations across different data cohorts to identify segments
    with different feature impact patterns.
    
    Parameters
    ----------
    model : Any
        The LightGBM model to explain. Can be a raw LightGBM model or wrapped by freamon.
    feature_perturbation : str, default='tree_path_dependent'
        Method used to explain the model. Options:
        - 'tree_path_dependent': Accounts for correlations (default for tree models)
        - 'interventional': Breaks dependencies between features, simpler interpretation
    background_samples : Optional[int], default=100
        Number of background samples to use for TreeExplainer. If None, use all data.
    """
    
    def __init__(
        self, 
        model: Any, 
        feature_perturbation: str = 'tree_path_dependent',
        background_samples: Optional[int] = 100
    ):
        """Initialize the LightGBM SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("shap package is required for LightGBMShapExplainer.")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm package is required for LightGBMShapExplainer.")
        
        self.model = model
        self.feature_perturbation = feature_perturbation
        self.background_samples = background_samples
        self.explainer = None
        self.is_fitted = False
        self.feature_names = None
        self.shap_values = None
        self.shap_interaction_values = None
        self.clustering_model = None
        self.cohort_labels = None
        self.cohort_shap_values = None
        
        # Extract the actual LightGBM model if it's wrapped
        if hasattr(model, 'model') and isinstance(model.model, lgb.Booster):
            self.lgb_model = model.model
        elif hasattr(model, 'model_') and isinstance(model.model_, lgb.Booster):
            self.lgb_model = model.model_
        elif isinstance(model, lgb.Booster):
            self.lgb_model = model
        elif hasattr(model, '_Booster') and isinstance(model._Booster, lgb.Booster):
            self.lgb_model = model._Booster
        else:
            raise ValueError("Model doesn't appear to be a LightGBM model or compatible wrapper.")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LightGBMShapExplainer':
        """
        Fit the LightGBM SHAP explainer to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to use for the explainer.
        y : Optional[pd.Series], default=None
            The target values, not used but kept for API consistency.
        
        Returns
        -------
        LightGBMShapExplainer
            The fitted explainer.
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Sample background data if needed
        if self.background_samples is not None and self.background_samples < len(X):
            background_data = shap.sample(X, self.background_samples)
        else:
            background_data = X
        
        # Create the TreeExplainer with LightGBM-specific optimizations
        self.explainer = shap.TreeExplainer(
            self.lgb_model, 
            data=background_data,
            feature_perturbation=self.feature_perturbation,
            model_output='raw'
        )
        
        self.is_fitted = True
        return self
    
    def explain(
        self, 
        X: pd.DataFrame, 
        interactions: bool = False,
        approximate: bool = False
    ) -> pd.DataFrame:
        """
        Generate SHAP values to explain predictions for LightGBM model.
        
        Parameters
        ----------
        X : pd.DataFrame
            The data to explain predictions for.
        interactions : bool, default=False
            Whether to compute SHAP interaction values (much slower).
        approximate : bool, default=False
            Whether to use approximation for faster computation (for large datasets).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with SHAP values for each feature and instance.
        """
        if not self.is_fitted:
            raise ValueError("Explainer is not fitted. Call fit() first.")
        
        # Calculate SHAP values
        if approximate:
            self.shap_values = self.explainer.shap_values(X, approximate=True, check_additivity=False)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        # Convert to DataFrame for easier analysis
        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
            # For multi-class models
            result_dfs = []
            for class_idx, class_shap_values in enumerate(self.shap_values):
                df = pd.DataFrame(class_shap_values, columns=self.feature_names, index=X.index)
                df['_class'] = class_idx
                df['_expected_value'] = self.explainer.expected_value[class_idx]
                result_dfs.append(df)
            shap_df = pd.concat(result_dfs)
        else:
            # For binary/regression models
            values = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values
            expected_value = self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            shap_df = pd.DataFrame(values, columns=self.feature_names, index=X.index)
            shap_df['_expected_value'] = expected_value
        
        # Also compute interaction values if requested
        if interactions:
            if approximate:
                self.shap_interaction_values = self.explainer.shap_interaction_values(
                    X.iloc[:100],  # Use a subset for interactions when approximating
                    approximate=True
                )
            else:
                self.shap_interaction_values = self.explainer.shap_interaction_values(X)
        
        return shap_df
    
    def create_cohorts(
        self,
        X: pd.DataFrame,
        shap_df: pd.DataFrame,
        n_clusters: int = 3,
        method: str = 'kmeans',
        features_to_use: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[pd.DataFrame]]:
        """
        Create cohorts (clusters) based on SHAP values to identify segments with similar explanations.
        
        Parameters
        ----------
        X : pd.DataFrame
            The original feature data.
        shap_df : pd.DataFrame
            DataFrame with SHAP values from the explain method.
        n_clusters : int, default=3
            Number of clusters/cohorts to create.
        method : str, default='kmeans'
            Clustering method to use. Currently only 'kmeans' is supported.
        features_to_use : Optional[List[str]], default=None
            Specific features to use for clustering. If None, use all features.
        
        Returns
        -------
        Tuple[np.ndarray, List[pd.DataFrame]]
            - Cluster labels for each instance
            - List of DataFrames with SHAP values for each cohort
        """
        # Handle multi-class case
        has_class_col = '_class' in shap_df.columns
        
        if has_class_col:
            # For multi-class, separate by class before clustering
            cohorts_by_class = []
            all_labels = []
            
            for class_idx in shap_df['_class'].unique():
                class_shap_df = shap_df[shap_df['_class'] == class_idx].drop('_class', axis=1)
                
                # Get clustering features
                if features_to_use:
                    cluster_features = [f for f in features_to_use if f in class_shap_df.columns]
                    if not cluster_features:
                        raise ValueError("None of the specified features exist in the SHAP values DataFrame")
                else:
                    # Use all features except metadata columns
                    cluster_features = [c for c in class_shap_df.columns if not c.startswith('_')]
                
                # Cluster
                if method == 'kmeans':
                    clustering = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = clustering.fit_predict(class_shap_df[cluster_features])
                else:
                    raise ValueError(f"Unsupported clustering method: {method}")
                
                # Store results
                all_labels.append(labels)
                class_cohorts = []
                for i in range(n_clusters):
                    mask = labels == i
                    cohort_df = class_shap_df[mask].copy()
                    cohort_df['_cohort'] = i
                    cohort_df['_class'] = class_idx
                    class_cohorts.append(cohort_df)
                
                cohorts_by_class.append(class_cohorts)
            
            # Flatten the cohorts
            self.cohort_labels = np.concatenate(all_labels)
            self.cohort_shap_values = [df for class_cohorts in cohorts_by_class for df in class_cohorts]
            
        else:
            # For binary/regression models
            # Get clustering features
            if features_to_use:
                cluster_features = [f for f in features_to_use if f in shap_df.columns]
                if not cluster_features:
                    raise ValueError("None of the specified features exist in the SHAP values DataFrame")
            else:
                # Use all features except metadata columns
                cluster_features = [c for c in shap_df.columns if not c.startswith('_')]
            
            # Cluster
            if method == 'kmeans':
                self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
                self.cohort_labels = self.clustering_model.fit_predict(shap_df[cluster_features])
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Create cohort DataFrames
            self.cohort_shap_values = []
            for i in range(n_clusters):
                mask = self.cohort_labels == i
                cohort_df = shap_df[mask].copy()
                cohort_df['_cohort'] = i
                self.cohort_shap_values.append(cohort_df)
        
        # Also create X cohorts for reference
        self.X_cohorts = []
        unique_cohort_indices = np.unique([(df['_cohort'].iloc[0], df['_class'].iloc[0] if '_class' in df.columns else 0) 
                                           for df in self.cohort_shap_values])
        
        for cohort_idx, class_idx in unique_cohort_indices:
            if has_class_col:
                mask = (self.cohort_labels == cohort_idx) & (shap_df['_class'] == class_idx)
            else:
                mask = self.cohort_labels == cohort_idx
            
            # Get the corresponding rows from X
            cohort_X = X.loc[shap_df[mask].index].copy()
            cohort_X['_cohort'] = cohort_idx
            if has_class_col:
                cohort_X['_class'] = class_idx
            self.X_cohorts.append(cohort_X)
        
        return self.cohort_labels, self.cohort_shap_values
    
    def analyze_cohorts(
        self, 
        X: pd.DataFrame, 
        top_n_features: int = 5
    ) -> pd.DataFrame:
        """
        Analyze differences between cohorts to understand segment-specific explanations.
        
        Parameters
        ----------
        X : pd.DataFrame
            The original feature data.
        top_n_features : int, default=5
            Number of top features to include in the analysis for each cohort.
        
        Returns
        -------
        pd.DataFrame
            Summary of cohort characteristics and key features.
        """
        if self.cohort_shap_values is None or self.X_cohorts is None:
            raise ValueError("No cohorts available. Call create_cohorts() first.")
        
        # Create analysis dataframe
        cohort_analysis = []
        
        for i, (cohort_shap, cohort_X) in enumerate(zip(self.cohort_shap_values, self.X_cohorts)):
            cohort_id = cohort_shap['_cohort'].iloc[0]
            class_id = cohort_shap['_class'].iloc[0] if '_class' in cohort_shap.columns else None
            
            # Get feature names (excluding metadata columns)
            feature_cols = [c for c in cohort_shap.columns if not c.startswith('_')]
            
            # Calculate the mean absolute SHAP value for each feature
            mean_abs_shap = cohort_shap[feature_cols].abs().mean().sort_values(ascending=False)
            top_features = mean_abs_shap.head(top_n_features)
            
            # Calculate feature statistics for top features
            feature_stats = {}
            for feat in top_features.index:
                if feat in cohort_X.columns:
                    # Calculate statistics differently based on data type
                    if np.issubdtype(cohort_X[feat].dtype, np.number):
                        feat_mean = cohort_X[feat].mean()
                        feat_std = cohort_X[feat].std()
                        feat_min = cohort_X[feat].min()
                        feat_max = cohort_X[feat].max()
                        feature_stats[feat] = f"mean={feat_mean:.2f}, std={feat_std:.2f}, range=[{feat_min:.2f}, {feat_max:.2f}]"
                    else:
                        # For categorical features, show top 3 most common values
                        value_counts = cohort_X[feat].value_counts(normalize=True).head(3)
                        top_values = [f"{val}({cnt:.1%})" for val, cnt in value_counts.items()]
                        feature_stats[feat] = ", ".join(top_values)
            
            # Create analysis entry
            analysis_entry = {
                "cohort_id": cohort_id,
                "class_id": class_id,
                "size": len(cohort_shap),
                "proportion": len(cohort_shap) / len(X),
                "top_features": ", ".join(top_features.index),
                "mean_impact": {feat: float(val) for feat, val in top_features.items()},
                "feature_stats": feature_stats
            }
            
            cohort_analysis.append(analysis_entry)
        
        return pd.DataFrame(cohort_analysis)
    
    def plot_cohort_comparison(
        self, 
        top_n_features: int = 10, 
        plot_type: str = 'bar',
        figsize: Tuple[int, int] = (14, 8)
    ) -> None:
        """
        Plot a comparison of feature importance across different cohorts.
        
        Parameters
        ----------
        top_n_features : int, default=10
            Number of top features to include in the plot.
        plot_type : str, default='bar'
            Type of plot to create ('bar' or 'heatmap').
        figsize : Tuple[int, int], default=(14, 8)
            Figure size (width, height) in inches.
        """
        if self.cohort_shap_values is None:
            raise ValueError("No cohorts available. Call create_cohorts() first.")
        
        # Get all feature columns (excluding metadata)
        feature_cols = [c for c in self.cohort_shap_values[0].columns if not c.startswith('_')]
        
        # Calculate mean absolute SHAP values for each cohort
        cohort_importances = []
        
        for cohort_df in self.cohort_shap_values:
            cohort_id = cohort_df['_cohort'].iloc[0]
            class_id = cohort_df['_class'].iloc[0] if '_class' in cohort_df.columns else 0
            
            # Calculate mean absolute SHAP
            mean_abs_shap = cohort_df[feature_cols].abs().mean()
            mean_abs_shap.name = f"Cohort {cohort_id}" + (f" (Class {class_id})" if '_class' in cohort_df.columns else "")
            
            cohort_importances.append(mean_abs_shap)
        
        # Combine into a single DataFrame
        importance_df = pd.concat(cohort_importances, axis=1)
        
        # Identify top features across all cohorts
        overall_top_features = importance_df.mean(axis=1).sort_values(ascending=False).head(top_n_features).index
        
        # Plot comparison
        plt.figure(figsize=figsize)
        
        if plot_type == 'bar':
            importance_df.loc[overall_top_features].plot(kind='bar', figsize=figsize)
            plt.title('Feature Importance Comparison Across Cohorts')
            plt.ylabel('Mean |SHAP Value|')
            plt.xlabel('Feature')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Cohort')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        elif plot_type == 'heatmap':
            try:
                import seaborn as sns
                plt.figure(figsize=figsize)
                sns.heatmap(
                    importance_df.loc[overall_top_features].T,
                    annot=True, 
                    cmap='viridis',
                    fmt='.3f',
                    cbar_kws={'label': 'Mean |SHAP Value|'}
                )
                plt.title('Feature Importance Heatmap Across Cohorts')
                plt.ylabel('Cohort')
                plt.xlabel('Feature')
            except ImportError:
                plt.figure(figsize=figsize)
                plt.imshow(importance_df.loc[overall_top_features].T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Mean |SHAP Value|')
                plt.xticks(range(len(overall_top_features)), overall_top_features, rotation=45, ha='right')
                plt.yticks(range(len(importance_df.columns)), importance_df.columns)
                plt.title('Feature Importance Heatmap Across Cohorts')
                plt.ylabel('Cohort')
                plt.xlabel('Feature')
        
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'bar' or 'heatmap'.")
        
        plt.tight_layout()
        plt.show()
    
    def plot_dependence(
        self, 
        feature: str, 
        interaction_feature: Optional[str] = None,
        X: Optional[pd.DataFrame] = None,
        max_points: int = 1000,
        cohort_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 7)
    ) -> None:
        """
        Create a SHAP dependence plot for a feature, optionally colored by an interaction feature.
        
        Parameters
        ----------
        feature : str
            The feature to plot on the x-axis.
        interaction_feature : Optional[str], default=None
            Feature to use for coloring points. If None, automatically selects the strongest interaction.
        X : Optional[pd.DataFrame], default=None
            Original data for x-axis values. If None, uses the stored SHAP values.
        max_points : int, default=1000
            Maximum number of points to display (samples if more exist).
        cohort_idx : Optional[int], default=None
            If provided, plot only for the specified cohort.
        figsize : Tuple[int, int], default=(10, 7)
            Figure size (width, height) in inches.
        """
        if not self.is_fitted:
            raise ValueError("Explainer is not fitted. Call fit() first.")
        
        if self.shap_values is None:
            raise ValueError("No SHAP values available. Call explain() first.")
        
        # Select data based on cohort
        if cohort_idx is not None:
            if self.cohort_shap_values is None:
                raise ValueError("No cohorts available. Call create_cohorts() first.")
            
            # Find the cohort dataframe
            cohort_df = next((df for df in self.cohort_shap_values if df['_cohort'].iloc[0] == cohort_idx), None)
            if cohort_df is None:
                raise ValueError(f"Cohort {cohort_idx} not found")
            
            # Get the corresponding X data
            if X is not None:
                X_cohort = X.loc[cohort_df.index]
            else:
                X_cohort = None
            
            # Get SHAP values for the cohort
            feature_cols = [c for c in cohort_df.columns if not c.startswith('_')]
            shap_values_cohort = cohort_df[feature_cols].values
            
            # Create the plot
            plt.figure(figsize=figsize)
            if X_cohort is not None:
                if interaction_feature:
                    shap.dependence_plot(
                        feature, 
                        shap_values_cohort, 
                        X_cohort, 
                        interaction_index=interaction_feature,
                        alpha=0.8,
                        max_display=max_points
                    )
                else:
                    shap.dependence_plot(
                        feature, 
                        shap_values_cohort, 
                        X_cohort,
                        alpha=0.8,
                        max_display=max_points
                    )
            else:
                # Without original X, we can still plot SHAP values but without actual x-axis values
                plt.scatter(
                    range(len(shap_values_cohort)), 
                    shap_values_cohort[:, feature_cols.index(feature)], 
                    alpha=0.8
                )
                plt.xlabel("Index")
                plt.ylabel(f"SHAP value for {feature}")
            
            plt.title(f"SHAP Dependence Plot for {feature} (Cohort {cohort_idx})")
        
        else:
            # Plot for all data
            plt.figure(figsize=figsize)
            
            if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
                # For multi-class, we'll plot the first class by default
                values_to_plot = self.shap_values[0]
            else:
                values_to_plot = self.shap_values
            
            if X is not None:
                # Use SHAP's built-in dependence plot
                if interaction_feature:
                    shap.dependence_plot(
                        feature, 
                        values_to_plot, 
                        X, 
                        interaction_index=interaction_feature,
                        alpha=0.8,
                        max_display=max_points
                    )
                else:
                    shap.dependence_plot(
                        feature, 
                        values_to_plot, 
                        X,
                        alpha=0.8,
                        max_display=max_points
                    )
            else:
                # Without X, plot simpler version
                if isinstance(values_to_plot, list):
                    values_to_plot = values_to_plot[0]
                
                feature_idx = self.feature_names.index(feature)
                plt.scatter(
                    range(len(values_to_plot)), 
                    values_to_plot[:, feature_idx], 
                    alpha=0.8
                )
                plt.xlabel("Index")
                plt.ylabel(f"SHAP value for {feature}")
            
            plt.title(f"SHAP Dependence Plot for {feature}")
        
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()