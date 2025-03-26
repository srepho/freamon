"""
Base model class for Freamon.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import lightgbm

from freamon.explainability.shap_explainer import ShapExplainer, ShapIQExplainer


class Model:
    """
    Base class for all models in Freamon.
    
    This class wraps various model implementations (scikit-learn, LightGBM,
    XGBoost, CatBoost) and provides a consistent interface for training,
    prediction, and evaluation.
    
    Parameters
    ----------
    model : Any
        The underlying model object.
    model_type : str
        The type of model ('sklearn', 'lightgbm', 'xgboost', 'catboost').
    params : Dict[str, Any]
        The parameters used to initialize the model.
    feature_names : Optional[List[str]]
        The names of the features used by the model.
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str,
        params: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize the model wrapper."""
        self.model = model
        self.model_type = model_type
        self.params = params
        self.feature_names = feature_names
        self.is_fitted = False
        
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        This method is required by scikit-learn for cloning estimators.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters of all sub-objects that are estimators.
            
        Returns
        -------
        Dict[str, Any]
            Parameter names mapped to their values.
        """
        params = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names
        }
        
        # If deep=True, also include parameters from the underlying model
        if deep and hasattr(self.model, 'get_params'):
            model_params = self.model.get_params(deep=deep)
            # Prefix with 'model__' to avoid name conflicts
            model_params = {f'model__{key}': val for key, val in model_params.items()}
            params.update(model_params)
            
        return params
        
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        This method is required by scikit-learn for cloning estimators.
        
        Parameters
        ----------
        **params : Dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        valid_params = self.get_params(deep=False)
        
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            elif key.startswith('model__') and hasattr(self.model, 'set_params'):
                # Handle nested parameters for the underlying model
                model_key = key[7:]  # Remove 'model__' prefix
                self.model.set_params(**{model_key: value})
            else:
                raise ValueError(f'Invalid parameter {key} for estimator {self.__class__.__name__}')
                
        return self
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> 'Model':
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The training features.
        y : Union[pd.Series, np.ndarray]
            The target values.
        **kwargs
            Additional keyword arguments to pass to the underlying model's fit method.
            These may include validation data, sample weights, etc.
        
        Returns
        -------
        Model
            The fitted model.
        """
        # Store feature names if X is a dataframe
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Extract validation data if provided
        eval_set = kwargs.pop('eval_set', None)
        
        # Fit the model
        if self.model_type == 'sklearn':
            self.model.fit(X, y, **kwargs)
        
        elif self.model_type == 'lightgbm':
            # For LightGBM, we need to handle eval_set and early_stopping_rounds differently
            # LGBMClassifier.fit() doesn't accept early_stopping_rounds directly
            early_stopping_rounds = kwargs.pop('early_stopping_rounds', None)
            
            # Handle eval_set
            if eval_set is not None:
                # Check if eval_set is already a list of tuples
                if isinstance(eval_set, list) and all(isinstance(x, tuple) for x in eval_set):
                    if early_stopping_rounds is not None:
                        self.model.fit(X, y, eval_set=eval_set, callbacks=[
                            lightgbm.early_stopping(stopping_rounds=early_stopping_rounds)
                        ], **kwargs)
                    else:
                        self.model.fit(X, y, eval_set=eval_set, **kwargs)
                else:
                    # Assume it's a single validation set tuple
                    X_val, y_val = eval_set
                    if early_stopping_rounds is not None:
                        self.model.fit(
                            X, y,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lightgbm.early_stopping(stopping_rounds=early_stopping_rounds)],
                            **kwargs
                        )
                    else:
                        self.model.fit(
                            X, y,
                            eval_set=[(X_val, y_val)],
                            **kwargs
                        )
            else:
                # Without eval_set, early stopping can't be used
                self.model.fit(X, y, **kwargs)
        
        elif self.model_type == 'xgboost':
            # For XGBoost, we can pass eval_set directly
            self.model.fit(X, y, eval_set=eval_set, **kwargs)
        
        elif self.model_type == 'catboost':
            # For CatBoost, we need to handle eval_set differently
            if eval_set is not None:
                X_val, y_val = eval_set
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    **kwargs
                )
            else:
                self.model.fit(X, y, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features.
        
        Returns
        -------
        np.ndarray
            The predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Check feature names if X is a dataframe and feature_names is set
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                # Instead of failing, add the missing features with zeros
                import warnings
                warnings.warn(f"Input is missing features during prediction: {missing_features}. "
                             f"These will be filled with zeros.", UserWarning)
                
                # Create missing columns with zeros
                for feature in missing_features:
                    X[feature] = 0.0
            
            # Ensure all needed features exist and are in the right order
            feature_df = pd.DataFrame(index=X.index)
            for feature in self.feature_names:
                if feature in X.columns:
                    feature_df[feature] = X[feature]
                else:
                    feature_df[feature] = 0.0
            
            X = feature_df
        
        return self.model.predict(X)
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Generate class probabilities for the input data.
        
        Only applicable for classification models.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features.
        
        Returns
        -------
        np.ndarray
            The predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Check feature names if X is a dataframe and feature_names is set
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                # Instead of failing, add the missing features with zeros
                import warnings
                warnings.warn(f"Input is missing features during prediction_proba: {missing_features}. "
                             f"These will be filled with zeros.", UserWarning)
                
                # Create missing columns with zeros
                for feature in missing_features:
                    X[feature] = 0.0
            
            # Ensure all needed features exist and are in the right order
            feature_df = pd.DataFrame(index=X.index)
            for feature in self.feature_names:
                if feature in X.columns:
                    feature_df[feature] = X[feature]
                else:
                    feature_df[feature] = 0.0
            
            X = feature_df
        
        # Check if the model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(
                f"Model of type {self.model_type} does not support predict_proba"
            )
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, method: str = 'native', X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Get the feature importance from the model.
        
        Parameters
        ----------
        method : str, default='native'
            The method to use for computing feature importance.
            Options: 'native', 'shap', 'shapiq'
        X : Optional[pd.DataFrame], default=None
            The data to use for computing SHAP values.
            Required if method is 'shap' or 'shapiq'.
        
        Returns
        -------
        pd.Series
            A Series mapping feature names to importance values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Using SHAP to compute feature importance
        if method == 'shap':
            if X is None:
                raise ValueError("X must be provided when method='shap'")
            
            try:
                # Create and fit SHAP explainer
                model_type = 'tree' if self.model_type in ['lightgbm', 'xgboost', 'catboost'] else 'kernel'
                explainer = ShapExplainer(self.model, model_type=model_type)
                explainer.fit(X)
                
                # Compute SHAP values
                shap_values = explainer.explain(X)
                
                # Calculate global feature importance (mean absolute SHAP value)
                if isinstance(shap_values, pd.DataFrame):
                    # For single output models
                    importance = shap_values.abs().mean()
                else:
                    # For multi-class models
                    importance_list = []
                    for class_idx, df in enumerate(shap_values):
                        if isinstance(df, pd.DataFrame):
                            class_df = df[df['_class'] == class_idx].drop('_class', axis=1)
                            importance_list.append(class_df.abs().mean())
                    importance = pd.concat(importance_list).groupby(level=0).mean()
                
                return importance.sort_values(ascending=False)
            
            except ImportError:
                print("Warning: shap package not available. Falling back to native importance.")
                return self._get_native_importance()
        
        # Using ShapIQ to compute feature importance with interactions
        elif method == 'shapiq':
            if X is None:
                raise ValueError("X must be provided when method='shapiq'")
            
            try:
                # Create and fit ShapIQ explainer
                explainer = ShapIQExplainer(self.model, max_order=2)
                explainer.fit(X)
                
                # Compute interaction values (using a small subset to be more efficient)
                sample_size = min(100, len(X))  # Use at most 100 samples for efficiency
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                sample_X = X.iloc[sample_indices]
                
                interactions = explainer.explain(sample_X)
                
                # Extract first-order effects (main effects)
                main_effects = interactions.get_order(1)
                
                # Calculate global feature importance (mean absolute main effect)
                importance_values = np.abs(main_effects.values).mean(axis=0)
                
                # Create Series with feature names
                importance = pd.Series(importance_values, index=X.columns)
                
                return importance.sort_values(ascending=False)
            
            except ImportError:
                print("Warning: shapiq package not available. Falling back to native importance.")
                return self._get_native_importance()
        
        # Using native feature importance from the model
        elif method == 'native':
            return self._get_native_importance()
        
        else:
            raise ValueError(f"Unknown method: {method}. Options: 'native', 'shap', 'shapiq'")
    
    def _get_native_importance(self) -> pd.Series:
        """Get the native feature importance from the model."""
        # Different models have different feature importance attributes
        if self.model_type == 'sklearn':
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_).flatten()
            else:
                raise AttributeError(
                    f"Model of type {type(self.model).__name__} does not provide feature importances"
                )
        
        elif self.model_type == 'lightgbm':
            importances = self.model.feature_importances_
        
        elif self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        
        elif self.model_type == 'catboost':
            importances = self.model.feature_importances_
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create a Series with feature names and importances
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Ensure the number of feature names matches the number of importance values
        if len(feature_names) != len(importances):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) does not match "
                f"number of importance values ({len(importances)})"
            )
        
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        import joblib
        
        # Create a dictionary with all the necessary information
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
        }
        
        # Save the model data
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'Model':
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            The path to load the model from.
        
        Returns
        -------
        Model
            The loaded model.
        """
        import joblib
        
        # Load the model data
        model_data = joblib.load(path)
        
        # Create a new model instance
        model = cls(
            model=model_data['model'],
            model_type=model_data['model_type'],
            params=model_data['params'],
            feature_names=model_data['feature_names'],
        )
        
        # Set the fitted flag
        model.is_fitted = model_data['is_fitted']
        
        return model