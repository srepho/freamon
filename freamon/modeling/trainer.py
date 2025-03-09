"""
Module for training and evaluating models.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from freamon.modeling.factory import create_model
from freamon.modeling.metrics import calculate_metrics
from freamon.modeling.model import Model


class ModelTrainer:
    """
    Class for training and evaluating models.
    
    Parameters
    ----------
    model_type : Literal['sklearn', 'lightgbm', 'xgboost', 'catboost']
        The type of model to train.
    model_name : str
        The name of the model within the specified library.
    problem_type : Literal['classification', 'regression']
        The type of problem (classification or regression).
    params : Optional[Dict[str, Any]], default=None
        Parameters to pass to the model constructor.
    random_state : Optional[int], default=None
        Random state to use for reproducibility.
    """
    
    def __init__(
        self,
        model_type: Literal['sklearn', 'lightgbm', 'xgboost', 'catboost'],
        model_name: str,
        problem_type: Literal['classification', 'regression'],
        params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the model trainer."""
        self.model_type = model_type
        self.model_name = model_name
        self.problem_type = problem_type
        self.params = params or {}
        self.random_state = random_state
        
        # Create the model
        self.model = create_model(
            model_type=model_type,
            model_name=model_name,
            params=params,
            random_state=random_state,
        )
        
        # Initialize training history
        self.history: Dict[str, List[float]] = {}
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model on the given data.
        
        Parameters
        ----------
        X_train : Union[pd.DataFrame, np.ndarray]
            The training features.
        y_train : Union[pd.Series, np.ndarray]
            The training target values.
        X_val : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            The validation features. If provided, used for early stopping.
        y_val : Optional[Union[pd.Series, np.ndarray]], default=None
            The validation target values. Required if X_val is provided.
        **kwargs
            Additional keyword arguments to pass to the model's fit method.
        
        Returns
        -------
        Dict[str, float]
            A dictionary of validation metrics if validation data is provided,
            otherwise training metrics.
        """
        # Check if validation data is provided
        if X_val is not None and y_val is None:
            raise ValueError("y_val must be provided if X_val is provided")
        
        # Prepare validation data for early stopping if provided
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        else:
            eval_set = None
        
        # Train the model
        self.model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
        
        # Calculate metrics
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
        else:
            y_pred = self.predict(X_train)
            metrics = self._calculate_metrics(y_train, y_pred)
        
        # Update history
        for metric, value in metrics.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)
        
        return metrics
    
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
        if self.problem_type != 'classification':
            raise ValueError(
                "predict_proba is only applicable for classification problems"
            )
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features.
        y : Union[pd.Series, np.ndarray]
            The true target values.
        
        Returns
        -------
        Dict[str, float]
            A dictionary of performance metrics.
        """
        # Generate predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        return self._calculate_metrics(y, y_pred)
    
    def _calculate_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        y_true : Union[pd.Series, np.ndarray]
            The true target values.
        y_pred : Union[pd.Series, np.ndarray]
            The predicted target values.
        
        Returns
        -------
        Dict[str, float]
            A dictionary of performance metrics.
        """
        # For classification, get predicted probabilities if available
        y_proba = None
        if self.problem_type == 'classification':
            try:
                y_proba = self.predict_proba(X=None)  # X=None as a placeholder
            except (ValueError, AttributeError):
                pass
        
        # Calculate metrics
        return calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            problem_type=self.problem_type,
            y_proba=y_proba,
        )
    
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
        return self.model.get_feature_importance(method=method, X=X)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        self.model.save(path)
    
    @classmethod
    def load(
        cls,
        path: str,
        problem_type: Literal['classification', 'regression'],
    ) -> 'ModelTrainer':
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            The path to load the model from.
        problem_type : Literal['classification', 'regression']
            The type of problem (classification or regression).
        
        Returns
        -------
        ModelTrainer
            The loaded model trainer.
        """
        # Load the model
        model = Model.load(path)
        
        # Create a new trainer
        trainer = cls(
            model_type=model.model_type,
            model_name=type(model.model).__name__,
            problem_type=problem_type,
            params=model.params,
        )
        
        # Replace the model
        trainer.model = model
        
        return trainer