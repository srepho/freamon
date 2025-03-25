"""
Cross-validation training step for pipelines.

This module provides a PipelineStep implementation for cross-validated model training
within the pipeline framework.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from freamon.pipeline.steps import PipelineStep
from freamon.model_selection.cv_trainer import CrossValidationTrainer


class CrossValidationTrainingStep(PipelineStep):
    """
    Pipeline step that trains a model using cross-validation.
    
    This step extends PipelineStep to provide cross-validated model training
    within the pipeline framework. It supports different cross-validation
    strategies and ensemble methods.
    
    Attributes
    ----------
    model_type : str
        The type of model to train
    problem_type : str
        The type of problem
    cv_trainer : CrossValidationTrainer
        The underlying cross-validated trainer
    eval_metric : Optional[str]
        The evaluation metric
    """
    
    def __init__(
        self,
        name: str,
        model_type: str,
        problem_type: str = "classification",
        cv_strategy: str = "kfold",
        n_splits: int = 5,
        ensemble_method: str = "best",
        hyperparameters: Optional[Dict[str, Any]] = None,
        eval_metric: Optional[str] = None,
        early_stopping_rounds: Optional[int] = None,
        random_state: int = 42,
        **cv_kwargs
    ):
        """
        Initialize cross-validation training step.
        
        Parameters
        ----------
        name : str
            Name of the step
        model_type : str
            The type of model to train ('lightgbm', 'xgboost', etc.)
        problem_type : str, default='classification'
            The type of problem ('classification' or 'regression')
        cv_strategy : str, default='kfold'
            The cross-validation strategy to use.
            Options: 'kfold', 'stratified', 'timeseries', 'walk_forward'
        n_splits : int, default=5
            Number of cross-validation splits
        ensemble_method : str, default='best'
            Method for combining fold models.
            Options: 'best', 'average', 'weighted', 'stacking'
        hyperparameters : Optional[Dict[str, Any]], default=None
            Model hyperparameters
        eval_metric : Optional[str], default=None
            Evaluation metric to optimize
        early_stopping_rounds : Optional[int], default=None
            Number of rounds for early stopping
        random_state : int, default=42
            Random seed for reproducibility
        **cv_kwargs : Dict[str, Any]
            Additional kwargs for cross-validation
        """
        super().__init__(name)
        self.model_type = model_type
        self.problem_type = problem_type
        self.hyperparameters = hyperparameters or {}
        self.eval_metric = eval_metric
        self.random_state = random_state
        
        # Initialize cross-validated trainer
        self.cv_trainer = CrossValidatedTrainer(
            model_type=model_type,
            problem_type=problem_type,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            ensemble_method=ensemble_method,
            hyperparameters=hyperparameters,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            **cv_kwargs
        )
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        **kwargs
    ) -> "CrossValidationTrainingStep":
        """
        Fit the cross-validation training step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series], default=None
            Target series
        **kwargs : Dict[str, Any]
            Additional parameters
            
        Returns
        -------
        CrossValidationTrainingStep
            The fitted step
            
        Raises
        ------
        ValueError
            If target is not provided
        """
        if y is None:
            raise ValueError("Target variable y is required for model training")
        
        # Train model with cross-validation
        self.cv_trainer.fit(X, y, **kwargs)
        
        self._is_fitted = True
        return self
    
    def transform(
        self, 
        X: pd.DataFrame, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Transform method for the cross-validation step.
        
        For CrossValidationTrainingStep, transform returns the dataframe unchanged.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        **kwargs : Dict[str, Any]
            Additional parameters
            
        Returns
        -------
        pd.DataFrame
            Input dataframe unchanged
        """
        return X
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        **kwargs
    ) -> pd.DataFrame:
        """
        Fit and transform the cross-validation training step.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        y : Optional[pd.Series], default=None
            Target series
        **kwargs : Dict[str, Any]
            Additional parameters
            
        Returns
        -------
        pd.DataFrame
            Input dataframe unchanged
        """
        self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)
    
    def predict(
        self, 
        X: pd.DataFrame, 
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        **kwargs : Dict[str, Any]
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Predictions
            
        Raises
        ------
        ValueError
            If the step is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Step has not been fitted yet")
        
        return self.cv_trainer.predict(X)
    
    def predict_proba(
        self, 
        X: pd.DataFrame, 
        **kwargs
    ) -> np.ndarray:
        """
        Make probability predictions with the trained model.
        
        Only applicable for classification problems.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe
        **kwargs : Dict[str, Any]
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Probability predictions
            
        Raises
        ------
        ValueError
            If the step is not fitted or not a classifier
        """
        if not self._is_fitted:
            raise ValueError("Step has not been fitted yet")
        
        return self.cv_trainer.predict_proba(X)
    
    def get_cv_results(self) -> Dict[str, List[float]]:
        """
        Get cross-validation results.
        
        Returns
        -------
        Dict[str, List[float]]
            Cross-validation metrics for each fold
            
        Raises
        ------
        ValueError
            If the step is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Step has not been fitted yet")
        
        return self.cv_trainer.get_cv_results()
    
    def get_feature_importances(self) -> pd.Series:
        """
        Get feature importances from the trained model.
        
        Returns
        -------
        pd.Series
            Feature importances
            
        Raises
        ------
        ValueError
            If the step is not fitted or feature importances are not available
        """
        if not self._is_fitted:
            raise ValueError("Step has not been fitted yet")
        
        return self.cv_trainer.get_feature_importances()
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get the trained models.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with ensemble model and fold models
            
        Raises
        ------
        ValueError
            If the step is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Step has not been fitted yet")
        
        return self.cv_trainer.get_models()