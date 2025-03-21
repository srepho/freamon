"""
Module for LightGBM modeling capabilities in Freamon.

This module provides a high-level interface for training LightGBM models
with intelligent hyperparameter tuning, custom objectives, advanced callbacks,
and enhanced model inspection capabilities.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from freamon.modeling.model import Model
from freamon.modeling.factory import create_model
from freamon.modeling.tuning import LightGBMTuner

# Set up logger
logger = logging.getLogger(__name__)


class LightGBMModel:
    """
    High-level interface for LightGBM models with intelligent hyperparameter tuning.
    
    This class provides a streamlined workflow for training LightGBM models,
    including automatic hyperparameter tuning and proper handling of categorical features.
    
    Parameters
    ----------
    problem_type : Literal['classification', 'regression']
        The type of problem (classification or regression).
    objective : Optional[str], default=None
        LightGBM objective function. If None, it will be set automatically based on problem_type.
        For classification: 'binary' or 'multiclass'
        For regression: 'regression', 'mse', 'mae', etc.
    metric : Optional[str], default=None
        The metric to optimize. If None, it will be set automatically based on problem_type.
        For classification: 'auc', 'binary_logloss', 'multi_logloss', etc.
        For regression: 'rmse', 'mae', 'mape', etc.
    tuning_cv : int, default=5
        Number of cross-validation folds for hyperparameter tuning.
    tuning_trials : int, default=100
        Maximum number of trials for hyperparameter optimization.
    early_stopping_rounds : int, default=50
        Number of early stopping rounds during training.
    random_state : Optional[int], default=None
        Random state for reproducibility.
    probability_threshold : Optional[float], default=None
        Classification probability threshold for positive class prediction.
        If None, uses the default threshold of 0.5.
    
    Attributes
    ----------
    model : Optional[Model]
        The trained model, or None if not trained yet.
    tuner : LightGBMTuner
        The hyperparameter tuner.
    best_params : Optional[Dict[str, Any]]
        The best hyperparameters found during tuning.
    is_fitted : bool
        Whether the model has been fitted.
    feature_names : Optional[List[str]]
        The names of the features used for training.
    categorical_features : Optional[List[Union[int, str]]]
        The list of categorical features.
    probability_threshold : Optional[float]
        The probability threshold for binary classification.
    """
    
    def __init__(
        self,
        problem_type: Literal['classification', 'regression'],
        objective: Optional[Union[str, Any]] = None,
        metric: Optional[str] = None,
        tuning_cv: int = 5,
        tuning_trials: int = 100,
        early_stopping_rounds: int = 50,
        random_state: Optional[int] = None,
        custom_objective: Optional[Any] = None,
        custom_eval_metric: Optional[Any] = None,
        probability_threshold: Optional[float] = None,
    ):
        """Initialize the LightGBM model.
        
        Parameters
        ----------
        problem_type : Literal['classification', 'regression']
            The type of problem (classification or regression).
        objective : Optional[Union[str, Any]], default=None
            LightGBM objective function. If None, it will be set automatically based on problem_type.
            For classification: 'binary' or 'multiclass'
            For regression: 'regression', 'mse', 'mae', etc.
            Can be a custom objective function if custom_objective is not provided.
        metric : Optional[str], default=None
            The metric to optimize. If None, it will be set automatically based on problem_type.
            For classification: 'auc', 'binary_logloss', 'multi_logloss', etc.
            For regression: 'rmse', 'mae', 'mape', etc.
        tuning_cv : int, default=5
            Number of cross-validation folds for hyperparameter tuning.
        tuning_trials : int, default=100
            Maximum number of trials for hyperparameter optimization.
        early_stopping_rounds : int, default=50
            Number of early stopping rounds during training.
        random_state : Optional[int], default=None
            Random state for reproducibility.
        custom_objective : Optional[Any], default=None
            Custom objective function that takes (y_true, y_pred) and returns (grad, hess).
            If provided, it takes precedence over the objective parameter.
        custom_eval_metric : Optional[Any], default=None
            Custom evaluation metric that takes (y_true, y_pred) and returns (name, value, is_higher_better).
            If provided, it will be used in addition to the metric parameter.
        probability_threshold : Optional[float], default=None
            Classification probability threshold for positive class prediction.
            If None, uses the default threshold of 0.5.
        """
        self.problem_type = problem_type
        self.objective = objective
        self.metric = metric
        self.tuning_cv = tuning_cv
        self.tuning_trials = tuning_trials
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.custom_objective = custom_objective
        self.custom_eval_metric = custom_eval_metric
        self.probability_threshold = probability_threshold
        
        # If objective is a callable but custom_objective is None, use it as custom objective
        if callable(objective) and custom_objective is None:
            self.custom_objective = objective
            self.objective = None  # Reset objective to None
        
        # Initialize the tuner
        self.tuner = LightGBMTuner(
            problem_type=problem_type,
            objective=objective if not callable(objective) else None,
            metric=metric,
            cv=tuning_cv,
            cv_type='stratified' if problem_type == 'classification' else 'kfold',
            random_state=random_state,
            n_trials=tuning_trials,
            n_jobs=-1,  # Use all cores by default
        )
        
        # Initialize model attributes
        self.model = None
        self.best_params = None
        self.is_fitted = False
        self.feature_names = None
        self.categorical_features = None
        self.fixed_params = None  # Will store default params if not tuning
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        categorical_features: Optional[List[Union[int, str]]] = None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        validation_size: float = 0.2,
        tune_hyperparameters: bool = True,
        fixed_params: Optional[Dict[str, Any]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> 'LightGBMModel':
        """
        Fit the LightGBM model to the training data.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The training features.
        y : Union[pd.Series, np.ndarray]
            The training target.
        categorical_features : Optional[List[Union[int, str]]], default=None
            The list of categorical features (column indices or names).
        X_val : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            The validation features. If None, a validation set will be created from
            X and y using validation_size.
        y_val : Optional[Union[pd.Series, np.ndarray]], default=None
            The validation target. Required if X_val is provided.
        validation_size : float, default=0.2
            The proportion of the training data to use for validation if X_val is None.
        tune_hyperparameters : bool, default=True
            Whether to tune hyperparameters using the tuner.
        fixed_params : Optional[Dict[str, Any]], default=None
            Parameters to fix during tuning.
        fit_params : Optional[Dict[str, Any]], default=None
            Additional parameters to pass to the fit method.
        
        Returns
        -------
        LightGBMModel
            The fitted model.
        """
        # Store feature names if X is a dataframe
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Store categorical features
        self.categorical_features = categorical_features
        
        # Create validation set if not provided
        if X_val is None or y_val is None:
            logger.info(f"Creating validation set with {validation_size*100:.0f}% of training data")
            if self.problem_type == 'classification':
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_size, random_state=self.random_state, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_size, random_state=self.random_state
                )
        else:
            X_train, y_train = X, y
        
        # Tune hyperparameters if requested
        if tune_hyperparameters:
            logger.info("Tuning hyperparameters...")
            self.best_params = self.tuner.tune(
                X_train, y_train,
                categorical_features=categorical_features,
                fixed_params=fixed_params,
                progressive_tuning=True,
                early_stopping_rounds=self.early_stopping_rounds,
            )
            
            logger.info("Creating model with best parameters...")
            self.model = self.tuner.create_model()
        else:
            # Create model with default or fixed parameters
            logger.info("Creating model with default parameters...")
            if self.problem_type == 'classification':
                model_name = 'LGBMClassifier'
            else:
                model_name = 'LGBMRegressor'
            
            # Start with either fixed_params from method arg, or from instance, or empty dict
            params = fixed_params or self.fixed_params or {}
            
            if self.objective is not None:
                params['objective'] = self.objective
            if self.random_state is not None:
                params['random_state'] = self.random_state
            
            self.model = create_model('lightgbm', model_name, params, self.random_state)
        
        # Prepare fit parameters
        fit_params = fit_params or {}
        fit_params['eval_set'] = [(X_val, y_val)]
        fit_params['early_stopping_rounds'] = self.early_stopping_rounds
        fit_params['verbose'] = fit_params.get('verbose', False)
        
        # Add categorical features if provided
        if categorical_features is not None:
            fit_params['categorical_feature'] = categorical_features
        
        # Add custom objective and eval metric if provided
        if self.custom_objective:
            fit_params['obj'] = self.custom_objective
        
        if self.custom_eval_metric:
            # Since LightGBM's API distinguishes between a feval function and a 
            # custom eval metric name, we need to handle both cases
            if callable(self.custom_eval_metric):
                fit_params['eval_metric'] = self.custom_eval_metric
            else:
                fit_params['metric'] = self.custom_eval_metric
        
        # Fit the model
        logger.info("Fitting model...")
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        
        # Log evaluation metrics
        final_iteration = self.model.model.best_iteration_
        val_metric = self.model.model.best_score_
        logger.info(f"Model fitted with early stopping at iteration {final_iteration}")
        logger.info(f"Validation metrics: {val_metric}")
        
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        For classification problems, allows optional probability threshold customization.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The input features.
        threshold : Optional[float], default=None
            Custom probability threshold for binary classification. 
            If None, uses the instance's threshold if set, otherwise uses 0.5.
            Ignored for regression or multiclass problems.
        
        Returns
        -------
        np.ndarray
            The predicted values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # For regression, simply return the predictions
        if self.problem_type != 'classification':
            return self.model.predict(X)
        
        # For binary classification, apply threshold if needed
        probas = self.predict_proba(X)
        
        # Check if binary classification (2 classes)
        is_binary = probas.shape[1] == 2 if probas.ndim == 2 else True
        
        # Apply custom threshold only for binary classification
        if is_binary and (threshold is not None or self.probability_threshold is not None):
            # Use provided threshold, fallback to instance threshold, or default to 0.5
            active_threshold = threshold if threshold is not None else (
                self.probability_threshold if self.probability_threshold is not None else 0.5
            )
            
            # Get probabilities for positive class
            if probas.ndim == 2:
                positive_proba = probas[:, 1]
            else:
                positive_proba = probas
                
            # Apply threshold
            return (positive_proba >= active_threshold).astype(int)
        
        # For multiclass or no threshold customization, use standard predict
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
        
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only applicable for classification problems")
        
        return self.model.predict_proba(X)
    
    def find_optimal_threshold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metric: Union[str, Callable] = 'f1',
        thresholds: Optional[Union[int, List[float], np.ndarray]] = 100,
        set_as_default: bool = False
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Find the optimal probability threshold for binary classification.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The features to generate predictions on.
        y : Union[pd.Series, np.ndarray] 
            The true class labels.
        metric : Union[str, Callable], default='f1'
            The metric to optimize. See metrics.find_optimal_threshold for available options.
        thresholds : Optional[Union[int, List[float], np.ndarray]], default=100
            The thresholds to evaluate. If integer, generates that many thresholds.
        set_as_default : bool, default=False
            Whether to set the found optimal threshold as the model's default threshold.
        
        Returns
        -------
        Tuple[float, float, pd.DataFrame]
            A tuple containing:
            - The optimal threshold value
            - The score achieved at the optimal threshold
            - A DataFrame with threshold values and resulting metric scores
            
        Raises
        ------
        ValueError
            If the model is not fitted or not a binary classification model.
        
        Examples
        --------
        >>> # Find optimal threshold to maximize F1 score
        >>> threshold, score, results = model.find_optimal_threshold(X_val, y_val)
        >>> 
        >>> # Find optimal threshold to maximize precision and set as default
        >>> threshold, score, results = model.find_optimal_threshold(
        ...     X_val, y_val, metric='precision', set_as_default=True
        ... )
        >>> 
        >>> # Make predictions with the optimal threshold
        >>> y_pred = model.predict(X_test)  # Uses the optimal threshold automatically
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        if self.problem_type != 'classification':
            raise ValueError("Threshold optimization is only applicable for classification problems")
        
        # Get predicted probabilities
        y_proba = self.predict_proba(X)
        
        # Check if binary classification
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            # For binary classification, use the probability of the positive class
            y_proba_pos = y_proba[:, 1]
        elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
            raise ValueError("Threshold optimization is only applicable for binary classification")
        else:
            # If already 1D array, use as is
            y_proba_pos = y_proba
        
        # Import the threshold optimization function
        from freamon.modeling.metrics import find_optimal_threshold
        
        # Find optimal threshold
        optimal_threshold, optimal_score, results_df = find_optimal_threshold(
            y_true=y,
            y_proba=y_proba_pos,
            metric=metric,
            thresholds=thresholds
        )
        
        # Set as default if requested
        if set_as_default:
            self.probability_threshold = optimal_threshold
            logger.info(f"Optimal threshold of {optimal_threshold:.4f} set as default")
        
        return optimal_threshold, optimal_score, results_df
    
    def get_feature_importance(
        self, 
        method: str = 'native', 
        X: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Get the feature importance from the model.
        
        Parameters
        ----------
        method : str, default='native'
            The method to use for computing feature importance.
            Options: 'native', 'shap', 'shapiq', 'permutation'
        X : Optional[pd.DataFrame], default=None
            The data to use for computing SHAP or permutation values.
            Required if method is 'shap', 'shapiq', or 'permutation'.
        
        Returns
        -------
        pd.Series
            A Series mapping feature names to importance values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        if method == 'permutation':
            if X is None:
                raise ValueError("X must be provided when method='permutation'")
            
            try:
                from sklearn.inspection import permutation_importance
                import pandas as pd
                
                y_pred = self.model.predict(X)
                result = permutation_importance(
                    self.model.model, X, y_pred, 
                    n_repeats=10, 
                    random_state=self.random_state
                )
                
                importance = pd.Series(
                    result.importances_mean, 
                    index=X.columns if isinstance(X, pd.DataFrame) else None
                )
                return importance.sort_values(ascending=False)
            
            except ImportError:
                logger.warning("scikit-learn>=0.22 is required for permutation_importance")
                logger.warning("Falling back to native importance")
                method = 'native'
        
        # Use the model's feature importance for other methods
        return self.model.get_feature_importance(method=method, X=X)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        import joblib
        
        # Create a dictionary with all the necessary information
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'problem_type': self.problem_type,
            'objective': self.objective,
            'metric': self.metric,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'is_fitted': self.is_fitted,
            'probability_threshold': self.probability_threshold,
        }
        
        # Save the model data
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        Load a model from disk.
        
        Parameters
        ----------
        path : str
            The path to load the model from.
        
        Returns
        -------
        LightGBMModel
            The loaded model.
        """
        import joblib
        
        # Load the model data
        model_data = joblib.load(path)
        
        # Create a new model instance
        model = cls(
            problem_type=model_data['problem_type'],
            objective=model_data['objective'],
            metric=model_data['metric'],
            probability_threshold=model_data.get('probability_threshold'),  # Backward compatibility
        )
        
        # Restore the model state
        model.model = model_data['model']
        model.best_params = model_data['best_params']
        model.feature_names = model_data['feature_names']
        model.categorical_features = model_data['categorical_features']
        model.is_fitted = model_data['is_fitted']
        
        return model
    
    def __repr__(self) -> str:
        """Return a string representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        problem = self.problem_type
        obj = f"objective='{self.objective}'" if self.objective else ""
        params = f"params={len(self.best_params) if self.best_params else 'None'}"
        threshold = f"threshold={self.probability_threshold}" if self.probability_threshold is not None else ""
        
        components = [status, problem, obj, params, threshold]
        # Filter out empty strings
        components = [c for c in components if c]
        
        return f"LightGBMModel({', '.join(components)})"