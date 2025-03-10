"""
Cross-validated model training module.

This module provides classes and functions for training models with 
cross-validation as the standard approach.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor

from freamon.modeling.trainer import ModelTrainer
from freamon.modeling.factory import create_model
from freamon.modeling.metrics import calculate_metrics
from freamon.model_selection.cross_validation import (
    cross_validate, 
    time_series_cross_validate,
    walk_forward_validation
)


class CrossValidatedTrainer:
    """
    A trainer that uses cross-validation as the standard mechanism for model training.
    
    This trainer extends the functionality of ModelTrainer by incorporating
    cross-validation into the training process and providing options for
    different ensemble methods to combine models from different folds.
    
    Attributes
    ----------
    model_type : str
        The type of model to train ('lightgbm', 'xgboost', etc.)
    problem_type : str
        The type of problem ('classification' or 'regression')
    cv_strategy : str
        The cross-validation strategy to use
    n_splits : int
        Number of cross-validation splits
    ensemble_method : str
        Method for combining fold models
    hyperparameters : Dict[str, Any]
        Model hyperparameters
    random_state : int
        Random seed for reproducibility
    fold_metrics : Dict[str, List[float]]
        Metrics for each cross-validation fold
    fold_models : List[Dict[str, Any]]
        Models trained on each fold
    ensemble_model : Any
        The final ensemble model
    feature_importances : pd.Series
        Feature importance scores
    """
    
    def __init__(
        self,
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
        Initialize cross-validated trainer.
        
        Parameters
        ----------
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
            
        Examples
        --------
        >>> from freamon.model_selection.cv_trainer import CrossValidatedTrainer
        >>> # Create trainer for classification with LightGBM
        >>> trainer = CrossValidatedTrainer(
        ...     model_type="lightgbm",
        ...     problem_type="classification",
        ...     n_splits=5,
        ...     ensemble_method="average",
        ... )
        >>> # Fit model with cross-validation
        >>> trainer.fit(X_train, y_train)
        >>> # Make predictions
        >>> y_pred = trainer.predict(X_test)
        >>> # Get cross-validation results
        >>> cv_results = trainer.get_cv_results()
        """
        self.model_type = model_type
        self.problem_type = problem_type
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.ensemble_method = ensemble_method
        self.hyperparameters = hyperparameters or {}
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.cv_kwargs = cv_kwargs
        
        # Will be populated during training
        self.fold_metrics = {}
        self.fold_models = []
        self.ensemble_model = None
        self.feature_importances = None
        self._is_fitted = False
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray], 
        **kwargs
    ) -> "CrossValidatedTrainer":
        """
        Fit the model using cross-validation.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
        y : Union[pd.Series, np.ndarray]
            Target vector
        **kwargs : Dict[str, Any]
            Additional kwargs for model training
            
        Returns
        -------
        CrossValidatedTrainer
            The fitted trainer
            
        Raises
        ------
        ValueError
            If invalid parameters are provided
        """
        # Convert to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")
            
        # Create combined DataFrame for CV
        df = X.copy()
        target_name = y.name if hasattr(y, "name") else "target"
        df[target_name] = y
        
        # Handle different CV strategies
        if self.cv_strategy == "kfold":
            self._perform_kfold_cv(df, target_name, **kwargs)
        elif self.cv_strategy == "stratified":
            self._perform_stratified_cv(df, target_name, **kwargs)
        elif self.cv_strategy == "timeseries":
            self._perform_timeseries_cv(df, target_name, **kwargs)
        elif self.cv_strategy == "walk_forward":
            self._perform_walk_forward_cv(df, target_name, **kwargs)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
        
        # Create ensemble model
        self._create_ensemble_model(X, y)
        
        self._is_fitted = True
        return self
    
    def _perform_kfold_cv(self, df: pd.DataFrame, target_name: str, **kwargs):
        """
        Perform k-fold cross-validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target
        target_name : str
            Name of the target column
        **kwargs : Dict[str, Any]
            Additional kwargs for cross-validation
        """
        # Create a function to initialize models with the same parameters
        def model_fn(**inner_kwargs):
            params = {**self.hyperparameters, **inner_kwargs}
            return self._create_model(params)
        
        # Run cross-validation
        self.fold_metrics = cross_validate(
            df=df,
            target_column=target_name,
            model_fn=model_fn,
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
            problem_type=self.problem_type,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.cv_kwargs
        )
        
        # Collect fold models if needed for ensemble
        if self.ensemble_method != "best":
            self._train_fold_models(df, target_name)
    
    def _perform_stratified_cv(self, df: pd.DataFrame, target_name: str, **kwargs):
        """
        Perform stratified k-fold cross-validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target
        target_name : str
            Name of the target column
        **kwargs : Dict[str, Any]
            Additional kwargs for cross-validation
        """
        # Create a function to initialize models with the same parameters
        def model_fn(**inner_kwargs):
            params = {**self.hyperparameters, **inner_kwargs}
            return self._create_model(params)
        
        # Run cross-validation with stratification
        stratify_by = self.cv_kwargs.get("stratify_by", target_name)
        
        self.fold_metrics = cross_validate(
            df=df,
            target_column=target_name,
            model_fn=model_fn,
            n_splits=self.n_splits,
            shuffle=True,
            stratify_by=stratify_by,
            random_state=self.random_state,
            problem_type=self.problem_type,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.cv_kwargs
        )
        
        # Collect fold models if needed for ensemble
        if self.ensemble_method != "best":
            self._train_fold_models(df, target_name, stratify_by=stratify_by)
    
    def _perform_timeseries_cv(self, df: pd.DataFrame, target_name: str, **kwargs):
        """
        Perform time series cross-validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target
        target_name : str
            Name of the target column
        **kwargs : Dict[str, Any]
            Additional kwargs for cross-validation
        """
        # Get date column
        date_column = self.cv_kwargs.get("date_column")
        if date_column is None:
            raise ValueError("date_column is required for time series CV")
        
        # Create a function to initialize models with the same parameters
        def model_fn(**inner_kwargs):
            params = {**self.hyperparameters, **inner_kwargs}
            return self._create_model(params)
        
        # Run time series cross-validation
        self.fold_metrics = time_series_cross_validate(
            df=df,
            target_column=target_name,
            date_column=date_column,
            model_fn=model_fn,
            n_splits=self.n_splits,
            problem_type=self.problem_type,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.cv_kwargs
        )
        
        # Collect fold models if needed for ensemble
        if self.ensemble_method != "best":
            self._train_fold_models(df, target_name, is_timeseries=True)
    
    def _perform_walk_forward_cv(self, df: pd.DataFrame, target_name: str, **kwargs):
        """
        Perform walk-forward validation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target
        target_name : str
            Name of the target column
        **kwargs : Dict[str, Any]
            Additional kwargs for cross-validation
        """
        # Get date column
        date_column = self.cv_kwargs.get("date_column")
        if date_column is None:
            raise ValueError("date_column is required for walk-forward CV")
        
        # Create a function to initialize models with the same parameters
        def model_fn(**inner_kwargs):
            params = {**self.hyperparameters, **inner_kwargs}
            return self._create_model(params)
        
        # Run walk-forward validation
        self.fold_metrics = walk_forward_validation(
            df=df,
            target_column=target_name,
            date_column=date_column,
            model_fn=model_fn,
            n_splits=self.n_splits,
            problem_type=self.problem_type,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.cv_kwargs
        )
        
        # Collect fold models if needed for ensemble
        if self.ensemble_method != "best":
            self._train_fold_models(df, target_name, is_timeseries=True)
    
    def _train_fold_models(
        self, 
        df: pd.DataFrame, 
        target_name: str, 
        stratify_by: Optional[str] = None,
        is_timeseries: bool = False
    ):
        """
        Train models on each fold for ensembling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame including target
        target_name : str
            Name of the target column
        stratify_by : Optional[str], default=None
            Column to stratify by for stratified CV
        is_timeseries : bool, default=False
            Whether to use time series splitting
        """
        # Create CV splitter based on strategy
        if is_timeseries:
            from sklearn.model_selection import TimeSeriesSplit
            splitter = TimeSeriesSplit(n_splits=self.n_splits)
            split_args = (df.index,)
        elif self.cv_strategy == "stratified":
            splitter = StratifiedKFold(
                n_splits=self.n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
            split_args = (df.index, df[stratify_by])
        else:
            splitter = KFold(
                n_splits=self.n_splits, 
                shuffle=True, 
                random_state=self.random_state
            )
            split_args = (df.index,)
        
        # Clear existing models
        self.fold_models = []
        
        # Train a model for each fold
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
            # Split data
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            X_train = train_df.drop(columns=[target_name])
            y_train = train_df[target_name]
            X_test = test_df.drop(columns=[target_name])
            y_test = test_df[target_name]
            
            # Create and train model
            model = self._create_model(self.hyperparameters)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = {}
            
            # Predict based on problem type
            if self.problem_type == "classification":
                y_pred = model.predict(X_test)
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)
                    metrics = calculate_metrics(
                        y_true=y_test, 
                        y_pred=y_pred, 
                        y_prob=y_prob, 
                        problem_type="classification"
                    )
                else:
                    metrics = calculate_metrics(
                        y_true=y_test, 
                        y_pred=y_pred, 
                        problem_type="classification"
                    )
            else:
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(
                    y_true=y_test, 
                    y_pred=y_pred, 
                    problem_type="regression"
                )
            
            # Save model
            self.fold_models.append({
                'fold': fold_idx,
                'model': model,
                'train_indices': train_idx,
                'test_indices': test_idx,
                'metrics': metrics
            })
    
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """
        Create a model instance with the given parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Model parameters
            
        Returns
        -------
        Any
            The model instance
        """
        # Create ModelTrainer
        trainer = ModelTrainer(
            model_type=self.model_type,
            problem_type=self.problem_type,
            hyperparameters=params,
            random_state=self.random_state
        )
        
        return trainer.model
    
    def _get_best_fold_index(self) -> int:
        """
        Find the index of the best performing fold.
        
        Returns
        -------
        int
            The index of the best fold
        """
        # Determine the metric to optimize
        metric = self.eval_metric or ('accuracy' if self.problem_type == 'classification' else 'r2')
        
        # Check if metric exists in fold_metrics
        if metric not in self.fold_metrics:
            # Fall back to a default metric
            metric = next(iter(self.fold_metrics))
        
        # Find the best fold (maximize or minimize depending on metric)
        minimize_metrics = {'mse', 'rmse', 'mae', 'log_loss', 'error'}
        
        if any(m in metric.lower() for m in minimize_metrics):
            # Lower is better
            best_idx = np.argmin(self.fold_metrics[metric])
        else:
            # Higher is better
            best_idx = np.argmax(self.fold_metrics[metric])
        
        return best_idx
    
    def _calculate_fold_weights(self) -> List[float]:
        """
        Calculate weights for each fold based on performance.
        
        Returns
        -------
        List[float]
            The weights for each fold
        """
        # Determine the metric to optimize
        metric = self.eval_metric or ('accuracy' if self.problem_type == 'classification' else 'r2')
        
        # Check if metric exists in fold_metrics
        if metric not in self.fold_metrics:
            # Fall back to a default metric
            metric = next(iter(self.fold_metrics))
        
        # Extract metric values
        values = np.array(self.fold_metrics[metric])
        
        # Adjust weights based on whether higher or lower is better
        minimize_metrics = {'mse', 'rmse', 'mae', 'log_loss', 'error'}
        
        if any(m in metric.lower() for m in minimize_metrics):
            # Lower is better, invert values
            if np.all(values > 0):
                weights = 1.0 / values
            else:
                # Handle zero or negative values
                weights = np.max(values) - values + 1e-10
        else:
            # Higher is better
            weights = values
            
            # Handle negative values
            if np.any(weights < 0):
                weights = weights - np.min(weights) + 1e-10
        
        # Normalize weights to sum to 1
        return weights / np.sum(weights)
    
    def _create_ensemble_model(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray]
    ):
        """
        Create an ensemble model from fold models.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : Union[pd.Series, np.ndarray]
            Target vector
        """
        if self.ensemble_method == "best":
            # Train a new model on the whole dataset using the best hyperparameters
            model = self._create_model(self.hyperparameters)
            model.fit(X, y)
            self.ensemble_model = model
        elif self.ensemble_method == "average":
            # Use VotingClassifier or VotingRegressor
            models = [model_info['model'] for model_info in self.fold_models]
            
            if self.problem_type == "classification":
                self.ensemble_model = VotingClassifier(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)],
                    voting="soft" if hasattr(models[0], "predict_proba") else "hard"
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)]
                )
                
            # Fit the voting ensemble (necessary to make predictions)
            self.ensemble_model.fit(X, y)
        elif self.ensemble_method == "weighted":
            # Use VotingClassifier or VotingRegressor with weights
            models = [model_info['model'] for model_info in self.fold_models]
            weights = self._calculate_fold_weights()
            
            if self.problem_type == "classification":
                self.ensemble_model = VotingClassifier(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)],
                    voting="soft" if hasattr(models[0], "predict_proba") else "hard",
                    weights=weights
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)],
                    weights=weights
                )
                
            # Fit the weighted voting ensemble
            self.ensemble_model.fit(X, y)
        elif self.ensemble_method == "stacking":
            # Use StackingClassifier or StackingRegressor
            models = [model_info['model'] for model_info in self.fold_models]
            
            # Create a simple meta-model
            final_model = self._create_model(self.hyperparameters)
            
            if self.problem_type == "classification":
                self.ensemble_model = StackingClassifier(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)],
                    final_estimator=final_model,
                    cv=5
                )
            else:
                self.ensemble_model = StackingRegressor(
                    estimators=[(f"fold_{i}", model) for i, model in enumerate(models)],
                    final_estimator=final_model,
                    cv=5
                )
                
            # Fit the stacking ensemble
            self.ensemble_model.fit(X, y)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Calculate feature importances
        self._calculate_feature_importances(X)
    
    def _calculate_feature_importances(self, X: pd.DataFrame):
        """
        Calculate feature importances from models.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        """
        # Try different approaches to get feature importances
        if hasattr(self.ensemble_model, "feature_importances_"):
            # Direct attribute access (e.g., for RandomForest)
            importances = self.ensemble_model.feature_importances_
            self.feature_importances = pd.Series(
                importances, 
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(self.ensemble_model, "coef_"):
            # Linear models
            if self.ensemble_model.coef_.ndim > 1:
                # For multi-class classification, use the mean of absolute coefficients
                importances = np.abs(self.ensemble_model.coef_).mean(axis=0)
            else:
                importances = np.abs(self.ensemble_model.coef_)
                
            self.feature_importances = pd.Series(
                importances, 
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(self.ensemble_model, "get_feature_importance"):
            # LightGBM
            importances = self.ensemble_model.get_feature_importance()
            self.feature_importances = pd.Series(
                importances, 
                index=X.columns
            ).sort_values(ascending=False)
        elif hasattr(self.ensemble_model, "feature_importances"):
            # XGBoost
            importances = self.ensemble_model.feature_importances
            self.feature_importances = pd.Series(
                importances, 
                index=X.columns
            ).sort_values(ascending=False)
        elif isinstance(self.ensemble_model, (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)):
            # For ensemble models, average importances from base models
            base_importances = []
            
            for model_info in self.fold_models:
                model = model_info['model']
                if hasattr(model, "feature_importances_"):
                    base_importances.append(
                        pd.Series(model.feature_importances_, index=X.columns)
                    )
                elif hasattr(model, "coef_"):
                    if model.coef_.ndim > 1:
                        base_importances.append(
                            pd.Series(np.abs(model.coef_).mean(axis=0), index=X.columns)
                        )
                    else:
                        base_importances.append(
                            pd.Series(np.abs(model.coef_), index=X.columns)
                        )
                elif hasattr(model, "get_feature_importance"):
                    base_importances.append(
                        pd.Series(model.get_feature_importance(), index=X.columns)
                    )
                elif hasattr(model, "feature_importances"):
                    base_importances.append(
                        pd.Series(model.feature_importances, index=X.columns)
                    )
            
            if base_importances:
                # Combine and normalize
                self.feature_importances = pd.concat(base_importances, axis=1) \
                    .mean(axis=1) \
                    .sort_values(ascending=False)
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions with the model.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predictions
            
        Raises
        ------
        ValueError
            If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make probability predictions with the model.
        
        Only applicable for classification problems.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Probability predictions
            
        Raises
        ------
        ValueError
            If the model is not fitted or not a classifier
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        if self.problem_type != "classification":
            raise ValueError("predict_proba() is only available for classification problems")
        
        # Check if the model has predict_proba
        if not hasattr(self.ensemble_model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        return self.ensemble_model.predict_proba(X)
    
    def get_feature_importances(self) -> pd.Series:
        """
        Get feature importances from the model.
        
        Returns
        -------
        pd.Series
            Feature importances
            
        Raises
        ------
        ValueError
            If feature importances are not available
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        if self.feature_importances is None:
            raise ValueError("Feature importances are not available for this model")
        
        return self.feature_importances
    
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
            If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        return self.fold_metrics
    
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
            If the model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        return {
            "ensemble_model": self.ensemble_model,
            "fold_models": self.fold_models
        }
"""

from freamon.pipeline.steps import PipelineStep


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
    cv_trainer : CrossValidatedTrainer
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