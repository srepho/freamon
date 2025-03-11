"""Pipeline steps implementation for ML workflows.

This module provides implementations of common pipeline steps for feature
engineering, feature selection, model training, hyperparameter tuning, and evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

from freamon.pipeline.pipeline import PipelineStep
from freamon.features.engineer import FeatureEngineer
from freamon.features.selector import FeatureSelector
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer as ShapIQEngineer
from freamon.modeling.trainer import ModelTrainer
from freamon.modeling.factory import create_model
from freamon.modeling.tuning import LightGBMTuner
from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.importance import calculate_permutation_importance


class FeatureEngineeringStep(PipelineStep):
    """Pipeline step for feature engineering.
    
    This step wraps the FeatureEngineer class to provide a pipeline-compatible
    interface for feature engineering operations.
    """
    
    def __init__(
        self, 
        name: str,
        operations: Optional[List[Dict[str, Any]]] = None,
        drop_original: bool = False
    ):
        """Initialize feature engineering step.
        
        Args:
            name: Name of the step
            operations: List of operations to perform, each as a dict with 'method' and 'params' keys
            drop_original: Whether to drop original features after engineering
        """
        super().__init__(name)
        self.operations = operations or []
        self.drop_original = drop_original
        self.engineer = FeatureEngineer()
        self.original_columns = set()
        
    def add_operation(self, method: str, **params) -> FeatureEngineeringStep:
        """Add a feature engineering operation.
        
        Args:
            method: Name of operation to perform
            **params: Parameters to pass to the method
            
        Returns:
            self: For method chaining
        """
        # Map legacy 'add_' method names to 'create_' method names for backward compatibility
        if method.startswith("add_") and not hasattr(FeatureEngineer, method):
            create_method = method.replace("add_", "create_")
            if hasattr(FeatureEngineer, create_method):
                method = create_method
        
        self.operations.append({
            'method': method,
            'params': params
        })
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> FeatureEngineeringStep:
        """Fit the feature engineering step.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        self.original_columns = set(X.columns)
        
        # Create a fresh engineer
        self.engineer = FeatureEngineer()
        
        # Apply each operation
        for op in self.operations:
            method_name = op['method']
            params = op['params']
            
            # Get the method from the engineer
            method = getattr(self.engineer, method_name)
            
            # Call the method with parameters
            method(**params)
        
        # Fit the engineer
        self.engineer.fit(X, y)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data with fitted feature engineering.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe
        """
        if not self._is_fitted:
            raise ValueError("This FeatureEngineeringStep instance is not fitted yet.")
        
        # Transform the data
        result = self.engineer.transform(X)
        
        # Drop original columns if specified
        if self.drop_original:
            cols_to_drop = self.original_columns.intersection(set(result.columns))
            result = result.drop(columns=list(cols_to_drop))
            
        return result


class ShapIQFeatureEngineeringStep(PipelineStep):
    """Pipeline step for ShapIQ feature engineering.
    
    This step uses ShapIQ to detect and generate interaction features.
    """
    
    def __init__(
        self, 
        name: str,
        model_type: str = "lightgbm",
        n_interactions: int = 10,
        max_interaction_size: int = 2,
        categorical_features: Optional[List[str]] = None,
        drop_original: bool = False
    ):
        """Initialize ShapIQ feature engineering step.
        
        Args:
            name: Name of the step
            model_type: Type of model to use for ShapIQ
            n_interactions: Number of top interactions to create
            max_interaction_size: Maximum interaction size (2 for pairwise)
            categorical_features: List of categorical feature names
            drop_original: Whether to drop original features after engineering
        """
        super().__init__(name)
        self.model_type = model_type
        self.n_interactions = n_interactions
        self.max_interaction_size = max_interaction_size
        self.categorical_features = categorical_features or []
        self.drop_original = drop_original
        self.shapiq_engineer = None
        self.original_columns = set()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> ShapIQFeatureEngineeringStep:
        """Fit the ShapIQ feature engineering step.
        
        Args:
            X: Feature dataframe
            y: Target series (required for ShapIQ)
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        if y is None:
            raise ValueError("Target variable y is required for ShapIQ feature engineering")
            
        self.original_columns = set(X.columns)
        
        # Create ShapIQ engineer
        self.shapiq_engineer = ShapIQEngineer(
            model_type=self.model_type,
            n_interactions=self.n_interactions,
            max_interaction_size=self.max_interaction_size,
            categorical_features=self.categorical_features
        )
        
        # Fit the engineer
        self.shapiq_engineer.fit(X, y)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data with ShapIQ feature engineering.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe with interaction features
        """
        if not self._is_fitted:
            raise ValueError("This ShapIQFeatureEngineeringStep instance is not fitted yet.")
        
        # Transform the data
        result = self.shapiq_engineer.transform(X)
        
        # Drop original columns if specified
        if self.drop_original:
            cols_to_drop = self.original_columns.intersection(set(result.columns))
            result = result.drop(columns=list(cols_to_drop))
            
        return result


class FeatureSelectionStep(PipelineStep):
    """Pipeline step for feature selection.
    
    This step wraps the FeatureSelector class to provide a pipeline-compatible
    interface for feature selection operations.
    """
    
    def __init__(
        self,
        name: str,
        method: str,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        features_to_keep: Optional[List[str]] = None,
        **method_params
    ):
        """Initialize feature selection step.
        
        Args:
            name: Name of the step
            method: Selection method ('correlation', 'variance', 'mutual_info', 'model_based', etc.)
            n_features: Number of features to select (if applicable)
            threshold: Threshold for feature selection (if applicable)
            features_to_keep: List of features to always keep regardless of selection
            **method_params: Additional parameters for the selection method
        """
        super().__init__(name)
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.features_to_keep = features_to_keep or []
        self.method_params = method_params
        self.selector = None
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> FeatureSelectionStep:
        """Fit the feature selection step.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        # Create selector
        self.selector = FeatureSelector()
        
        # Configure and fit the selector
        if self.method == 'correlation':
            self.selector.correlation_selection(
                threshold=self.threshold,
                n_features=self.n_features,
                **self.method_params
            )
        elif self.method == 'variance':
            self.selector.variance_selection(
                threshold=self.threshold,
                **self.method_params
            )
        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("Target variable y is required for mutual information selection")
            self.selector.mutual_info_selection(
                n_features=self.n_features,
                **self.method_params
            )
        elif self.method == 'model_based':
            if y is None:
                raise ValueError("Target variable y is required for model-based selection")
            self.selector.model_based_selection(
                n_features=self.n_features,
                threshold=self.threshold,
                **self.method_params
            )
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        # Fit the selector
        self.selector.fit(X, y)
        
        # Get selected features
        self.selected_features = set(self.selector.get_selected_features())
        
        # Add features_to_keep
        self.selected_features.update(self.features_to_keep)
        
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data by selecting features.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Dataframe with selected features
        """
        if not self._is_fitted:
            raise ValueError("This FeatureSelectionStep instance is not fitted yet.")
        
        # Get only the columns that exist in X
        valid_columns = [col for col in self.selected_features if col in X.columns]
        
        # Select the features
        return X[valid_columns]


class ModelTrainingStep(PipelineStep):
    """Pipeline step for model training.
    
    This step wraps the ModelTrainer class to provide a pipeline-compatible
    interface for model training.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str,
        problem_type: str = "classification",
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        cv_folds: int = 0,
        early_stopping_rounds: Optional[int] = None,
        random_state: int = 42
    ):
        """Initialize model training step.
        
        Args:
            name: Name of the step
            model_type: Model type ('lightgbm', 'xgboost', 'catboost', 'sklearn_rf', etc.)
            problem_type: Problem type ('classification' or 'regression')
            eval_metric: Evaluation metric
            hyperparameters: Model hyperparameters
            cv_folds: Number of cross-validation folds (0 for no CV)
            early_stopping_rounds: Number of rounds for early stopping
            random_state: Random state for reproducibility
        """
        super().__init__(name)
        self.model_type = model_type
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.hyperparameters = hyperparameters or {}
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.trainer = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> ModelTrainingStep:
        """Fit the model training step.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters, including val_X, val_y for validation
            
        Returns:
            self: The fitted step
        """
        if y is None:
            raise ValueError("Target variable y is required for model training")
        
        # Extract validation data if provided
        val_X = kwargs.get('val_X', None)
        val_y = kwargs.get('val_y', None)
        
        # Create trainer
        # Get default model name based on model type
        model_name = self._get_default_model_name(self.model_type)
        
        self.trainer = ModelTrainer(
            model_type=self.model_type,
            model_name=model_name,
            problem_type=self.problem_type,
            params=self.hyperparameters,
            random_state=self.random_state
        )
        
        # Fit the model
        self.trainer.fit(X, y, val_X, val_y)
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform method for model training step.
        
        For ModelTrainingStep, transform returns the dataframe unchanged,
        as the model doesn't transform the data.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Input dataframe unchanged
        """
        # Model training doesn't transform data
        return X
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions with the trained model.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("This ModelTrainingStep instance is not fitted yet.")
        
        return self.trainer.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make probability predictions with the trained model.
        
        Only applicable for classification problems.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Probability predictions
        """
        if not self._is_fitted:
            raise ValueError("This ModelTrainingStep instance is not fitted yet.")
        
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification problems")
        
        return self.trainer.predict_proba(X)
        
    def _get_default_model_name(self, model_type: str) -> str:
        """Get the default model name based on model type.
        
        Args:
            model_type: Model type
            
        Returns:
            Default model name
        """
        model_name_map = {
            'lightgbm': 'LGBMClassifier',
            'xgboost': 'XGBClassifier',
            'catboost': 'CatBoostClassifier',
            'sklearn': 'RandomForestClassifier',
        }
        
        if self.problem_type == 'regression':
            # Change classifier to regressor
            model_name = model_name_map.get(model_type, 'RandomForestClassifier')
            if 'Classifier' in model_name:
                model_name = model_name.replace('Classifier', 'Regressor')
        else:
            model_name = model_name_map.get(model_type, 'RandomForestClassifier')
            
        return model_name
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the trained model.
        
        Returns:
            DataFrame with feature importances
        """
        if not self._is_fitted:
            raise ValueError("This ModelTrainingStep instance is not fitted yet.")
        
        return self.trainer.get_feature_importances()


class HyperparameterTuningStep(PipelineStep):
    """Pipeline step for hyperparameter tuning.
    
    This step provides intelligent hyperparameter tuning capabilities.
    Currently supports LightGBM models with a focus on parameter-importance
    aware optimization.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str = "lightgbm",
        problem_type: str = "classification",
        metric: Optional[str] = None,
        n_trials: int = 100,
        cv: int = 5,
        cv_type: str = "auto",
        early_stopping_rounds: int = 50,
        categorical_features: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        progressive_tuning: bool = True,
        optimize_threshold: bool = False,
        threshold_metric: str = 'f1',
        random_state: int = 42
    ):
        """Initialize hyperparameter tuning step.
        
        Args:
            name: Name of the step
            model_type: Model type (currently only 'lightgbm' is supported)
            problem_type: Problem type ('classification' or 'regression')
            metric: Metric to optimize
            n_trials: Number of hyperparameter trials
            cv: Number of cross-validation folds
            cv_type: Type of cross-validation ('auto', 'kfold', 'stratified', 'timeseries')
            early_stopping_rounds: Number of rounds for early stopping
            categorical_features: List of categorical feature names
            fixed_params: Parameters to fix during tuning
            progressive_tuning: Whether to use progressive tuning (first tune key parameters)
            optimize_threshold: Whether to automatically optimize classification threshold
            threshold_metric: Metric to optimize threshold for ('f1', 'precision', 'recall', etc.)
            random_state: Random state for reproducibility
        """
        super().__init__(name)
        self.model_type = model_type
        self.problem_type = problem_type
        self.metric = metric
        self.n_trials = n_trials
        self.cv = cv
        
        # Automatically determine CV type based on problem type if 'auto'
        if cv_type == "auto":
            self.cv_type = "stratified" if problem_type == "classification" else "kfold"
        else:
            self.cv_type = cv_type
            
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_features = categorical_features or []
        self.fixed_params = fixed_params or {}
        self.progressive_tuning = progressive_tuning
        self.optimize_threshold = optimize_threshold
        self.threshold_metric = threshold_metric
        self.random_state = random_state
        
        # Will be set during fit
        self.tuner = None
        self.best_params = None
        self.model = None
        self.feature_importances = None
        self.param_importances = None
        self.optimal_threshold = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> HyperparameterTuningStep:
        """Fit the hyperparameter tuning step.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        if y is None:
            raise ValueError("Target variable y is required for hyperparameter tuning")
        
        # Currently only support LightGBM
        if self.model_type != "lightgbm":
            raise ValueError(f"Model type '{self.model_type}' not supported. Only 'lightgbm' is currently supported.")
        
        # Ensure categorical features exist in the data
        for feat in self.categorical_features:
            if feat not in X.columns:
                raise ValueError(f"Categorical feature '{feat}' not found in the dataframe")
        
        # Create and run tuner
        if self.model_type == "lightgbm":
            self.tuner = LightGBMTuner(
                problem_type=self.problem_type,
                objective=self.fixed_params.get('objective', None),
                metric=self.metric,
                cv=self.cv,
                cv_type=self.cv_type,
                random_state=self.random_state,
                n_trials=self.n_trials,
                n_jobs=-1,  # Use all cores
                verbose=True
            )
            
            # Run tuning
            self.best_params = self.tuner.tune(
                X, y,
                categorical_features=self.categorical_features,
                fixed_params=self.fixed_params,
                progressive_tuning=self.progressive_tuning,
                early_stopping_rounds=self.early_stopping_rounds,
                study_name=f"tuning_{self.name}"
            )
            
            # Store parameter importance
            self.param_importances = self.tuner.param_importance
            
            # Create optimized model
            self.model = self.tuner.create_model()
            
            # Fit the model with best parameters
            self.model.fit(
                X, y,
                categorical_feature=self.categorical_features,
                early_stopping_rounds=self.early_stopping_rounds
            )
            
            # Calculate feature importance
            self.feature_importances = self.model.get_feature_importance(method='native')
            
            # Optimize classification threshold if requested and it's a binary classification task
            if (self.optimize_threshold and self.problem_type == "classification" and 
                isinstance(self.model, LightGBMModel)):
                
                # If we have a validation set in kwargs, use it
                X_val = kwargs.get('val_X', None)
                y_val = kwargs.get('val_y', None)
                
                # If no explicit validation set provided, use a portion of the training data
                if X_val is None or y_val is None:
                    from sklearn.model_selection import train_test_split
                    X_val, _, y_val, _ = train_test_split(
                        X, y, test_size=0.2, random_state=self.random_state,
                        stratify=y if self.problem_type == "classification" else None
                    )
                
                try:
                    # Find optimal threshold
                    threshold, score, _ = self.model.find_optimal_threshold(
                        X_val, y_val, metric=self.threshold_metric, set_as_default=True
                    )
                    self.optimal_threshold = threshold
                    print(f"Optimal {self.threshold_metric} threshold: {threshold:.4f} (score: {score:.4f})")
                except Exception as e:
                    # Handle any errors during threshold optimization
                    print(f"Warning: Could not optimize threshold: {str(e)}")
                    self.optimal_threshold = None
            
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform method for hyperparameter tuning step.
        
        For HyperparameterTuningStep, transform returns the dataframe unchanged.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Input dataframe unchanged
        """
        # Tuning doesn't transform data
        return X
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions with the tuned model.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make probability predictions with the tuned model.
        
        Only applicable for classification problems.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Probability predictions
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification problems")
        
        return self.model.predict_proba(X)
    
    def get_feature_importances(self, method: str = 'native', data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Get feature importances from the tuned model.
        
        Args:
            method: Method to use for computing importances ('native', 'shap', 'permutation')
            data: Data to use for computing importances (required for 'permutation' method)
            
        Returns:
            Series with feature importances
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        if method == 'permutation':
            if data is None:
                raise ValueError("data must be provided when method='permutation'")
            
            # Calculate permutation importance
            return self.model.get_feature_importance(method='permutation', X=data)
        else:
            # Use cached feature importance for 'native' or other methods
            if self.feature_importances is not None and method == 'native':
                return self.feature_importances
            
            # Calculate feature importance with the specified method
            return self.model.get_feature_importance(method=method, X=data)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found during tuning.
        
        Returns:
            Dictionary of best hyperparameters
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        return self.best_params
    
    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances from the tuning process.
        
        Returns:
            Dictionary mapping parameter names to importance values
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        return self.param_importances or {}
    
    def plot_tuning_history(self):
        """Plot the optimization history.
        
        Returns:
            Plot of the optimization history
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        if self.tuner is None:
            raise ValueError("Tuner not available")
        
        return self.tuner.plot_optimization_history()
    
    def plot_param_importances(self):
        """Plot parameter importances.
        
        Returns:
            Plot of parameter importances
        """
        if not self._is_fitted:
            raise ValueError("This HyperparameterTuningStep instance is not fitted yet.")
        
        if self.tuner is None:
            raise ValueError("Tuner not available")
        
        return self.tuner.plot_param_importances()


class EvaluationStep(PipelineStep):
    """Pipeline step for model evaluation.
    
    This step evaluates model performance using various metrics.
    """
    
    def __init__(
        self,
        name: str,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        problem_type: str = "classification"
    ):
        """Initialize evaluation step.
        
        Args:
            name: Name of the step
            metrics: List of metrics to compute
            custom_metrics: Dictionary of custom metrics as {name: function}
            problem_type: Problem type ('classification' or 'regression')
        """
        super().__init__(name)
        
        # Default metrics based on problem type
        if metrics is None:
            if problem_type == "classification":
                metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            else:  # regression
                metrics = ["mse", "rmse", "mae", "r2"]
        
        self.metrics = metrics
        self.custom_metrics = custom_metrics or {}
        self.problem_type = problem_type
        self.results = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> EvaluationStep:
        """Fit the evaluation step.
        
        For EvaluationStep, fit does nothing since evaluation doesn't need fitting.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            self: The step
        """
        # Evaluation doesn't require fitting
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform method for evaluation step.
        
        For EvaluationStep, transform returns the dataframe unchanged.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Input dataframe unchanged
        """
        # Evaluation doesn't transform data
        return X
    
    def evaluate(
        self, 
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model predictions against true values.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            y_prob: Predicted probabilities (for classification)
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        # Convert to numpy if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
            
        # Classification metrics
        if self.problem_type == "classification":
            for metric in self.metrics:
                if metric == "accuracy":
                    results[metric] = accuracy_score(y_true, y_pred)
                elif metric == "precision":
                    results[metric] = precision_score(y_true, y_pred, average="weighted")
                elif metric == "recall":
                    results[metric] = recall_score(y_true, y_pred, average="weighted")
                elif metric == "f1":
                    results[metric] = f1_score(y_true, y_pred, average="weighted")
                elif metric == "roc_auc" and y_prob is not None:
                    # Handle binary and multiclass cases
                    if y_prob.ndim > 1 and y_prob.shape[1] > 2:
                        # Multiclass case
                        results[metric] = roc_auc_score(
                            pd.get_dummies(y_true),  # One-hot encode targets
                            y_prob,
                            multi_class="ovr",
                            average="weighted"
                        )
                    else:
                        # Binary case
                        prob_values = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                        results[metric] = roc_auc_score(y_true, prob_values)
        # Regression metrics
        else:
            for metric in self.metrics:
                if metric == "mse":
                    results[metric] = mean_squared_error(y_true, y_pred)
                elif metric == "rmse":
                    results[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
                elif metric == "mae":
                    results[metric] = mean_absolute_error(y_true, y_pred)
                elif metric == "r2":
                    results[metric] = r2_score(y_true, y_pred)
        
        # Custom metrics
        for name, metric_func in self.custom_metrics.items():
            if name == "roc_auc" and y_prob is not None:
                results[name] = metric_func(y_true, y_prob)
            else:
                results[name] = metric_func(y_true, y_pred)
        
        # Store results
        self.results = results
        
        return results