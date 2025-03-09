"""
Module for intelligent hyperparameter tuning.

This module provides sophisticated hyperparameter tuning capabilities
with a special focus on LightGBM optimization.
"""
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import logging
import time
from functools import partial
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.base import BaseEstimator
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, CmaEsSampler

from freamon.modeling.model import Model
from freamon.modeling.factory import create_model

# Set up logger
logger = logging.getLogger(__name__)


class LightGBMTuner:
    """
    Intelligent hyperparameter tuning for LightGBM models.
    
    This tuner specifically optimizes LightGBM models using a parameter-importance
    aware approach, focusing tuning efforts on the most impactful parameters.
    
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
    cv : int, default=5
        Number of cross-validation folds.
    cv_type : Literal['kfold', 'stratified', 'timeseries'], default='kfold'
        Type of cross-validation to use.
    random_state : Optional[int], default=None
        Random state for reproducibility.
    n_trials : int, default=100
        Maximum number of trials for hyperparameter optimization.
    timeout : Optional[int], default=None
        Time limit in seconds for the optimization process.
    n_jobs : int, default=1
        Number of parallel jobs for optimization.
    verbose : bool, default=True
        Whether to print optimization progress.
    """
    
    def __init__(
        self,
        problem_type: Literal['classification', 'regression'],
        objective: Optional[str] = None,
        metric: Optional[str] = None,
        cv: int = 5,
        cv_type: Literal['kfold', 'stratified', 'timeseries'] = 'kfold',
        random_state: Optional[int] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        """Initialize the LightGBM tuner."""
        self.problem_type = problem_type
        self.cv = cv
        self.cv_type = cv_type
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Set default objective and metric based on problem type
        if objective is None:
            if problem_type == 'classification':
                self.objective = 'binary'
            else:
                self.objective = 'regression'
        else:
            self.objective = objective
        
        if metric is None:
            if problem_type == 'classification':
                self.metric = 'auc'
            else:
                self.metric = 'rmse'
        else:
            self.metric = metric
        
        # Initialize storage for optimization results
        self.study = None
        self.best_params = None
        self.param_importance = None
        self.optimization_history = None
    
    def tune(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        categorical_features: Optional[List[Union[int, str]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        progressive_tuning: bool = True,
        early_stopping_rounds: int = 100,
        study_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for LightGBM.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            The feature data.
        y : Union[pd.Series, np.ndarray]
            The target data.
        categorical_features : Optional[List[Union[int, str]]], default=None
            List of categorical features (indices or column names).
        fixed_params : Optional[Dict[str, Any]], default=None
            Parameters to fix during tuning.
        progressive_tuning : bool, default=True
            Whether to use progressive tuning (tune key parameters first,
            then fine-tune additional parameters).
        early_stopping_rounds : int, default=100
            Number of early stopping rounds.
        study_name : Optional[str], default=None
            Name for the Optuna study.
        
        Returns
        -------
        Dict[str, Any]
            The best hyperparameters found.
        """
        self.feature_names = None
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Handle categorical features
        self.categorical_features = categorical_features
        
        # Prepare fixed parameters
        if fixed_params is None:
            fixed_params = {}
        self.fixed_params = fixed_params
        
        # Setup cross-validation
        cv_splitter = self._create_cv_splitter()
        
        # Configure pruning
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Configure the sampler
        sampler = TPESampler(seed=self.random_state)
        
        if progressive_tuning:
            logger.info("Starting progressive tuning...")
            # Phase 1: Tune core parameters
            phase1_params = self._get_core_param_ranges()
            study_name_1 = f"{study_name}_phase1" if study_name else "lightgbm_phase1"
            
            study1 = optuna.create_study(
                direction='maximize' if self.metric in ['auc', 'accuracy', 'f1'] else 'minimize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name_1
            )
            
            objective_func1 = self._create_objective_func(
                X, y, cv_splitter, early_stopping_rounds, phase1_params
            )
            
            n_trials_phase1 = min(50, self.n_trials // 2)
            timeout_phase1 = self.timeout // 2 if self.timeout else None
            
            study1.optimize(
                objective_func1,
                n_trials=n_trials_phase1,
                timeout=timeout_phase1,
                n_jobs=self.n_jobs,
                show_progress_bar=self.verbose
            )
            
            # Get best params from Phase 1
            best_params_phase1 = study1.best_params
            
            # Add fixed parameters
            for param, value in self.fixed_params.items():
                best_params_phase1[param] = value
            
            # Get param importance
            if len(study1.trials) > 5:
                try:
                    phase1_importance = optuna.importance.get_param_importances(study1)
                    logger.info(f"Phase 1 parameter importance: {phase1_importance}")
                    
                    # Keep only important parameters (contributes >5% to improvement)
                    important_params = {
                        k: v for k, v in phase1_importance.items() if v > 0.05
                    }
                    
                    # Update fixed params with optimized values of unimportant parameters
                    unimportant_params = set(best_params_phase1.keys()) - set(important_params.keys())
                    for param in unimportant_params:
                        self.fixed_params[param] = best_params_phase1[param]
                except:
                    # If importance calculation fails, use all parameters
                    self.fixed_params.update(best_params_phase1)
            else:
                # Not enough trials for importance calculation
                self.fixed_params.update(best_params_phase1)
            
            # Phase 2: Fine-tune with all parameters
            all_params = self._get_all_param_ranges()
            study_name_2 = f"{study_name}_phase2" if study_name else "lightgbm_phase2"
            
            study = optuna.create_study(
                direction='maximize' if self.metric in ['auc', 'accuracy', 'f1'] else 'minimize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name_2
            )
            
            objective_func = self._create_objective_func(
                X, y, cv_splitter, early_stopping_rounds, all_params
            )
            
            n_trials_phase2 = self.n_trials - n_trials_phase1
            timeout_phase2 = self.timeout - time.time() + study1.best_trial.datetime_start.timestamp() if self.timeout else None
            
            study.optimize(
                objective_func,
                n_trials=n_trials_phase2,
                timeout=timeout_phase2,
                n_jobs=self.n_jobs,
                show_progress_bar=self.verbose
            )
            
            self.study = study
        
        else:
            # Regular tuning with all parameters at once
            all_params = self._get_all_param_ranges()
            
            study = optuna.create_study(
                direction='maximize' if self.metric in ['auc', 'accuracy', 'f1'] else 'minimize',
                sampler=sampler,
                pruner=pruner,
                study_name=study_name or "lightgbm_tuning"
            )
            
            objective_func = self._create_objective_func(
                X, y, cv_splitter, early_stopping_rounds, all_params
            )
            
            study.optimize(
                objective_func,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                show_progress_bar=self.verbose
            )
            
            self.study = study
        
        # Extract results
        self.best_params = self.study.best_params.copy()
        
        # Add fixed parameters
        for param, value in self.fixed_params.items():
            if param not in self.best_params:
                self.best_params[param] = value
        
        # Calculate parameter importance
        if len(self.study.trials) > 5:
            try:
                self.param_importance = optuna.importance.get_param_importances(self.study)
            except:
                logger.warning("Failed to calculate parameter importance")
                self.param_importance = {}
        
        # Extract optimization history
        self.optimization_history = {
            'value': [t.value for t in self.study.trials if t.value is not None],
            'datetime': [t.datetime_start for t in self.study.trials if t.value is not None],
            'params': [t.params for t in self.study.trials if t.value is not None]
        }
        
        return self.best_params
    
    def _create_objective_func(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_splitter,
        early_stopping_rounds: int,
        param_ranges: Dict[str, Any]
    ) -> Callable:
        """Create the objective function for Optuna."""
        def objective(trial):
            # Generate hyperparameters
            params = self._sample_params(trial, param_ranges)
            
            # Add fixed parameters
            for param, value in self.fixed_params.items():
                params[param] = value
            
            # Add basic parameters
            params['objective'] = self.objective
            params['metric'] = self.metric
            params['verbosity'] = -1
            params['random_state'] = self.random_state
            if 'learning_rate' not in params:
                params['learning_rate'] = 0.05
            
            # Prepare for cross-validation
            models = []
            scores = []
            
            # Run cross-validation
            fold_idx = 0
            for train_idx, valid_idx in cv_splitter.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                else:
                    X_train, X_valid = X[train_idx], X[valid_idx]
                
                if isinstance(y, pd.Series):
                    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                else:
                    y_train, y_valid = y[train_idx], y[valid_idx]
                
                # Determine model class by the objective
                if self.problem_type == 'classification':
                    model_name = 'LGBMClassifier'
                else:
                    model_name = 'LGBMRegressor'
                
                # Create model for this fold
                model = create_model('lightgbm', model_name, params, self.random_state)
                
                # Prepare eval_set for early stopping
                eval_set = (X_valid, y_valid)
                
                try:
                    # Train the model with early stopping
                    model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False,
                        categorical_feature=self.categorical_features
                    )
                    
                    # Save model
                    models.append(model)
                    
                    # Evaluate on validation set
                    if self.metric == 'auc':
                        from sklearn.metrics import roc_auc_score
                        y_prob = model.model.predict_proba(X_valid)[:, 1]
                        score = roc_auc_score(y_valid, y_prob)
                    elif self.metric == 'accuracy':
                        from sklearn.metrics import accuracy_score
                        y_pred = model.predict(X_valid)
                        score = accuracy_score(y_valid, y_pred)
                    elif self.metric == 'f1':
                        from sklearn.metrics import f1_score
                        y_pred = model.predict(X_valid)
                        score = f1_score(y_valid, y_pred, average='weighted')
                    elif self.metric in ['rmse', 'mse']:
                        from sklearn.metrics import mean_squared_error
                        y_pred = model.predict(X_valid)
                        score = -mean_squared_error(y_valid, y_pred, squared=self.metric=='mse')
                    elif self.metric == 'mae':
                        from sklearn.metrics import mean_absolute_error
                        y_pred = model.predict(X_valid)
                        score = -mean_absolute_error(y_valid, y_pred)
                    else:
                        # Use the model's best score if metric is not supported
                        score = model.model.best_score_['valid_0'][self.metric]
                        if self.metric not in ['auc', 'accuracy', 'f1']:
                            score = -score  # For metrics where lower is better
                    
                    scores.append(score)
                    
                    # Report for pruning
                    trial.report(score, fold_idx)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                    fold_idx += 1
                    
                except Exception as e:
                    logger.warning(f"Error in fold {fold_idx}: {str(e)}")
                    return float('-inf') if self.metric in ['auc', 'accuracy', 'f1'] else float('inf')
            
            # Return the mean score across folds
            return np.mean(scores)
        
        return objective
    
    def _sample_params(self, trial, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from defined ranges."""
        params = {}
        
        for param_name, param_config in param_ranges.items():
            param_type = param_config.get('type', 'continuous')
            
            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', 1),
                    log=param_config.get('log', False)
                )
            elif param_type == 'continuous':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', None),
                    log=param_config.get('log', False)
                )
                
        return params
    
    def _get_core_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get the core parameter ranges for the first tuning phase."""
        # Focus on the most impactful parameters first
        return {
            'num_leaves': {
                'type': 'int',
                'low': 20,
                'high': 150,
                'log': True
            },
            'max_depth': {
                'type': 'int',
                'low': 3,
                'high': 12
            },
            'learning_rate': {
                'type': 'continuous',
                'low': 0.01,
                'high': 0.3,
                'log': True
            },
            'subsample': {
                'type': 'continuous',
                'low': 0.5,
                'high': 1.0
            },
            'colsample_bytree': {
                'type': 'continuous',
                'low': 0.5,
                'high': 1.0
            },
            'min_child_samples': {
                'type': 'int',
                'low': 5,
                'high': 100,
                'log': True
            }
        }
    
    def _get_all_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get all parameter ranges for the second tuning phase."""
        # Start with core parameters
        params = self._get_core_param_ranges()
        
        # Add additional parameters
        additional_params = {
            'reg_alpha': {
                'type': 'continuous',
                'low': 0.0,
                'high': 10.0,
                'log': True
            },
            'reg_lambda': {
                'type': 'continuous',
                'low': 0.0,
                'high': 10.0,
                'log': True
            },
            'min_split_gain': {
                'type': 'continuous',
                'low': 0.0,
                'high': 1.0
            },
            'feature_fraction': {
                'type': 'continuous',
                'low': 0.5,
                'high': 1.0
            },
            'bagging_fraction': {
                'type': 'continuous',
                'low': 0.5,
                'high': 1.0
            },
            'bagging_freq': {
                'type': 'int',
                'low': 0,
                'high': 10
            },
            'min_data_in_leaf': {
                'type': 'int',
                'low': 10,
                'high': 200,
                'log': True
            }
        }
        
        params.update(additional_params)
        return params
    
    def _create_cv_splitter(self):
        """Create cross-validation splitter based on cv_type."""
        if self.cv_type == 'kfold':
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        elif self.cv_type == 'stratified':
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        elif self.cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.cv)
        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type}")
    
    def create_model(self) -> Model:
        """
        Create a LightGBM model with the optimized hyperparameters.
        
        Returns
        -------
        Model
            A Model instance with the best hyperparameters.
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Call tune() first.")
        
        params = self.best_params.copy()
        
        # Add basic parameters
        params['objective'] = self.objective
        params['metric'] = self.metric
        params['random_state'] = self.random_state
        
        # Determine model class by the objective
        if self.problem_type == 'classification':
            model_name = 'LGBMClassifier'
        else:
            model_name = 'LGBMRegressor'
        
        # Create and return model
        return create_model('lightgbm', model_name, params, self.random_state)
    
    def plot_optimization_history(self):
        """
        Plot optimization history.
        
        Returns
        -------
        matplotlib.figure.Figure
            The optimization history figure.
        """
        if self.study is None:
            raise ValueError("No optimization performed. Call tune() first.")
        
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            return fig
        except ImportError:
            logger.warning("Cannot plot optimization history. Make sure plotly is installed.")
            return None
    
    def plot_param_importances(self):
        """
        Plot parameter importances.
        
        Returns
        -------
        matplotlib.figure.Figure
            The parameter importance figure.
        """
        if self.study is None or self.param_importance is None:
            raise ValueError("No optimization performed or parameter importance not available.")
        
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            return fig
        except ImportError:
            logger.warning("Cannot plot parameter importances. Make sure plotly is installed.")
            return None
    
    def print_study_summary(self):
        """Print a summary of the optimization study."""
        if self.study is None:
            raise ValueError("No optimization performed. Call tune() first.")
        
        print(f"Number of finished trials: {len(self.study.trials)}")
        print(f"Best trial:")
        trial = self.study.best_trial
        
        print(f"  Value: {trial.value}")
        print(f"  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        if self.param_importance is not None:
            print("\nParameter importance:")
            for key, value in self.param_importance.items():
                print(f"  {key}: {value:.4f}")