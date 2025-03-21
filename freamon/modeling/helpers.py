"""
Helper functions for creating common model types.

This module provides simplified functions to create various model types
with sensible defaults, making it easier for users to get started.
"""
from typing import Any, Dict, Literal, Optional, Union

from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.factory import create_model


def create_lightgbm_regressor(
    objective: str = 'regression',
    metric: str = 'rmse',
    n_estimators: int = 100,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 50,
    tune_hyperparameters: bool = False,
    tuning_trials: int = 100,
    random_state: Optional[int] = None,
    **kwargs
) -> LightGBMModel:
    """
    Create a LightGBM regressor with sensible defaults.
    
    Parameters
    ----------
    objective : str, default='regression'
        The objective function for LightGBM.
        Options: 'regression', 'mse', 'mae', 'huber', 'quantile', etc.
    metric : str, default='rmse'
        The metric to use for evaluation.
        Options: 'rmse', 'mae', 'mape', etc.
    n_estimators : int, default=100
        Number of boosting iterations.
    max_depth : int, default=-1
        Maximum tree depth. -1 means no limit.
    num_leaves : int, default=31
        Maximum number of leaves in each tree.
    learning_rate : float, default=0.1
        Learning rate.
    early_stopping_rounds : int, default=50
        Number of rounds with no improvement before early stopping.
    tune_hyperparameters : bool, default=False
        Whether to tune hyperparameters automatically.
    tuning_trials : int, default=100
        Number of trials for hyperparameter tuning.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    **kwargs
        Additional parameters to pass to LightGBMModel.
        
    Returns
    -------
    LightGBMModel
        A configured LightGBM regression model.
        
    Examples
    --------
    >>> # Create a simple regressor
    >>> model = create_lightgbm_regressor()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Create a tuned regressor
    >>> model = create_lightgbm_regressor(tune_hyperparameters=True, tuning_trials=50)
    >>> model.fit(X_train, y_train)
    """
    # Set up fixed parameters for when tuning is disabled
    fixed_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
    }
    
    # Create the LightGBM model with specified parameters
    model = LightGBMModel(
        problem_type='regression',
        objective=objective,
        metric=metric,
        early_stopping_rounds=early_stopping_rounds,
        tuning_trials=tuning_trials,
        random_state=random_state,
        **kwargs
    )
    
    # Set fixed parameters to use when tuning is disabled
    model.fixed_params = fixed_params
    
    return model


def create_lightgbm_classifier(
    objective: str = 'binary',
    metric: Union[str, None] = None,
    n_estimators: int = 100,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 50,
    tune_hyperparameters: bool = False,
    tuning_trials: int = 100,
    random_state: Optional[int] = None,
    probability_threshold: Optional[float] = None,
    **kwargs
) -> LightGBMModel:
    """
    Create a LightGBM classifier with sensible defaults.
    
    Parameters
    ----------
    objective : str, default='binary'
        The objective function for LightGBM.
        Options: 'binary' for binary classification, 'multiclass' for multi-class.
    metric : Union[str, None], default=None
        The metric to use for evaluation. If None, uses 'auc' for binary and 'multi_logloss'
        for multiclass.
    n_estimators : int, default=100
        Number of boosting iterations.
    max_depth : int, default=-1
        Maximum tree depth. -1 means no limit.
    num_leaves : int, default=31
        Maximum number of leaves in each tree.
    learning_rate : float, default=0.1
        Learning rate.
    early_stopping_rounds : int, default=50
        Number of rounds with no improvement before early stopping.
    tune_hyperparameters : bool, default=False
        Whether to tune hyperparameters automatically.
    tuning_trials : int, default=100
        Number of trials for hyperparameter tuning.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    probability_threshold : Optional[float], default=None
        Threshold for binary classification. If None, uses 0.5.
    **kwargs
        Additional parameters to pass to LightGBMModel.
        
    Returns
    -------
    LightGBMModel
        A configured LightGBM classification model.
        
    Examples
    --------
    >>> # Create a simple binary classifier
    >>> model = create_lightgbm_classifier()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Create a tuned multiclass classifier
    >>> model = create_lightgbm_classifier(
    ...     objective='multiclass',
    ...     tune_hyperparameters=True,
    ...     tuning_trials=50
    ... )
    >>> model.fit(X_train, y_train)
    """
    # Set default metric based on objective if not provided
    if metric is None:
        metric = 'auc' if objective == 'binary' else 'multi_logloss'
    
    # Set up fixed parameters for when tuning is disabled
    fixed_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
    }
    
    # Create the LightGBM model with specified parameters
    model = LightGBMModel(
        problem_type='classification',
        objective=objective,
        metric=metric,
        early_stopping_rounds=early_stopping_rounds,
        tuning_trials=tuning_trials,
        random_state=random_state,
        probability_threshold=probability_threshold,
        **kwargs
    )
    
    # Set fixed parameters to use when tuning is disabled
    model.fixed_params = fixed_params
    
    return model


def create_sklearn_model(
    model_type: Literal['classifier', 'regressor'],
    algorithm: str = 'random_forest',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create a scikit-learn model with sensible defaults.
    
    Parameters
    ----------
    model_type : Literal['classifier', 'regressor']
        The type of model to create (classifier or regressor).
    algorithm : str, default='random_forest'
        The algorithm to use.
        Options for classifiers: 'random_forest', 'gradient_boosting', 'logistic_regression', etc.
        Options for regressors: 'random_forest', 'gradient_boosting', 'linear_regression', etc.
    n_estimators : int, default=100
        Number of estimators for ensemble methods.
    max_depth : Optional[int], default=None
        Maximum tree depth for tree-based methods.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    **kwargs
        Additional parameters to pass to the model.
        
    Returns
    -------
    Model
        A configured scikit-learn model wrapped in a Freamon Model instance.
        
    Examples
    --------
    >>> # Create a random forest classifier
    >>> model = create_sklearn_model('classifier', 'random_forest')
    >>> 
    >>> # Create a gradient boosting regressor
    >>> model = create_sklearn_model('regressor', 'gradient_boosting', n_estimators=200)
    """
    model_name_map = {
        'classifier': {
            'random_forest': 'RandomForestClassifier',
            'gradient_boosting': 'GradientBoostingClassifier',
            'logistic_regression': 'LogisticRegression',
            'decision_tree': 'DecisionTreeClassifier',
            'svm': 'SVC',
            'knn': 'KNeighborsClassifier',
        },
        'regressor': {
            'random_forest': 'RandomForestRegressor',
            'gradient_boosting': 'GradientBoostingRegressor',
            'linear_regression': 'LinearRegression',
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'decision_tree': 'DecisionTreeRegressor',
            'svr': 'SVR',
            'knn': 'KNeighborsRegressor',
        }
    }
    
    # Get the model name for the specified algorithm
    try:
        model_name = model_name_map[model_type][algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}' for model type '{model_type}'. "
            f"Available algorithms for {model_type}: {list(model_name_map[model_type].keys())}"
        )
    
    # Set parameters based on the algorithm
    params = {}
    if 'random_forest' in algorithm or 'gradient_boosting' in algorithm or 'tree' in algorithm:
        params['n_estimators'] = n_estimators
        if max_depth is not None:
            params['max_depth'] = max_depth
    
    # Add additional parameters
    params.update(kwargs)
    
    # Create the model
    return create_model('sklearn', model_name, params, random_state)