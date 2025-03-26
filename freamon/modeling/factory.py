"""
Module for creating models from different libraries.
"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from freamon.modeling.model import Model


def create_model(
    model_type: str,
    model_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> Model:
    """
    Create a model of the specified type and name.
    
    Parameters
    ----------
    model_type : str
        The type of model to create. Can be one of:
        - Direct library types: 'sklearn', 'lightgbm', 'xgboost', 'catboost'
        - Shorthand model types: 'lgbm_classifier', 'lgbm_regressor', 'xgb_classifier', 'xgb_regressor'
        
        Using shorthand model types is recommended as they automatically select the correct model class
        without needing to specify model_name.
    model_name : str, optional
        The name of the model within the specified library.
        Not needed when using shorthand model types like 'lgbm_classifier'.
    params : Optional[Dict[str, Any]], default=None
        Parameters to pass to the model constructor.
    random_state : Optional[int], default=None
        Random state to use for reproducibility.
    **kwargs
        Additional keyword arguments to pass to the model constructor.
        
        Note: Internal parameters (cv, metrics, etc.) are automatically filtered out
        and will not be passed to the underlying model constructor.
    
    Returns
    -------
    Model
        A Model instance wrapping the created model.
    
    Examples
    --------
    >>> # Create a scikit-learn RandomForestClassifier
    >>> model = create_model('sklearn', 'RandomForestClassifier', {'n_estimators': 100}, random_state=42)
    >>> 
    >>> # Create a LightGBM classifier using library type and model name
    >>> model = create_model('lightgbm', 'LGBMClassifier', {'num_leaves': 31}, random_state=42)
    >>>
    >>> # Recommended: Use shorthand model type for LightGBM classifier (no model_name needed)
    >>> model = create_model('lgbm_classifier', params={'num_leaves': 31}, random_state=42)
    >>>
    >>> # Recommended: Use shorthand model type for LightGBM regressor
    >>> model = create_model('lgbm_regressor', params={'num_leaves': 31}, random_state=42)
    """
    # Combine params and kwargs
    if params is None:
        params = {}
    all_params = {**params, **kwargs}
    
    # Add random_state if provided
    if random_state is not None:
        all_params['random_state'] = random_state
    
    # Filter out internal parameters not meant for the model constructor
    model_params = all_params.copy()
    # These parameters are used internally but shouldn't be passed to model constructors
    internal_params = [
        'cv', 'early_stopping_rounds', 'metrics', 'callbacks',
        'params', 'model_name', 'model_type', 'feature_names',
        'problem_type', 'categorical_features'
    ]
    for param in internal_params:
        if param in model_params:
            model_params.pop(param)
    
    # Create the model based on model_type
    if model_type == 'sklearn':
        if model_name is None:
            raise ValueError("model_name is required for sklearn models")
        return _create_sklearn_model(model_name, model_params)
    elif model_type == 'lightgbm':
        if model_name is None:
            raise ValueError("model_name is required for lightgbm models")
        return _create_lightgbm_model(model_name, model_params)
    elif model_type == 'xgboost':
        if model_name is None:
            raise ValueError("model_name is required for xgboost models")
        return _create_xgboost_model(model_name, model_params)
    elif model_type == 'catboost':
        if model_name is None:
            raise ValueError("model_name is required for catboost models")
        return _create_catboost_model(model_name, model_params)
    # Handle shorthand model types
    elif model_type == 'lgbm_classifier':
        return _create_lightgbm_model('LGBMClassifier', model_params)
    elif model_type == 'lgbm_regressor':
        return _create_lightgbm_model('LGBMRegressor', model_params)
    elif model_type == 'xgb_classifier':
        return _create_xgboost_model('XGBClassifier', model_params)
    elif model_type == 'xgb_regressor':
        return _create_xgboost_model('XGBRegressor', model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _create_sklearn_model(model_name: str, params: Dict[str, Any]) -> Model:
    """
    Create a scikit-learn model.
    
    Parameters
    ----------
    model_name : str
        The name of the scikit-learn model to create.
    params : Dict[str, Any]
        Parameters to pass to the model constructor.
    
    Returns
    -------
    Model
        A Model instance wrapping the created scikit-learn model.
    """
    try:
        import sklearn
    except ImportError:
        raise ImportError(
            "scikit-learn is not installed. "
            "Install it with 'pip install scikit-learn'."
        )
    
    # Import the module containing the model class
    if model_name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        from sklearn.linear_model import (
            LinearRegression, Ridge, Lasso, ElasticNet
        )
        model_classes = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
        }
    
    elif model_name in ['LogisticRegression', 'SGDClassifier', 'SGDRegressor']:
        from sklearn.linear_model import (
            LogisticRegression, SGDClassifier, SGDRegressor
        )
        model_classes = {
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier,
            'SGDRegressor': SGDRegressor,
        }
    
    elif model_name in ['RandomForestClassifier', 'RandomForestRegressor',
                       'ExtraTreesClassifier', 'ExtraTreesRegressor']:
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor
        )
        model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForestRegressor': RandomForestRegressor,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'ExtraTreesRegressor': ExtraTreesRegressor,
        }
    
    elif model_name in ['GradientBoostingClassifier', 'GradientBoostingRegressor',
                       'AdaBoostClassifier', 'AdaBoostRegressor']:
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            AdaBoostClassifier, AdaBoostRegressor
        )
        model_classes = {
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'AdaBoostClassifier': AdaBoostClassifier,
            'AdaBoostRegressor': AdaBoostRegressor,
        }
    
    elif model_name in ['KNeighborsClassifier', 'KNeighborsRegressor']:
        from sklearn.neighbors import (
            KNeighborsClassifier, KNeighborsRegressor
        )
        model_classes = {
            'KNeighborsClassifier': KNeighborsClassifier,
            'KNeighborsRegressor': KNeighborsRegressor,
        }
    
    elif model_name in ['SVC', 'SVR', 'LinearSVC', 'LinearSVR']:
        from sklearn.svm import (
            SVC, SVR, LinearSVC, LinearSVR
        )
        model_classes = {
            'SVC': SVC,
            'SVR': SVR,
            'LinearSVC': LinearSVC,
            'LinearSVR': LinearSVR,
        }
    
    elif model_name in ['DecisionTreeClassifier', 'DecisionTreeRegressor']:
        from sklearn.tree import (
            DecisionTreeClassifier, DecisionTreeRegressor
        )
        model_classes = {
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTreeRegressor': DecisionTreeRegressor,
        }
    
    else:
        raise ValueError(f"Unknown scikit-learn model: {model_name}")
    
    # Get the model class and instantiate it
    if model_name not in model_classes:
        raise ValueError(f"Unknown scikit-learn model: {model_name}")
    
    model_class = model_classes[model_name]
    model_instance = model_class(**params)
    
    # Wrap the model in our Model class
    return Model(
        model=model_instance,
        model_type='sklearn',
        params=params,
    )


def _create_lightgbm_model(model_name: str, params: Dict[str, Any]) -> Model:
    """
    Create a LightGBM model.
    
    Parameters
    ----------
    model_name : str
        The name of the LightGBM model to create.
    params : Dict[str, Any]
        Parameters to pass to the model constructor.
    
    Returns
    -------
    Model
        A Model instance wrapping the created LightGBM model.
    """
    try:
        import lightgbm
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. "
            "Install it with 'pip install lightgbm'."
        )
    
    # Import the module containing the model class
    from lightgbm import LGBMClassifier, LGBMRegressor
    
    model_classes = {
        'LGBMClassifier': LGBMClassifier,
        'LGBMRegressor': LGBMRegressor,
    }
    
    # Get the model class and instantiate it
    if model_name not in model_classes:
        raise ValueError(f"Unknown LightGBM model: {model_name}")
    
    model_class = model_classes[model_name]
    model_instance = model_class(**params)
    
    # Wrap the model in our Model class
    return Model(
        model=model_instance,
        model_type='lightgbm',
        params=params,
    )


def _create_xgboost_model(model_name: str, params: Dict[str, Any]) -> Model:
    """
    Create an XGBoost model.
    
    Parameters
    ----------
    model_name : str
        The name of the XGBoost model to create.
    params : Dict[str, Any]
        Parameters to pass to the model constructor.
    
    Returns
    -------
    Model
        A Model instance wrapping the created XGBoost model.
    """
    try:
        import xgboost
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. "
            "Install it with 'pip install xgboost'."
        )
    
    # Import the module containing the model class
    from xgboost import XGBClassifier, XGBRegressor, XGBRanker
    
    model_classes = {
        'XGBClassifier': XGBClassifier,
        'XGBRegressor': XGBRegressor,
        'XGBRanker': XGBRanker,
    }
    
    # Get the model class and instantiate it
    if model_name not in model_classes:
        raise ValueError(f"Unknown XGBoost model: {model_name}")
    
    model_class = model_classes[model_name]
    model_instance = model_class(**params)
    
    # Wrap the model in our Model class
    return Model(
        model=model_instance,
        model_type='xgboost',
        params=params,
    )


def _create_catboost_model(model_name: str, params: Dict[str, Any]) -> Model:
    """
    Create a CatBoost model.
    
    Parameters
    ----------
    model_name : str
        The name of the CatBoost model to create.
    params : Dict[str, Any]
        Parameters to pass to the model constructor.
    
    Returns
    -------
    Model
        A Model instance wrapping the created CatBoost model.
    """
    try:
        import catboost
    except ImportError:
        raise ImportError(
            "CatBoost is not installed. "
            "Install it with 'pip install catboost'."
        )
    
    # Import the module containing the model class
    from catboost import CatBoostClassifier, CatBoostRegressor
    
    model_classes = {
        'CatBoostClassifier': CatBoostClassifier,
        'CatBoostRegressor': CatBoostRegressor,
    }
    
    # Get the model class and instantiate it
    if model_name not in model_classes:
        raise ValueError(f"Unknown CatBoost model: {model_name}")
    
    model_class = model_classes[model_name]
    model_instance = model_class(**params)
    
    # Wrap the model in our Model class
    return Model(
        model=model_instance,
        model_type='catboost',
        params=params,
    )