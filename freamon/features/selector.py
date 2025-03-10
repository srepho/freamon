"""
Feature selection module for selecting the most informative features.

This module provides various methods for feature selection, including:
- Correlation-based selection
- Importance-based selection
- Variance-based selection
- Mutual information-based selection
- Recursive feature elimination with cross-validation
- Stability selection
- Genetic algorithm-based selection
- Multi-objective feature selection
- Time series-specific feature selection
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import (
    SelectKBest, 
    SelectPercentile, 
    SelectFromModel,
    VarianceThreshold,
    f_classif, 
    f_regression, 
    mutual_info_classif, 
    mutual_info_regression,
    RFECV
)
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.utils import check_random_state

# Optional imports for specialized methods
try:
    import deap
    from deap import algorithms, base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

try:
    import pymoo
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import grangercausalitytests, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from freamon.utils import check_dataframe_type, convert_dataframe


class FeatureSelector:
    """Feature selector for choosing the most informative features.
    
    This class provides methods for selecting features based on various criteria
    such as correlation, importance, variance, mutual information, etc.
    
    Attributes
    ----------
    selected_features_ : List[str]
        Names of selected features after calling fit or fit_transform
    scores_ : pd.Series
        Feature scores (if available) after calling fit or fit_transform
    """
    
    def __init__(self):
        """Initialize the feature selector."""
        self.selected_features_ = None
        self.scores_ = None
    
    def fit(self, X, y=None, **kwargs):
        """Fit the feature selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters for the selection method
            
        Returns
        -------
        self : FeatureSelector
            The fitted selector
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    def transform(self, X):
        """Transform X by selecting features.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
            
        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted. Call fit first.")
        
        # Return only the selected features
        return X[self.selected_features_]
    
    def fit_transform(self, X, y=None, **kwargs):
        """Fit the selector and transform X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters for the selection method
            
        Returns
        -------
        pd.DataFrame
            DataFrame with only the selected features
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)


def select_features(
    df: Any,
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    method: str = 'correlation',
    k: Optional[int] = None,
    k_percent: Optional[float] = None,
    threshold: Optional[float] = None,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    exclude_columns: Optional[List[str]] = None,
) -> Union[Any, List[str]]:
    """
    Select features from the dataframe using the specified method.
    
    This is a high-level function that redirects to specific selection methods
    based on the `method` parameter.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Optional[Union[str, pd.Series, np.ndarray]], default=None
        The target variable for supervised selection methods. Can be a column name
        in the dataframe or a separate series/array. Required for supervised methods.
    method : str, default='correlation'
        The feature selection method to use. Options: 'correlation', 'importance',
        'variance', 'mutual_info', 'kbest', 'percentile', 'model'.
    k : Optional[int], default=None
        Number of top features to select. Used by 'kbest' method.
    k_percent : Optional[float], default=None
        Percentage of top features to select. Used by 'percentile' method.
    threshold : Optional[float], default=None
        Threshold for feature selection. Used by 'correlation', 'variance', and 'model' methods.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    exclude_columns : Optional[List[str]], default=None
        Columns to exclude from selection.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1],
    ...     'target': [0, 0, 1, 1, 1]
    ... })
    >>> select_features(df, 'target', method='correlation', threshold=0.5, return_names_only=True)
    ['A', 'B']
    """
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        # Unsupervised method
        y = None
        df_without_target = df
    
    # Exclude columns if specified
    if exclude_columns:
        df_without_target = df_without_target.drop(columns=[
            col for col in exclude_columns if col in df_without_target.columns
        ])
    
    # Infer problem type if not provided
    if problem_type is None and y is not None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Select features based on method
    if method == 'correlation':
        return select_by_correlation(
            df_without_target, target=y, threshold=threshold, 
            return_names_only=return_names_only
        )
    elif method == 'importance':
        return select_by_importance(
            df_without_target, target=y, threshold=threshold, k=k,
            problem_type=problem_type, return_names_only=return_names_only
        )
    elif method == 'variance':
        return select_by_variance(
            df_without_target, threshold=threshold, return_names_only=return_names_only
        )
    elif method == 'mutual_info':
        return select_by_mutual_info(
            df_without_target, target=y, k=k, problem_type=problem_type,
            return_names_only=return_names_only
        )
    elif method == 'kbest':
        if k is None:
            raise ValueError("k must be provided for 'kbest' method")
        return select_by_kbest(
            df_without_target, target=y, k=k, problem_type=problem_type,
            return_names_only=return_names_only
        )
    elif method == 'percentile':
        if k_percent is None:
            raise ValueError("k_percent must be provided for 'percentile' method")
        return select_by_percentile(
            df_without_target, target=y, percentile=k_percent, 
            problem_type=problem_type, return_names_only=return_names_only
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def select_by_correlation(
    df: Any,
    target: Optional[Union[pd.Series, np.ndarray]] = None,
    threshold: float = 0.5,
    method: str = 'pearson',
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select features based on correlation with the target variable.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Optional[Union[pd.Series, np.ndarray]], default=None
        The target variable. If None, performs feature-feature correlation analysis.
    threshold : float, default=0.5
        Correlation threshold for selection.
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1]
    ... })
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> select_by_correlation(df, target, threshold=0.5, return_names_only=True)
    ['A', 'B']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Calculate correlations
    if target is not None:
        # Calculate correlation with target
        correlations = {}
        for col in numeric_df.columns:
            corr = numeric_df[col].corr(pd.Series(target), method=method)
            correlations[col] = abs(corr)  # Use absolute correlation
        
        # Convert to Series
        correlations = pd.Series(correlations)
    else:
        # Calculate pairwise correlations
        corr_matrix = numeric_df.corr(method=method).abs()
        
        # For each column, find the maximum correlation with other columns
        correlations = corr_matrix.max()
    
    # Select features based on threshold
    selected_features = correlations[correlations >= threshold].index.tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


def select_by_importance(
    df: Any,
    target: Union[pd.Series, np.ndarray],
    model_type: str = 'forest',
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select features based on importance scores from a model.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[pd.Series, np.ndarray]
        The target variable.
    model_type : str, default='forest'
        Type of model to use: 'forest', 'gbdt', 'linear', or 'svm'.
    threshold : Optional[float], default=None
        Importance threshold for selection. Features with importance >= threshold are selected.
    k : Optional[int], default=None
        Number of top features to select. Overrides threshold if provided.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1]
    ... })
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> select_by_importance(df, target, k=2, return_names_only=True)
    ['A', 'B']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(target, 'nunique'):
            if target.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Create model based on type and problem type
    if model_type == 'forest':
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gbdt':
        if problem_type == 'classification':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=42)
    elif model_type == 'linear':
        if problem_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
    elif model_type == 'svm':
        if problem_type == 'classification':
            from sklearn.svm import SVC
            model = SVC(kernel='linear', random_state=42)
        else:
            from sklearn.svm import SVR
            model = SVR(kernel='linear')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit the model
    model.fit(numeric_df, target)
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
    else:
        raise ValueError("Model does not provide feature importances or coefficients")
    
    # Create Series of importances
    importances = pd.Series(importances, index=numeric_df.columns)
    
    # Select features
    if k is not None:
        # Select top k features
        selected_features = importances.nlargest(k).index.tolist()
    elif threshold is not None:
        # Select features above threshold
        selected_features = importances[importances >= threshold].index.tolist()
    else:
        # Default to selecting all features
        selected_features = numeric_df.columns.tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


def select_by_variance(
    df: Any,
    threshold: float = 0.0,
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select features based on their variance.
    
    This method removes features with low variance, which can be considered as 
    having little information.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    threshold : float, default=0.0
        Features with variance <= threshold will be removed.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [1, 1, 1, 1, 1],
    ...     'C': [1, 2, 1, 2, 1]
    ... })
    >>> select_by_variance(df, threshold=0.1, return_names_only=True)
    ['A', 'C']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Create variance selector
    selector = VarianceThreshold(threshold)
    selector.fit(numeric_df)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = numeric_df.columns[selected_indices].tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


def select_by_mutual_info(
    df: Any,
    target: Union[pd.Series, np.ndarray],
    k: Optional[int] = 10,
    problem_type: Optional[str] = None,
    discrete_features: Union[str, List[bool], List[int]] = 'auto',
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select features based on mutual information with the target.
    
    Mutual information measures the dependency between variables. It is always 
    non-negative, with higher values indicating stronger dependency.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[pd.Series, np.ndarray]
        The target variable.
    k : Optional[int], default=10
        Number of top features to select.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    discrete_features : Union[str, List[bool], List[int]], default='auto'
        Indicates which features are discrete. Options are:
        - 'auto': Automatically detect discrete features
        - array of booleans: Boolean mask indicating discrete features
        - array of indices: Array of indices of discrete features
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1]
    ... })
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> select_by_mutual_info(df, target, k=2, return_names_only=True)
    ['A', 'B']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(target, 'nunique'):
            if target.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            target_array = np.array(target)
            if np.issubdtype(target_array.dtype, np.integer) and len(np.unique(target_array)) < 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
    
    # Select the mutual information function based on problem type
    if problem_type == 'classification':
        score_func = mutual_info_classif
    else:
        score_func = mutual_info_regression
    
    # Create selector
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(numeric_df, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = numeric_df.columns[selected_indices].tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


def select_by_kbest(
    df: Any,
    target: Union[pd.Series, np.ndarray],
    k: int = 10,
    problem_type: Optional[str] = None,
    score_func: Optional[Callable] = None,
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select top k features based on univariate statistical tests.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[pd.Series, np.ndarray]
        The target variable.
    k : int, default=10
        Number of top features to select.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    score_func : Optional[Callable], default=None
        Function for scoring features. If None, f_classif or f_regression will be used
        based on the problem type.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1]
    ... })
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> select_by_kbest(df, target, k=2, return_names_only=True)
    ['A', 'B']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Ensure k is valid
    if k > len(numeric_df.columns):
        k = len(numeric_df.columns)
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(target, 'nunique'):
            if target.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            target_array = np.array(target)
            if np.issubdtype(target_array.dtype, np.integer) and len(np.unique(target_array)) < 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
    
    # Set score function if not provided
    if score_func is None:
        if problem_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
    
    # Create selector
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(numeric_df, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = numeric_df.columns[selected_indices].tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


def select_by_percentile(
    df: Any,
    target: Union[pd.Series, np.ndarray],
    percentile: int = 10,
    problem_type: Optional[str] = None,
    score_func: Optional[Callable] = None,
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select top percentile of features based on univariate statistical tests.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[pd.Series, np.ndarray]
        The target variable.
    percentile : int, default=10
        Percentage of features to keep.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    score_func : Optional[Callable], default=None
        Function for scoring features. If None, f_classif or f_regression will be used
        based on the problem type.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1],
    ...     'D': [2, 2, 2, 2, 2]
    ... })
    >>> target = np.array([0, 0, 1, 1, 1])
    >>> select_by_percentile(df, target, percentile=50, return_names_only=True)
    ['A', 'B']
    """
    # Check dataframe type and convert to pandas if needed
    df_type = check_dataframe_type(df)
    if df_type != 'pandas':
        df_pandas = convert_dataframe(df, 'pandas')
    else:
        df_pandas = df
    
    # Only use numeric columns
    numeric_df = df_pandas.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        raise ValueError("No numeric columns found in dataframe")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(target, 'nunique'):
            if target.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            target_array = np.array(target)
            if np.issubdtype(target_array.dtype, np.integer) and len(np.unique(target_array)) < 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
    
    # Set score function if not provided
    if score_func is None:
        if problem_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
    
    # Create selector
    selector = SelectPercentile(score_func=score_func, percentile=percentile)
    selector.fit(numeric_df, target)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = numeric_df.columns[selected_indices].tolist()
    
    # Return results
    if return_names_only:
        return selected_features
    else:
        return df_pandas[selected_features]


class RecursiveFeatureEliminationCV(FeatureSelector):
    """Feature selector that performs Recursive Feature Elimination with Cross-Validation.
    
    This selector uses an estimator's feature importance to recursively eliminate 
    features while using cross-validation to determine the optimal number of features.
    
    Attributes
    ----------
    estimator : BaseEstimator
        The base estimator from which the feature importances will be determined.
    n_features_to_select : int or float, default=None
        The minimum number of features to be selected. If None, half of the features
        are selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to select.
    cv : int or cross-validation generator, default=5
        Determines the cross-validation splitting strategy.
    scoring : str or callable, default=None
        A str (see sklearn.metrics.SCORERS) or a scorer callable object that returns
        a single value. If None, the estimator's default scorer is used.
    verbose : int, default=0
        Controls verbosity of output.
    step : int or float, default=1
        If >0, number of features to remove at each iteration. If int, removes
        that many features. If float between 0 and 1, removes that proportion
        of features.
    min_features_to_select : int, default=1
        The minimum number of features to be selected.
    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance from the estimator. If callable,
        it is called on the estimator and returns the feature importance.
    selected_features_ : list
        Names of selected features after calling fit.
    cv_results_ : dict
        A dict with keys:
        - 'mean_test_score': Mean scores across CV folds.
        - 'std_test_score': Standard deviation of scores across CV folds.
        - 'n_features': Number of features used in each CV fold.
    n_features_ : int
        The number of selected features.
    ranking_ : ndarray of shape (n_features,)
        The feature ranking. Features are ranked according to their
        elimination order, i.e. the larger the ranking, the later the feature was
        eliminated.
    grid_scores_ : ndarray of shape (n_subsets_of_features,)
        The cross-validation scores for each number of features.
    """
    
    def __init__(self, 
                 estimator: BaseEstimator,
                 n_features_to_select: Optional[Union[int, float]] = None,
                 cv: Union[int, Any] = 5,
                 scoring: Optional[Union[str, Callable]] = None,
                 verbose: int = 0,
                 step: Union[int, float] = 1,
                 min_features_to_select: int = 1,
                 importance_getter: Union[str, Callable] = 'auto',
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """Initialize the RecursiveFeatureEliminationCV.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator from which the feature importances will be determined.
        n_features_to_select : int or float, default=None
            The minimum number of features to be selected. If None, half of the features
            are selected. If integer, the parameter is the absolute number of features
            to select. If float between 0 and 1, it is the fraction of features to select.
        cv : int or cross-validation generator, default=5
            Determines the cross-validation splitting strategy.
        scoring : str or callable, default=None
            A str (see sklearn.metrics.SCORERS) or a scorer callable object that returns
            a single value. If None, the estimator's default scorer is used.
        verbose : int, default=0
            Controls verbosity of output.
        step : int or float, default=1
            If >0, number of features to remove at each iteration.
        min_features_to_select : int, default=1
            The minimum number of features to be selected.
        importance_getter : str or callable, default='auto'
            If 'auto', uses the feature importance from the estimator.
        random_state : int or RandomState, default=None
            Controls the randomness of the estimator and CV splitter.
        """
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.importance_getter = importance_getter
        self.random_state = random_state
        
        # Attributes that will be set during fit
        self.cv_results_ = None
        self.n_features_ = None
        self.ranking_ = None
        self.grid_scores_ = None
        self.rfecv_ = None
        self.feature_importances_ = None
        
    def fit(self, X, y=None, **kwargs):
        """Fit the RFECV model to identify the most relevant features.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters passed to the estimator
            
        Returns
        -------
        self : RecursiveFeatureEliminationCV
            The fitted selector
        """
        # Check dataframe type and convert to pandas if needed
        X_type = check_dataframe_type(X)
        if X_type != 'pandas':
            X_pandas = convert_dataframe(X, 'pandas')
        else:
            X_pandas = X
        
        # Only use numeric columns
        numeric_X = X_pandas.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) == 0:
            raise ValueError("No numeric columns found in dataframe")
        
        # Determine CV strategy based on problem type
        if y is not None:
            if hasattr(y, 'nunique') and y.nunique() < 10:  # Classification
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:  # Regression
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv
            
        # Initialize RFECV
        self.rfecv_ = RFECV(
            estimator=clone(self.estimator),
            min_features_to_select=self.min_features_to_select,
            step=self.step,
            cv=cv,
            scoring=self.scoring,
            verbose=self.verbose,
            n_jobs=-1  # Use all available cores
        )
        
        # Fit RFECV
        self.rfecv_.fit(numeric_X, y, **kwargs)
        
        # Store attributes from RFECV
        self.n_features_ = self.rfecv_.n_features_
        self.ranking_ = self.rfecv_.ranking_
        self.grid_scores_ = self.rfecv_.grid_scores_
        self.cv_results_ = {
            'mean_test_score': self.rfecv_.grid_scores_,
            'std_test_score': np.std(self.rfecv_.cv_results_['split*_test_score'], axis=1) 
                if hasattr(self.rfecv_, 'cv_results_') and 'split0_test_score' in self.rfecv_.cv_results_ 
                else np.zeros_like(self.rfecv_.grid_scores_),
            'n_features': np.arange(len(self.rfecv_.grid_scores_)) + self.min_features_to_select
        }
        
        # Get feature importances if possible
        if hasattr(self.rfecv_.estimator_, 'feature_importances_'):
            self.feature_importances_ = self.rfecv_.estimator_.feature_importances_
        elif hasattr(self.rfecv_.estimator_, 'coef_'):
            self.feature_importances_ = np.abs(self.rfecv_.estimator_.coef_)
            if self.feature_importances_.ndim > 1:
                self.feature_importances_ = np.mean(self.feature_importances_, axis=0)
        else:
            self.feature_importances_ = None
            
        # Get selected features
        self.selected_features_ = numeric_X.columns[self.rfecv_.support_].tolist()
        
        # Create feature importance scores if possible
        if self.feature_importances_ is not None:
            self.scores_ = pd.Series(
                self.feature_importances_,
                index=numeric_X.columns
            )
        
        return self
    
    def plot_cv_results(self, ax=None, **kwargs):
        """Plot the cross-validation results.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The axes on which to plot. If None, a new figure and axes are created.
        **kwargs : dict
            Additional parameters passed to the plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.cv_results_ is None:
            raise ValueError("No CV results available. Call fit first.")
            
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot mean CV score vs number of features
        ax.plot(
            self.cv_results_['n_features'],
            self.cv_results_['mean_test_score'],
            'o-',
            **kwargs
        )
        
        # Add error bars if available
        if np.any(self.cv_results_['std_test_score'] > 0):
            ax.fill_between(
                self.cv_results_['n_features'],
                self.cv_results_['mean_test_score'] - self.cv_results_['std_test_score'],
                self.cv_results_['mean_test_score'] + self.cv_results_['std_test_score'],
                alpha=0.2
            )
            
        # Mark the selected number of features
        optimal_idx = np.argmax(self.cv_results_['mean_test_score'])
        optimal_n_features = self.cv_results_['n_features'][optimal_idx]
        optimal_score = self.cv_results_['mean_test_score'][optimal_idx]
        
        ax.axvline(
            x=optimal_n_features,
            color='r',
            linestyle='--',
            alpha=0.7,
            label=f'Optimal: {optimal_n_features} features'
        )
        ax.plot(
            optimal_n_features,
            optimal_score,
            'ro',
            markersize=10
        )
        
        # Add labels and legend
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('Recursive Feature Elimination with Cross-Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_feature_ranking(self):
        """Get features ranked by their elimination order.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features and their rankings
        """
        if self.ranking_ is None:
            raise ValueError("No feature ranking available. Call fit first.")
            
        # Create dataframe with feature names and rankings
        ranking_df = pd.DataFrame({
            'Feature': self.rfecv_.estimator.feature_names_in_,
            'Ranking': self.ranking_
        })
        
        # Add feature importance if available
        if self.feature_importances_ is not None:
            ranking_df['Importance'] = self.feature_importances_
            
        # Sort by ranking (smaller is better)
        ranking_df = ranking_df.sort_values('Ranking')
        
        return ranking_df


class StabilitySelector(FeatureSelector):
    """Feature selector that uses stability selection.
    
    Stability selection is a method for feature selection that is based on
    subsampling in combination with selection algorithms. It addresses the
    instability of feature selection by running the selection algorithm on
    different subsamples of the data and considering the frequency of feature
    selection across these subsamples.
    
    Attributes
    ----------
    estimator : BaseEstimator
        The base estimator for feature selection.
    n_features : int or float, default=None
        Number of features to select. If None, half of the features are selected.
        If int, it is the absolute number of features to select. If float between
        0 and 1, it is the fraction of features to select.
    threshold : float, default=0.5
        Threshold for selection. Features that are selected in more than
        threshold fraction of subsamples are considered stable.
    subsample_size : float, default=0.5
        Fraction of samples to use in each subsample.
    n_subsamples : int, default=100
        Number of subsamples to generate.
    random_state : int or RandomState, default=None
        Controls the randomness of the subsampling process.
    selected_features_ : list
        Names of selected features after calling fit.
    selection_frequencies_ : pd.Series
        The frequency of selection for each feature.
    """
    
    def __init__(self,
                 estimator: BaseEstimator,
                 n_features: Optional[Union[int, float]] = None,
                 threshold: float = 0.5,
                 subsample_size: float = 0.5,
                 n_subsamples: int = 100,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """Initialize the StabilitySelector.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator for feature selection.
        n_features : int or float, default=None
            Number of features to select. If None, half of the features are selected.
            If int, it is the absolute number of features to select. If float between
            0 and 1, it is the fraction of features to select.
        threshold : float, default=0.5
            Threshold for selection. Features that are selected in more than
            threshold fraction of subsamples are considered stable.
        subsample_size : float, default=0.5
            Fraction of samples to use in each subsample.
        n_subsamples : int, default=100
            Number of subsamples to generate.
        random_state : int or RandomState, default=None
            Controls the randomness of the subsampling process.
        """
        super().__init__()
        self.estimator = estimator
        self.n_features = n_features
        self.threshold = threshold
        self.subsample_size = subsample_size
        self.n_subsamples = n_subsamples
        self.random_state = random_state
        
        # Attributes that will be set during fit
        self.selection_frequencies_ = None
    
    def fit(self, X, y=None, **kwargs):
        """Fit the stability selection model.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters passed to the estimator
            
        Returns
        -------
        self : StabilitySelector
            The fitted selector
        """
        # Check dataframe type and convert to pandas if needed
        X_type = check_dataframe_type(X)
        if X_type != 'pandas':
            X_pandas = convert_dataframe(X, 'pandas')
        else:
            X_pandas = X
        
        # Only use numeric columns
        numeric_X = X_pandas.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) == 0:
            raise ValueError("No numeric columns found in dataframe")
        
        # Initialize random state
        random_state = check_random_state(self.random_state)
        
        # Calculate number of features to select if given as a fraction
        n_samples, n_features = numeric_X.shape
        if self.n_features is None:
            n_features_to_select = n_features // 2
        elif isinstance(self.n_features, float):
            n_features_to_select = max(1, int(self.n_features * n_features))
        else:
            n_features_to_select = min(self.n_features, n_features)
        
        # Calculate subsample size
        subsample_size = max(1, int(self.subsample_size * n_samples))
        
        # Initialize selection frequencies
        selection_frequencies = np.zeros(n_features)
        
        # Perform stability selection
        for i in range(self.n_subsamples):
            # Create random subsample
            subsample_indices = random_state.choice(
                np.arange(n_samples),
                size=subsample_size,
                replace=False
            )
            
            # Create subsample
            X_subsample = numeric_X.iloc[subsample_indices]
            if y is not None:
                if isinstance(y, pd.Series):
                    y_subsample = y.iloc[subsample_indices]
                else:
                    y_subsample = y[subsample_indices]
            else:
                y_subsample = None
            
            # Clone estimator to avoid modifying the original
            estimator = clone(self.estimator)
            
            # Fit estimator on subsample
            estimator.fit(X_subsample, y_subsample, **kwargs)
            
            # Get selected features
            if hasattr(estimator, 'get_support'):
                # For scikit-learn feature selectors
                support = estimator.get_support()
            elif hasattr(estimator, 'feature_importances_'):
                # For tree-based models
                importances = estimator.feature_importances_
                # Select top n_features_to_select features
                threshold = np.sort(importances)[-n_features_to_select]
                support = importances >= threshold
            elif hasattr(estimator, 'coef_'):
                # For linear models
                coef = estimator.coef_
                if coef.ndim > 1:
                    # For multi-class or multi-target problems
                    coef = np.abs(coef).mean(axis=0)
                else:
                    coef = np.abs(coef)
                # Select top n_features_to_select features
                threshold = np.sort(coef)[-n_features_to_select]
                support = coef >= threshold
            else:
                raise ValueError(
                    "Estimator has no feature selection capability. "
                    "It must have get_support, feature_importances_, or coef_ attribute."
                )
            
            # Update selection frequencies
            selection_frequencies += support
        
        # Normalize selection frequencies
        selection_frequencies /= self.n_subsamples
        
        # Store selection frequencies
        self.selection_frequencies_ = pd.Series(
            selection_frequencies,
            index=numeric_X.columns
        )
        
        # Select features based on threshold
        selected_indices = np.where(selection_frequencies >= self.threshold)[0]
        self.selected_features_ = numeric_X.columns[selected_indices].tolist()
        
        # Store scores
        self.scores_ = self.selection_frequencies_.copy()
        
        return self
    
    def plot_stability_path(self, ax=None, **kwargs):
        """Plot the stability path.
        
        The stability path shows the selection frequency of each feature
        as the selection threshold varies.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The axes on which to plot. If None, a new figure and axes are created.
        **kwargs : dict
            Additional parameters passed to the plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.selection_frequencies_ is None:
            raise ValueError("No selection frequencies available. Call fit first.")
            
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        # Sort features by selection frequency
        sorted_freq = self.selection_frequencies_.sort_values(ascending=False)
        
        # Plot selection frequencies
        ax.bar(
            range(len(sorted_freq)),
            sorted_freq.values,
            **kwargs
        )
        
        # Add threshold line
        ax.axhline(
            y=self.threshold,
            color='r',
            linestyle='--',
            alpha=0.7,
            label=f'Selection Threshold: {self.threshold}'
        )
        
        # Add labels and legend
        ax.set_xlabel('Features')
        ax.set_ylabel('Selection Frequency')
        ax.set_title('Stability Selection Path')
        ax.set_xticks(range(len(sorted_freq)))
        ax.set_xticklabels(sorted_freq.index, rotation=90)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_support_probabilities(self):
        """Get the selection probability for each feature.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features and their selection probabilities
        """
        if self.selection_frequencies_ is None:
            raise ValueError("No selection frequencies available. Call fit first.")
            
        # Create dataframe with feature names and selection frequencies
        prob_df = pd.DataFrame({
            'Feature': self.selection_frequencies_.index,
            'Selection_Probability': self.selection_frequencies_.values
        })
        
        # Add selection status
        prob_df['Selected'] = prob_df['Selection_Probability'] >= self.threshold
        
        # Sort by selection probability
        prob_df = prob_df.sort_values('Selection_Probability', ascending=False)
        
        return prob_df


class GeneticFeatureSelector(FeatureSelector):
    """Feature selector that uses a genetic algorithm to select features.
    
    This selector uses a genetic algorithm to evolve a population of feature
    subsets, evaluating each subset using a specified estimator and scoring
    function. The genetic algorithm performs selection, crossover, and mutation
    operations to evolve the population toward better feature subsets.
    
    Note: This class requires the DEAP library to be installed.
    
    Attributes
    ----------
    estimator : BaseEstimator
        The base estimator for evaluating feature subsets.
    n_features_to_select : int or float, default=None
        Target number of features to select. If None, the algorithm will
        optimize for the best performing subset regardless of size.
        If int, the algorithm will try to select exactly that many features.
        If float between 0 and 1, it is the fraction of features to select.
    scoring : str or callable, default='accuracy'
        Scoring metric to use for evaluating feature subsets.
    cv : int or cross-validation generator, default=5
        Cross-validation strategy for evaluating feature subsets.
    population_size : int, default=50
        Size of the population in the genetic algorithm.
    n_generations : int, default=40
        Number of generations to evolve.
    crossover_probability : float, default=0.5
        Probability of crossover between two individuals.
    mutation_probability : float, default=0.2
        Probability of mutation for each attribute in an individual.
    random_state : int or RandomState, default=None
        Controls the randomness of the genetic algorithm.
    selected_features_ : list
        Names of selected features after calling fit.
    feature_importances_ : pd.Series
        Importance of each feature based on selection frequency.
    best_individual_ : list
        Binary representation of the best individual found.
    best_score_ : float
        Score of the best individual found.
    generations_history_ : list
        History of the best and average scores for each generation.
    """
    
    def __init__(self,
                 estimator: BaseEstimator,
                 n_features_to_select: Optional[Union[int, float]] = None,
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: Union[int, Any] = 5,
                 population_size: int = 50,
                 n_generations: int = 40,
                 crossover_probability: float = 0.5,
                 mutation_probability: float = 0.2,
                 tournament_size: int = 3,
                 early_stopping_generations: Optional[int] = None,
                 early_stopping_threshold: float = 1e-4,
                 penalty_factor: float = 0.001,
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """Initialize the GeneticFeatureSelector.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator for evaluating feature subsets.
        n_features_to_select : int or float, default=None
            Target number of features to select. If None, the algorithm will
            optimize for the best performing subset regardless of size.
            If int, the algorithm will try to select exactly that many features.
            If float between 0 and 1, it is the fraction of features to select.
        scoring : str or callable, default='accuracy'
            Scoring metric to use for evaluating feature subsets.
        cv : int or cross-validation generator, default=5
            Cross-validation strategy for evaluating feature subsets.
        population_size : int, default=50
            Size of the population in the genetic algorithm.
        n_generations : int, default=40
            Number of generations to evolve.
        crossover_probability : float, default=0.5
            Probability of crossover between two individuals.
        mutation_probability : float, default=0.2
            Probability of mutation for each attribute in an individual.
        tournament_size : int, default=3
            Number of individuals participating in tournament selection.
        early_stopping_generations : Optional[int], default=None
            Number of generations with no improvement to wait before stopping.
            If None, early stopping is not used.
        early_stopping_threshold : float, default=1e-4
            Minimum improvement required to reset early stopping counter.
        penalty_factor : float, default=0.001
            Penalty factor for the number of features. Higher values encourage
            smaller feature subsets.
        random_state : int or RandomState, default=None
            Controls the randomness of the genetic algorithm.
        """
        if not DEAP_AVAILABLE:
            raise ImportError(
                "The DEAP library is required for genetic feature selection. "
                "Install it using: pip install deap"
            )
            
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_threshold = early_stopping_threshold
        self.penalty_factor = penalty_factor
        self.random_state = random_state
        
        # Attributes that will be set during fit
        self.best_individual_ = None
        self.best_score_ = None
        self.generations_history_ = []
        self.feature_importances_ = None
    
    def fit(self, X, y=None, **kwargs):
        """Fit the genetic feature selection model.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters passed to the estimator
            
        Returns
        -------
        self : GeneticFeatureSelector
            The fitted selector
        """
        # Check dataframe type and convert to pandas if needed
        X_type = check_dataframe_type(X)
        if X_type != 'pandas':
            X_pandas = convert_dataframe(X, 'pandas')
        else:
            X_pandas = X
        
        # Only use numeric columns
        numeric_X = X_pandas.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) == 0:
            raise ValueError("No numeric columns found in dataframe")
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        
        # Calculate target number of features if given as a fraction
        n_features = numeric_X.shape[1]
        if self.n_features_to_select is None:
            target_n_features = None
        elif isinstance(self.n_features_to_select, float):
            target_n_features = max(1, int(self.n_features_to_select * n_features))
        else:
            target_n_features = min(self.n_features_to_select, n_features)
        
        # Configure DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Attribute generator: binary values representing feature selection
        toolbox.register("attr_bool", np.random.choice, [0, 1])
        
        # Define initial population: random binary lists
        toolbox.register(
            "individual", 
            tools.initRepeat, 
            creator.Individual, 
            toolbox.attr_bool, 
            n=n_features
        )
        toolbox.register(
            "population", 
            tools.initRepeat, 
            list, 
            toolbox.individual
        )
        
        # Define evaluation function
        def evaluate_subset(individual):
            # Ensure at least one feature is selected
            if sum(individual) == 0:
                return (-np.inf,)
                
            # Get selected feature indices
            selected_indices = [i for i, val in enumerate(individual) if val == 1]
            
            # If we have a target number of features, add penalty based on difference
            if target_n_features is not None:
                feature_count_penalty = self.penalty_factor * abs(sum(individual) - target_n_features)
            else:
                # Otherwise, penalize for larger feature sets
                feature_count_penalty = self.penalty_factor * sum(individual)
            
            # Define cross-validation strategy
            if isinstance(self.cv, int):
                if hasattr(y, 'nunique') and y.nunique() < 10:  # Classification
                    cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                else:  # Regression
                    cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                cv = self.cv
            
            # Extract selected features
            X_subset = numeric_X.iloc[:, selected_indices]
            
            # Evaluate using cross-validation
            try:
                scores = []
                for train_idx, test_idx in cv.split(X_subset, y):
                    X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
                    if y is not None:
                        if isinstance(y, pd.Series):
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        else:
                            y_train, y_test = y[train_idx], y[test_idx]
                    else:
                        y_train, y_test = None, None
                    
                    # Clone estimator to avoid modifying the original
                    estimator = clone(self.estimator)
                    
                    # Fit estimator
                    estimator.fit(X_train, y_train)
                    
                    # Score estimator
                    if isinstance(self.scoring, str):
                        if self.scoring == 'accuracy' and hasattr(estimator, 'score'):
                            score = estimator.score(X_test, y_test)
                        else:
                            # Use scikit-learn's scoring function
                            from sklearn.metrics import get_scorer
                            scorer = get_scorer(self.scoring)
                            score = scorer(estimator, X_test, y_test)
                    else:
                        # Custom scoring function
                        score = self.scoring(estimator, X_test, y_test)
                    
                    scores.append(score)
                
                # Calculate mean score
                mean_score = np.mean(scores)
                
                # Return fitness score with penalty
                return (mean_score - feature_count_penalty,)
            except Exception as e:
                print(f"Error in evaluation: {e}")
                return (-np.inf,)
        
        # Register genetic operators
        toolbox.register("evaluate", evaluate_subset)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutation_probability)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Create initial population
        pop = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual and score
        best_individual = tools.selBest(pop, 1)[0]
        best_score = best_individual.fitness.values[0]
        
        # Initialize early stopping counter if enabled
        if self.early_stopping_generations is not None:
            early_stopping_counter = 0
            
        # Variables for tracking generation statistics
        gen_fitnesses = [ind.fitness.values[0] for ind in pop]
        self.generations_history_.append({
            'gen': 0,
            'best_score': best_score,
            'avg_score': np.mean(gen_fitnesses),
            'feature_count': sum(best_individual)
        })
        
        # Evolve population
        for g in range(1, self.n_generations + 1):
            # Select next generation
            offspring = toolbox.select(pop, len(pop))
            
            # Clone selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.crossover_probability:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    # Invalidate fitness values
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Apply mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_probability:
                    offspring[i], = toolbox.mutate(offspring[i])
                    # Invalidate fitness values
                    del offspring[i].fitness.values
            
            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Update best individual and score
            current_best = tools.selBest(pop, 1)[0]
            current_best_score = current_best.fitness.values[0]
            
            if current_best_score > best_score:
                improvement = current_best_score - best_score
                best_individual = toolbox.clone(current_best)
                best_score = current_best_score
                
                # Reset early stopping counter if significant improvement
                if self.early_stopping_generations is not None:
                    if improvement > self.early_stopping_threshold:
                        early_stopping_counter = 0
            elif self.early_stopping_generations is not None:
                # Increment early stopping counter
                early_stopping_counter += 1
                if early_stopping_counter >= self.early_stopping_generations:
                    print(f"Early stopping at generation {g}")
                    break
            
            # Update generation statistics
            gen_fitnesses = [ind.fitness.values[0] for ind in pop]
            self.generations_history_.append({
                'gen': g,
                'best_score': best_score,
                'avg_score': np.mean(gen_fitnesses),
                'feature_count': sum(best_individual)
            })
        
        # Store best individual and score
        self.best_individual_ = best_individual
        self.best_score_ = best_score
        
        # Get selected features
        selected_indices = [i for i, val in enumerate(self.best_individual_) if val == 1]
        self.selected_features_ = numeric_X.columns[selected_indices].tolist()
        
        # Calculate feature importances based on occurrence in population
        importances = np.zeros(n_features)
        for ind in pop:
            importances += np.array(ind)
        
        importances /= len(pop)
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            importances,
            index=numeric_X.columns
        )
        
        # Store scores (same as feature importances)
        self.scores_ = self.feature_importances_.copy()
        
        # Clean up DEAP globals to avoid conflicts with future runs
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        return self
    
    def plot_fitness_evolution(self, ax=None, **kwargs):
        """Plot the fitness evolution across generations.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The axes on which to plot. If None, a new figure and axes are created.
        **kwargs : dict
            Additional parameters passed to the plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if not self.generations_history_:
            raise ValueError("No generations history available. Call fit first.")
            
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Extract data from generations history
        generations = [g['gen'] for g in self.generations_history_]
        best_scores = [g['best_score'] for g in self.generations_history_]
        avg_scores = [g['avg_score'] for g in self.generations_history_]
        feature_counts = [g['feature_count'] for g in self.generations_history_]
        
        # Create second y-axis for feature count
        ax2 = ax.twinx()
        
        # Plot best and average scores
        ax.plot(
            generations,
            best_scores,
            'b-',
            label='Best Score',
            **kwargs
        )
        ax.plot(
            generations,
            avg_scores,
            'g--',
            label='Average Score',
            alpha=0.7,
            **kwargs
        )
        
        # Plot feature count
        ax2.plot(
            generations,
            feature_counts,
            'r:',
            label='Feature Count',
            alpha=0.7,
            **kwargs
        )
        
        # Add labels and legend
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Score')
        ax2.set_ylabel('Feature Count')
        ax.set_title('Genetic Algorithm Fitness Evolution')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_feature_importance(self):
        """Get the importance of each feature based on selection frequency.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features and their importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("No feature importances available. Call fit first.")
            
        # Create dataframe with feature names and importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_importances_.index,
            'Importance': self.feature_importances_.values,
            'Selected': [feat in self.selected_features_ for feat in self.feature_importances_.index]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df


class MultiObjectiveFeatureSelector(FeatureSelector):
    """Feature selector that uses multi-objective optimization for feature selection.
    
    This selector tries to find the best trade-off between multiple competing objectives,
    such as model performance, number of features, and model complexity. It uses
    multi-objective optimization algorithms to find a set of Pareto-optimal solutions,
    which represent different trade-offs between the objectives.
    
    Note: This class requires the pymoo library to be installed.
    
    Attributes
    ----------
    estimator : BaseEstimator
        The base estimator for evaluating feature subsets.
    n_features_to_select : int or float, default=None
        The number of features to select. If None, the algorithm will optimize
        for the best trade-offs between performance and number of features.
        If int, it is the absolute number of features to select. If float between
        0 and 1, it is the fraction of features to select.
    scoring : str or callable, default='accuracy'
        Scoring metric to use for evaluating feature subsets.
    cv : int or cross-validation generator, default=5
        Cross-validation strategy for evaluating feature subsets.
    objectives : list, default=['score', 'n_features']
        List of objectives to optimize. Available options:
        - 'score': maximize model performance
        - 'n_features': minimize number of features
        - 'complexity': minimize model complexity (if applicable)
    population_size : int, default=100
        Size of the population in the multi-objective optimization algorithm.
    n_generations : int, default=100
        Number of generations to evolve.
    selected_solution_method : str, default='knee'
        Method to select a single solution from the Pareto front:
        - 'knee': select the knee point of the Pareto front
        - 'best_score': select the solution with the best score
        - 'min_features': select the solution with the fewest features
        - 'compromise': select a solution that balances all objectives
    random_state : int or RandomState, default=None
        Controls the randomness of the optimization algorithm.
    selected_features_ : list
        Names of selected features after calling fit.
    pareto_front_ : pd.DataFrame
        Dataframe with the Pareto-optimal solutions.
    feature_importances_ : pd.Series
        Importance of each feature based on selection frequency in the Pareto front.
    """
    
    def __init__(self,
                 estimator: BaseEstimator,
                 n_features_to_select: Optional[Union[int, float]] = None,
                 scoring: Union[str, Callable] = 'accuracy',
                 cv: Union[int, Any] = 5,
                 objectives: List[str] = ['score', 'n_features'],
                 population_size: int = 100,
                 n_generations: int = 100,
                 selected_solution_method: str = 'knee',
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        """Initialize the MultiObjectiveFeatureSelector.
        
        Parameters
        ----------
        estimator : BaseEstimator
            The base estimator for evaluating feature subsets.
        n_features_to_select : int or float, default=None
            The number of features to select. If None, the algorithm will optimize
            for the best trade-offs between performance and number of features.
            If int, it is the absolute number of features to select. If float between
            0 and 1, it is the fraction of features to select.
        scoring : str or callable, default='accuracy'
            Scoring metric to use for evaluating feature subsets.
        cv : int or cross-validation generator, default=5
            Cross-validation strategy for evaluating feature subsets.
        objectives : list, default=['score', 'n_features']
            List of objectives to optimize. Available options:
            - 'score': maximize model performance
            - 'n_features': minimize number of features
            - 'complexity': minimize model complexity (if applicable)
        population_size : int, default=100
            Size of the population in the multi-objective optimization algorithm.
        n_generations : int, default=100
            Number of generations to evolve.
        selected_solution_method : str, default='knee'
            Method to select a single solution from the Pareto front:
            - 'knee': select the knee point of the Pareto front
            - 'best_score': select the solution with the best score
            - 'min_features': select the solution with the fewest features
            - 'compromise': select a solution that balances all objectives
        random_state : int or RandomState, default=None
            Controls the randomness of the optimization algorithm.
        """
        if not PYMOO_AVAILABLE:
            raise ImportError(
                "The pymoo library is required for multi-objective feature selection. "
                "Install it using: pip install pymoo"
            )
            
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.objectives = objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.selected_solution_method = selected_solution_method
        self.random_state = random_state
        
        # Validate objectives
        valid_objectives = ['score', 'n_features', 'complexity']
        for obj in self.objectives:
            if obj not in valid_objectives:
                raise ValueError(f"Invalid objective: {obj}. "
                                f"Valid objectives are: {valid_objectives}")
        
        # Attributes that will be set during fit
        self.pareto_front_ = None
        self.feature_importances_ = None
        self.best_solution_ = None
    
    def fit(self, X, y=None, **kwargs):
        """Fit the multi-objective feature selection model.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        y : pd.Series or np.ndarray, optional
            The target variable
        **kwargs : dict
            Additional parameters passed to the estimator
            
        Returns
        -------
        self : MultiObjectiveFeatureSelector
            The fitted selector
        """
        # Check dataframe type and convert to pandas if needed
        X_type = check_dataframe_type(X)
        if X_type != 'pandas':
            X_pandas = convert_dataframe(X, 'pandas')
        else:
            X_pandas = X
        
        # Only use numeric columns
        numeric_X = X_pandas.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) == 0:
            raise ValueError("No numeric columns found in dataframe")
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate target number of features if given as a fraction
        n_features = numeric_X.shape[1]
        if self.n_features_to_select is None:
            target_n_features = None
        elif isinstance(self.n_features_to_select, float):
            target_n_features = max(1, int(self.n_features_to_select * n_features))
        else:
            target_n_features = min(self.n_features_to_select, n_features)
        
        # Define cross-validation strategy
        if isinstance(self.cv, int):
            if hasattr(y, 'nunique') and y.nunique() < 10:  # Classification
                cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:  # Regression
                cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv
        
        # Define the multi-objective problem
        class FeatureSelectionProblem(Problem):
            def __init__(self, X, y, estimator, cv, scoring, objectives, **kwargs):
                self.X = X
                self.y = y
                self.estimator = estimator
                self.cv = cv
                self.scoring = scoring
                self.objectives = objectives
                self.kwargs = kwargs
                
                # Define problem dimensions
                n_var = X.shape[1]  # Number of variables (features)
                n_obj = len(objectives)  # Number of objectives
                n_constr = 0  # Number of constraints
                
                # Define variable bounds: binary variables (0 or 1)
                xl = np.zeros(n_var)
                xu = np.ones(n_var)
                
                # Initialize problem
                super().__init__(
                    n_var=n_var,
                    n_obj=n_obj,
                    n_constr=n_constr,
                    xl=xl,
                    xu=xu,
                    type_var=np.bool
                )
            
            def _evaluate(self, x, out, *args, **kwargs):
                # Initialize objective values
                n_individuals = x.shape[0]
                f = np.zeros((n_individuals, len(self.objectives)))
                
                # Evaluate each individual (feature subset)
                for i in range(n_individuals):
                    # Get selected feature indices
                    selected_indices = np.where(x[i] == 1)[0]
                    
                    # If no features selected, penalize heavily
                    if len(selected_indices) == 0:
                        f[i, :] = np.inf
                        continue
                    
                    # Extract selected features
                    X_subset = self.X.iloc[:, selected_indices]
                    
                    # Evaluate using cross-validation
                    try:
                        scores = []
                        for train_idx, test_idx in self.cv.split(X_subset, self.y):
                            X_train = X_subset.iloc[train_idx]
                            X_test = X_subset.iloc[test_idx]
                            
                            if self.y is not None:
                                if isinstance(self.y, pd.Series):
                                    y_train = self.y.iloc[train_idx]
                                    y_test = self.y.iloc[test_idx]
                                else:
                                    y_train = self.y[train_idx]
                                    y_test = self.y[test_idx]
                            else:
                                y_train, y_test = None, None
                            
                            # Clone estimator to avoid modifying the original
                            estimator = clone(self.estimator)
                            
                            # Fit estimator
                            estimator.fit(X_train, y_train, **self.kwargs)
                            
                            # Score estimator
                            if isinstance(self.scoring, str):
                                if self.scoring == 'accuracy' and hasattr(estimator, 'score'):
                                    score = estimator.score(X_test, y_test)
                                else:
                                    # Use scikit-learn's scoring function
                                    from sklearn.metrics import get_scorer
                                    scorer = get_scorer(self.scoring)
                                    score = scorer(estimator, X_test, y_test)
                            else:
                                # Custom scoring function
                                score = self.scoring(estimator, X_test, y_test)
                            
                            scores.append(score)
                        
                        # Calculate mean score
                        mean_score = np.mean(scores)
                        
                        # Set objective values
                        obj_idx = 0
                        for obj in self.objectives:
                            if obj == 'score':
                                # Negative because pymoo minimizes by default
                                f[i, obj_idx] = -mean_score
                            elif obj == 'n_features':
                                f[i, obj_idx] = len(selected_indices)
                            elif obj == 'complexity':
                                # Measure model complexity (if applicable)
                                if hasattr(estimator, 'get_depth'):
                                    # For tree-based models
                                    f[i, obj_idx] = estimator.get_depth()
                                elif hasattr(estimator, 'n_layers_'):
                                    # For neural networks
                                    f[i, obj_idx] = estimator.n_layers_
                                else:
                                    # Default to number of features
                                    f[i, obj_idx] = len(selected_indices)
                            
                            obj_idx += 1
                        
                    except Exception as e:
                        print(f"Error in evaluation: {e}")
                        f[i, :] = np.inf
                
                out["F"] = f
        
        # Initialize the problem
        problem = FeatureSelectionProblem(
            X=numeric_X,
            y=y,
            estimator=self.estimator,
            cv=cv,
            scoring=self.scoring,
            objectives=self.objectives,
            **kwargs
        )
        
        # Initialize the algorithm
        algorithm = NSGA2(
            pop_size=self.population_size,
            eliminate_duplicates=True,
            sampling=np.random.random((self.population_size, n_features)) < 0.5
        )
        
        # Run the optimization
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            verbose=False,
            seed=self.random_state
        )
        
        # Extract Pareto front
        pareto_solutions = res.X
        pareto_fitness = res.F
        
        # Create dataframe with Pareto solutions
        pareto_data = []
        for i, solution in enumerate(pareto_solutions):
            # Get selected feature indices
            selected_indices = np.where(solution == 1)[0]
            selected_feature_names = numeric_X.columns[selected_indices].tolist()
            
            # Create solution record
            solution_data = {
                'solution_id': i,
                'n_features': len(selected_indices),
                'selected_features': selected_feature_names
            }
            
            # Add objective values
            for j, obj in enumerate(self.objectives):
                if obj == 'score':
                    # Convert back to positive (we negated for minimization)
                    solution_data['score'] = -pareto_fitness[i, j]
                else:
                    solution_data[obj] = pareto_fitness[i, j]
                    
            pareto_data.append(solution_data)
        
        # Create Pareto front dataframe
        self.pareto_front_ = pd.DataFrame(pareto_data)
        
        # Select a solution based on the specified method
        if self.selected_solution_method == 'knee':
            # Find the knee point of the Pareto front
            if 'score' in self.objectives and 'n_features' in self.objectives:
                # Create artificial utility
                score_idx = self.objectives.index('score')
                n_features_idx = self.objectives.index('n_features')
                
                # Normalize scores
                scores = -pareto_fitness[:, score_idx]  # Convert back to positive
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                # Normalize feature counts
                n_features = pareto_fitness[:, n_features_idx]
                n_features_norm = (n_features - n_features.min()) / (n_features.max() - n_features.min() + 1e-10)
                
                # Calculate utility as a combination of normalized scores and feature counts
                utility = scores_norm - n_features_norm
                
                # Select solution with highest utility
                best_idx = np.argmax(utility)
            else:
                # Default to best score
                score_idx = 0  # Assume first objective is score
                best_idx = np.argmin(pareto_fitness[:, score_idx])
        elif self.selected_solution_method == 'best_score':
            # Select solution with best score
            score_idx = self.objectives.index('score') if 'score' in self.objectives else 0
            best_idx = np.argmin(pareto_fitness[:, score_idx])
        elif self.selected_solution_method == 'min_features':
            # Select solution with fewest features
            n_features_idx = self.objectives.index('n_features') if 'n_features' in self.objectives else 0
            best_idx = np.argmin(pareto_fitness[:, n_features_idx])
        elif self.selected_solution_method == 'compromise':
            # Select a compromise solution
            # Normalize all objectives and find the solution with the best sum
            normalized_fitness = np.zeros_like(pareto_fitness)
            for j in range(pareto_fitness.shape[1]):
                if j == self.objectives.index('score') if 'score' in self.objectives else -1:
                    # For score (maximize), normalize and negate
                    normalized_fitness[:, j] = 1.0 - (
                        (pareto_fitness[:, j] - pareto_fitness[:, j].min()) / 
                        (pareto_fitness[:, j].max() - pareto_fitness[:, j].min() + 1e-10)
                    )
                else:
                    # For other objectives (minimize), normalize
                    normalized_fitness[:, j] = (
                        (pareto_fitness[:, j] - pareto_fitness[:, j].min()) / 
                        (pareto_fitness[:, j].max() - pareto_fitness[:, j].min() + 1e-10)
                    )
            
            # Find solution with best sum
            compromise_scores = np.sum(normalized_fitness, axis=1)
            best_idx = np.argmin(compromise_scores)
        else:
            raise ValueError(f"Invalid solution selection method: {self.selected_solution_method}")
        
        # Set best solution
        self.best_solution_ = pareto_solutions[best_idx]
        
        # Get selected features
        selected_indices = np.where(self.best_solution_ == 1)[0]
        self.selected_features_ = numeric_X.columns[selected_indices].tolist()
        
        # Calculate feature importances based on selection frequency in Pareto front
        importances = np.zeros(n_features)
        for solution in pareto_solutions:
            importances += solution
        
        importances /= len(pareto_solutions)
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            importances,
            index=numeric_X.columns
        )
        
        # Store scores (same as feature importances)
        self.scores_ = self.feature_importances_.copy()
        
        return self
    
    def plot_pareto_front(self, x_objective='n_features', y_objective='score', ax=None, **kwargs):
        """Plot the Pareto front.
        
        Parameters
        ----------
        x_objective : str, default='n_features'
            The objective to plot on the x-axis.
        y_objective : str, default='score'
            The objective to plot on the y-axis.
        ax : matplotlib.axes.Axes, default=None
            The axes on which to plot. If None, a new figure and axes are created.
        **kwargs : dict
            Additional parameters passed to the plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.pareto_front_ is None:
            raise ValueError("No Pareto front available. Call fit first.")
        
        # Check if objectives are in the Pareto front
        if x_objective not in self.pareto_front_.columns:
            raise ValueError(f"Objective {x_objective} not found in Pareto front")
        if y_objective not in self.pareto_front_.columns:
            raise ValueError(f"Objective {y_objective} not found in Pareto front")
            
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot Pareto front
        ax.scatter(
            self.pareto_front_[x_objective],
            self.pareto_front_[y_objective],
            alpha=0.7,
            **kwargs
        )
        
        # Find selected solution
        selected_solution = self.pareto_front_[
            self.pareto_front_['selected_features'].apply(
                lambda x: set(x) == set(self.selected_features_)
            )
        ]
        
        if not selected_solution.empty:
            # Highlight selected solution
            ax.scatter(
                selected_solution[x_objective],
                selected_solution[y_objective],
                color='r',
                s=100,
                marker='*',
                label='Selected Solution'
            )
        
        # Add labels and legend
        ax.set_xlabel(x_objective)
        ax.set_ylabel(y_objective)
        ax.set_title('Pareto Front')
        
        # For 'score', higher is better
        if y_objective == 'score':
            ax.invert_yaxis()
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        if not selected_solution.empty:
            ax.legend()
        
        return ax
    
    def select_from_pareto(self, method='knee'):
        """Select a solution from the Pareto front.
        
        Parameters
        ----------
        method : str, default='knee'
            Method to select a solution:
            - 'knee': select the knee point of the Pareto front
            - 'best_score': select the solution with the best score
            - 'min_features': select the solution with the fewest features
            - 'compromise': select a solution that balances all objectives
            
        Returns
        -------
        dict
            Dictionary with the selected solution information
        """
        if self.pareto_front_ is None:
            raise ValueError("No Pareto front available. Call fit first.")
            
        if method == 'knee':
            # Find the knee point of the Pareto front
            if 'score' in self.pareto_front_.columns and 'n_features' in self.pareto_front_.columns:
                # Create artificial utility
                scores = self.pareto_front_['score']
                scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
                
                n_features = self.pareto_front_['n_features']
                n_features_norm = (n_features - n_features.min()) / (n_features.max() - n_features.min() + 1e-10)
                
                # Calculate utility as a combination of normalized scores and feature counts
                utility = scores_norm - n_features_norm
                
                # Select solution with highest utility
                best_idx = utility.idxmax()
            else:
                # Default to best score
                best_idx = self.pareto_front_['score'].idxmax() if 'score' in self.pareto_front_.columns else 0
        elif method == 'best_score':
            # Select solution with best score
            best_idx = self.pareto_front_['score'].idxmax() if 'score' in self.pareto_front_.columns else 0
        elif method == 'min_features':
            # Select solution with fewest features
            best_idx = self.pareto_front_['n_features'].idxmin() if 'n_features' in self.pareto_front_.columns else 0
        elif method == 'compromise':
            # Select a compromise solution
            # Normalize all objectives and find the solution with the best sum
            normalized_scores = pd.DataFrame(index=self.pareto_front_.index)
            
            for col in self.objectives:
                if col in self.pareto_front_.columns:
                    if col == 'score':
                        # For score (maximize), normalize
                        normalized_scores[col] = (
                            (self.pareto_front_[col] - self.pareto_front_[col].min()) / 
                            (self.pareto_front_[col].max() - self.pareto_front_[col].min() + 1e-10)
                        )
                    else:
                        # For other objectives (minimize), normalize and negate
                        normalized_scores[col] = 1.0 - (
                            (self.pareto_front_[col] - self.pareto_front_[col].min()) / 
                            (self.pareto_front_[col].max() - self.pareto_front_[col].min() + 1e-10)
                        )
            
            # Find solution with best sum
            normalized_scores['sum'] = normalized_scores.sum(axis=1)
            best_idx = normalized_scores['sum'].idxmax()
        else:
            raise ValueError(f"Invalid solution selection method: {method}")
        
        # Update selected features based on the chosen solution
        self.selected_features_ = self.pareto_front_.loc[best_idx, 'selected_features']
        
        # Return the selected solution
        return self.pareto_front_.loc[best_idx].to_dict()


class TimeSeriesFeatureSelector(FeatureSelector):
    """Feature selector specifically designed for time series data.
    
    This selector implements time series-specific feature selection methods,
    taking into account the temporal dependencies in the data. It can select
    features based on causality, autocorrelation, and forecasting impact.
    
    Note: Some methods require the statsmodels library to be installed.
    
    Attributes
    ----------
    method : str
        The feature selection method to use: 'causality', 'autocorrelation',
        'forecasting_impact', or 'combined'.
    target : str
        The name of the target variable column.
    time_col : str
        The name of the time column.
    group_col : Optional[str]
        The name of the group column for panel data.
    max_lag : int
        Maximum lag to consider for causality and autocorrelation methods.
    threshold : float
        Threshold for feature selection.
    n_features : int or float
        Number of features to select.
    estimator : Optional[BaseEstimator]
        Estimator to use for forecasting impact method.
    selected_features_ : list
        Names of selected features after calling fit.
    p_values_ : pd.DataFrame
        P-values from causality tests.
    acf_values_ : pd.DataFrame
        Autocorrelation values.
    forecasting_scores_ : pd.Series
        Forecasting impact scores.
    """
    
    def __init__(self,
                 method: str = 'combined',
                 target: Optional[str] = None,
                 time_col: Optional[str] = None,
                 group_col: Optional[str] = None,
                 max_lag: int = 5,
                 threshold: float = 0.05,
                 n_features: Optional[Union[int, float]] = None,
                 estimator: Optional[BaseEstimator] = None):
        """Initialize the TimeSeriesFeatureSelector.
        
        Parameters
        ----------
        method : str, default='combined'
            The feature selection method to use: 'causality', 'autocorrelation',
            'forecasting_impact', or 'combined'.
        target : Optional[str], default=None
            The name of the target variable column.
        time_col : Optional[str], default=None
            The name of the time column.
        group_col : Optional[str], default=None
            The name of the group column for panel data.
        max_lag : int, default=5
            Maximum lag to consider for causality and autocorrelation methods.
        threshold : float, default=0.05
            Threshold for feature selection. For causality method, features with
            p-value < threshold are selected. For other methods, it's used as a 
            relative threshold for the importance scores.
        n_features : Optional[Union[int, float]], default=None
            Number of features to select. If None, threshold is used instead.
            If int, the absolute number of features to select. If float between
            0 and 1, it is the fraction of features to select.
        estimator : Optional[BaseEstimator], default=None
            Estimator to use for forecasting impact method. If None, a default
            estimator is used.
        """
        if method == 'causality' and not STATSMODELS_AVAILABLE:
            raise ImportError(
                "The statsmodels library is required for causality-based feature selection. "
                "Install it using: pip install statsmodels"
            )
            
        super().__init__()
        self.method = method
        self.target = target
        self.time_col = time_col
        self.group_col = group_col
        self.max_lag = max_lag
        self.threshold = threshold
        self.n_features = n_features
        self.estimator = estimator
        
        # Attributes that will be set during fit
        self.p_values_ = None
        self.acf_values_ = None
        self.forecasting_scores_ = None
    
    def fit(self, X, y=None, **kwargs):
        """Fit the time series feature selector.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features, including the target and time columns.
        y : pd.Series or np.ndarray, optional
            Not used, present for API consistency.
        **kwargs : dict
            Additional parameters passed to the underlying methods.
            
        Returns
        -------
        self : TimeSeriesFeatureSelector
            The fitted selector
        """
        # Check dataframe type and convert to pandas if needed
        X_type = check_dataframe_type(X)
        if X_type != 'pandas':
            X_pandas = convert_dataframe(X, 'pandas')
        else:
            X_pandas = X
        
        # Validate inputs
        if self.target is None:
            raise ValueError("Target column must be specified")
        if self.target not in X_pandas.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe")
        
        # Extract target
        target_series = X_pandas[self.target]
        
        # Get feature columns (excluding target, time, and group)
        exclude_cols = [col for col in [self.target, self.time_col, self.group_col] if col is not None]
        feature_cols = [col for col in X_pandas.columns if col not in exclude_cols]
        
        # Select features based on the specified method
        if self.method == 'causality':
            self._select_by_causality(X_pandas, feature_cols)
        elif self.method == 'autocorrelation':
            self._select_by_autocorrelation(X_pandas, feature_cols)
        elif self.method == 'forecasting_impact':
            self._select_by_forecasting_impact(X_pandas, feature_cols, target_series)
        elif self.method == 'combined':
            self._select_by_combined(X_pandas, feature_cols, target_series)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _select_by_causality(self, X, feature_cols):
        """Select features based on Granger causality.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        feature_cols : list
            List of feature column names
        """
        # Check if statsmodels is available
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "The statsmodels library is required for causality-based feature selection. "
                "Install it using: pip install statsmodels"
            )
        
        # Prepare data for causality tests
        if self.time_col is not None:
            # Sort by time
            X = X.sort_values(self.time_col)
        
        # Initialize p-values dictionary
        p_values = {}
        
        # Perform causality tests for each feature
        for col in feature_cols:
            # Skip constant columns
            if X[col].nunique() <= 1:
                p_values[col] = 1.0
                continue
            
            # Create dataframe with feature and target
            df_pair = pd.DataFrame({
                'target': X[self.target],
                'feature': X[col]
            }).dropna()
            
            if len(df_pair) <= self.max_lag + 1:
                # Not enough data for the test
                p_values[col] = 1.0
                continue
            
            try:
                # Perform Granger causality test
                test_results = grangercausalitytests(
                    df_pair, 
                    maxlag=self.max_lag, 
                    verbose=False
                )
                
                # Get minimum p-value across all lags
                min_p_value = min([
                    test_results[lag][0]['ssr_chi2test'][1]
                    for lag in range(1, self.max_lag + 1)
                ])
                
                p_values[col] = min_p_value
            except Exception as e:
                print(f"Error in causality test for {col}: {e}")
                p_values[col] = 1.0
        
        # Create dataframe with p-values
        self.p_values_ = pd.DataFrame({
            'Feature': list(p_values.keys()),
            'P_Value': list(p_values.values())
        })
        
        # Sort by p-value
        self.p_values_ = self.p_values_.sort_values('P_Value')
        
        # Select features based on threshold or n_features
        if self.n_features is not None:
            if isinstance(self.n_features, float):
                # Select top percentage of features
                n = max(1, int(self.n_features * len(feature_cols)))
            else:
                # Select top n features
                n = min(self.n_features, len(feature_cols))
            
            # Select top n features
            self.selected_features_ = self.p_values_['Feature'].head(n).tolist()
        else:
            # Select features with p-value < threshold
            self.selected_features_ = self.p_values_[
                self.p_values_['P_Value'] < self.threshold
            ]['Feature'].tolist()
        
        # Create scores (1 - p_value)
        scores = {
            feature: 1.0 - p_value
            for feature, p_value in zip(self.p_values_['Feature'], self.p_values_['P_Value'])
        }
        
        self.scores_ = pd.Series(scores)
    
    def _select_by_autocorrelation(self, X, feature_cols):
        """Select features based on autocorrelation with the target.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        feature_cols : list
            List of feature column names
        """
        # Check if statsmodels is available
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "The statsmodels library is required for autocorrelation-based feature selection. "
                "Install it using: pip install statsmodels"
            )
        
        # Prepare data for autocorrelation tests
        if self.time_col is not None:
            # Sort by time
            X = X.sort_values(self.time_col)
        
        # Create lag features for target
        lag_features = {}
        for lag in range(1, self.max_lag + 1):
            lag_features[f'target_lag_{lag}'] = X[self.target].shift(lag)
        
        # Add lag features to the dataframe
        X_with_lags = X.copy()
        for col, values in lag_features.items():
            X_with_lags[col] = values
        
        # Calculate correlation between each feature and lagged target
        acf_values = {}
        for col in feature_cols:
            # Skip constant columns
            if X[col].nunique() <= 1:
                acf_values[col] = 0.0
                continue
            
            # Calculate correlation with each lagged target
            correlations = []
            for lag in range(1, self.max_lag + 1):
                lag_col = f'target_lag_{lag}'
                # Calculate correlation
                corr = X_with_lags[[col, lag_col]].corr().iloc[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(abs(corr))
            
            # Use maximum correlation as feature importance
            acf_values[col] = max(correlations) if correlations else 0.0
        
        # Create dataframe with autocorrelation values
        self.acf_values_ = pd.DataFrame({
            'Feature': list(acf_values.keys()),
            'ACF_Value': list(acf_values.values())
        })
        
        # Sort by autocorrelation value
        self.acf_values_ = self.acf_values_.sort_values('ACF_Value', ascending=False)
        
        # Select features based on threshold or n_features
        if self.n_features is not None:
            if isinstance(self.n_features, float):
                # Select top percentage of features
                n = max(1, int(self.n_features * len(feature_cols)))
            else:
                # Select top n features
                n = min(self.n_features, len(feature_cols))
            
            # Select top n features
            self.selected_features_ = self.acf_values_['Feature'].head(n).tolist()
        else:
            # Apply a relative threshold based on the maximum value
            max_acf = self.acf_values_['ACF_Value'].max()
            rel_threshold = max_acf * self.threshold
            
            # Select features with ACF value > rel_threshold
            self.selected_features_ = self.acf_values_[
                self.acf_values_['ACF_Value'] > rel_threshold
            ]['Feature'].tolist()
        
        # Store scores
        self.scores_ = pd.Series(
            {feature: acf for feature, acf in zip(
                self.acf_values_['Feature'], self.acf_values_['ACF_Value']
            )}
        )
    
    def _select_by_forecasting_impact(self, X, feature_cols, target_series):
        """Select features based on their impact on forecasting performance.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        feature_cols : list
            List of feature column names
        target_series : pd.Series
            The target variable
        """
        # Prepare data for forecasting tests
        if self.time_col is not None:
            # Sort by time
            X = X.sort_values(self.time_col)
        
        # Create train/test split for time series
        train_size = int(len(X) * 0.8)
        train_idx = X.index[:train_size]
        test_idx = X.index[train_size:]
        
        # Initialize estimator if not provided
        if self.estimator is None:
            from sklearn.ensemble import RandomForestRegressor
            self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Create baseline model using all features
        X_train = X.loc[train_idx, feature_cols]
        X_test = X.loc[test_idx, feature_cols]
        y_train = target_series.loc[train_idx]
        y_test = target_series.loc[test_idx]
        
        # Train baseline model
        baseline_model = clone(self.estimator)
        baseline_model.fit(X_train, y_train)
        
        # Evaluate baseline model
        baseline_score = baseline_model.score(X_test, y_test)
        
        # Evaluate impact of removing each feature
        feature_scores = {}
        for col in feature_cols:
            # Create feature subset without current feature
            subset_cols = [c for c in feature_cols if c != col]
            
            if not subset_cols:
                # No features left, skip
                feature_scores[col] = 0.0
                continue
            
            # Train model without this feature
            X_train_subset = X.loc[train_idx, subset_cols]
            X_test_subset = X.loc[test_idx, subset_cols]
            
            subset_model = clone(self.estimator)
            subset_model.fit(X_train_subset, y_train)
            
            # Evaluate subset model
            subset_score = subset_model.score(X_test_subset, y_test)
            
            # Calculate feature importance as drop in performance
            feature_scores[col] = baseline_score - subset_score
        
        # Create dataframe with forecasting impact scores
        self.forecasting_scores_ = pd.Series(feature_scores).sort_values(ascending=False)
        
        # Select features based on threshold or n_features
        if self.n_features is not None:
            if isinstance(self.n_features, float):
                # Select top percentage of features
                n = max(1, int(self.n_features * len(feature_cols)))
            else:
                # Select top n features
                n = min(self.n_features, len(feature_cols))
            
            # Select top n features
            self.selected_features_ = self.forecasting_scores_.index[:n].tolist()
        else:
            # Apply a relative threshold based on the maximum value
            max_score = self.forecasting_scores_.max()
            rel_threshold = max_score * self.threshold
            
            # Select features with score > rel_threshold
            self.selected_features_ = self.forecasting_scores_[
                self.forecasting_scores_ > rel_threshold
            ].index.tolist()
        
        # Store scores
        self.scores_ = self.forecasting_scores_.copy()
    
    def _select_by_combined(self, X, feature_cols, target_series):
        """Select features based on a combination of causality, autocorrelation, and forecasting impact.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input features
        feature_cols : list
            List of feature column names
        target_series : pd.Series
            The target variable
        """
        # Create separate selectors for each method
        causality_selector = TimeSeriesFeatureSelector(
            method='causality',
            target=self.target,
            time_col=self.time_col,
            group_col=self.group_col,
            max_lag=self.max_lag,
            threshold=self.threshold,
            n_features=None  # Use threshold
        )
        
        autocorr_selector = TimeSeriesFeatureSelector(
            method='autocorrelation',
            target=self.target,
            time_col=self.time_col,
            group_col=self.group_col,
            max_lag=self.max_lag,
            threshold=self.threshold,
            n_features=None  # Use threshold
        )
        
        # Fit each selector
        causality_selector.fit(X)
        autocorr_selector.fit(X)
        
        # Combine selected features
        self.selected_features_ = list(set(
            causality_selector.selected_features_ +
            autocorr_selector.selected_features_
        ))
        
        # If n_features is specified, select top n features
        if self.n_features is not None:
            if isinstance(self.n_features, float):
                # Select top percentage of features
                n = max(1, int(self.n_features * len(feature_cols)))
            else:
                # Select top n features
                n = min(self.n_features, len(feature_cols))
            
            # If we have too many features, we need to prioritize
            if len(self.selected_features_) > n:
                # Create combined score
                combined_scores = {}
                for feature in self.selected_features_:
                    causality_score = causality_selector.scores_.get(feature, 0)
                    autocorr_score = autocorr_selector.scores_.get(feature, 0)
                    
                    # Normalize scores
                    if causality_selector.scores_.max() > 0:
                        causality_score /= causality_selector.scores_.max()
                    if autocorr_selector.scores_.max() > 0:
                        autocorr_score /= autocorr_selector.scores_.max()
                    
                    # Combine scores (weighted average)
                    combined_scores[feature] = 0.5 * causality_score + 0.5 * autocorr_score
                
                # Create series from combined scores
                combined_series = pd.Series(combined_scores).sort_values(ascending=False)
                
                # Select top n features
                self.selected_features_ = combined_series.index[:n].tolist()
                
                # Store combined scores
                self.scores_ = combined_series.copy()
            else:
                # We have fewer features than n, store the combined scores
                combined_scores = {}
                for feature in self.selected_features_:
                    causality_score = causality_selector.scores_.get(feature, 0)
                    autocorr_score = autocorr_selector.scores_.get(feature, 0)
                    
                    # Normalize scores
                    if causality_selector.scores_.max() > 0:
                        causality_score /= causality_selector.scores_.max()
                    if autocorr_selector.scores_.max() > 0:
                        autocorr_score /= autocorr_selector.scores_.max()
                    
                    # Combine scores (weighted average)
                    combined_scores[feature] = 0.5 * causality_score + 0.5 * autocorr_score
                
                self.scores_ = pd.Series(combined_scores)
        else:
            # Store combined scores
            combined_scores = {}
            for feature in self.selected_features_:
                causality_score = causality_selector.scores_.get(feature, 0)
                autocorr_score = autocorr_selector.scores_.get(feature, 0)
                
                # Normalize scores
                if causality_selector.scores_.max() > 0:
                    causality_score /= causality_selector.scores_.max()
                if autocorr_selector.scores_.max() > 0:
                    autocorr_score /= autocorr_selector.scores_.max()
                
                # Combine scores (weighted average)
                combined_scores[feature] = 0.5 * causality_score + 0.5 * autocorr_score
            
            self.scores_ = pd.Series(combined_scores)
    
    def plot_feature_scores(self, ax=None, top_n=None, **kwargs):
        """Plot the feature importance scores.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, default=None
            The axes on which to plot. If None, a new figure and axes are created.
        top_n : int, default=None
            Number of top features to show. If None, all features are shown.
        **kwargs : dict
            Additional parameters passed to the plot function
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.scores_ is None:
            raise ValueError("No feature scores available. Call fit first.")
            
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Sort scores
        sorted_scores = self.scores_.sort_values(ascending=False)
        
        # Limit to top_n features if specified
        if top_n is not None:
            sorted_scores = sorted_scores.head(top_n)
            
        # Create bar plot
        ax.bar(
            range(len(sorted_scores)),
            sorted_scores.values,
            tick_label=sorted_scores.index,
            **kwargs
        )
        
        # Add threshold line if applicable
        if self.method == 'causality' and self.threshold is not None and self.n_features is None:
            ax.axhline(
                y=1.0 - self.threshold,
                color='r',
                linestyle='--',
                alpha=0.7,
                label=f'Threshold: {self.threshold}'
            )
        
        # Add labels and legend
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Feature Importance ({self.method.replace("_", " ").title()})')
        ax.set_xticks(range(len(sorted_scores)))
        ax.set_xticklabels(sorted_scores.index, rotation=90)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return ax


def select_features_rfecv(
    df: Any,
    target: Union[str, pd.Series, np.ndarray],
    estimator: Optional[BaseEstimator] = None,
    cv: Union[int, Any] = 5,
    scoring: Optional[Union[str, Callable]] = None,
    step: Union[int, float] = 1,
    min_features_to_select: int = 1,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    random_state: Optional[int] = None,
) -> Union[Any, List[str]]:
    """
    Select features using Recursive Feature Elimination with Cross-Validation.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[str, pd.Series, np.ndarray]
        The target variable. Can be a column name in the dataframe or a separate series/array.
    estimator : Optional[BaseEstimator], default=None
        The base estimator to use for feature importance. If None, a RandomForest model
        is used based on the problem type.
    cv : Union[int, Any], default=5
        Cross-validation strategy. If int, KFold or StratifiedKFold is used depending
        on the problem type.
    scoring : Optional[Union[str, Callable]], default=None
        Scoring metric for cross-validation. If None, the estimator's default scorer is used.
    step : Union[int, float], default=1
        Number of features to remove at each iteration. If int, removes that many features.
        If float between 0 and 1, removes that proportion of features.
    min_features_to_select : int, default=1
        The minimum number of features to select.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    random_state : Optional[int], default=None
        Random state for reproducibility.
        
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100),
    ...     'target': np.random.choice([0, 1], size=100)
    ... })
    >>> select_features_rfecv(df, 'target', return_names_only=True)
    ['A', 'B']
    """
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        raise ValueError("Target is required for RFECV")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Create estimator if not provided
    if estimator is None:
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # Create and fit RFECV selector
    selector = RecursiveFeatureEliminationCV(
        estimator=estimator,
        cv=cv,
        scoring=scoring,
        step=step,
        min_features_to_select=min_features_to_select,
        random_state=random_state
    )
    
    selector.fit(df_without_target, y)
    
    # Return results
    if return_names_only:
        return selector.selected_features_
    else:
        return df[selector.selected_features_]


def select_features_stability(
    df: Any,
    target: Union[str, pd.Series, np.ndarray],
    estimator: Optional[BaseEstimator] = None,
    n_features: Optional[Union[int, float]] = None,
    threshold: float = 0.5,
    subsample_size: float = 0.5,
    n_subsamples: int = 100,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    random_state: Optional[int] = None,
) -> Union[Any, List[str]]:
    """
    Select features using stability selection.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[str, pd.Series, np.ndarray]
        The target variable. Can be a column name in the dataframe or a separate series/array.
    estimator : Optional[BaseEstimator], default=None
        The base estimator to use for feature selection. If None, a model is selected
        based on the problem type.
    n_features : Optional[Union[int, float]], default=None
        Number of features to select. If None, half of the features are selected.
        If int, it is the absolute number of features to select. If float between
        0 and 1, it is the fraction of features to select.
    threshold : float, default=0.5
        Threshold for selection. Features that are selected in more than
        threshold fraction of subsamples are considered stable.
    subsample_size : float, default=0.5
        Fraction of samples to use in each subsample.
    n_subsamples : int, default=100
        Number of subsamples to generate.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    random_state : Optional[int], default=None
        Random state for reproducibility.
        
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100),
    ...     'target': np.random.choice([0, 1], size=100)
    ... })
    >>> select_features_stability(df, 'target', return_names_only=True)
    ['A', 'B']
    """
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        raise ValueError("Target is required for stability selection")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Create estimator if not provided
    if estimator is None:
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # Create and fit stability selector
    selector = StabilitySelector(
        estimator=estimator,
        n_features=n_features,
        threshold=threshold,
        subsample_size=subsample_size,
        n_subsamples=n_subsamples,
        random_state=random_state
    )
    
    selector.fit(df_without_target, y)
    
    # Return results
    if return_names_only:
        return selector.selected_features_
    else:
        return df[selector.selected_features_]


def select_features_genetic(
    df: Any,
    target: Union[str, pd.Series, np.ndarray],
    estimator: Optional[BaseEstimator] = None,
    n_features_to_select: Optional[Union[int, float]] = None,
    scoring: Union[str, Callable] = 'accuracy',
    cv: Union[int, Any] = 5,
    population_size: int = 50,
    n_generations: int = 40,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    random_state: Optional[int] = None,
) -> Union[Any, List[str]]:
    """
    Select features using a genetic algorithm.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[str, pd.Series, np.ndarray]
        The target variable. Can be a column name in the dataframe or a separate series/array.
    estimator : Optional[BaseEstimator], default=None
        The base estimator to use for evaluating feature subsets. If None, a model
        is selected based on the problem type.
    n_features_to_select : Optional[Union[int, float]], default=None
        Target number of features to select. If None, the algorithm will optimize
        for the best performing subset regardless of size. If int, the algorithm
        will try to select exactly that many features. If float between 0 and 1,
        it is the fraction of features to select.
    scoring : Union[str, Callable], default='accuracy'
        Scoring metric to use for evaluating feature subsets.
    cv : Union[int, Any], default=5
        Cross-validation strategy for evaluating feature subsets.
    population_size : int, default=50
        Size of the population in the genetic algorithm.
    n_generations : int, default=40
        Number of generations to evolve.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    random_state : Optional[int], default=None
        Random state for reproducibility.
        
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100),
    ...     'target': np.random.choice([0, 1], size=100)
    ... })
    >>> select_features_genetic(df, 'target', return_names_only=True)
    ['A', 'B']
    """
    if not DEAP_AVAILABLE:
        raise ImportError(
            "The DEAP library is required for genetic feature selection. "
            "Install it using: pip install deap"
        )
    
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        raise ValueError("Target is required for genetic selection")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Create estimator if not provided
    if estimator is None:
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # Create and fit genetic selector
    selector = GeneticFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        scoring=scoring,
        cv=cv,
        population_size=population_size,
        n_generations=n_generations,
        random_state=random_state
    )
    
    selector.fit(df_without_target, y)
    
    # Return results
    if return_names_only:
        return selector.selected_features_
    else:
        return df[selector.selected_features_]


def select_features_multi_objective(
    df: Any,
    target: Union[str, pd.Series, np.ndarray],
    estimator: Optional[BaseEstimator] = None,
    n_features_to_select: Optional[Union[int, float]] = None,
    scoring: Union[str, Callable] = 'accuracy',
    cv: Union[int, Any] = 5,
    objectives: List[str] = ['score', 'n_features'],
    selected_solution_method: str = 'knee',
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    random_state: Optional[int] = None,
) -> Union[Any, List[str]]:
    """
    Select features using multi-objective optimization.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Union[str, pd.Series, np.ndarray]
        The target variable. Can be a column name in the dataframe or a separate series/array.
    estimator : Optional[BaseEstimator], default=None
        The base estimator to use for evaluating feature subsets. If None, a model
        is selected based on the problem type.
    n_features_to_select : Optional[Union[int, float]], default=None
        The number of features to select. If None, the algorithm will optimize
        for the best trade-offs between performance and number of features.
        If int, it is the absolute number of features to select. If float between
        0 and 1, it is the fraction of features to select.
    scoring : Union[str, Callable], default='accuracy'
        Scoring metric to use for evaluating feature subsets.
    cv : Union[int, Any], default=5
        Cross-validation strategy for evaluating feature subsets.
    objectives : List[str], default=['score', 'n_features']
        List of objectives to optimize. Available options:
        - 'score': maximize model performance
        - 'n_features': minimize number of features
        - 'complexity': minimize model complexity (if applicable)
    selected_solution_method : str, default='knee'
        Method to select a single solution from the Pareto front:
        - 'knee': select the knee point of the Pareto front
        - 'best_score': select the solution with the best score
        - 'min_features': select the solution with the fewest features
        - 'compromise': select a solution that balances all objectives
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    random_state : Optional[int], default=None
        Random state for reproducibility.
        
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100),
    ...     'target': np.random.choice([0, 1], size=100)
    ... })
    >>> select_features_multi_objective(df, 'target', return_names_only=True)
    ['A', 'B']
    """
    if not PYMOO_AVAILABLE:
        raise ImportError(
            "The pymoo library is required for multi-objective feature selection. "
            "Install it using: pip install pymoo"
        )
    
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        raise ValueError("Target is required for multi-objective selection")
    
    # Infer problem type if not provided
    if problem_type is None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Create estimator if not provided
    if estimator is None:
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    
    # Create and fit multi-objective selector
    selector = MultiObjectiveFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        scoring=scoring,
        cv=cv,
        objectives=objectives,
        selected_solution_method=selected_solution_method,
        random_state=random_state
    )
    
    selector.fit(df_without_target, y)
    
    # Return results
    if return_names_only:
        return selector.selected_features_
    else:
        return df[selector.selected_features_]


def select_features_time_series(
    df: Any,
    target: str,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    method: str = 'combined',
    max_lag: int = 5,
    threshold: float = 0.05,
    n_features: Optional[Union[int, float]] = None,
    estimator: Optional[BaseEstimator] = None,
    return_names_only: bool = False,
) -> Union[Any, List[str]]:
    """
    Select features for time series data.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : str
        The name of the target variable column.
    time_col : Optional[str], default=None
        The name of the time column.
    group_col : Optional[str], default=None
        The name of the group column for panel data.
    method : str, default='combined'
        The feature selection method to use: 'causality', 'autocorrelation',
        'forecasting_impact', or 'combined'.
    max_lag : int, default=5
        Maximum lag to consider for causality and autocorrelation methods.
    threshold : float, default=0.05
        Threshold for feature selection. For causality method, features with
        p-value < threshold are selected. For other methods, it's used as a 
        relative threshold for the importance scores.
    n_features : Optional[Union[int, float]], default=None
        Number of features to select. If None, threshold is used instead.
        If int, the absolute number of features to select. If float between
        0 and 1, it is the fraction of features to select.
    estimator : Optional[BaseEstimator], default=None
        Estimator to use for forecasting impact method. If None, a default
        estimator is used.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
        
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2020-01-01', periods=100)
    >>> df = pd.DataFrame({
    ...     'date': dates,
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100),
    ...     'C': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })
    >>> select_features_time_series(df, 'target', time_col='date', return_names_only=True)
    ['A', 'B']
    """
    # Create and fit time series selector
    selector = TimeSeriesFeatureSelector(
        method=method,
        target=target,
        time_col=time_col,
        group_col=group_col,
        max_lag=max_lag,
        threshold=threshold,
        n_features=n_features,
        estimator=estimator
    )
    
    selector.fit(df)
    
    # Return results
    if return_names_only:
        return selector.selected_features_
    else:
        return df[selector.selected_features_]


# Update the main select_features function to include the new methods
def select_features(
    df: Any,
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    method: str = 'correlation',
    k: Optional[int] = None,
    k_percent: Optional[float] = None,
    threshold: Optional[float] = None,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    exclude_columns: Optional[List[str]] = None,
    **kwargs
) -> Union[Any, List[str]]:
    """
    Select features from the dataframe using the specified method.
    
    This is a high-level function that redirects to specific selection methods
    based on the `method` parameter.
    
    Parameters
    ----------
    df : Any
        The dataframe to process. Can be pandas, polars, or dask.
    target : Optional[Union[str, pd.Series, np.ndarray]], default=None
        The target variable for supervised selection methods. Can be a column name
        in the dataframe or a separate series/array. Required for supervised methods.
    method : str, default='correlation'
        The feature selection method to use. Options:
        - 'correlation': Select features based on correlation with target
        - 'importance': Select features based on model importance
        - 'variance': Select features based on variance
        - 'mutual_info': Select features based on mutual information
        - 'kbest': Select top k features based on statistical tests
        - 'percentile': Select top percentile of features
        - 'rfecv': Recursive feature elimination with cross-validation
        - 'stability': Stability selection
        - 'genetic': Genetic algorithm-based selection
        - 'multi_objective': Multi-objective optimization
        - 'time_series': Time series-specific feature selection
    k : Optional[int], default=None
        Number of top features to select. Used by 'kbest' and other methods.
    k_percent : Optional[float], default=None
        Percentage of top features to select. Used by 'percentile' method.
    threshold : Optional[float], default=None
        Threshold for feature selection. Used by 'correlation', 'variance', etc.
    problem_type : Optional[str], default=None
        Type of problem: 'classification' or 'regression'. Inferred if not provided.
    return_names_only : bool, default=False
        If True, return only the names of selected features.
    exclude_columns : Optional[List[str]], default=None
        Columns to exclude from selection.
    **kwargs : dict
        Additional parameters passed to the specific selection method.
    
    Returns
    -------
    Union[Any, List[str]]
        Dataframe with selected features or list of selected feature names.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 1, 1, 1, 1],
    ...     'target': [0, 0, 1, 1, 1]
    ... })
    >>> select_features(df, 'target', method='correlation', threshold=0.5, return_names_only=True)
    ['A', 'B']
    
    >>> # Using recursive feature elimination with cross-validation
    >>> select_features(df, 'target', method='rfecv', return_names_only=True)
    ['A', 'B']
    
    >>> # Using time series feature selection
    >>> dates = pd.date_range('2020-01-01', periods=5)
    >>> df['date'] = dates
    >>> select_features(df, 'target', method='time_series', time_col='date', return_names_only=True)
    ['A', 'B']
    """
    # Advanced/new methods
    if method == 'rfecv':
        # Use the RFECV selection method
        estimator = kwargs.get('estimator', None)
        cv = kwargs.get('cv', 5)
        scoring = kwargs.get('scoring', None)
        step = kwargs.get('step', 1)
        min_features_to_select = kwargs.get('min_features_to_select', 1)
        random_state = kwargs.get('random_state', None)
        
        return select_features_rfecv(
            df, target, estimator=estimator, cv=cv, scoring=scoring,
            step=step, min_features_to_select=min_features_to_select,
            problem_type=problem_type, return_names_only=return_names_only,
            random_state=random_state
        )
    elif method == 'stability':
        # Use the stability selection method
        estimator = kwargs.get('estimator', None)
        n_features = kwargs.get('n_features', k)
        subsample_size = kwargs.get('subsample_size', 0.5)
        n_subsamples = kwargs.get('n_subsamples', 100)
        random_state = kwargs.get('random_state', None)
        
        return select_features_stability(
            df, target, estimator=estimator, n_features=n_features,
            threshold=threshold, subsample_size=subsample_size,
            n_subsamples=n_subsamples, problem_type=problem_type,
            return_names_only=return_names_only, random_state=random_state
        )
    elif method == 'genetic':
        # Use the genetic algorithm-based selection
        estimator = kwargs.get('estimator', None)
        n_features_to_select = kwargs.get('n_features_to_select', k)
        scoring = kwargs.get('scoring', 'accuracy')
        cv = kwargs.get('cv', 5)
        population_size = kwargs.get('population_size', 50)
        n_generations = kwargs.get('n_generations', 40)
        random_state = kwargs.get('random_state', None)
        
        return select_features_genetic(
            df, target, estimator=estimator, n_features_to_select=n_features_to_select,
            scoring=scoring, cv=cv, population_size=population_size,
            n_generations=n_generations, problem_type=problem_type,
            return_names_only=return_names_only, random_state=random_state
        )
    elif method == 'multi_objective':
        # Use the multi-objective optimization
        estimator = kwargs.get('estimator', None)
        n_features_to_select = kwargs.get('n_features_to_select', k)
        scoring = kwargs.get('scoring', 'accuracy')
        cv = kwargs.get('cv', 5)
        objectives = kwargs.get('objectives', ['score', 'n_features'])
        selected_solution_method = kwargs.get('selected_solution_method', 'knee')
        random_state = kwargs.get('random_state', None)
        
        return select_features_multi_objective(
            df, target, estimator=estimator, n_features_to_select=n_features_to_select,
            scoring=scoring, cv=cv, objectives=objectives,
            selected_solution_method=selected_solution_method,
            problem_type=problem_type, return_names_only=return_names_only,
            random_state=random_state
        )
    elif method == 'time_series':
        # Use the time series feature selection
        time_col = kwargs.get('time_col', None)
        group_col = kwargs.get('group_col', None)
        ts_method = kwargs.get('ts_method', 'combined')
        max_lag = kwargs.get('max_lag', 5)
        n_features = kwargs.get('n_features', k)
        estimator = kwargs.get('estimator', None)
        
        return select_features_time_series(
            df, target, time_col=time_col, group_col=group_col,
            method=ts_method, max_lag=max_lag, threshold=threshold,
            n_features=n_features, estimator=estimator,
            return_names_only=return_names_only
        )
    else:
        # Use the original methods
        return select_features_original(
            df, target, method=method, k=k, k_percent=k_percent,
            threshold=threshold, problem_type=problem_type,
            return_names_only=return_names_only, exclude_columns=exclude_columns
        )


# Create a copy of the original select_features function
def select_features_original(
    df: Any,
    target: Optional[Union[str, pd.Series, np.ndarray]] = None,
    method: str = 'correlation',
    k: Optional[int] = None,
    k_percent: Optional[float] = None,
    threshold: Optional[float] = None,
    problem_type: Optional[str] = None,
    return_names_only: bool = False,
    exclude_columns: Optional[List[str]] = None,
) -> Union[Any, List[str]]:
    """
    Select features from the dataframe using basic selection methods.
    
    This function is a copy of the original select_features function, used
    internally by the updated select_features function.
    """
    # Get target as a separate variable if it's a column name
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        y = df[target]
        df_without_target = df.drop(columns=[target])
    elif target is not None:
        # Use the provided target
        y = target
        df_without_target = df
    else:
        # Unsupervised method
        y = None
        df_without_target = df
    
    # Exclude columns if specified
    if exclude_columns:
        df_without_target = df_without_target.drop(columns=[
            col for col in exclude_columns if col in df_without_target.columns
        ])
    
    # Infer problem type if not provided
    if problem_type is None and y is not None:
        # Check if target is categorical
        if hasattr(y, 'nunique'):
            if y.nunique() < 10:  # Arbitrary threshold
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            # Default to regression
            problem_type = 'regression'
    
    # Select features based on method
    if method == 'correlation':
        return select_by_correlation(
            df_without_target, target=y, threshold=threshold, 
            return_names_only=return_names_only
        )
    elif method == 'importance':
        return select_by_importance(
            df_without_target, target=y, threshold=threshold, k=k,
            problem_type=problem_type, return_names_only=return_names_only
        )
    elif method == 'variance':
        return select_by_variance(
            df_without_target, threshold=threshold, return_names_only=return_names_only
        )
    elif method == 'mutual_info':
        return select_by_mutual_info(
            df_without_target, target=y, k=k, problem_type=problem_type,
            return_names_only=return_names_only
        )
    elif method == 'kbest':
        if k is None:
            raise ValueError("k must be provided for 'kbest' method")
        return select_by_kbest(
            df_without_target, target=y, k=k, problem_type=problem_type,
            return_names_only=return_names_only
        )
    elif method == 'percentile':
        if k_percent is None:
            raise ValueError("k_percent must be provided for 'percentile' method")
        return select_by_percentile(
            df_without_target, target=y, percentile=k_percent, 
            problem_type=problem_type, return_names_only=return_names_only
        )
    else:
        raise ValueError(f"Unknown method: {method}")