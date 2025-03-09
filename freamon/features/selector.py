"""
Feature selection module for selecting the most informative features.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Literal

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
    mutual_info_regression
)

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