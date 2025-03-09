"""
Base encoder classes for categorical variables.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)


class EncoderWrapper:
    """
    Base wrapper class for various encoders.
    
    Parameters
    ----------
    encoder : Any
        The encoder instance to wrap.
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    """
    
    def __init__(self, encoder: Any, columns: Optional[List[str]] = None):
        """Initialize the encoder wrapper."""
        self.encoder = encoder
        self.columns = columns
        self.is_fitted = False
        self.input_columns: List[str] = []
        self.output_columns: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'EncoderWrapper':
        """
        Fit the encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        
        Returns
        -------
        EncoderWrapper
            The fitted encoder wrapper.
        """
        raise NotImplementedError("Subclasses must implement fit")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        raise NotImplementedError("Subclasses must implement transform")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder to the dataframe and transform it.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit and transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        return self.fit(df).transform(df)


class OneHotEncoderWrapper(EncoderWrapper):
    """
    Wrapper for sklearn's OneHotEncoder.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    drop : str, default='first'
        Whether to drop one of the categories for each feature.
        Options: 'first', 'if_binary', None.
    sparse_output : bool, default=False
        Whether to return a sparse matrix. Not recommended for pandas integration.
    handle_unknown : str, default='error'
        Whether to raise an error or ignore if an unknown category is present.
        Options: 'error', 'ignore'.
    min_frequency : float or int, default=None
        Minimum frequency for a category to be considered.
    """
    
    def __init__(
        self, 
        columns: Optional[List[str]] = None,
        drop: str = 'first',
        sparse_output: bool = False,
        handle_unknown: str = 'ignore',
        min_frequency: Optional[Union[float, int]] = None,
    ):
        """Initialize the OneHotEncoderWrapper."""
        encoder = OneHotEncoder(
            drop=drop,
            sparse_output=sparse_output,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
        )
        super().__init__(encoder, columns)
        self.feature_names: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'OneHotEncoderWrapper':
        """
        Fit the one-hot encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        
        Returns
        -------
        OneHotEncoderWrapper
            The fitted encoder wrapper.
        """
        # Determine columns to encode
        if self.columns is None:
            self.input_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.input_columns = [col for col in self.columns if col in df.columns]
        
        if not self.input_columns:
            self.is_fitted = True
            return self
        
        # Fit the encoder
        self.encoder.fit(df[self.input_columns])
        self.is_fitted = True
        
        # Get the output feature names
        if hasattr(self.encoder, 'get_feature_names_out'):
            self.feature_names = self.encoder.get_feature_names_out(self.input_columns).tolist()
        else:
            # Backwards compatibility for older sklearn versions
            self.feature_names = []
            for i, col in enumerate(self.input_columns):
                categories = self.encoder.categories_[i]
                if self.encoder.drop == 'first':
                    categories = categories[1:]
                elif self.encoder.drop == 'if_binary' and len(categories) == 2:
                    categories = categories[1:]
                for cat in categories:
                    self.feature_names.append(f"{col}_{cat}")
        
        self.output_columns = self.feature_names
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted one-hot encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with one-hot encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Transform the columns
        encoded_array = self.encoder.transform(df[self.input_columns])
        
        # Convert to dataframe
        if hasattr(encoded_array, "toarray"):
            encoded_array = encoded_array.toarray()
        
        encoded_df = pd.DataFrame(
            encoded_array,
            index=df.index,
            columns=self.output_columns
        )
        
        # Drop the original columns and add the encoded ones
        result = result.drop(columns=self.input_columns)
        result = pd.concat([result, encoded_df], axis=1)
        
        return result


class OrdinalEncoderWrapper(EncoderWrapper):
    """
    Wrapper for sklearn's OrdinalEncoder.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    handle_unknown : str, default='error'
        Whether to raise an error or use an additional category for unknown categories.
        Options: 'error', 'use_encoded_value'.
    unknown_value : int, default=-1
        The value to use for unknown categories when handle_unknown='use_encoded_value'.
    """
    
    def __init__(
        self, 
        columns: Optional[List[str]] = None,
        handle_unknown: str = 'use_encoded_value',
        unknown_value: int = -1,
    ):
        """Initialize the OrdinalEncoderWrapper."""
        encoder = OrdinalEncoder(
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
        )
        super().__init__(encoder, columns)
        self.mapping: Dict[str, Dict[str, int]] = {}
    
    def fit(self, df: pd.DataFrame) -> 'OrdinalEncoderWrapper':
        """
        Fit the ordinal encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        
        Returns
        -------
        OrdinalEncoderWrapper
            The fitted encoder wrapper.
        """
        # Determine columns to encode
        if self.columns is None:
            self.input_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.input_columns = [col for col in self.columns if col in df.columns]
        
        if not self.input_columns:
            self.is_fitted = True
            return self
        
        # Fit the encoder
        self.encoder.fit(df[self.input_columns])
        self.is_fitted = True
        
        # Store the mapping for each column
        self.mapping = {}
        for i, col in enumerate(self.input_columns):
            self.mapping[col] = {
                category: int(idx) for idx, category in enumerate(self.encoder.categories_[i])
            }
        
        # Output columns are the same as input columns
        self.output_columns = self.input_columns
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted ordinal encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with ordinally encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Transform the columns
        encoded_array = self.encoder.transform(df[self.input_columns])
        
        # Replace the original columns with the encoded ones
        for i, col in enumerate(self.input_columns):
            result[col] = encoded_array[:, i].astype(int)
        
        return result


class TargetEncoderWrapper(EncoderWrapper):
    """
    Target encoder for categorical features.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    smoothing : float, default=10.0
        Smoothing factor to prevent overfitting to rare categories.
    min_samples_leaf : int, default=1
        Minimum number of samples required for a category to be encoded separately.
    handle_unknown : str, default='value'
        How to handle unknown categories.
        Options: 'value' (use the global mean), 'error'.
    handle_missing : str, default='value'
        How to handle missing values.
        Options: 'value' (use the global mean), 'error'.
    """
    
    def __init__(
        self, 
        columns: Optional[List[str]] = None,
        smoothing: float = 10.0,
        min_samples_leaf: int = 1,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
    ):
        """Initialize the TargetEncoderWrapper."""
        super().__init__(None, columns)
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.target_mean: float = 0.0
        self.mapping: Dict[str, Dict[Any, float]] = {}
    
    def fit(self, df: pd.DataFrame, target: str) -> 'TargetEncoderWrapper':
        """
        Fit the target encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        target : str
            The target column name.
        
        Returns
        -------
        TargetEncoderWrapper
            The fitted encoder wrapper.
        """
        # Determine columns to encode
        if self.columns is None:
            self.input_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.input_columns = [col for col in self.columns if col in df.columns]
        
        # Remove target from columns to encode if present
        if target in self.input_columns:
            self.input_columns.remove(target)
        
        if not self.input_columns:
            self.is_fitted = True
            return self
        
        # Calculate global target mean
        self.target_mean = df[target].mean()
        
        # Calculate target encoding for each column
        self.mapping = {}
        for col in self.input_columns:
            # Group by the categorical column and calculate stats
            stats = df.groupby(col)[target].agg(['count', 'mean'])
            # Apply smoothing
            smoothed_mean = (
                stats['count'] * stats['mean'] + self.smoothing * self.target_mean
            ) / (stats['count'] + self.smoothing)
            # Filter out categories with too few samples
            smoothed_mean = smoothed_mean[stats['count'] >= self.min_samples_leaf]
            # Create mapping
            self.mapping[col] = smoothed_mean.to_dict()
        
        self.is_fitted = True
        
        # Output columns are the same as input columns
        self.output_columns = self.input_columns
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted target encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with target-encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Create a copy of the dataframe to avoid modifying the original
        result = df.copy()
        
        # Transform each column
        for col in self.input_columns:
            # Create a Series to hold the encoded values
            encoded_values = pd.Series(index=df.index, dtype=float)
            
            # Map each category to its encoded value
            for cat, encoded_value in self.mapping[col].items():
                mask = df[col] == cat
                encoded_values[mask] = encoded_value
            
            # Handle unknown categories
            unknown_mask = ~df[col].isin(self.mapping[col].keys())
            if unknown_mask.any():
                if self.handle_unknown == 'error':
                    unknown_cats = df.loc[unknown_mask, col].unique()
                    raise ValueError(f"Found unknown categories in column {col}: {unknown_cats}")
                elif self.handle_unknown == 'value':
                    encoded_values[unknown_mask] = self.target_mean
            
            # Handle missing values
            missing_mask = df[col].isna()
            if missing_mask.any():
                if self.handle_missing == 'error':
                    raise ValueError(f"Found missing values in column {col}")
                elif self.handle_missing == 'value':
                    encoded_values[missing_mask] = self.target_mean
            
            # Replace the original column with the encoded one
            result[col] = encoded_values
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Fit the target encoder to the dataframe and transform it.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit and transform.
        target : str
            The target column name.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        return self.fit(df, target).transform(df)