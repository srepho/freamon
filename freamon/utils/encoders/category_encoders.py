"""
Wrapper implementations for the category_encoders package.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import category_encoders as ce
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False
    warnings.warn(
        "category_encoders package not found. "
        "The advanced encoders will not be available."
    )

from freamon.utils.encoders.base import EncoderWrapper


class BinaryEncoderWrapper(EncoderWrapper):
    """
    Wrapper for category_encoders' BinaryEncoder.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize the BinaryEncoderWrapper."""
        if not CATEGORY_ENCODERS_AVAILABLE:
            raise ImportError(
                "category_encoders package is required for BinaryEncoderWrapper."
            )
        encoder = ce.BinaryEncoder(cols=columns)
        super().__init__(encoder, columns)
    
    def fit(self, df: pd.DataFrame) -> 'BinaryEncoderWrapper':
        """
        Fit the binary encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        
        Returns
        -------
        BinaryEncoderWrapper
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
        
        # Set the cols parameter for the encoder
        self.encoder.cols = self.input_columns
        
        # Fit the encoder
        self.encoder.fit(df)
        self.is_fitted = True
        
        # Get output column names
        transformed_df = self.encoder.transform(df.head(1))
        self.output_columns = [col for col in transformed_df.columns if col not in df.columns or col in self.input_columns]
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted binary encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with binary encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Transform using the encoder
        result = self.encoder.transform(df)
        
        return result


class HashingEncoderWrapper(EncoderWrapper):
    """
    Wrapper for category_encoders' HashingEncoder.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    n_components : int, default=8
        The number of features to create.
    """
    
    def __init__(
        self, 
        columns: Optional[List[str]] = None,
        n_components: int = 8
    ):
        """Initialize the HashingEncoderWrapper."""
        if not CATEGORY_ENCODERS_AVAILABLE:
            raise ImportError(
                "category_encoders package is required for HashingEncoderWrapper."
            )
        encoder = ce.HashingEncoder(cols=columns, n_components=n_components)
        super().__init__(encoder, columns)
        self.n_components = n_components
    
    def fit(self, df: pd.DataFrame) -> 'HashingEncoderWrapper':
        """
        Fit the hashing encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        
        Returns
        -------
        HashingEncoderWrapper
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
        
        # Set the cols parameter for the encoder
        self.encoder.cols = self.input_columns
        
        # Fit the encoder
        self.encoder.fit(df)
        self.is_fitted = True
        
        # Get output column names
        transformed_df = self.encoder.transform(df.head(1))
        self.output_columns = [col for col in transformed_df.columns if col not in df.columns or col in self.input_columns]
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted hashing encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with hashing encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Transform using the encoder
        result = self.encoder.transform(df)
        
        return result


class WOEEncoderWrapper(EncoderWrapper):
    """
    Wrapper for category_encoders' WOEEncoder.
    
    Parameters
    ----------
    columns : Optional[List[str]]
        The columns to encode. If None, all object/category columns are used.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """Initialize the WOEEncoderWrapper."""
        if not CATEGORY_ENCODERS_AVAILABLE:
            raise ImportError(
                "category_encoders package is required for WOEEncoderWrapper."
            )
        encoder = ce.WOEEncoder(cols=columns)
        super().__init__(encoder, columns)
    
    def fit(self, df: pd.DataFrame, target: str) -> 'WOEEncoderWrapper':
        """
        Fit the WOE encoder to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to fit the encoder to.
        target : str
            The target column name.
        
        Returns
        -------
        WOEEncoderWrapper
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
        
        # Set the cols parameter for the encoder
        self.encoder.cols = self.input_columns
        
        # Fit the encoder
        self.encoder.fit(df, df[target])
        self.is_fitted = True
        
        # Output columns are the same as input columns
        self.output_columns = self.input_columns
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe using the fitted WOE encoder.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed dataframe with WOE encoded columns.
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted. Call fit() first.")
        
        # If no columns to encode, return the original dataframe
        if not self.input_columns:
            return df.copy()
        
        # Transform using the encoder
        result = self.encoder.transform(df)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Fit the WOE encoder to the dataframe and transform it.
        
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