"""Integration module for connecting Freamon with Allyanonimiser.

This module provides components that bridge Freamon's pipeline system
with Allyanonimiser's PII detection and anonymization capabilities.
"""

from typing import Optional, Dict, Any, List, Union

import pandas as pd

from freamon.pipeline.steps import PreprocessingStep


class AnonymizationError(Exception):
    """Exception raised for anonymization-related errors."""
    pass


class AnonymizationStep(PreprocessingStep):
    """Pipeline step for anonymizing personally identifiable information (PII).
    
    This step integrates with the Allyanonimiser library to detect and anonymize
    PII in text data, while maintaining index tracking required for deduplication.
    
    Note: Requires the Allyanonimiser library to be installed.
    """
    
    def __init__(
        self,
        text_column: str,
        output_column: Optional[str] = None,
        anonymization_config: Optional[Dict[str, Any]] = None,
        preserve_original: bool = True,
        name: str = "anonymize"
    ):
        """Initialize the anonymization step.
        
        Args:
            text_column: Name of the column containing text to anonymize
            output_column: Name of column to store anonymized text (defaults to text_column if None)
            anonymization_config: Configuration options for Allyanonimiser
            preserve_original: Whether to keep the original text column
            name: Name of the pipeline step
        """
        super().__init__(name=name)
        self.text_column = text_column
        self.output_column = output_column or f"anonymized_{text_column}"
        self.anonymization_config = anonymization_config or {}
        self.preserve_original = preserve_original
        self.anonymizer = None
        
        # Lazy import to avoid hard dependency
        try:
            from allyanonimiser import Anonymizer
            self.anonymizer = Anonymizer(**self.anonymization_config)
        except ImportError:
            # We'll check for this in fit/transform
            pass
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'AnonymizationStep':
        """Fit the anonymization step (no-op).
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        if self.anonymizer is None:
            try:
                from allyanonimiser import Anonymizer
                self.anonymizer = Anonymizer(**self.anonymization_config)
            except ImportError:
                raise AnonymizationError(
                    "The Allyanonimiser library is required for anonymization. "
                    "Please install it with: pip install allyanonimiser"
                )
        
        if self.text_column not in X.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in the dataframe")
        
        self._is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data by anonymizing text.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Dataframe with anonymized text
        """
        if not self._is_fitted:
            raise ValueError("This AnonymizationStep instance is not fitted yet.")
            
        if self.anonymizer is None:
            raise AnonymizationError("Anonymizer not initialized. Call fit() first.")
        
        # Create a copy of the dataframe
        df = X.copy()
        
        # Apply anonymization preserving index
        df[self.output_column] = df[self.text_column].apply(self.anonymizer.anonymize_text)
        
        # Drop original text column if not preserving
        if not self.preserve_original and self.output_column != self.text_column:
            df = df.drop(columns=[self.text_column])
            
        return df


class EnhancedTextPreprocessingStep(PreprocessingStep):
    """Enhanced text preprocessing with optional anonymization.
    
    This step extends text preprocessing with optional anonymization capabilities,
    combining preprocessed text normalization with PII anonymization in a
    pipeline-compatible way that preserves index tracking.
    """
    
    def __init__(
        self,
        text_column: str,
        output_column: Optional[str] = None,
        anonymize: bool = False,
        anonymization_config: Optional[Dict[str, Any]] = None,
        preprocessing_options: Optional[Dict[str, Any]] = None,
        preserve_original: bool = True,
        name: str = "text_preprocessing"
    ):
        """Initialize the enhanced text preprocessing step.
        
        Args:
            text_column: Name of the column containing text to process
            output_column: Name of column to store processed text (defaults to processed_{text_column})
            anonymize: Whether to anonymize PII in addition to preprocessing
            anonymization_config: Configuration options for Allyanonimiser
            preprocessing_options: Options for text preprocessing
            preserve_original: Whether to keep the original text column
            name: Name of the pipeline step
        """
        super().__init__(name=name)
        self.text_column = text_column
        self.output_column = output_column or f"processed_{text_column}"
        self.anonymize = anonymize
        self.anonymization_config = anonymization_config or {}
        self.preprocessing_options = preprocessing_options or {}
        self.preserve_original = preserve_original
        
        # Initialize components
        from freamon.utils.text_utils import TextProcessor
        self.text_processor = TextProcessor()
        
        # Lazy import for anonymizer
        self.anonymizer = None
        if self.anonymize:
            try:
                from allyanonimiser import Anonymizer
                self.anonymizer = Anonymizer(**self.anonymization_config)
            except ImportError:
                # We'll check for this in fit/transform if anonymize=True
                pass
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'EnhancedTextPreprocessingStep':
        """Fit the preprocessing step.
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        if self.text_column not in X.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in the dataframe")
        
        # Initialize anonymizer if needed
        if self.anonymize and self.anonymizer is None:
            try:
                from allyanonimiser import Anonymizer
                self.anonymizer = Anonymizer(**self.anonymization_config)
            except ImportError:
                raise AnonymizationError(
                    "The Allyanonimiser library is required for anonymization. "
                    "Please install it with: pip install allyanonimiser"
                )
        
        self._is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data by preprocessing and optionally anonymizing text.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Dataframe with processed text
        """
        if not self._is_fitted:
            raise ValueError("This EnhancedTextPreprocessingStep instance is not fitted yet.")
        
        # Create a copy of the dataframe
        df = X.copy()
        
        # Process the text
        if self.anonymize and self.anonymizer is not None:
            # First anonymize
            anonymized_texts = df[self.text_column].apply(self.anonymizer.anonymize_text)
            
            # Then preprocess
            df[self.output_column] = anonymized_texts.apply(
                lambda x: self.text_processor.preprocess_text(x, **self.preprocessing_options)
            )
        else:
            # Just preprocess
            df[self.output_column] = df[self.text_column].apply(
                lambda x: self.text_processor.preprocess_text(x, **self.preprocessing_options)
            )
        
        # Drop original text column if not preserving
        if not self.preserve_original and self.output_column != self.text_column:
            df = df.drop(columns=[self.text_column])
            
        return df


class PIIDetectionStep(PreprocessingStep):
    """Pipeline step for detecting personally identifiable information (PII).
    
    This step integrates with the Allyanonimiser library to detect PII
    in text data without anonymizing it, adding detection flags and
    metadata to the dataframe.
    """
    
    def __init__(
        self,
        text_column: str,
        detection_config: Optional[Dict[str, Any]] = None,
        prefix: str = "pii_",
        include_details: bool = False,
        name: str = "pii_detection"
    ):
        """Initialize the PII detection step.
        
        Args:
            text_column: Name of the column containing text to analyze
            detection_config: Configuration options for PII detection
            prefix: Prefix for PII detection columns
            include_details: Whether to include detailed PII findings
            name: Name of the pipeline step
        """
        super().__init__(name=name)
        self.text_column = text_column
        self.detection_config = detection_config or {}
        self.prefix = prefix
        self.include_details = include_details
        self.analyzer = None
        
        # Lazy import to avoid hard dependency
        try:
            from allyanonimiser import Analyzer
            self.analyzer = Analyzer(**self.detection_config)
        except ImportError:
            # We'll check for this in fit/transform
            pass
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'PIIDetectionStep':
        """Fit the PII detection step (no-op).
        
        Args:
            X: Feature dataframe
            y: Target series (optional)
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        if self.analyzer is None:
            try:
                from allyanonimiser import Analyzer
                self.analyzer = Analyzer(**self.detection_config)
            except ImportError:
                raise AnonymizationError(
                    "The Allyanonimiser library is required for PII detection. "
                    "Please install it with: pip install allyanonimiser"
                )
        
        if self.text_column not in X.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in the dataframe")
        
        self._is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data by detecting PII.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Dataframe with PII detection results
        """
        if not self._is_fitted:
            raise ValueError("This PIIDetectionStep instance is not fitted yet.")
            
        if self.analyzer is None:
            raise AnonymizationError("Analyzer not initialized. Call fit() first.")
        
        # Create a copy of the dataframe
        df = X.copy()
        
        # Apply PII detection
        pii_results = df[self.text_column].apply(self.analyzer.analyze_text)
        
        # Extract key metrics
        df[f"{self.prefix}has_pii"] = pii_results.apply(lambda x: x["has_pii"])
        df[f"{self.prefix}pii_count"] = pii_results.apply(lambda x: len(x["findings"]))
        df[f"{self.prefix}pii_types"] = pii_results.apply(
            lambda x: [finding["entity_type"] for finding in x["findings"]]
        )
        
        # Include detailed findings if requested
        if self.include_details:
            df[f"{self.prefix}details"] = pii_results
            
        return df