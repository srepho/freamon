"""
Automated modeling workflow that handles text processing, feature engineering, and time series.

This module provides a high-level interface for automated machine learning workflows
that intelligently process datasets containing text, determine optimal feature
representations, perform proper train/test splitting based on temporal data (if available),
and build optimized models.
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from freamon.pipeline import Pipeline, PipelineStep
from freamon.pipeline.steps import (
    FeatureEngineeringStep, 
    FeatureSelectionStep, 
    ModelTrainingStep,
    HyperparameterTuningStep,
    EvaluationStep
)
from freamon.utils.text_utils import TextProcessor, create_topic_model_optimized
from freamon.model_selection.cross_validation import (
    create_time_series_cv, 
    create_stratified_cv,
    create_kfold_cv
)
from freamon.model_selection.cv_trainer import CrossValidationTrainer
from freamon.modeling.metrics import calculate_metrics
from freamon.modeling.factory import create_model
from freamon.modeling.visualization import (
    plot_cv_metrics,
    plot_feature_importance,
    plot_time_series_predictions
)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AutoModelFlow:
    """Automated modeling workflow for text, tabular, and time series data.
    
    This class provides an end-to-end automated machine learning framework that:
    1. Analyzes datasets to identify different types of features (text, categorical, numeric, date)
    2. Applies intelligent preprocessing for each column type:
       - Text columns: Topic modeling with automatic topic number detection, feature extraction
       - Date columns: Time series feature engineering (lags, rolling statistics, date components)
       - Categorical columns: Encoding
    3. Selects appropriate cross-validation strategy (time series CV or standard CV)
    4. Performs hyperparameter tuning and model training
    5. Provides evaluation, visualization, and interpretation tools
    
    The class is designed to simplify building ML pipelines for complex datasets that contain
    a mixture of text data, structured features, and time series, requiring minimal code while
    still offering fine-grained control through configuration options.
    
    Key Features:
    - Automatic column type detection and appropriate processing
    - Intelligent text processing with topic modeling
    - Time series feature engineering for temporal data
    - Appropriate cross-validation strategies based on data type
    - Built-in visualization for model evaluation and feature importance
    - Support for classification and regression problems
    - Integration with multiple model types (LightGBM, XGBoost, CatBoost, scikit-learn)
    """
    
    def __init__(
        self,
        model_type: str = "lightgbm",
        problem_type: str = "classification",
        text_processing: bool = True,
        time_series_features: bool = True,
        feature_selection: bool = True,
        hyperparameter_tuning: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Initialize AutoModelFlow.
        
        Args:
            model_type: Type of model to use ('lightgbm', 'xgboost', 'catboost', 'sklearn')
            problem_type: Type of problem ('classification', 'regression')
            text_processing: Whether to automatically process text columns
            time_series_features: Whether to create time series features for date columns
            feature_selection: Whether to perform automated feature selection
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            random_state: Random state for reproducibility
            verbose: Whether to print progress and information
        """
        self.model_type = model_type.lower()
        self.problem_type = problem_type.lower()
        self.text_processing = text_processing
        self.time_series_features = time_series_features
        self.feature_selection = feature_selection
        self.hyperparameter_tuning = hyperparameter_tuning
        self.random_state = random_state
        self.verbose = verbose
        
        # Pipeline components that will be created during fit
        self.pipeline = None
        self.text_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.target_column = None
        self.date_column = None
        self.cv_results = None
        self.feature_importance = None
        self.text_topics = {}
        self.text_features = {}
        
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
    
    def analyze_dataset(
        self, 
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        ignore_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze a dataset to identify column types and statistics.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            date_column: Name of main date/time column (for time series)
            text_columns: List of text column names (if None, will be auto-detected)
            categorical_columns: List of categorical column names (if None, will be auto-detected)
            ignore_columns: List of columns to ignore
            
        Returns:
            Dictionary with dataset analysis results
        """
        logger.info("Analyzing dataset...")
        
        # Store target column name
        self.target_column = target_column
        self.date_column = date_column
        
        # Initialize column lists
        self.text_columns = text_columns or []
        self.categorical_columns = categorical_columns or []
        ignore_columns = ignore_columns or []
        ignore_columns.append(target_column)  # Always ignore target column
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataframe")
        
        # Check date column if provided
        if date_column and date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in the dataframe")
            
        # Auto-detect column types if not specified
        if not self.text_columns:
            self.text_columns = self._detect_text_columns(df, ignore_columns)
            logger.info(f"Detected text columns: {self.text_columns}")
            
        if not self.categorical_columns:
            self.categorical_columns = self._detect_categorical_columns(df, ignore_columns)
            logger.info(f"Detected categorical columns: {self.categorical_columns}")
        
        # Identify date columns
        self.date_columns = self._detect_date_columns(df, ignore_columns)
        
        # Make sure date_column is in date_columns if specified
        if date_column and date_column not in self.date_columns:
            self.date_columns.append(date_column)
            
        logger.info(f"Detected date columns: {self.date_columns}")
        
        # Numeric columns are those that aren't text, categorical, date, or ignored
        all_special_columns = set(self.text_columns + self.categorical_columns + 
                                self.date_columns + ignore_columns)
        self.numeric_columns = [col for col in df.columns if col not in all_special_columns]
        
        logger.info(f"Identified numeric columns: {self.numeric_columns}")
        
        # Create dataset summary
        summary = {
            'dataset_shape': df.shape,
            'target_column': target_column,
            'target_distribution': df[target_column].value_counts().to_dict() if self.problem_type == 'classification' else {
                'min': df[target_column].min(),
                'max': df[target_column].max(),
                'mean': df[target_column].mean(),
                'median': df[target_column].median()
            },
            'text_columns': self.text_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'date_columns': self.date_columns,
            'primary_date_column': date_column,
            'is_time_series': bool(date_column),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary
    
    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        ignore_columns: Optional[List[str]] = None,
        validation_size: float = 0.2,
        cv_folds: int = 5,
        metrics: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        text_processing_options: Optional[Dict[str, Any]] = None,
        time_series_options: Optional[Dict[str, Any]] = None,
        feature_selection_options: Optional[Dict[str, Any]] = None,
        tuning_options: Optional[Dict[str, Any]] = None
    ) -> 'AutoModelFlow':
        """Fit an automated machine learning model on the provided dataset.
        
        This method implements the complete end-to-end modeling workflow:
        1. Analyzes the dataset structure to identify column types
        2. Processes text columns with topic modeling and extracts features
        3. Creates time series features from date columns if applicable
        4. Encodes categorical features
        5. Selects and trains an appropriate model with cross-validation
        6. Tunes hyperparameters if requested
        7. Calculates feature importance and evaluation metrics
        
        Args:
            df (pd.DataFrame): Input DataFrame containing both features and target column
            
            target_column (str): Name of the column to predict. Must exist in df.
            
            date_column (Optional[str]): Name of the primary date/time column for temporal data.
                If provided, will enable time series feature engineering and time-based 
                cross-validation. Should be in datetime format or convertible to datetime.
            
            text_columns (Optional[List[str]]): List of column names containing text data.
                If None, text columns will be auto-detected based on content characteristics.
                Each text column will be processed with topic modeling and feature extraction.
            
            categorical_columns (Optional[List[str]]): List of column names containing categorical data.
                If None, categorical columns will be auto-detected based on cardinality and type.
                Will be one-hot encoded during modeling.
            
            ignore_columns (Optional[List[str]]): List of column names to exclude from analysis.
                These columns will be ignored during all processing steps.
            
            validation_size (float): Proportion of data to use for validation when performing
                train/test splitting. Only used when not using cross-validation. Default: 0.2.
            
            cv_folds (int): Number of cross-validation folds. For time series data, will use
                time-based splitting to avoid data leakage. Default: 5.
            
            metrics (Optional[List[str]]): List of evaluation metrics to compute.
                If None, defaults based on problem_type:
                - Classification: ["accuracy", "precision", "recall", "f1", "roc_auc"]
                - Regression: ["mse", "rmse", "mae", "r2"]
            
            model_params (Optional[Dict[str, Any]]): Parameters to pass to the underlying model.
                Only used if hyperparameter_tuning=False. If None, default parameters will be used
                based on the model_type and problem_type.
            
            text_processing_options (Optional[Dict[str, Any]]): Configuration options for text processing:
                - topic_modeling (Dict): Options for topic modeling
                    - method (str): 'nmf' or 'lda'
                    - auto_topics_range (Tuple[int, int]): Range of topics to try
                    - auto_topics_method (str): 'coherence' or 'stability'
                - text_features (Dict): Options for text feature extraction
                    - extract_features (bool): Whether to extract additional features
                    - include_stats (bool): Whether to include statistical features
                    - include_readability (bool): Whether to include readability metrics
                    - include_sentiment (bool): Whether to include sentiment analysis
                    - extract_keywords (bool): Whether to extract and use keywords
            
            time_series_options (Optional[Dict[str, Any]]): Configuration for time series features:
                - create_target_lags (bool): Whether to create lag features of the target
                - lag_periods (List[int]): List of periods for creating lag features
                - rolling_windows (List[int]): List of window sizes for rolling statistics
                - include_numeric_columns (bool): Whether to create features for numeric columns
                - numeric_lags (List[int]): Periods for numeric column lags
                - numeric_rolling_windows (List[int]): Windows for numeric rolling stats
            
            feature_selection_options (Optional[Dict[str, Any]]): Configuration for feature selection:
                - method (str): Feature selection method to use
                - n_features (int): Number of features to select
                - scoring (str): Metric to use for feature selection
            
            tuning_options (Optional[Dict[str, Any]]): Configuration for hyperparameter tuning:
                - n_trials (int): Number of hyperparameter tuning trials
                - timeout (Optional[int]): Timeout in seconds for tuning
                - eval_metric (str): Metric to optimize during tuning
                - early_stopping_rounds (int): Number of rounds without improvement before stopping
            
        Returns:
            AutoModelFlow: The fitted AutoModelFlow instance with trained model, 
                feature importance, and evaluation metrics.
                
        Raises:
            ValueError: If target_column or date_column (if specified) are not found in df
            
        Examples:
        ---------
        # Basic classification model
        >>> model_flow = AutoModelFlow(model_type='lightgbm', problem_type='classification')
        >>> model_flow.fit(df=customer_data, target_column='churn')
        >>> predictions = model_flow.predict(new_customers)
        
        # Time series forecasting with custom features
        >>> model_flow = AutoModelFlow(model_type='lightgbm', problem_type='regression')
        >>> model_flow.fit(
        ...     df=sales_data,
        ...     target_column='weekly_sales',
        ...     date_column='date',
        ...     time_series_options={
        ...         'lag_periods': [1, 2, 4, 8, 12],  # Weekly, bi-weekly, monthly, etc.
        ...         'rolling_windows': [4, 12, 26]    # Monthly, quarterly, semi-annual
        ...     }
        ... )
        
        # Text classification with custom processing
        >>> model_flow = AutoModelFlow()
        >>> model_flow.fit(
        ...     df=document_data,
        ...     target_column='category',
        ...     text_columns=['title', 'body'],
        ...     text_processing_options={
        ...         'topic_modeling': {
        ...             'method': 'nmf',
        ...             'auto_topics_method': 'stability'
        ...         },
        ...         'text_features': {
        ...             'include_sentiment': True,
        ...             'extract_keywords': True
        ...         }
        ...     }
        ... )
        """
        logger.info(f"Starting automated modeling flow with {self.model_type} for {self.problem_type}")
        
        # Analyze dataset to identify column types
        self.analyze_dataset(
            df=df, 
            target_column=target_column,
            date_column=date_column,
            text_columns=text_columns,
            categorical_columns=categorical_columns,
            ignore_columns=ignore_columns
        )
        
        # Create default options if not provided
        text_processing_options = text_processing_options or {}
        time_series_options = time_series_options or {}
        feature_selection_options = feature_selection_options or {}
        tuning_options = tuning_options or {}
        
        # Create working copy of dataframe
        dataset = df.copy()
        
        # Step 1: Process text columns
        if self.text_processing and self.text_columns:
            dataset = self._process_text_columns(dataset, text_processing_options)
        
        # Step 2: Create time series features if applicable
        if self.time_series_features and self.date_column:
            dataset = self._create_time_series_features(dataset, time_series_options)
        
        # Step 3: Prepare data for modeling
        X, y = self._prepare_modeling_data(dataset)
        
        # Step 4: Create cross-validation
        if self.date_column:
            # Time series cross-validation if date column is provided
            cv = self._create_time_series_cv(dataset, cv_folds)
        else:
            # Regular cross-validation
            cv = self._create_standard_cv(dataset, cv_folds)
        
        # Step 5: Create and train the model
        if self.hyperparameter_tuning:
            model, metrics_results = self._train_model_with_tuning(
                X, y, cv, metrics, tuning_options)
        else:
            model, metrics_results = self._train_model(
                X, y, cv, metrics, model_params)
            
        # Step 6: Feature importance
        self.feature_importance = self._get_feature_importance(model, X)
        
        # Store cross-validation results
        self.cv_results = metrics_results
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            df: DataFrame with the same structure as training data
            
        Returns:
            Array of predictions
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create working copy of dataframe
        dataset = df.copy()
        
        # Step 1: Process text columns - apply same transformations as during fit
        if self.text_processing and self.text_columns:
            dataset = self._transform_text_columns(dataset)
        
        # Step 2: Create time series features if applicable
        if self.time_series_features and self.date_column:
            dataset = self._transform_time_series_features(dataset)
        
        # Step 3: Prepare data for prediction - same as in fit
        X = self._prepare_prediction_data(dataset)
        
        # Step 4: Make predictions
        if self.problem_type == 'classification':
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Make probability predictions on new data (classification only).
        
        Args:
            df: DataFrame with the same structure as training data
            
        Returns:
            Array of probability predictions
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba() is only available for classification problems")
            
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create working copy of dataframe
        dataset = df.copy()
        
        # Apply same transformations as in predict()
        if self.text_processing and self.text_columns:
            dataset = self._transform_text_columns(dataset)
        
        if self.time_series_features and self.date_column:
            dataset = self._transform_time_series_features(dataset)
        
        # Prepare data
        X = self._prepare_prediction_data(dataset)
        
        # Make probability predictions
        return self.model.predict_proba(X)
    
    def plot_metrics(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot cross-validation metrics from model training with error bars.
        
        This method visualizes the cross-validation metrics computed during model training,
        showing both the mean performance and variance across folds. This helps assess
        model stability and reliability.
        
        Args:
            figsize (Tuple[int, int]): Figure size as (width, height) in inches.
                Default is (12, 8) for a readable visualization.
            
        Returns:
            plt.Figure: Matplotlib Figure object containing the visualization.
                The figure can be further customized or saved.
                
        Raises:
            ValueError: If no cross-validation results are available (model not fitted).
            
        Example:
            >>> model = AutoModelFlow().fit(df, 'target', metrics=['accuracy', 'f1', 'precision'])
            >>> fig = model.plot_metrics()
            >>> fig.savefig('cv_metrics.png', dpi=300)  # Save high-resolution image
        """
        if self.cv_results is None:
            raise ValueError("No cross-validation results available. Call fit() first.")
        
        return plot_cv_metrics(self.cv_results, figsize=figsize)
    
    def plot_importance(
        self, 
        top_n: int = 20, 
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Plot feature importance to visualize which features most influence the model.
        
        This method creates a bar chart showing the most influential features in the model,
        ordered by importance. For tree-based models (LightGBM, XGBoost), this represents
        the gain or information value of each feature. This visualization helps understand
        which features (including automatically generated text topics and time series features)
        drive the model's predictions.
        
        Args:
            top_n (int): Number of top features to display. Default is 20.
                Set to a smaller value for cleaner visualization or a larger value
                to see more features.
            figsize (Tuple[int, int]): Figure size as (width, height) in inches.
                Default is (12, 8) for a readable visualization.
            
        Returns:
            plt.Figure: Matplotlib Figure object containing the visualization.
                The figure can be further customized or saved.
                
        Raises:
            ValueError: If no feature importance is available (model not fitted).
            
        Example:
            >>> model = AutoModelFlow().fit(df, 'target')
            >>> # Plot top 15 most important features
            >>> fig = model.plot_importance(top_n=15, figsize=(10, 6))
            >>> fig.savefig('feature_importance.png')
            
        Note:
            For text columns, feature names will be prefixed with the column name
            followed by a topic number (e.g., 'description_Topic_1') or text feature
            type (e.g., 'description_word_count').
            
            For time series features, names will include the source column and feature
            type (e.g., 'sales_lag_7', 'date_month').
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Call fit() first.")
        
        return plot_feature_importance(
            self.feature_importance, 
            top_n=top_n, 
            figsize=figsize
        )
    
    def plot_predictions_over_time(
        self, 
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 8),
        date_format: str = '%Y-%m-%d'
    ) -> plt.Figure:
        """Plot model predictions over time compared to actual values.
        
        This method creates a time series visualization that shows both the actual values
        and model predictions over time. This is particularly useful for time series
        forecasting and regression tasks to visualize how well the model captures
        trends, seasonality, and specific events.
        
        Args:
            df (pd.DataFrame): DataFrame containing test or validation data.
                Must include:
                - The date column specified during model fitting
                - The target column with actual values
                - All feature columns needed for prediction
            figsize (Tuple[int, int]): Figure size as (width, height) in inches.
                Default is (15, 8) which provides good visibility of temporal patterns.
            date_format (str): Format string for date labels on the x-axis.
                Default is '%Y-%m-%d'. Common alternatives:
                - '%Y-%m': Year-Month
                - '%b %Y': Month name and year
                - '%d %b': Day and month name
                - '%H:%M:%S': Hour-minute-second
            
        Returns:
            plt.Figure: Matplotlib Figure object containing the visualization.
                The figure shows two lines:
                - Blue line: Actual values from the target column
                - Red dashed line: Model predictions
                
        Raises:
            ValueError: If no date column was specified during model fitting.
            
        Example:
            >>> # Fit model on historical sales data
            >>> model = AutoModelFlow(problem_type='regression')
            >>> model.fit(train_df, target_column='sales', date_column='date')
            >>>
            >>> # Visualize predictions on test data
            >>> fig = model.plot_predictions_over_time(test_df)
            >>> fig.savefig('sales_forecast.png')
            
        Note:
            The date column must be in datetime format or convertible to datetime.
            The dataframe will be automatically sorted by the date column before plotting.
        """
        if not self.date_column:
            raise ValueError("No date column specified. This method is only for time series data.")
            
        # Get actual values
        y_true = df[self.target_column]
        
        # Get predictions
        y_pred = self.predict(df)
        
        # Get dates
        dates = pd.to_datetime(df[self.date_column])
        
        return plot_time_series_predictions(
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            figsize=figsize,
            date_format=date_format
        )
    
    def get_topic_terms(self, text_column: str, n_terms: int = 10) -> Dict[int, List[str]]:
        """Get the most representative terms for each topic in a text column.
        
        This method returns the top terms for each topic discovered during topic modeling
        of the specified text column. These terms provide insight into the semantic
        meaning of each automatically detected topic.
        
        Args:
            text_column (str): Name of the text column for which to retrieve topics.
                Must be a column that was processed during fit().
            n_terms (int): Number of representative terms to return for each topic.
                Default is 10 terms per topic.
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping topic IDs (0-based) to lists of 
                their most representative terms, ordered by importance.
                
        Raises:
            ValueError: If the specified text_column was not processed during fitting
                or doesn't have topic information available.
                
        Example:
            >>> model = AutoModelFlow().fit(df, 'target', text_columns=['description'])
            >>> topics = model.get_topic_terms('description', n_terms=5)
            >>> for topic_id, terms in topics.items():
            ...     print(f"Topic {topic_id + 1}: {', '.join(terms)}")
            Topic 1: finance, banking, markets, investment, stocks
            Topic 2: sports, football, player, team, game
            ...
        """
        if text_column not in self.text_topics:
            raise ValueError(f"No topic model found for column '{text_column}'")
        
        topic_model = self.text_topics[text_column]['topic_model']
        
        # Extract top terms for each topic
        topics = {}
        for topic_id, words in topic_model['topics']:
            topics[topic_id] = words[:n_terms]
        
        return topics
    
    def get_document_topics(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Get topic distributions for new documents in a specific text column.
        
        This method applies the previously trained topic model to new text data,
        returning the distribution of topics for each document. This allows you to
        understand the thematic composition of new documents using the topics
        discovered during training.
        
        The returned DataFrame contains one row per input document and one column
        per topic, with values representing the strength of each topic in that document
        (higher values indicate stronger presence of the topic).
        
        Args:
            df (pd.DataFrame): DataFrame containing the text column to analyze.
                Must contain the specified text_column.
            text_column (str): Name of the text column to process.
                Must be a column that was processed during fit().
            
        Returns:
            pd.DataFrame: DataFrame with document-topic distributions.
                - Index matches the original DataFrame
                - Columns are named "Topic 1", "Topic 2", etc.
                - Values represent topic weights for each document (0.0 to 1.0)
                
        Raises:
            ValueError: If the specified text_column was not processed during fitting
                or doesn't have topic information available.
                
        Example:
            >>> model = AutoModelFlow().fit(train_df, 'target', text_columns=['description'])
            >>> # Get topic distributions for new documents
            >>> new_docs = pd.DataFrame({'description': ["Financial markets report", 
            ...                                         "Sports game highlights"]})
            >>> topic_dist = model.get_document_topics(new_docs, 'description')
            >>> print(topic_dist)
               Topic 1  Topic 2  Topic 3  ...
            0     0.82     0.05     0.13  ...
            1     0.07     0.75     0.18  ...
        """
        if text_column not in self.text_topics:
            raise ValueError(f"No topic model found for column '{text_column}'")
        
        # Process the text with the saved processor
        processor = self.text_topics[text_column]['processor']
        topic_model = self.text_topics[text_column]['topic_model']
        
        # Preprocess text
        processed_texts = []
        for text in df[text_column]:
            processed_text = processor.preprocess_text(
                text,
                remove_stopwords=True,
                remove_punctuation=True,
                lemmatize=True
            )
            processed_texts.append(processed_text)
        
        # Convert to document-term matrix
        vectorizer = topic_model['vectorizer']
        dtm = vectorizer.transform(processed_texts)
        
        # Get topic distribution
        model = topic_model['model']
        doc_topic_matrix = model.transform(dtm)
        
        # Create DataFrame
        topic_cols = [f"Topic {i+1}" for i in range(topic_model['n_topics'])]
        return pd.DataFrame(doc_topic_matrix, columns=topic_cols, index=df.index)
    
    def _detect_text_columns(self, df: pd.DataFrame, ignore_columns: List[str]) -> List[str]:
        """Detect columns that likely contain text data.
        
        Args:
            df: Input DataFrame
            ignore_columns: Columns to ignore
            
        Returns:
            List of detected text column names
        """
        text_columns = []
        
        for col in df.columns:
            if col in ignore_columns:
                continue
                
            # Check if column is string/object type
            if df[col].dtype == 'object':
                # Sample some values and check for text characteristics
                sample = df[col].dropna().sample(min(100, len(df))).astype(str)
                
                # If average word count is >= 5, likely text data
                word_counts = sample.str.split().str.len()
                avg_words = word_counts.mean() if not word_counts.empty else 0
                
                if avg_words >= 5:
                    text_columns.append(col)
        
        return text_columns
    
    def _detect_categorical_columns(self, df: pd.DataFrame, ignore_columns: List[str]) -> List[str]:
        """Detect columns that likely contain categorical data.
        
        Args:
            df: Input DataFrame
            ignore_columns: Columns to ignore
            
        Returns:
            List of detected categorical column names
        """
        categorical_columns = []
        
        for col in df.columns:
            if col in ignore_columns:
                continue
                
            # Skip non-numeric and date columns
            if df[col].dtype == 'datetime64[ns]':
                continue
                
            # Check cardinality relative to dataset size
            nunique = df[col].nunique()
            
            # If low cardinality relative to data size, likely categorical
            if nunique < min(100, len(df) * 0.05):
                categorical_columns.append(col)
                
            # Also consider boolean columns
            elif df[col].dtype == bool or set(df[col].dropna().unique()).issubset({0, 1}):
                categorical_columns.append(col)
        
        return categorical_columns
    
    def _detect_date_columns(self, df: pd.DataFrame, ignore_columns: List[str]) -> List[str]:
        """Detect columns that contain date/time data.
        
        Args:
            df: Input DataFrame
            ignore_columns: Columns to ignore
            
        Returns:
            List of detected date column names
        """
        date_columns = []
        
        for col in df.columns:
            if col in ignore_columns:
                continue
                
            # Check if already datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
                continue
                
            # Try to convert to datetime if string
            if df[col].dtype == 'object':
                try:
                    # Try to parse dates
                    pd.to_datetime(df[col], errors='raise')
                    date_columns.append(col)
                except (ValueError, TypeError):
                    # Not a date column
                    pass
        
        return date_columns
    
    def _process_text_columns(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Process text columns with topic modeling and comprehensive feature extraction.
        
        This method applies advanced text processing to each text column:
        1. Performs automatic topic modeling to discover underlying themes
        2. Creates document-topic distributions as features
        3. Extracts text statistics (length, word count, etc.)
        4. Calculates readability metrics
        5. Performs sentiment analysis
        6. Extracts and counts significant keywords
        
        All extracted features are added as new columns to the dataframe with appropriate
        prefixes to identify their source column.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing text columns
            options (Dict[str, Any]): Text processing configuration options including:
                - topic_modeling (Dict): Configuration for topic modeling
                    - method (str): 'nmf' or 'lda' algorithm to use
                    - auto_topics_range (Tuple[int,int]): Range of topics to try
                    - auto_topics_method (str): Method for automatic topic selection
                - text_features (Dict): Configuration for feature extraction
                    - extract_features (bool): Whether to extract additional text features
                    - include_stats (bool): Whether to include statistical features
                    - include_readability (bool): Whether to include readability metrics 
                    - include_sentiment (bool): Whether to include sentiment analysis
                    - extract_keywords (bool): Whether to extract keyword features
            
        Returns:
            pd.DataFrame: The original dataframe augmented with text-derived features
            
        Note:
            This method stores the fitted topic models and processors in self.text_topics
            and self.text_features for later use during prediction.
        """
        logger.info("Processing text columns...")
        
        # Get options with defaults
        topic_options = options.get('topic_modeling', {})
        feature_options = options.get('text_features', {})
        
        # Preprocessing options for topic modeling
        preprocessing_opts = {
            'enabled': True,
            'use_lemmatization': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
        }
        
        # Create a TextProcessor
        processor = TextProcessor(use_spacy=preprocessing_opts['use_lemmatization'])
        
        # Process each text column
        result_df = df.copy()
        
        for col in self.text_columns:
            logger.info(f"Processing text column: {col}")
            
            # Store the processor for later use
            if col not in self.text_topics:
                self.text_topics[col] = {'processor': processor}
            
            # 1. Topic modeling
            topic_result = create_topic_model_optimized(
                df=df,
                text_column=col,
                n_topics='auto',
                method=topic_options.get('method', 'nmf'),
                preprocessing_options=preprocessing_opts,
                auto_topics_range=topic_options.get('auto_topics_range', (2, 15)),
                auto_topics_method=topic_options.get('auto_topics_method', 'coherence')
            )
            
            # Store topic model results for later use
            self.text_topics[col]['topic_model'] = topic_result['topic_model']
            self.text_topics[col]['topics'] = topic_result['topics']
            self.text_topics[col]['n_topics'] = topic_result['topic_model']['n_topics']
            
            # Add document-topic distribution features
            topic_features = topic_result['document_topics']
            
            # Rename columns to include text column name
            topic_features.columns = [f"{col}_{c}" for c in topic_features.columns]
            
            # Add to result dataframe
            result_df = pd.concat([result_df, topic_features], axis=1)
            
            # 2. Extract additional text features
            if feature_options.get('extract_features', True):
                # Create text statistics, readability, and sentiment features
                text_stats = processor.create_text_features(
                    df,
                    col,
                    include_stats=feature_options.get('include_stats', True),
                    include_readability=feature_options.get('include_readability', True),
                    include_sentiment=feature_options.get('include_sentiment', True),
                    prefix=f"{col}_"
                )
                
                # Extract keywords if requested
                if feature_options.get('extract_keywords', False):
                    # Extract and count keywords for each document
                    all_keywords = []
                    for text in df[col].fillna(""):
                        keywords = processor.extract_keywords_rake(
                            text, max_keywords=5)
                        all_keywords.extend([kw for kw, _ in keywords])
                    
                    # Get top keywords across all documents
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    top_keywords = [kw for kw, _ in keyword_counts.most_common(20)]
                    
                    # Create keyword features
                    for keyword in top_keywords:
                        # Check for keyword presence in each document
                        result_df[f"{col}_kw_{keyword.replace(' ', '_')}"] = df[col].fillna("").str.contains(
                            keyword, case=False, regex=False).astype(int)
                
                # Add text features to result dataframe
                for feature_col in text_stats.columns:
                    if feature_col not in result_df.columns:
                        result_df[feature_col] = text_stats[feature_col]
                
                # Store feature info for later transformation
                self.text_features[col] = {
                    'feature_columns': list(text_stats.columns),
                    'keywords': top_keywords if feature_options.get('extract_keywords', False) else []
                }
        
        return result_df
    
    def _transform_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform text columns using the fitted text processors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with text features
        """
        result_df = df.copy()
        
        for col in self.text_columns:
            if col not in self.text_topics:
                logger.warning(f"No fitted processor found for text column '{col}'")
                continue
                
            # Get processor and topic model
            processor = self.text_topics[col]['processor']
            topic_model = self.text_topics[col]['topic_model']
            
            # 1. Generate topic features
            # Preprocess text
            processed_texts = []
            for text in df[col].fillna(""):
                processed_text = processor.preprocess_text(
                    text,
                    remove_stopwords=True,
                    remove_punctuation=True,
                    lemmatize=True
                )
                processed_texts.append(processed_text)
            
            # Convert to document-term matrix
            dtm = topic_model['vectorizer'].transform(processed_texts)
            
            # Get topic distribution
            doc_topic_matrix = topic_model['model'].transform(dtm)
            
            # Create DataFrame and add to result
            topic_cols = [f"{col}_{i}" for i in range(topic_model['n_topics'])]
            for i, topic_col in enumerate(topic_cols):
                result_df[topic_col] = doc_topic_matrix[:, i]
            
            # 2. Generate text features
            if col in self.text_features:
                # Create text statistics, readability, and sentiment features
                text_stats = processor.create_text_features(
                    df,
                    col,
                    include_stats=True,
                    include_readability=True,
                    include_sentiment=True,
                    prefix=f"{col}_"
                )
                
                # Add text features to result dataframe
                for feature_col in self.text_features[col]['feature_columns']:
                    if feature_col in text_stats.columns:
                        result_df[feature_col] = text_stats[feature_col]
                
                # Add keyword features
                for keyword in self.text_features[col].get('keywords', []):
                    feature_name = f"{col}_kw_{keyword.replace(' ', '_')}"
                    result_df[feature_name] = df[col].fillna("").str.contains(
                        keyword, case=False, regex=False).astype(int)
        
        return result_df
    
    def _create_time_series_features(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Create time series features for date columns.
        
        Args:
            df: Input DataFrame
            options: Time series feature creation options
            
        Returns:
            DataFrame with time series features
        """
        if not self.date_column:
            logger.warning("No date column specified for time series feature creation")
            return df
            
        logger.info(f"Creating time series features for date column: {self.date_column}")
        
        # Copy dataframe
        result_df = df.copy()
        
        # Ensure date column is datetime
        result_df[self.date_column] = pd.to_datetime(result_df[self.date_column])
        
        # Sort by date
        result_df = result_df.sort_values(by=self.date_column)
        
        # Create date-based features
        result_df[f"{self.date_column}_year"] = result_df[self.date_column].dt.year
        result_df[f"{self.date_column}_month"] = result_df[self.date_column].dt.month
        result_df[f"{self.date_column}_day"] = result_df[self.date_column].dt.day
        result_df[f"{self.date_column}_dayofweek"] = result_df[self.date_column].dt.dayofweek
        result_df[f"{self.date_column}_quarter"] = result_df[self.date_column].dt.quarter
        
        # Create lag features for target if requested
        if options.get('create_target_lags', True):
            target = df[self.target_column]
            
            # Create n lag features
            for lag in options.get('lag_periods', [1, 7, 14, 30]):
                result_df[f"{self.target_column}_lag_{lag}"] = target.shift(lag)
                
            # Create rolling window features
            for window in options.get('rolling_windows', [7, 14, 30]):
                result_df[f"{self.target_column}_rolling_mean_{window}"] = target.rolling(window).mean().shift(1)
                result_df[f"{self.target_column}_rolling_std_{window}"] = target.rolling(window).std().shift(1)
        
        # Create features for other numeric columns if requested
        if options.get('include_numeric_columns', True):
            for col in self.numeric_columns:
                # Skip the target itself
                if col == self.target_column:
                    continue
                    
                # Create lag features
                for lag in options.get('numeric_lags', [1, 7]):
                    result_df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                
                # Create rolling features
                for window in options.get('numeric_rolling_windows', [7, 14]):
                    result_df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean().shift(1)
        
        return result_df
    
    def _transform_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform time series features for new data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time series features
        """
        if not self.date_column:
            return df
            
        # Copy dataframe
        result_df = df.copy()
        
        # Ensure date column is datetime
        result_df[self.date_column] = pd.to_datetime(result_df[self.date_column])
        
        # Create date-based features
        result_df[f"{self.date_column}_year"] = result_df[self.date_column].dt.year
        result_df[f"{self.date_column}_month"] = result_df[self.date_column].dt.month
        result_df[f"{self.date_column}_day"] = result_df[self.date_column].dt.day
        result_df[f"{self.date_column}_dayofweek"] = result_df[self.date_column].dt.dayofweek
        result_df[f"{self.date_column}_quarter"] = result_df[self.date_column].dt.quarter
        
        # Note: Lag features can't be properly created without historical context
        # In a real prediction setting, you would need to append new data to historical data
        
        return result_df
    
    def _prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling by encoding categorical features, etc.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y) for modeling
        """
        # Copy dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Drop the target column
        y = result_df[self.target_column]
        X = result_df.drop(columns=[self.target_column])
        
        # Drop date columns
        X = X.drop(columns=self.date_columns, errors='ignore')
        
        # Drop text columns (we've already extracted features from them)
        X = X.drop(columns=self.text_columns, errors='ignore')
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in X.columns:
                # One-hot encode the categorical column
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(columns=[col])
        
        # Fill missing values
        X = X.fillna(0)
        
        return X, y
    
    def _prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction, applying the same transformations as in training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Features DataFrame ready for prediction
        """
        # Copy dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Drop the target column if it exists
        if self.target_column in result_df.columns:
            result_df = result_df.drop(columns=[self.target_column])
        
        # Drop date columns
        result_df = result_df.drop(columns=self.date_columns, errors='ignore')
        
        # Drop text columns (we've already extracted features from them)
        result_df = result_df.drop(columns=self.text_columns, errors='ignore')
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in result_df.columns:
                # One-hot encode the categorical column
                dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=True)
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df = result_df.drop(columns=[col])
        
        # Fill missing values
        result_df = result_df.fillna(0)
        
        return result_df
    
    def _create_time_series_cv(self, df: pd.DataFrame, cv_folds: int) -> Any:
        """Create time series cross-validation splits.
        
        Args:
            df: Input DataFrame
            cv_folds: Number of folds
            
        Returns:
            Cross-validation splitter
        """
        if not self.date_column:
            raise ValueError("No date column specified for time series cross-validation")
            
        # Ensure date column is datetime
        dates = pd.to_datetime(df[self.date_column])
        
        # Create time series CV
        cv = create_time_series_cv(
            dates=dates,
            n_splits=cv_folds,
            test_size=None,  # Use equal sized folds
            gap=0  # No gap between train and test
        )
        
        return cv
    
    def _create_standard_cv(self, df: pd.DataFrame, cv_folds: int) -> Any:
        """Create standard cross-validation splits.
        
        Args:
            df: Input DataFrame
            cv_folds: Number of folds
            
        Returns:
            Cross-validation splitter
        """
        # Get target values
        y = df[self.target_column]
        
        # Create CV based on problem type
        if self.problem_type == 'classification':
            cv = create_stratified_cv(
                y=y,
                n_splits=cv_folds,
                random_state=self.random_state
            )
        else:  # regression
            cv = create_kfold_cv(
                n_splits=cv_folds,
                random_state=self.random_state
            )
        
        return cv
    
    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Any,
        metrics: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a model with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Cross-validation splitter
            metrics: List of metrics to compute
            model_params: Model parameters
            
        Returns:
            Tuple of (model, metrics_results)
        """
        logger.info(f"Training {self.model_type} model for {self.problem_type} with cross-validation")
        
        # Default metrics based on problem type
        if metrics is None:
            if self.problem_type == 'classification':
                metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            else:  # regression
                metrics = ["mse", "rmse", "mae", "r2"]
        
        # Get default parameters based on model type
        default_params = self._get_default_model_params()
        
        # Merge with user-provided params
        params = {**default_params, **(model_params or {})}
        
        # Create CV trainer
        trainer = CrossValidationTrainer(
            model_type=self.model_type,
            problem_type=self.problem_type,
            cv=cv,
            metrics=metrics,
            params=params,
            random_state=self.random_state
        )
        
        # Train the model
        trainer.fit(X, y)
        
        # Get CV results
        cv_results = trainer.get_cv_results()
        
        # Get the final model
        model = trainer.get_final_model()
        
        return model, cv_results
    
    def _train_model_with_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Any,
        metrics: Optional[List[str]] = None,
        tuning_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train a model with hyperparameter tuning and cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Cross-validation splitter
            metrics: List of metrics to compute
            tuning_options: Hyperparameter tuning options
            
        Returns:
            Tuple of (model, metrics_results)
        """
        logger.info(f"Training {self.model_type} model with hyperparameter tuning")
        
        # Default metrics based on problem type
        if metrics is None:
            if self.problem_type == 'classification':
                metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            else:  # regression
                metrics = ["mse", "rmse", "mae", "r2"]
        
        # Default tuning options
        default_tuning = {
            'n_trials': 50,
            'timeout': None,
            'use_optuna': True,
            'early_stopping_rounds': 50
        }
        
        # Merge with user-provided options
        tuning_opts = {**default_tuning, **(tuning_options or {})}
        
        # Default eval metric based on problem type
        if 'eval_metric' not in tuning_opts:
            if self.problem_type == 'classification':
                tuning_opts['eval_metric'] = 'auc'
            else:  # regression
                tuning_opts['eval_metric'] = 'rmse'
        
        # Create pipeline with tuning step
        pipeline = Pipeline()
        
        # Add hyperparameter tuning step
        tuning_step = HyperparameterTuningStep(
            name="hyperparameter_tuning",
            model_type=self.model_type,
            problem_type=self.problem_type,
            metric=tuning_opts['eval_metric'],
            n_trials=tuning_opts['n_trials'],
            cv=cv.n_splits if hasattr(cv, 'n_splits') else 5,
            early_stopping_rounds=tuning_opts['early_stopping_rounds'],
            random_state=self.random_state
        )
        
        pipeline.add_step(tuning_step)
        
        # Add evaluation step
        eval_step = EvaluationStep(
            name="evaluation",
            metrics=metrics,
            problem_type=self.problem_type
        )
        
        pipeline.add_step(eval_step)
        
        # Fit the pipeline
        pipeline.fit(X, y)
        
        # Get tuned model
        model = tuning_step.model
        
        # Get evaluation results
        # Use cross-validation to get metrics
        cv_results = {}
        
        # Store model for later use
        self.model = model
        
        # Get cross-validation results
        cv_metrics = {}
        for metric in metrics:
            cv_metrics[metric] = []
            
        # Perform cross-validation with the tuned model
        for train_idx, test_idx in cv:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clone and fit the model
            from sklearn.base import clone
            cv_model = clone(model)
            cv_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = cv_model.predict(X_test)
            
            # Make probability predictions if classification
            y_prob = None
            if self.problem_type == 'classification' and hasattr(cv_model, 'predict_proba'):
                y_prob = cv_model.predict_proba(X_test)
                if y_prob.shape[1] == 2:  # Binary classification
                    y_prob = y_prob[:, 1]
            
            # Calculate metrics
            fold_metrics = calculate_metrics(
                y_test, y_pred, y_prob, metrics, problem_type=self.problem_type)
            
            # Store results
            for metric, value in fold_metrics.items():
                cv_metrics[metric].append(value)
        
        # Calculate mean and std for each metric
        for metric, values in cv_metrics.items():
            cv_results[f"{metric}_mean"] = np.mean(values)
            cv_results[f"{metric}_std"] = np.std(values)
            cv_results[f"{metric}_values"] = values
        
        return model, cv_results
    
    def _get_feature_importance(self, model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance from the model.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature importances
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            feature_names = X.columns
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
            feature_names = X.columns
        elif hasattr(model, 'get_feature_importance'):
            # LightGBM model wrapper
            importance_dict = model.get_feature_importance()
            feature_names = importance_dict.index
            importances = importance_dict.values
        else:
            # Try permutation importance as a fallback
            try:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(
                    model, X, y=None, n_repeats=10, random_state=self.random_state)
                importances = perm_importance.importances_mean
                feature_names = X.columns
            except Exception as e:
                logger.warning(f"Could not compute feature importance: {str(e)}")
                return pd.DataFrame({'feature': X.columns, 'importance': np.zeros(len(X.columns))})
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def _get_default_model_params(self) -> Dict[str, Any]:
        """Get default model parameters based on model type and problem type.
        
        Returns:
            Dictionary of default parameters
        """
        # Default parameters for popular models
        if self.model_type == 'lightgbm':
            if self.problem_type == 'classification':
                return {
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'n_estimators': 100,
                    'verbosity': -1,
                    'random_state': self.random_state
                }
            else:  # regression
                return {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'n_estimators': 100,
                    'verbosity': -1,
                    'random_state': self.random_state
                }
        elif self.model_type == 'xgboost':
            if self.problem_type == 'classification':
                return {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 100,
                    'random_state': self.random_state
                }
            else:  # regression
                return {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 100,
                    'random_state': self.random_state
                }
        else:  # other models
            return {
                'random_state': self.random_state
            }


# Create a direct function interface for easier usage
def auto_model(
    df: pd.DataFrame,
    target_column: str,
    date_column: Optional[str] = None,
    model_type: str = "lightgbm",
    problem_type: str = "classification",
    text_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    cv_folds: int = 5,
    metrics: Optional[List[str]] = None,
    tuning: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Automated modeling function for tabular, text, and time series data.
    
    This function provides a simplified interface to the AutoModelFlow class, automatically
    handling the complete modeling workflow from data analysis to model training and evaluation.
    It intelligently processes different types of features (text, categorical, date/time) and
    selects appropriate methods based on the data characteristics.
    
    The function performs the following steps:
    1. Analyzes the dataset to identify column types
    2. Processes text columns with topic modeling and feature extraction
    3. Creates time series features if date column is provided
    4. Trains a model with appropriate cross-validation (time series CV for temporal data)
    5. Performs hyperparameter tuning if requested
    6. Evaluates model performance with multiple metrics
    7. Extracts feature importance
    
    Args:
        df (pd.DataFrame): Input DataFrame containing all features and target
        target_column (str): Name of the target column to predict
        date_column (Optional[str]): Name of main date/time column for time series analysis.
            If provided, time series features and time-based cross-validation will be used.
        model_type (str): Type of model to use:
            - 'lightgbm': LightGBM (default, good for most tasks)
            - 'xgboost': XGBoost
            - 'catboost': CatBoost
            - 'sklearn': Generic scikit-learn model
        problem_type (str): Type of prediction problem:
            - 'classification': For predicting categorical targets
            - 'regression': For predicting continuous targets
        text_columns (Optional[List[str]]): List of text column names to process with topic modeling.
            If None, will be auto-detected based on text characteristics.
        categorical_columns (Optional[List[str]]): List of categorical column names.
            If None, will be auto-detected based on cardinality.
        cv_folds (int): Number of cross-validation folds
        metrics (Optional[List[str]]): List of metrics to compute during evaluation.
            If None, default metrics will be selected based on problem_type:
            - Classification: ["accuracy", "precision", "recall", "f1", "roc_auc"]
            - Regression: ["mse", "rmse", "mae", "r2"]
        tuning (bool): Whether to perform hyperparameter tuning (default=True)
        **kwargs: Additional options including:
            - random_state (int): Random seed for reproducibility
            - verbose (bool): Whether to print progress information
            - ignore_columns (List[str]): Columns to ignore during processing
            - model_params (Dict[str, Any]): Parameters for the model if not using tuning
            - text_options (Dict[str, Any]): Options for text processing, including:
                - topic_modeling (Dict): Options for topic modeling
                - text_features (Dict): Options for text feature extraction
            - time_options (Dict[str, Any]): Options for time series feature creation, including:
                - lag_periods (List[int]): Periods for lag features
                - rolling_windows (List[int]): Window sizes for rolling statistics
            - tuning_options (Dict[str, Any]): Options for hyperparameter tuning, including:
                - n_trials (int): Number of tuning trials
                - eval_metric (str): Metric to optimize
                - early_stopping_rounds (int): Patience for early stopping
            
    Returns:
        Dict[str, Any]: Dictionary with comprehensive modeling results:
            - model: The trained model ready for predictions
            - metrics: Dictionary of cross-validation metrics (mean and std for each metric)
            - feature_importance: DataFrame with feature importance scores
            - text_topics: Dictionary of topic models for each text column
            - autoflow: The full AutoModelFlow instance for additional operations
            - dataset_info: Dictionary with information about the analyzed dataset
    
    Examples:
    ---------
    # Basic classification with automatic feature detection
    >>> from freamon.modeling import auto_model
    >>> results = auto_model(
    ...     df=train_df,
    ...     target_column='is_fraudulent',
    ...     problem_type='classification'
    ... )
    >>> # Make predictions on new data
    >>> predictions = results['model'].predict(test_df)
    
    # Time series forecasting with custom options
    >>> results = auto_model(
    ...     df=historical_data, 
    ...     target_column='sales',
    ...     date_column='date',
    ...     problem_type='regression',
    ...     time_options={
    ...         'lag_periods': [1, 7, 14, 28],  # Daily, weekly, bi-weekly, monthly lags
    ...         'rolling_windows': [7, 14, 30]   # Weekly, bi-weekly, monthly rolling stats
    ...     },
    ...     metrics=['rmse', 'mae']
    ... )
    >>> # Plot predictions over time
    >>> results['autoflow'].plot_predictions_over_time(test_df)
    
    # Text classification with topic modeling
    >>> results = auto_model(
    ...     df=document_df,
    ...     target_column='category',
    ...     text_columns=['content', 'title'],
    ...     text_options={
    ...         'topic_modeling': {
    ...             'method': 'nmf',
    ...             'auto_topics_range': (2, 20),
    ...             'auto_topics_method': 'stability'
    ...         }
    ...     }
    ... )
    >>> # Examine topics found in the text
    >>> for column, topic_info in results['text_topics'].items():
    ...     print(f"\nTopics in '{column}':")
    ...     for topic_idx, terms in topic_info['topics']:
    ...         print(f"Topic {topic_idx+1}: {', '.join(terms[:7])}")
    """
    # Create AutoModelFlow
    autoflow = AutoModelFlow(
        model_type=model_type,
        problem_type=problem_type,
        text_processing=True,
        time_series_features=date_column is not None,
        hyperparameter_tuning=tuning,
        random_state=kwargs.get('random_state', 42),
        verbose=kwargs.get('verbose', True)
    )
    
    # Analyze dataset
    dataset_info = autoflow.analyze_dataset(
        df=df,
        target_column=target_column,
        date_column=date_column,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        ignore_columns=kwargs.get('ignore_columns', None)
    )
    
    # Train model
    autoflow.fit(
        df=df,
        target_column=target_column,
        date_column=date_column,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        ignore_columns=kwargs.get('ignore_columns', None),
        cv_folds=cv_folds,
        metrics=metrics,
        model_params=kwargs.get('model_params', None),
        text_processing_options=kwargs.get('text_options', None),
        time_series_options=kwargs.get('time_options', None),
        tuning_options=kwargs.get('tuning_options', None)
    )
    
    # Return comprehensive results
    return {
        'model': autoflow.model,
        'metrics': autoflow.cv_results,
        'feature_importance': autoflow.feature_importance,
        'text_topics': autoflow.text_topics,
        'autoflow': autoflow,  # Return the full autoflow for additional operations
        'dataset_info': dataset_info
    }