"""
Main module for the EDA analyzer class.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Literal, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from freamon.utils import check_dataframe_type, convert_dataframe
from freamon.utils.datatype_detector import DataTypeDetector
from freamon.utils.matplotlib_fixes import configure_matplotlib_for_currency, patch_freamon_eda
from freamon.eda.univariate import (
    analyze_numeric,
    analyze_categorical,
    analyze_datetime,
)
from freamon.eda.bivariate import (
    analyze_correlation,
    analyze_feature_target,
)
from freamon.eda.time_series import (
    analyze_timeseries,
    analyze_seasonality,
)
from freamon.eda.multivariate import analyze_multivariate
from freamon.eda.report import generate_html_report
from freamon.eda.markdown_report import generate_markdown_report


class EDAAnalyzer:
    """
    Class for performing exploratory data analysis on a dataframe.
    
    This class provides methods for analyzing a dataframe, including univariate
    analysis, bivariate analysis, and time series analysis. The results can be
    visualized and saved to an HTML report.
    
    Parameters
    ----------
    df : Any
        The dataframe to analyze. Can be pandas, polars, or dask.
    target_column : Optional[str], default=None
        The name of the target column for supervised learning analysis.
    date_column : Optional[str], default=None
        The name of the datetime column for time series analysis.
    
    Attributes
    ----------
    df : pd.DataFrame
        The dataframe to analyze, converted to pandas.
    target_column : Optional[str]
        The name of the target column.
    date_column : Optional[str]
        The name of the datetime column.
    numeric_columns : List[str]
        The list of numeric columns in the dataframe.
    categorical_columns : List[str]
        The list of categorical columns in the dataframe.
    datetime_columns : List[str]
        The list of datetime columns in the dataframe.
    n_rows : int
        The number of rows in the dataframe.
    n_cols : int
        The number of columns in the dataframe.
    dataframe_type : str
        The type of the input dataframe ('pandas', 'polars', 'dask', or 'unknown').
    """
    
    def __init__(
        self,
        df: Any,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        use_sampling: bool = False,
        sample_size: Optional[int] = None,
    ):
        """
        Initialize the EDAAnalyzer.
        
        Parameters
        ----------
        df : Any
            The dataframe to analyze. Can be pandas, polars, or dask.
        target_column : Optional[str], default=None
            The name of the target column for supervised learning analysis.
        date_column : Optional[str], default=None
            The name of the datetime column for time series analysis.
        custom_patterns : Optional[Dict[str, str]], default=None
            Dictionary of custom regex patterns for semantic type detection
            in the format {'type_name': 'regex_pattern'}.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        """
        # Apply matplotlib fixes before any plotting
        try:
            configure_matplotlib_for_currency()
            patch_freamon_eda()
        except Exception as e:
            import warnings
            warnings.warn(f"Could not apply matplotlib fixes: {str(e)}")
            
        self.dataframe_type = check_dataframe_type(df)
        self.custom_patterns = custom_patterns
        self.use_sampling = use_sampling
        self.sample_size = sample_size
        
        # Convert to pandas for analysis
        if self.dataframe_type != 'pandas':
            self.df = convert_dataframe(df, 'pandas')
        else:
            self.df = df.copy()
        
        self.target_column = target_column
        self.date_column = date_column
        
        # Validate target column
        if self.target_column is not None and self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
        
        # Validate date column
        if self.date_column is not None and self.date_column not in self.df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in dataframe")
        
        # Set column types
        self._set_column_types()
        
        # Set basic stats
        self.n_rows, self.n_cols = self.df.shape
        
        # Initialize result storage and caches
        self.analysis_results = {}
        self._multivariate_cache = {}
    
    def _set_column_types(self) -> None:
        """Identify numeric, categorical, and datetime columns with advanced type detection."""
        # Use advanced type detection with custom patterns if provided
        detector = DataTypeDetector(
            self.df,
            custom_patterns=self.custom_patterns
        )
        detected_types = detector.detect_all_types()
        
        # Store the detected types for reference
        self.detected_types = detected_types
        
        # Initialize column type lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        
        # Classify columns based on detected logical type
        for col, info in detected_types.items():
            logical_type = info.get('logical_type', 'unknown')
            
            if logical_type in ['datetime']:
                self.datetime_columns.append(col)
            elif logical_type in ['integer', 'float', 'continuous_integer', 'continuous_float']:
                self.numeric_columns.append(col)
            elif logical_type == 'boolean' or (
                # Special case for binary columns - treat as numeric if they are 0/1
                logical_type in ['categorical_integer'] and 
                self.df[col].dropna().nunique() == 2 and
                set(self.df[col].dropna().unique()).issubset({0, 1, True, False})
            ):
                # Binary columns should be in both numeric and categorical
                self.numeric_columns.append(col)
                self.categorical_columns.append(col)
            elif logical_type in ['categorical', 'categorical_integer', 'categorical_float', 'boolean', 'string']:
                self.categorical_columns.append(col)
            else:
                # Fallback to basic type detection based on storage type
                dtype = self.df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    self.numeric_columns.append(col)
                elif pd.api.types.is_datetime64_dtype(dtype):
                    self.datetime_columns.append(col)
                else:
                    self.categorical_columns.append(col)
        
        # If date_column is specified but not detected as datetime, try to convert
        if (
            self.date_column is not None 
            and self.date_column not in self.datetime_columns
            and self.date_column in self.df.columns
        ):
            try:
                self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
                self.datetime_columns.append(self.date_column)
                
                # Remove from other type lists if present
                if self.date_column in self.numeric_columns:
                    self.numeric_columns.remove(self.date_column)
                if self.date_column in self.categorical_columns:
                    self.categorical_columns.remove(self.date_column)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {self.date_column} to datetime")
    
    def analyze_basic_stats(self) -> Dict[str, Any]:
        """
        Calculate basic statistics about the dataframe.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary with basic statistics about the dataframe.
        """
        stats = {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_numeric": len(self.numeric_columns),
            "n_categorical": len(self.categorical_columns),
            "n_datetime": len(self.datetime_columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            "has_missing": self.df.isna().any().any(),
            "missing_count": self.df.isna().sum().sum(),
            "missing_percent": (self.df.isna().sum().sum() / (self.n_rows * self.n_cols)) * 100,
        }
        
        # Add column lists
        stats["numeric_columns"] = self.numeric_columns
        stats["categorical_columns"] = self.categorical_columns
        stats["datetime_columns"] = self.datetime_columns
        
        # Add detected types information if available
        if hasattr(self, 'detected_types'):
            stats["detected_types"] = self.detected_types
            
            # Extract semantic types for easier reference
            semantic_types = {}
            for col, info in self.detected_types.items():
                if 'semantic_type' in info:
                    semantic_types[col] = info['semantic_type']
            
            if semantic_types:
                stats["semantic_types"] = semantic_types
            
            # Extract conversion suggestions
            conversion_suggestions = {}
            for col, info in self.detected_types.items():
                if 'suggested_conversion' in info:
                    conversion_suggestions[col] = info['suggested_conversion']
            
            if conversion_suggestions:
                stats["conversion_suggestions"] = conversion_suggestions
        
        # Calculate missing values by column
        missing_by_col = self.df.isna().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        stats["missing_by_column"] = missing_cols.to_dict()
        
        # Store results
        self.analysis_results["basic_stats"] = stats
        
        return stats
    
    def analyze_univariate(
        self, 
        columns: Optional[List[str]] = None,
        max_categories: int = 20,
        bins: Optional[int] = None,
        sample_size: Optional[int] = None,
        use_sampling: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform univariate analysis on the specified columns.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            The columns to analyze. If None, all columns are analyzed.
        max_categories : int, default=20
            The maximum number of categories to display for categorical columns.
        bins : Optional[int], default=None
            The number of bins for histograms. If None, a suitable number is chosen.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary with analysis results for each column.
        """
        if columns is None:
            # Analyze all columns
            columns = self.df.columns.tolist()
        
        result = {}
        
        # Determine if sampling should be used and the sample size
        df_to_analyze = self.df
        sampling_info = {}
        
        if use_sampling and self.n_rows > 10000:
            if sample_size is None:
                # Choose a reasonable sample size based on dataframe size
                if self.n_rows > 1000000:
                    sample_size = 100000
                elif self.n_rows > 100000:
                    sample_size = 50000
                else:
                    sample_size = 10000
            
            # Sample the dataframe
            df_to_analyze = self.df.sample(min(sample_size, self.n_rows), random_state=42)
            sampling_info = {
                "original_size": self.n_rows,
                "sample_size": len(df_to_analyze),
                "sampling_ratio": len(df_to_analyze) / self.n_rows
            }
        
        for col in columns:
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in dataframe")
                continue
            
            # Analyze based on column type
            if col in self.numeric_columns:
                col_result = analyze_numeric(df_to_analyze, col, bins=bins)
            elif col in self.categorical_columns:
                col_result = analyze_categorical(df_to_analyze, col, max_categories=max_categories)
            elif col in self.datetime_columns:
                col_result = analyze_datetime(df_to_analyze, col)
            else:
                # Try to determine type and analyze
                dtype = df_to_analyze[col].dtype
                if np.issubdtype(dtype, np.number):
                    col_result = analyze_numeric(df_to_analyze, col, bins=bins)
                elif np.issubdtype(dtype, np.datetime64):
                    col_result = analyze_datetime(df_to_analyze, col)
                else:
                    col_result = analyze_categorical(df_to_analyze, col, max_categories=max_categories)
            
            # Add sampling information if sampling was used
            if sampling_info:
                col_result["sampling_info"] = sampling_info
                
            result[col] = col_result
        
        # Store results
        self.analysis_results["univariate"] = result
        
        return result
    
    def analyze_bivariate(
        self,
        columns: Optional[List[str]] = None,
        target: Optional[str] = None,
        correlation_method: str = 'pearson',
        sample_size: Optional[int] = None,
        use_sampling: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform bivariate analysis on the specified columns.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            The columns to analyze. If None, all numeric columns are analyzed.
        target : Optional[str], default=None
            The target column for feature-target analysis. If None, the class's
            target_column is used.
        correlation_method : str, default='pearson'
            The correlation method to use. Options: 'pearson', 'spearman', 'kendall'.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with bivariate analysis results.
        """
        result = {}
        
        # Set target column
        if target is None:
            target = self.target_column
        
        # Set columns to analyze
        if columns is None:
            if target is not None:
                # Exclude target from feature columns
                columns = [col for col in self.df.columns if col != target]
            else:
                columns = self.df.columns.tolist()
        
        # Determine if sampling should be used and the sample size
        df_to_analyze = self.df
        sampling_info = {}
        
        if use_sampling and self.n_rows > 10000:
            if sample_size is None:
                # Choose a reasonable sample size based on dataframe size
                if self.n_rows > 1000000:
                    sample_size = 100000
                elif self.n_rows > 100000:
                    sample_size = 50000
                else:
                    sample_size = 10000
            
            # Sample the dataframe
            df_to_analyze = self.df.sample(min(sample_size, self.n_rows), random_state=42)
            sampling_info = {
                "original_size": self.n_rows,
                "sample_size": len(df_to_analyze),
                "sampling_ratio": len(df_to_analyze) / self.n_rows
            }
        
        # Calculate correlation matrix for numeric columns
        numeric_cols = [col for col in columns if col in self.numeric_columns]
        if len(numeric_cols) > 1:
            result["correlation"] = analyze_correlation(
                df_to_analyze, columns=numeric_cols, method=correlation_method
            )
            
            # Add sampling information if sampling was used
            if sampling_info:
                result["correlation"]["sampling_info"] = sampling_info
        
        # Analyze relationship with target if provided
        if target is not None:
            if target not in self.df.columns:
                raise ValueError(f"Target column '{target}' not found in dataframe")
            
            target_analysis = {}
            for col in columns:
                if col == target:
                    continue
                
                col_result = analyze_feature_target(df_to_analyze, feature=col, target=target)
                
                # Add sampling information if sampling was used
                if sampling_info:
                    col_result["sampling_info"] = sampling_info
                    
                target_analysis[col] = col_result
            
            result["feature_target"] = target_analysis
        
        # Store results
        self.analysis_results["bivariate"] = result
        
        return result
    
    def analyze_time_series(
        self,
        date_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        freq: Optional[str] = None,
        lags: int = 20,
    ) -> Dict[str, Any]:
        """
        Perform time series analysis on the specified columns.
        
        Parameters
        ----------
        date_col : Optional[str], default=None
            The datetime column to use. If None, the class's date_column is used.
        value_cols : Optional[List[str]], default=None
            The numeric columns to analyze. If None, all numeric columns are used.
        freq : Optional[str], default=None
            The frequency to use for resampling. If None, no resampling is done.
        lags : int, default=20
            The number of lags to use for autocorrelation analysis.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with time series analysis results.
        """
        # Set date column
        if date_col is None:
            date_col = self.date_column
        
        if date_col is None:
            # Try to find a datetime column
            if len(self.datetime_columns) == 0:
                raise ValueError("No datetime column found for time series analysis")
            date_col = self.datetime_columns[0]
        
        # Make sure date_col is a datetime column
        if not pd.api.types.is_datetime64_dtype(self.df[date_col]):
            try:
                # Try to convert to datetime
                self.df[date_col] = pd.to_datetime(self.df[date_col])
            except (ValueError, TypeError):
                raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
        
        # Set value columns
        if value_cols is None:
            value_cols = self.numeric_columns
        
        # Sort dataframe by date if needed
        if not self.df[date_col].equals(self.df[date_col].sort_values()):
            self.df = self.df.sort_values(by=date_col).reset_index(drop=True)
        
        result = {}
        
        # Analyze each value column
        for col in value_cols:
            if col not in self.df.columns:
                print(f"Warning: Column '{col}' not found in dataframe")
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"Warning: Column '{col}' is not numeric, skipping time series analysis")
                continue
            
            # Basic time series analysis
            ts_result = analyze_timeseries(self.df, date_col=date_col, value_col=col, freq=freq)
            
            # Seasonality analysis (if enough data)
            if len(self.df) >= 24:  # Need enough data for seasonality
                seasonal_result = analyze_seasonality(
                    self.df, date_col=date_col, value_col=col, freq=freq
                )
                ts_result["seasonality"] = seasonal_result
            
            result[col] = ts_result
        
        # Store results
        self.analysis_results["time_series"] = result
        
        return result
    
    def generate_report(
        self,
        output_path: str,
        title: str = "Exploratory Data Analysis Report",
        format: str = "html",
        theme: str = "cosmo",
        lazy_loading: bool = True,
        include_export_button: bool = True,
        convert_to_html: bool = False,
    ) -> str:
        """
        Generate a report with the analysis results.
        
        Parameters
        ----------
        output_path : str
            The path to save the report.
        title : str, default="Exploratory Data Analysis Report"
            The title of the report.
        format : str, default="html"
            The format of the report. Options: 'html', 'markdown'.
        theme : str, default="cosmo"
            The Bootstrap theme to use for HTML reports.
            Options: 'cosmo', 'flatly', 'journal', 'lumen', 'sandstone',
            'simplex', 'spacelab', 'united', 'yeti'.
        lazy_loading : bool, default=True
            Whether to enable lazy loading for images in HTML reports, which can improve 
            performance for reports with many visualizations.
        include_export_button : bool, default=True
            Whether to include a button that allows exporting the report 
            as a Jupyter notebook (HTML reports only).
        convert_to_html : bool, default=False
            If format is 'markdown', whether to also generate an HTML version of the report.
            
        Returns
        -------
        str
            The report as a string in the specified format.
        """
        # Make sure we have some results
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            print("No analysis results found. Run analyze_* methods first.")
            return
        
        # Run basic stats if not already done
        if "basic_stats" not in self.analysis_results:
            self.analyze_basic_stats()
        
        # Generate the report based on the format
        if format.lower() == "markdown":
            report = generate_markdown_report(
                df=self.df,
                analysis_results=self.analysis_results,
                output_path=output_path,
                title=title,
                convert_to_html=convert_to_html,
                include_export_button=include_export_button,
            )
        else:  # default to HTML
            report = generate_html_report(
                df=self.df,
                analysis_results=self.analysis_results,
                output_path=output_path,
                title=title,
                theme=theme,
                lazy_loading=lazy_loading,
                include_export_button=include_export_button,
            )
        
        return report
    
    def analyze_feature_importance(
        self,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        method: str = 'random_forest',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        sample_size: Optional[int] = None,
        use_sampling: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate and analyze feature importance for a target variable.
        
        Parameters
        ----------
        features : Optional[List[str]], default=None
            List of feature columns to use. If None, all numeric columns except target are used.
        target : Optional[str], default=None
            The name of the target column. If None, the class's target_column is used.
        method : str, default='random_forest'
            Method to use for importance calculation. Options: 'random_forest', 'permutation', 'shap'.
        n_estimators : int, default=100
            Number of estimators for tree-based methods.
        max_depth : Optional[int], default=None
            Maximum depth for tree-based methods.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with feature importance analysis results.
        """
        from freamon.eda.bivariate import calculate_feature_importance
        
        # Set target column
        if target is None:
            target = self.target_column
        
        if target is None:
            raise ValueError("Target column must be specified for feature importance analysis")
        
        # Set feature columns
        if features is None:
            # Use all numeric columns except target
            features = [col for col in self.numeric_columns if col != target]
        
        # Determine if sampling should be used and the sample size
        df_to_analyze = self.df
        sampling_info = {}
        
        if use_sampling and self.n_rows > 10000:
            if sample_size is None:
                # Choose a reasonable sample size based on dataframe size
                if self.n_rows > 1000000:
                    sample_size = 100000
                elif self.n_rows > 100000:
                    sample_size = 50000
                else:
                    sample_size = 10000
            
            # Sample the dataframe
            df_to_analyze = self.df.sample(min(sample_size, self.n_rows), random_state=42)
            sampling_info = {
                "original_size": self.n_rows,
                "sample_size": len(df_to_analyze),
                "sampling_ratio": len(df_to_analyze) / self.n_rows
            }
        
        # Calculate feature importance
        importance_result = calculate_feature_importance(
            df=df_to_analyze,
            features=features,
            target=target,
            method=method,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        
        # Add sampling information if sampling was used
        if sampling_info:
            importance_result["sampling_info"] = sampling_info
        
        # Store results
        if "feature_importance" not in self.analysis_results:
            self.analysis_results["feature_importance"] = {}
        
        self.analysis_results["feature_importance"][target] = importance_result
        
        return importance_result
        
    def analyze_multivariate(
        self,
        columns: Optional[List[str]] = None,
        method: Literal['pca', 'tsne', 'correlation_network', 'interaction_heatmap', 'all'] = 'all',
        n_components: int = 2,
        scale: bool = True,
        tsne_perplexity: float = 30.0,
        tsne_learning_rate: Union[float, str] = 'auto',
        tsne_n_iter: int = 1000,
        correlation_threshold: float = 0.5,
        correlation_method: str = 'pearson',
        correlation_layout: str = 'spring',
        max_heatmap_features: int = 20,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        use_sampling: bool = False,
        cache_results: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform multivariate analysis on the specified columns.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            The columns to use for analysis. If None, all numeric columns are used.
        method : Literal['pca', 'tsne', 'correlation_network', 'interaction_heatmap', 'all'], default='all'
            The analysis method(s) to use.
        n_components : int, default=2
            The number of components to extract for PCA and t-SNE.
        scale : bool, default=True
            Whether to standardize the data before PCA and t-SNE.
        tsne_perplexity : float, default=30.0
            The perplexity parameter for t-SNE.
        tsne_learning_rate : Union[float, str], default='auto'
            The learning rate for t-SNE.
        tsne_n_iter : int, default=1000
            The number of iterations for t-SNE.
        correlation_threshold : float, default=0.5
            Minimum absolute correlation value for the correlation network.
        correlation_method : str, default='pearson'
            The correlation method to use ('pearson', 'spearman', or 'kendall').
        correlation_layout : str, default='spring'
            The graph layout algorithm for the correlation network.
        max_heatmap_features : int, default=20
            Maximum number of features to include in the interaction heatmap.
        plot_kwargs : Optional[Dict[str, Any]], default=None
            Additional arguments to pass to the plot functions.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
        cache_results : bool, default=True
            Whether to cache results for expensive operations like PCA and t-SNE.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary with multivariate analysis results.
        """
        # Set columns to analyze
        if columns is None:
            columns = self.numeric_columns
        
        # Make sure there are enough numeric columns
        numeric_cols = [col for col in columns if col in self.numeric_columns]
        if len(numeric_cols) < 2:
            raise ValueError("At least 2 numeric columns are required for multivariate analysis")
        
        # Determine if sampling should be used and the sample size
        df_to_analyze = self.df
        sampling_info = {}
        
        if use_sampling and self.n_rows > 10000:
            if sample_size is None:
                # Choose a reasonable sample size based on dataframe size
                if self.n_rows > 1000000:
                    sample_size = 100000
                elif self.n_rows > 100000:
                    sample_size = 50000
                else:
                    sample_size = 10000
            
            # Sample the dataframe
            df_to_analyze = self.df.sample(min(sample_size, self.n_rows), random_state=42)
            sampling_info = {
                "original_size": self.n_rows,
                "sample_size": len(df_to_analyze),
                "sampling_ratio": len(df_to_analyze) / self.n_rows
            }
            
        # Check if we have cached results we can use
        cache_key = None
        if cache_results and hasattr(self, '_multivariate_cache'):
            # Create a cache key based on parameters
            cache_key = (
                tuple(sorted(numeric_cols)), 
                method,
                n_components,
                scale,
                tsne_perplexity if 'tsne' in method or method == 'all' else None,
                tsne_n_iter if 'tsne' in method or method == 'all' else None,
                correlation_threshold if 'correlation_network' in method or method == 'all' else None,
                correlation_method if 'correlation_network' in method or method == 'all' else None,
                len(df_to_analyze)  # Include dataframe size in cache key
            )
            
            if cache_key in self._multivariate_cache:
                # We have cached results we can use
                cached_result = self._multivariate_cache[cache_key].copy()
                
                # Add sampling information if sampling was used
                if sampling_info:
                    for method_name, method_results in cached_result.items():
                        method_results["sampling_info"] = sampling_info.copy()
                
                # Store results
                self.analysis_results["multivariate"] = cached_result
                return cached_result
        
        # Perform multivariate analysis
        result = analyze_multivariate(
            df=df_to_analyze,
            columns=numeric_cols,
            method=method,
            n_components=n_components,
            scale=scale,
            tsne_perplexity=tsne_perplexity,
            tsne_learning_rate=tsne_learning_rate,
            tsne_n_iter=tsne_n_iter,
            correlation_threshold=correlation_threshold,
            correlation_method=correlation_method,
            correlation_layout=correlation_layout,
            max_heatmap_features=max_heatmap_features,
            plot_kwargs=plot_kwargs,
        )
        
        # Add sampling information if sampling was used
        if sampling_info:
            for method_name, method_results in result.items():
                method_results["sampling_info"] = sampling_info.copy()
        
        # Cache the results if caching is enabled
        if cache_results:
            if not hasattr(self, '_multivariate_cache'):
                self._multivariate_cache = {}
            
            if cache_key is not None:
                self._multivariate_cache[cache_key] = result.copy()
        
        # Store results
        self.analysis_results["multivariate"] = result
        
        return result
        
    def run_full_analysis(
        self,
        output_path: Optional[str] = None,
        title: str = "Exploratory Data Analysis Report",
        include_multivariate: bool = False,  # Changed default to False to improve performance
        include_feature_importance: bool = True,
        sample_size: Optional[int] = None,
        use_sampling: bool = False,
        cache_results: bool = True,
        show_progress: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a complete analysis and optionally generate a report.
        
        Parameters
        ----------
        output_path : Optional[str], default=None
            The path to save the HTML report. If None, no report is generated.
        title : str, default="Exploratory Data Analysis Report"
            The title of the report.
        include_multivariate : bool, default=False
            Whether to include multivariate analysis in the full analysis. Set to False by default
            to improve performance since multivariate analysis can be computationally expensive.
        sample_size : Optional[int], default=None
            The number of rows to sample for analysis. If None and use_sampling is True,
            a suitable sample size is chosen based on the dataframe size.
        use_sampling : bool, default=False
            Whether to use sampling for large datasets to speed up analysis.
        cache_results : bool, default=True
            Whether to cache results for expensive operations like PCA and t-SNE.
        show_progress : bool, default=False
            Whether to show progress messages during analysis.
            
        Returns
        -------
        Dict[str, Any]
            A dictionary with all analysis results.
        """
        import time
        
        start_time = time.time()
        
        if show_progress:
            print("Starting full analysis...")
            print(f"DataFrame size: {self.n_rows} rows, {self.n_cols} columns")
            if use_sampling and self.n_rows > 10000:
                actual_sample = sample_size if sample_size is not None else (
                    100000 if self.n_rows > 1000000 else 
                    50000 if self.n_rows > 100000 else 
                    10000
                )
                print(f"Using sampling with sample size: {actual_sample} rows")
        
        # Run basic stats
        if show_progress:
            print("Running basic statistics analysis...")
            bs_start = time.time()
            
        self.analyze_basic_stats()
        
        if show_progress:
            bs_end = time.time()
            print(f"Basic statistics complete in {bs_end - bs_start:.2f} seconds")
            print("Running univariate analysis...")
            univ_start = time.time()
        
        # Run univariate analysis
        self.analyze_univariate(
            sample_size=sample_size,
            use_sampling=use_sampling
        )
        
        if show_progress:
            univ_end = time.time()
            print(f"Univariate analysis complete in {univ_end - univ_start:.2f} seconds")
            print("Running bivariate analysis...")
            biv_start = time.time()
        
        # Run bivariate analysis
        try:
            self.analyze_bivariate(
                sample_size=sample_size,
                use_sampling=use_sampling
            )
            if show_progress:
                biv_end = time.time()
                print(f"Bivariate analysis complete in {biv_end - biv_start:.2f} seconds")
        except Exception as e:
            print(f"Warning: Bivariate analysis failed: {e}")
        
        # Run time series analysis if datetime columns are present
        if self.date_column is not None or len(self.datetime_columns) > 0:
            if show_progress:
                print("Running time series analysis...")
                ts_start = time.time()
                
            try:
                self.analyze_time_series()
                if show_progress:
                    ts_end = time.time()
                    print(f"Time series analysis complete in {ts_end - ts_start:.2f} seconds")
            except Exception as e:
                print(f"Warning: Time series analysis failed: {e}")
        
        # Run multivariate analysis if requested
        if include_multivariate and len(self.numeric_columns) >= 2:
            if show_progress:
                print("Running multivariate analysis...")
                mv_start = time.time()
                
            try:
                self.analyze_multivariate(
                    sample_size=sample_size,
                    use_sampling=use_sampling,
                    cache_results=cache_results
                )
                if show_progress:
                    mv_end = time.time()
                    print(f"Multivariate analysis complete in {mv_end - mv_start:.2f} seconds")
            except Exception as e:
                print(f"Warning: Multivariate analysis failed: {e}")
        
        # Run feature importance analysis if requested and target column is set
        if include_feature_importance and self.target_column is not None:
            if show_progress:
                print("Running feature importance analysis...")
                fi_start = time.time()
                
            try:
                self.analyze_feature_importance(
                    target=self.target_column,
                    sample_size=sample_size,
                    use_sampling=use_sampling
                )
                if show_progress:
                    fi_end = time.time()
                    print(f"Feature importance analysis complete in {fi_end - fi_start:.2f} seconds")
            except Exception as e:
                print(f"Warning: Feature importance analysis failed: {e}")
        
        # Generate report if requested
        if output_path is not None:
            if show_progress:
                print("Generating HTML report...")
                report_start = time.time()
                
            self.generate_report(output_path=output_path, title=title)
            
            if show_progress:
                report_end = time.time()
                print(f"Report generation complete in {report_end - report_start:.2f} seconds")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if show_progress:
            print(f"Full analysis completed in {total_time:.2f} seconds")
        
        return self.analysis_results