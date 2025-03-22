"""
Module for Exploratory Data Analysis (EDA).
"""

from freamon.eda.analyzer import EDAAnalyzer
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
    analyze_autocorrelation,
    analyze_seasonality,
)
from freamon.eda.multivariate import (
    analyze_multivariate,
    perform_pca,
    perform_tsne,
)
from freamon.eda.report import generate_html_report
from freamon.eda.explainability_report import generate_interaction_report
from freamon.utils.datatype_detector import (
    DataTypeDetector,
    detect_column_types,
    optimize_dataframe_types,
)

__all__ = [
    "EDAAnalyzer",
    "analyze_numeric",
    "analyze_categorical",
    "analyze_datetime",
    "analyze_correlation",
    "analyze_feature_target",
    "analyze_timeseries",
    "analyze_autocorrelation",
    "analyze_seasonality",
    "analyze_multivariate",
    "perform_pca",
    "perform_tsne",
    "generate_html_report",
    "generate_interaction_report",
    "DataTypeDetector",
    "detect_column_types",
    "optimize_dataframe_types",
]