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
]