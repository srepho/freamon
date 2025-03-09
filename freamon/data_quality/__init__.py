"""
Module for assessing data quality and generating data quality reports.
"""

from freamon.data_quality.analyzer import DataQualityAnalyzer
from freamon.data_quality.missing_values import handle_missing_values
from freamon.data_quality.outliers import detect_outliers
from freamon.data_quality.duplicates import detect_duplicates, remove_duplicates
from freamon.data_quality.cardinality import analyze_cardinality

__all__ = [
    "DataQualityAnalyzer",
    "handle_missing_values",
    "detect_outliers",
    "detect_duplicates",
    "remove_duplicates",
    "analyze_cardinality",
]