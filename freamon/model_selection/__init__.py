"""
Module for model selection, train/test splitting, and cross-validation.
"""

from freamon.model_selection.splitter import (
    train_test_split,
    time_series_split,
    stratified_time_series_split,
)

from freamon.model_selection.cross_validation import (
    cross_validate,
    time_series_cross_validate,
)

__all__ = [
    "train_test_split",
    "time_series_split",
    "stratified_time_series_split",
    "cross_validate",
    "time_series_cross_validate",
]