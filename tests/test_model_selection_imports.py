"""
Tests to verify that imports from the model_selection module work correctly.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_import_crossvalidation_functions():
    """Test that cross-validation functions can be imported."""
    from freamon.model_selection.cross_validation import (
        create_time_series_cv,
        create_stratified_cv,
        create_kfold_cv,
        cross_validate,
        time_series_cross_validate,
        walk_forward_validation
    )
    
    # Test create_time_series_cv
    dates = pd.date_range(start='2023-01-01', periods=100)
    cv = create_time_series_cv(dates, n_splits=3)
    assert hasattr(cv, 'split')
    
    # Test create_stratified_cv
    y = np.random.choice([0, 1], size=100)
    cv = create_stratified_cv(y, n_splits=3)
    assert hasattr(cv, 'split')
    
    # Test create_kfold_cv
    cv = create_kfold_cv(n_splits=3)
    assert hasattr(cv, 'split')


def test_import_cv_trainer():
    """Test that CrossValidationTrainer can be imported and initialized."""
    from freamon.model_selection.cv_trainer import CrossValidationTrainer
    
    trainer = CrossValidationTrainer(
        model_type="sklearn",
        problem_type="classification",
        cv_strategy="kfold",
        n_splits=3
    )
    
    assert trainer.model_type == "sklearn"
    assert trainer.problem_type == "classification"
    assert trainer.cv_strategy == "kfold"
    assert trainer.n_splits == 3


def test_import_from_modeling_autoflow():
    """Test that imports in the modeling.autoflow module work."""
    from freamon.modeling.autoflow import AutoModelFlow
    
    # Create AutoModelFlow instance
    autoflow = AutoModelFlow(
        model_type="lightgbm",
        problem_type="classification",
        text_processing=True,
        time_series_features=True
    )
    
    assert autoflow.model_type == "lightgbm"
    assert autoflow.problem_type == "classification"
    assert autoflow.text_processing is True
    assert autoflow.time_series_features is True