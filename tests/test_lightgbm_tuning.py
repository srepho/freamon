"""
Tests for the LightGBM tuning module.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from freamon.modeling.tuning import LightGBMTuner
from freamon.modeling.model import Model


class TestLightGBMTuner:
    """Test cases for the LightGBMTuner class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data for testing."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        
        # Add categorical feature (as integer to avoid LightGBM dtype issues)
        X_df['cat_feature'] = np.random.choice([0, 1, 2], size=len(X))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data for testing."""
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y, name='target')
        
        # Add categorical feature (as integer to avoid LightGBM dtype issues)
        X_df['cat_feature'] = np.random.choice([0, 1, 2], size=len(X))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def test_init(self):
        """Test initialization of LightGBMTuner."""
        tuner = LightGBMTuner(
            problem_type='classification',
            objective='binary',
            metric='auc',
            cv=5,
            cv_type='stratified',
            random_state=42,
            n_trials=10,
            timeout=None,
            n_jobs=1,
            verbose=False
        )
        
        assert tuner.problem_type == 'classification'
        assert tuner.objective == 'binary'
        assert tuner.metric == 'auc'
        assert tuner.cv == 5
        assert tuner.cv_type == 'stratified'
        assert tuner.random_state == 42
        assert tuner.n_trials == 10
        assert tuner.n_jobs == 1
        assert tuner.verbose is False
    
    def test_init_defaults(self):
        """Test initialization with default values."""
        tuner = LightGBMTuner(problem_type='classification')
        
        assert tuner.problem_type == 'classification'
        assert tuner.objective == 'binary'
        assert tuner.metric == 'auc'
        assert tuner.cv == 5
        assert tuner.cv_type == 'kfold'
        
        tuner = LightGBMTuner(problem_type='regression')
        
        assert tuner.problem_type == 'regression'
        assert tuner.objective == 'regression'
        assert tuner.metric == 'rmse'
    
    def test_get_core_param_ranges(self):
        """Test getting core parameter ranges."""
        tuner = LightGBMTuner(problem_type='classification')
        param_ranges = tuner._get_core_param_ranges()
        
        assert 'num_leaves' in param_ranges
        assert 'max_depth' in param_ranges
        assert 'learning_rate' in param_ranges
        assert 'subsample' in param_ranges
        assert 'colsample_bytree' in param_ranges
        assert 'min_child_samples' in param_ranges
    
    def test_get_all_param_ranges(self):
        """Test getting all parameter ranges."""
        tuner = LightGBMTuner(problem_type='classification')
        param_ranges = tuner._get_all_param_ranges()
        
        # Core parameters
        assert 'num_leaves' in param_ranges
        assert 'max_depth' in param_ranges
        
        # Additional parameters
        assert 'reg_alpha' in param_ranges
        assert 'reg_lambda' in param_ranges
        assert 'min_split_gain' in param_ranges
        assert 'min_data_in_leaf' in param_ranges
    
    def test_create_cv_splitter(self):
        """Test creation of CV splitters."""
        from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
        
        # KFold splitter
        tuner = LightGBMTuner(problem_type='classification', cv_type='kfold')
        splitter = tuner._create_cv_splitter()
        assert isinstance(splitter, KFold)
        assert splitter.n_splits == 5
        
        # StratifiedKFold splitter
        tuner = LightGBMTuner(problem_type='classification', cv_type='stratified')
        splitter = tuner._create_cv_splitter()
        assert isinstance(splitter, StratifiedKFold)
        assert splitter.n_splits == 5
        
        # TimeSeriesSplit splitter
        tuner = LightGBMTuner(problem_type='regression', cv_type='timeseries')
        splitter = tuner._create_cv_splitter()
        assert isinstance(splitter, TimeSeriesSplit)
        assert splitter.n_splits == 5
        
        # Invalid CV type
        tuner = LightGBMTuner(problem_type='classification', cv_type='invalid')
        with pytest.raises(ValueError):
            tuner._create_cv_splitter()
    
    def test_sample_params(self):
        """Test parameter sampling from ranges."""
        import optuna
        
        tuner = LightGBMTuner(problem_type='classification')
        param_ranges = {
            'num_leaves': {
                'type': 'int',
                'low': 20,
                'high': 150
            },
            'learning_rate': {
                'type': 'continuous',
                'low': 0.01,
                'high': 0.3,
                'log': True
            },
            'boosting_type': {
                'type': 'categorical',
                'choices': ['gbdt', 'dart', 'goss']
            }
        }
        
        # Create a trial
        study = optuna.create_study()
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        
        # Sample parameters
        params = tuner._sample_params(trial, param_ranges)
        
        # Check that all parameters were sampled
        assert 'num_leaves' in params
        assert 'learning_rate' in params
        assert 'boosting_type' in params
        
        # Check types and ranges
        assert isinstance(params['num_leaves'], int)
        assert 20 <= params['num_leaves'] <= 150
        
        assert isinstance(params['learning_rate'], float)
        assert 0.01 <= params['learning_rate'] <= 0.3
        
        assert params['boosting_type'] in ['gbdt', 'dart', 'goss']
    
    @pytest.mark.parametrize("progressive", [True, False])
    def test_tune_classification(self, classification_data, progressive):
        """Test tuning for classification problems."""
        X_train, X_test, y_train, y_test = classification_data
        
        tuner = LightGBMTuner(
            problem_type='classification',
            objective='binary',
            metric='auc',
            cv=3,  # Reduced for testing
            cv_type='stratified',
            random_state=42,
            n_trials=5,  # Reduced for testing
            n_jobs=1,
            verbose=False
        )
        
        # Fixed parameters for faster testing
        fixed_params = {
            'n_estimators': 50,
            'verbose': -1
        }
        
        # Tune hyperparameters
        best_params = tuner.tune(
            X_train, y_train,
            categorical_features=['cat_feature'],
            fixed_params=fixed_params,
            progressive_tuning=progressive
        )
        
        # Check that best parameters were found
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        
        # Check that study was created
        assert tuner.study is not None
        
        # Create model with best parameters
        model = tuner.create_model()
        assert isinstance(model, Model)
        assert model.model_type == 'lightgbm'
        
        # Train and test the model
        model.fit(X_train, y_train, categorical_feature=['cat_feature'])
        y_pred_proba = model.predict_proba(X_test)
        
        # Check predictions shape
        assert y_pred_proba.shape[0] == len(X_test)
        assert y_pred_proba.shape[1] == 2  # Binary classification
    
    @pytest.mark.parametrize("progressive", [True, False])
    def test_tune_regression(self, regression_data, progressive):
        """Test tuning for regression problems."""
        X_train, X_test, y_train, y_test = regression_data
        
        tuner = LightGBMTuner(
            problem_type='regression',
            objective='regression',
            metric='rmse',
            cv=3,  # Reduced for testing
            cv_type='kfold',
            random_state=42,
            n_trials=5,  # Reduced for testing
            n_jobs=1,
            verbose=False
        )
        
        # Fixed parameters for faster testing
        fixed_params = {
            'n_estimators': 50,
            'verbose': -1
        }
        
        # Tune hyperparameters
        best_params = tuner.tune(
            X_train, y_train,
            categorical_features=['cat_feature'],
            fixed_params=fixed_params,
            progressive_tuning=progressive
        )
        
        # Check that best parameters were found
        assert best_params is not None
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        
        # Check that study was created
        assert tuner.study is not None
        
        # Create model with best parameters
        model = tuner.create_model()
        assert isinstance(model, Model)
        assert model.model_type == 'lightgbm'
        
        # Train and test the model
        model.fit(X_train, y_train, categorical_feature=['cat_feature'])
        y_pred = model.predict(X_test)
        
        # Check predictions shape
        assert y_pred.shape[0] == len(X_test)
    
    def test_create_model_without_tuning(self):
        """Test creating a model without tuning first."""
        tuner = LightGBMTuner(problem_type='classification')
        
        with pytest.raises(ValueError):
            tuner.create_model()
    
    def test_print_study_summary_without_tuning(self):
        """Test printing study summary without tuning first."""
        tuner = LightGBMTuner(problem_type='classification')
        
        with pytest.raises(ValueError):
            tuner.print_study_summary()
    
    def test_plot_optimization_history_without_tuning(self):
        """Test plotting optimization history without tuning first."""
        tuner = LightGBMTuner(problem_type='classification')
        
        with pytest.raises(ValueError):
            tuner.plot_optimization_history()
    
    def test_plot_param_importances_without_tuning(self):
        """Test plotting parameter importances without tuning first."""
        tuner = LightGBMTuner(problem_type='classification')
        
        with pytest.raises(ValueError):
            tuner.plot_param_importances()