"""
Tests for the EDA module.
"""
import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from freamon.eda import (
    EDAAnalyzer,
    analyze_numeric,
    analyze_categorical,
    analyze_datetime,
    analyze_correlation,
    analyze_feature_target,
    analyze_timeseries,
)


class TestEDAAnalyzer:
    """Test class for EDAAnalyzer."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe for testing."""
        # Create sample data with different column types
        np.random.seed(42)
        n = 100
        
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=n)
        
        # Create numeric columns
        numeric1 = np.random.normal(0, 1, n)
        numeric2 = np.random.normal(5, 2, n)
        numeric3 = np.random.randint(0, 100, n)
        
        # Create categorical columns
        cat1 = np.random.choice(['A', 'B', 'C'], n)
        cat2 = np.random.choice(['X', 'Y'], n)
        
        # Create a binary column
        binary = np.random.choice([0, 1], n)
        
        # Introduce some correlation
        correlated = numeric1 * 2 + np.random.normal(0, 0.5, n)
        
        # Introduce some missing values
        numeric1[np.random.choice(n, 5)] = np.nan
        cat1[np.random.choice(n, 5)] = None
        
        return pd.DataFrame({
            'date': dates,
            'numeric1': numeric1,
            'numeric2': numeric2,
            'numeric3': numeric3,
            'categorical1': cat1,
            'categorical2': cat2,
            'binary': binary,
            'correlated': correlated,
        })
    
    def test_initialization(self, sample_df):
        """Test that the EDAAnalyzer initializes correctly."""
        analyzer = EDAAnalyzer(sample_df)
        
        # Check that columns are correctly categorized
        assert set(analyzer.numeric_columns) == {'numeric1', 'numeric2', 'numeric3', 'binary', 'correlated'}
        assert set(analyzer.categorical_columns) == {'categorical1', 'categorical2'}
        assert set(analyzer.datetime_columns) == {'date'}
        
        # Check basic properties
        assert analyzer.n_rows == len(sample_df)
        assert analyzer.n_cols == len(sample_df.columns)
        
        # Test with target column
        analyzer_with_target = EDAAnalyzer(sample_df, target_column='binary')
        assert analyzer_with_target.target_column == 'binary'
        
        # Test with date column
        analyzer_with_date = EDAAnalyzer(sample_df, date_column='date')
        assert analyzer_with_date.date_column == 'date'
        
        # Test with invalid target column
        with pytest.raises(ValueError):
            EDAAnalyzer(sample_df, target_column='nonexistent')
        
        # Test with invalid date column
        with pytest.raises(ValueError):
            EDAAnalyzer(sample_df, date_column='nonexistent')
    
    def test_analyze_basic_stats(self, sample_df):
        """Test basic statistics calculation."""
        analyzer = EDAAnalyzer(sample_df)
        stats = analyzer.analyze_basic_stats()
        
        # Check that key statistics are present
        assert 'n_rows' in stats
        assert 'n_cols' in stats
        assert 'n_numeric' in stats
        assert 'n_categorical' in stats
        assert 'n_datetime' in stats
        assert 'memory_usage_mb' in stats
        assert 'has_missing' in stats
        assert 'missing_count' in stats
        assert 'missing_percent' in stats
        
        # Check that column lists are present
        assert 'numeric_columns' in stats
        assert 'categorical_columns' in stats
        assert 'datetime_columns' in stats
        
        # Check that missing values by column is correct
        assert 'missing_by_column' in stats
        assert 'numeric1' in stats['missing_by_column']
        assert 'categorical1' in stats['missing_by_column']
        
        # Check that results are stored
        assert 'basic_stats' in analyzer.analysis_results
    
    def test_analyze_univariate(self, sample_df):
        """Test univariate analysis."""
        analyzer = EDAAnalyzer(sample_df)
        result = analyzer.analyze_univariate()
        
        # Check that each column is analyzed
        for col in sample_df.columns:
            assert col in result
            
            # Check that key metrics are present
            assert 'count' in result[col]
            assert 'missing' in result[col]
        
        # Check numeric column specifics
        assert 'mean' in result['numeric1']
        assert 'median' in result['numeric1']
        assert 'std' in result['numeric1']
        
        # Check categorical column specifics
        assert 'unique' in result['categorical1']
        assert 'value_counts' in result['categorical1']
        
        # Check datetime column specifics
        assert 'min' in result['date']
        assert 'max' in result['date']
        
        # Check that plots are included
        assert 'plot' in result['numeric1']
        assert 'plot' in result['categorical1']
        assert 'plot' in result['date']
        
        # Check that results are stored
        assert 'univariate' in analyzer.analysis_results
    
    def test_analyze_bivariate(self, sample_df):
        """Test bivariate analysis."""
        # Create analyzer with target column
        analyzer = EDAAnalyzer(sample_df, target_column='binary')
        result = analyzer.analyze_bivariate()
        
        # Check that correlation analysis is present
        assert 'correlation' in result
        assert 'correlation_matrix' in result['correlation']
        assert 'top_correlations' in result['correlation']
        assert 'plot' in result['correlation']
        
        # Check that feature-target analysis is present
        assert 'feature_target' in result
        
        # Check numeric vs binary relationship
        assert 'numeric1' in result['feature_target']
        assert 'type' in result['feature_target']['numeric1']
        assert result['feature_target']['numeric1']['type'] == 'numeric_vs_categorical'
        
        # Check that results are stored
        assert 'bivariate' in analyzer.analysis_results
    
    def test_analyze_time_series(self, sample_df):
        """Test time series analysis."""
        # Create analyzer with date column
        analyzer = EDAAnalyzer(sample_df, date_column='date')
        result = analyzer.analyze_time_series()
        
        # Check that each numeric column is analyzed
        for col in analyzer.numeric_columns:
            assert col in result
            
            # Check that key metrics are present
            assert 'count' in result[col]
            assert 'mean' in result[col]
            assert 'std' in result[col]
            assert 'start_date' in result[col]
            assert 'end_date' in result[col]
            assert 'duration_days' in result[col]
            
            # Check that plot is included
            assert 'plot' in result[col]
        
        # Check that results are stored
        assert 'time_series' in analyzer.analysis_results
    
    def test_run_full_analysis(self, sample_df):
        """Test running a full analysis."""
        analyzer = EDAAnalyzer(sample_df, target_column='binary', date_column='date')
        results = analyzer.run_full_analysis()
        
        # Check that all analysis types are present
        assert 'basic_stats' in results
        assert 'univariate' in results
        assert 'bivariate' in results
        assert 'time_series' in results
    
    def test_generate_report(self, sample_df, tmp_path):
        """Test report generation."""
        analyzer = EDAAnalyzer(sample_df, target_column='binary', date_column='date')
        analyzer.run_full_analysis()
        
        # Generate report to a temporary file
        report_path = os.path.join(tmp_path, "eda_report.html")
        analyzer.generate_report(output_path=report_path)
        
        # Check that the report file was created
        assert os.path.exists(report_path)
        
        # Check that the file has some content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0
            assert "<html>" in content
            assert analyzer.target_column in content
            assert analyzer.date_column in content


class TestUnivariateFunctions:
    """Test class for individual univariate analysis functions."""
    
    @pytest.fixture
    def numeric_series(self):
        """Create a numeric series for testing."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1, 100), name='numeric')
    
    @pytest.fixture
    def categorical_series(self):
        """Create a categorical series for testing."""
        np.random.seed(42)
        return pd.Series(np.random.choice(['A', 'B', 'C'], 100), name='categorical')
    
    @pytest.fixture
    def datetime_series(self):
        """Create a datetime series for testing."""
        return pd.Series(pd.date_range(start='2023-01-01', periods=100), name='datetime')
    
    def test_analyze_numeric(self, numeric_series):
        """Test numeric column analysis."""
        df = pd.DataFrame({numeric_series.name: numeric_series})
        result = analyze_numeric(df, numeric_series.name)
        
        # Check that key statistics are present
        assert 'count' in result
        assert 'missing' in result
        assert 'mean' in result
        assert 'median' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result
        assert 'range' in result
        
        # Check that percentiles are included
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            assert f'percentile_{p}' in result
        
        # Check that plot is included with include_plot=True
        assert 'plot' in result
        
        # Check that plot is not included with include_plot=False
        result_no_plot = analyze_numeric(df, numeric_series.name, include_plot=False)
        assert 'plot' not in result_no_plot
    
    def test_analyze_categorical(self, categorical_series):
        """Test categorical column analysis."""
        df = pd.DataFrame({categorical_series.name: categorical_series})
        result = analyze_categorical(df, categorical_series.name)
        
        # Check that key statistics are present
        assert 'count' in result
        assert 'missing' in result
        assert 'unique' in result
        assert 'is_boolean' in result
        assert 'value_counts' in result
        
        # Check that value counts are correct
        for cat in categorical_series.unique():
            assert str(cat) in result['value_counts']
            assert 'count' in result['value_counts'][str(cat)]
            assert 'percentage' in result['value_counts'][str(cat)]
        
        # Check that plot is included with include_plot=True
        assert 'plot' in result
        
        # Check that plot is not included with include_plot=False
        result_no_plot = analyze_categorical(df, categorical_series.name, include_plot=False)
        assert 'plot' not in result_no_plot
    
    def test_analyze_datetime(self, datetime_series):
        """Test datetime column analysis."""
        df = pd.DataFrame({datetime_series.name: datetime_series})
        result = analyze_datetime(df, datetime_series.name)
        
        # Check that key statistics are present
        assert 'count' in result
        assert 'missing' in result
        assert 'min' in result
        assert 'max' in result
        assert 'range_days' in result
        assert 'components' in result
        
        # Check that date components are analyzed
        for component in ['year', 'month', 'day', 'weekday']:
            assert component in result['components']
        
        # Check distributions
        if 'year' in result['components']:
            assert 'year_counts' in result
        if 'month' in result['components']:
            assert 'month_counts' in result
        if 'weekday' in result['components']:
            assert 'weekday_counts' in result
        
        # Check that plot is included with include_plot=True
        assert 'plot' in result
        
        # Check that plot is not included with include_plot=False
        result_no_plot = analyze_datetime(df, datetime_series.name, include_plot=False)
        assert 'plot' not in result_no_plot


class TestBivariateFunctions:
    """Test class for individual bivariate analysis functions."""
    
    @pytest.fixture
    def correlation_df(self):
        """Create a dataframe with correlated variables for testing."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.8 + np.random.normal(0, 0.5, n)
        x3 = np.random.normal(0, 1, n)
        return pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
        })
    
    @pytest.fixture
    def feature_target_df(self):
        """Create a dataframe with features and targets for testing."""
        np.random.seed(42)
        n = 100
        
        # Create features
        num_feature = np.random.normal(0, 1, n)
        cat_feature = np.random.choice(['A', 'B', 'C'], n)
        
        # Create targets (numeric and categorical)
        num_target = num_feature * 2 + np.random.normal(0, 0.5, n)
        cat_target = np.where(num_feature > 0, 'Positive', 'Negative')
        
        return pd.DataFrame({
            'num_feature': num_feature,
            'cat_feature': cat_feature,
            'num_target': num_target,
            'cat_target': cat_target,
        })
    
    def test_analyze_correlation(self, correlation_df):
        """Test correlation analysis."""
        result = analyze_correlation(correlation_df)
        
        # Check that key components are present
        assert 'method' in result
        assert 'correlation_matrix' in result
        assert 'top_correlations' in result
        
        # Check that correlation matrix has all columns
        for col in correlation_df.columns:
            assert col in result['correlation_matrix']
        
        # Check that x1-x2 correlation is strong and positive
        for item in result['top_correlations']:
            if (item['column1'] == 'x1' and item['column2'] == 'x2') or \
               (item['column1'] == 'x2' and item['column2'] == 'x1'):
                assert item['correlation'] > 0.7
                break
        else:
            pytest.fail("Expected strong correlation between x1 and x2 not found")
        
        # Check that plot is included with include_plot=True
        assert 'plot' in result
        
        # Check that plot is not included with include_plot=False
        result_no_plot = analyze_correlation(correlation_df, include_plot=False)
        assert 'plot' not in result_no_plot
    
    def test_analyze_feature_target(self, feature_target_df):
        """Test feature-target analysis for different combinations."""
        # Test numeric feature vs numeric target
        num_num_result = analyze_feature_target(
            feature_target_df, 'num_feature', 'num_target'
        )
        assert num_num_result['type'] == 'numeric_vs_numeric'
        assert 'pearson_correlation' in num_num_result
        assert 'spearman_correlation' in num_num_result
        assert num_num_result['pearson_correlation'] > 0.9  # Should be strongly correlated
        
        # Test numeric feature vs categorical target
        num_cat_result = analyze_feature_target(
            feature_target_df, 'num_feature', 'cat_target'
        )
        assert num_cat_result['type'] == 'numeric_vs_categorical'
        assert 'grouped_stats' in num_cat_result
        assert 'anova_p_value' in num_cat_result
        assert num_cat_result['anova_p_value'] < 0.05  # Should be significant
        
        # Test categorical feature vs numeric target
        cat_num_result = analyze_feature_target(
            feature_target_df, 'cat_feature', 'num_target'
        )
        assert cat_num_result['type'] == 'categorical_vs_numeric'
        assert 'grouped_stats' in cat_num_result
        
        # Test categorical feature vs categorical target
        cat_cat_result = analyze_feature_target(
            feature_target_df, 'cat_feature', 'cat_target'
        )
        assert cat_cat_result['type'] == 'categorical_vs_categorical'
        assert 'contingency_table' in cat_cat_result
        assert 'chi2_p_value' in cat_cat_result
        
        # Check that plots are included with include_plot=True
        assert 'plot' in num_num_result
        assert 'plot' in num_cat_result
        assert 'plot' in cat_num_result
        assert 'plot' in cat_cat_result
        
        # Check that plots are not included with include_plot=False
        result_no_plot = analyze_feature_target(
            feature_target_df, 'num_feature', 'num_target', include_plot=False
        )
        assert 'plot' not in result_no_plot


class TestTimeSeriesFunctions:
    """Test class for individual time series analysis functions."""
    
    @pytest.fixture
    def timeseries_df(self):
        """Create a dataframe with time series data for testing."""
        # Create a date range with daily frequency
        dates = pd.date_range(start='2023-01-01', periods=100)
        
        # Create a time series with trend and seasonality
        t = np.arange(len(dates))
        trend = 0.1 * t
        seasonality = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 1, len(dates))
        
        values = trend + seasonality + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
        })
    
    def test_analyze_timeseries(self, timeseries_df):
        """Test time series analysis."""
        result = analyze_timeseries(timeseries_df, 'date', 'value')
        
        # Check that key statistics are present
        assert 'count' in result
        assert 'mean' in result
        assert 'std' in result
        assert 'min' in result
        assert 'max' in result
        assert 'start_date' in result
        assert 'end_date' in result
        assert 'duration_days' in result
        
        # Check trend analysis
        assert 'trend' in result
        assert result['trend'] == 'increasing'  # Should detect increasing trend
        assert 'absolute_change' in result
        assert 'percent_change' in result
        
        # Check that plot is included with include_plot=True
        assert 'plot' in result
        
        # Check that plot is not included with include_plot=False
        result_no_plot = analyze_timeseries(
            timeseries_df, 'date', 'value', include_plot=False
        )
        assert 'plot' not in result_no_plot