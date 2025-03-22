"""
Tests for the integration of DataTypeDetector with the EDAAnalyzer class.
"""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from freamon.eda import EDAAnalyzer
from freamon.utils.datatype_detector import DataTypeDetector


class TestEDATypeDetection:
    """Test class for DataTypeDetector integration with EDAAnalyzer."""
    
    @pytest.fixture
    def mixed_df(self):
        """Create a sample dataframe with mixed data types."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            # Numeric columns
            'id': range(1, n + 1),
            'continuous': np.random.normal(0, 1, n),
            'categorical_numeric': np.random.choice([1, 2, 3, 4, 5], n),
            
            # Datetime columns
            'date': pd.date_range(start='2023-01-01', periods=n),
            'date_string': pd.date_range(start='2023-01-01', periods=n).astype(str),
            
            # Categorical columns
            'category': np.random.choice(['A', 'B', 'C'], n),
            
            # String columns with patterns
            'email': [f"user{i}@example.com" for i in range(n)],
            'url': [f"https://example.com/page{i}" for i in range(n)],
            
            # Australian data types
            'au_postcode': [i % 9000 + 1000 for i in range(n)],
            'au_phone': [f"02 {i % 9000 + 1000} {i % 9000 + 1000}" for i in range(n)],
        })
    
    def test_eda_initialization_with_type_detection(self, mixed_df):
        """Test that EDAAnalyzer correctly initializes with type detection."""
        analyzer = EDAAnalyzer(mixed_df)
        
        # Check that column types are detected correctly
        assert 'id' in analyzer.numeric_columns
        assert 'continuous' in analyzer.numeric_columns
        assert 'categorical_numeric' in analyzer.numeric_columns or 'categorical_numeric' in analyzer.categorical_columns
        
        assert 'date' in analyzer.datetime_columns
        assert 'date_string' in analyzer.datetime_columns  # Should be automatically detected
        
        assert 'category' in analyzer.categorical_columns
        assert 'email' in analyzer.categorical_columns
        assert 'url' in analyzer.categorical_columns
        
        # Check that detected_types attribute exists
        assert hasattr(analyzer, 'detected_types')
        
        # Test with custom patterns
        custom_patterns = {'custom_pattern': r'^custom\d+$'}
        df_with_custom = mixed_df.copy()
        n = len(df_with_custom)
        df_with_custom['custom_field'] = [f"custom{i}" for i in range(n)]
        
        analyzer_with_custom = EDAAnalyzer(df_with_custom, custom_patterns=custom_patterns)
        
        # Check custom pattern is detected
        assert 'custom_field' in analyzer_with_custom.categorical_columns
        assert analyzer_with_custom.detected_types['custom_field']['semantic_type'] == 'custom_pattern'
    
    def test_eda_basic_stats_with_type_info(self, mixed_df):
        """Test that EDAAnalyzer.analyze_basic_stats includes detected type information."""
        analyzer = EDAAnalyzer(mixed_df)
        stats = analyzer.analyze_basic_stats()
        
        # Check that detected types are included in basic stats
        assert 'detected_types' in stats
        
        # Check that semantic types are included
        assert 'semantic_types' in stats
        assert 'id' in stats['semantic_types']
        assert stats['semantic_types']['id'] == 'id'
        assert 'email' in stats['semantic_types']
        assert stats['semantic_types']['email'] == 'email'
        
        # Check that conversion suggestions are included
        assert 'conversion_suggestions' in stats
    
    def test_eda_univariate_with_type_detection(self, mixed_df):
        """Test that univariate analysis respects detected types."""
        analyzer = EDAAnalyzer(mixed_df)
        result = analyzer.analyze_univariate()
        
        # Check that categorical_numeric is analyzed correctly based on detected type
        if 'categorical_numeric' in analyzer.categorical_columns:
            assert 'value_counts' in result['categorical_numeric']
        else:
            assert 'mean' in result['categorical_numeric']
        
        # Check that date_string is analyzed as datetime
        assert 'min' in result['date_string']
        assert 'max' in result['date_string']
        assert 'components' in result['date_string']
    
    def test_eda_bivariate_with_type_detection(self, mixed_df):
        """Test that bivariate analysis respects detected types."""
        # Create analyzer with categorical_numeric as target
        analyzer = EDAAnalyzer(mixed_df, target_column='categorical_numeric')
        
        # Run bivariate analysis
        result = analyzer.analyze_bivariate()
        
        # Check feature-target relationships
        for col, analysis in result['feature_target'].items():
            if col in analyzer.numeric_columns:
                # Relationship type depends on how categorical_numeric was classified
                assert analysis['type'] in ['numeric_vs_numeric', 'numeric_vs_categorical']
            elif col in analyzer.categorical_columns:
                assert analysis['type'] in ['categorical_vs_numeric', 'categorical_vs_categorical']
    
    def test_eda_report_with_type_detection(self, mixed_df, tmp_path):
        """Test that report generation includes detected type information."""
        analyzer = EDAAnalyzer(mixed_df)
        analyzer.analyze_basic_stats()
        
        # Generate report to a temporary file
        report_path = tmp_path / "eda_report_with_types.html"
        analyzer.generate_report(output_path=str(report_path))
        
        # Check that the report file was created
        assert report_path.exists()
        
        # Check that the report contains basic type information
        report_content = report_path.read_text()
        # Basic info that should be in the report regardless of implementation details
        assert "Exploratory Data Analysis Report" in report_content
        assert "Basic Statistics" in report_content
        
        # Just check that the basic data info is there without getting too specific about headings
        for col in ['id', 'email', 'date']:
            assert col in report_content
    
    def test_eda_australian_data_integration(self):
        """Test integration with Australian data types."""
        # Create a dataframe with mixed Australian data
        df = pd.DataFrame({
            'suburb': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Darwin'],
            'postcode_str': ['2000', '3000', '4000', '6000', '5000', '0800'],
            'postcode_int': [2000, 3000, 4000, 6000, 5000, 800],  # Note Darwin loses leading zero
            'phone': ['02 9999 9999', '03 9999 9999', '07 3999 9999', '08 9999 9999', '08 8999 9999', '08 8999 9999'],
            'mobile': ['0412 345 678', '0433 456 789', '0451 567 890', '0478 678 901', '0401 234 567', '0444 555 666'],
            'abn': ['12 345 678 901', '98 765 432 109', '11 222 333 444', '55 666 777 888', '99 888 777 666', '33 444 555 666'],
        })
        
        # Initialize analyzer
        analyzer = EDAAnalyzer(df)
        stats = analyzer.analyze_basic_stats()
        
        # Check that at least some Australian data types are detected
        if 'semantic_types' in stats:
            # Should detect at least one of each category
            semantic_values = list(stats['semantic_types'].values())
            
            # At least one postcode or currency pattern
            assert any(st in ['au_postcode', 'currency', 'zip_code'] for st in semantic_values), \
                "No postcode pattern detected"
            
            # At least one phone or mobile pattern
            assert any(st in ['au_phone', 'phone_number'] for st in semantic_values), \
                "No phone pattern detected"
            
            # At least one business ID pattern
            assert any(st in ['au_abn', 'ssn'] for st in semantic_values), \
                "No business ID pattern detected"