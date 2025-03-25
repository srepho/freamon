"""
Tests for the HTML report generation functionality of the DataTypeDetector class.
"""
import os
import pandas as pd
import numpy as np
import pytest
from bs4 import BeautifulSoup
import re
import tempfile

from freamon.utils.datatype_detector import DataTypeDetector


class TestHTMLReportGeneration:
    """Test class for HTML report generation functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample dataframe with diverse data types for testing."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            # Numeric columns
            'id': range(1, n + 1),
            'continuous': np.random.normal(0, 1, n),
            'integer_values': np.random.randint(1, 100, n),
            'categorical_numeric': np.random.choice([1, 2, 3, 4, 5], n),
            
            # Datetime columns
            'datetime': pd.date_range(start='2023-01-01', periods=n),
            'date_string': pd.date_range(start='2023-01-01', periods=n).astype(str),
            
            # Categorical columns
            'category': np.random.choice(['A', 'B', 'C'], n),
            
            # String columns with patterns
            'email': [f"user{i}@example.com" for i in range(n)],
            'currency': [f"${np.random.randint(100, 10000)}.{np.random.randint(0, 100):02d}" for i in range(n)],
            
            # Special cases
            'null_heavy': [None if i % 2 == 0 else i for i in range(n)],
            'mixed_content': [i if i % 2 == 0 else f"Value {i}" for i in range(n)],
        })
    
    def test_save_html_report(self, sample_df):
        """Test saving HTML report to a file."""
        detector = DataTypeDetector(sample_df)
        detector.detect_all_types()
        
        # Create a temporary file for the report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Generate and save the report
            report_path = detector.save_html_report(temp_path, include_stats=True)
            
            # Check that the file exists and has content
            assert os.path.exists(report_path)
            assert os.path.getsize(report_path) > 0
            
            # Read the file and check basic structure
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Check basic HTML structure
            assert '<!DOCTYPE html>' in content
            assert '<html>' in content
            assert '</html>' in content
            assert '<table' in content
            
            # Parse with BeautifulSoup for more detailed checks
            soup = BeautifulSoup(content, 'html.parser')
            
            # Check for title
            assert soup.title is not None
            
            # Check that all columns from the dataframe are in the report
            for column in sample_df.columns:
                assert column in content
            
            # Check for column labels in table
            headers = [th.text.strip() for th in soup.find_all('th')]
            for expected_header in ['Column', 'Storage Type', 'Logical Type']:
                assert any(expected_header in header for header in headers)
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_column_report_html(self, sample_df):
        """Test getting HTML report content as a string."""
        detector = DataTypeDetector(sample_df)
        detector.detect_all_types()
        
        # Get the HTML content
        html_content = detector.get_column_report_html()
        
        # Check that we got a non-empty string
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        
        # Check basic HTML structure
        assert '<!DOCTYPE html>' in html_content
        assert '<html>' in html_content
        assert '</html>' in html_content
        assert '<table' in html_content
        
        # Parse with BeautifulSoup for more detailed checks
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check for title
        assert soup.title is not None
        
        # Check for column headers in the table
        headers = [th.text.strip() for th in soup.find_all('th')]
        for expected_header in ['Column', 'Storage Type', 'Logical Type']:
            assert any(expected_header in header for header in headers)
        
        # Check that all columns from the dataframe are in the HTML
        for column in sample_df.columns:
            assert column in html_content
    
    def test_html_report_custom_content(self, sample_df):
        """Test custom usage of HTML report content."""
        detector = DataTypeDetector(sample_df)
        detector.detect_all_types()
        
        # Get HTML content
        html_content = detector.get_column_report_html()
        
        # Create a custom report with a modified title
        custom_html = html_content.replace('<title>', '<title>Custom ')
        
        # Save custom content to a file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(custom_html.encode('utf-8'))
        
        try:
            # Check that file exists
            assert os.path.exists(temp_path)
            
            # Read back the customized file
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Verify customization was applied
            assert '<title>Custom ' in content
            
            # Verify original data is still there
            for column in sample_df.columns:
                assert column in content
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_html_report_with_special_characters(self):
        """Test HTML report generation with special characters in column names."""
        # Create a dataframe with special characters in column names
        df = pd.DataFrame({
            'normal_column': [1, 2, 3],
            'with_underscore': [4, 5, 6],
            'with$dollar': [7, 8, 9],
            'with.dot': [10, 11, 12],
            'with space': [13, 14, 15],
        })
        
        detector = DataTypeDetector(df)
        detector.detect_all_types()
        
        # Get HTML content
        html_content = detector.get_column_report_html()
        
        # Check that special character columns are present in the HTML
        for column in df.columns:
            assert column in html_content or column.replace('$', r'\$') in html_content
    
    def test_html_report_styling(self, sample_df):
        """Test that HTML report includes proper styling."""
        detector = DataTypeDetector(sample_df)
        detector.detect_all_types()
        
        # Get HTML content
        html_content = detector.get_column_report_html()
        
        # Check for CSS styling
        assert '<style>' in html_content
        assert '</style>' in html_content
        
        # Look for critical styling elements
        assert 'table {' in html_content
        assert 'body {' in html_content
        
        # Parse with BeautifulSoup to check for styled elements
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Verify table structure
        tables = soup.find_all('table')
        assert len(tables) > 0
        
        # Verify rows and cells
        rows = soup.find_all('tr')
        assert len(rows) > 0
        
        # Verify headings
        headings = soup.find_all('h1') + soup.find_all('h2')
        assert len(headings) > 0