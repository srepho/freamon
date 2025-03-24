"""
Tests for Markdown report generation in the EDA module.
"""
import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from freamon.eda import EDAAnalyzer


class TestMarkdownReportGeneration:
    """Test class for Markdown report generation."""
    
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
    
    def test_markdown_report_generation(self, sample_df, tmp_path):
        """Test generating a Markdown report."""
        analyzer = EDAAnalyzer(sample_df, target_column='binary', date_column='date')
        analyzer.run_full_analysis()
        
        # Generate Markdown report
        md_report_path = os.path.join(tmp_path, "eda_report.md")
        analyzer.generate_report(output_path=md_report_path, format="markdown")
        
        # Check that the report file was created
        assert os.path.exists(md_report_path)
        
        # Check that the file has some content
        with open(md_report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0
            assert "# Exploratory Data Analysis Report" in content
            assert "## Dataset Overview" in content
            assert analyzer.target_column in content
            assert analyzer.date_column in content
    
    def test_markdown_with_html_conversion(self, sample_df, tmp_path):
        """Test generating a Markdown report with HTML conversion."""
        analyzer = EDAAnalyzer(sample_df, target_column='binary', date_column='date')
        analyzer.run_full_analysis()
        
        # Generate Markdown report with HTML conversion
        md_report_path = os.path.join(tmp_path, "eda_report.md")
        analyzer.generate_report(
            output_path=md_report_path, 
            format="markdown", 
            convert_to_html=True
        )
        
        # Check that both files were created
        assert os.path.exists(md_report_path)
        assert os.path.exists(md_report_path + ".html")
        
        # Check Markdown content
        with open(md_report_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
            assert len(md_content) > 0
            assert "# Exploratory Data Analysis Report" in md_content
        
        # Check HTML content
        with open(md_report_path + ".html", 'r', encoding='utf-8') as f:
            html_content = f.read()
            assert len(html_content) > 0
            assert "<html>" in html_content
            assert "<title>Exploratory Data Analysis Report</title>" in html_content