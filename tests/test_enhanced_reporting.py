"""
Unit tests for enhanced deduplication reporting functionality.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import the enhanced reporting module
try:
    from freamon.deduplication.enhanced_reporting import (
        EnhancedDeduplicationReport,
        generate_enhanced_report,
    )
    from freamon.deduplication.evaluation import (
        calculate_deduplication_metrics,
        flag_and_evaluate,
    )
    has_enhanced_reporting = True
except ImportError:
    has_enhanced_reporting = False

# Skip all tests if the enhanced reporting module is not available
pytestmark = pytest.mark.skipif(
    not has_enhanced_reporting,
    reason="Enhanced reporting module not available"
)


def create_test_dataset(size=100, duplicate_rate=0.2):
    """Create a test dataset with known duplicates."""
    # Create a dataset with unique records
    df = pd.DataFrame({
        'id': range(1, size + 1),
        'name': [f"Person {i}" for i in range(1, size + 1)],
        'email': [f"person{i}@example.com" for i in range(1, size + 1)],
        'age': np.random.randint(18, 80, size),
        'is_duplicate': [False] * size
    })
    
    # Create duplicates
    duplicate_size = int(size * duplicate_rate)
    duplicates = []
    for i in range(duplicate_size):
        idx = np.random.randint(0, size)
        duplicate = df.iloc[idx].copy()
        duplicate['is_duplicate'] = True
        duplicates.append(duplicate)
    
    # Add duplicates to the dataset
    duplicate_df = pd.DataFrame(duplicates)
    full_df = pd.concat([df, duplicate_df], ignore_index=True)
    
    # Add predicted duplicate column (with some errors for evaluation)
    full_df['predicted_duplicate'] = full_df['is_duplicate'].copy()
    
    # Add some false positives and false negatives (about 10%)
    error_count = int(len(full_df) * 0.1)
    error_indices = np.random.choice(full_df.index, error_count, replace=False)
    full_df.loc[error_indices, 'predicted_duplicate'] = ~full_df.loc[error_indices, 'predicted_duplicate']
    
    return full_df


def test_enhanced_deduplication_report_creation():
    """Test that the EnhancedDeduplicationReport class can be instantiated."""
    # Create test dataset
    df = create_test_dataset()
    
    # Create results dictionary
    results = {
        'dataframe': df,
        'flag_column': 'predicted_duplicate',
        'truth_column': 'is_duplicate',
        'metrics': calculate_deduplication_metrics(
            df=df,
            prediction_column='predicted_duplicate',
            truth_column='is_duplicate'
        )
    }
    
    # Create reporter instance
    with tempfile.TemporaryDirectory() as tmpdirname:
        reporter = EnhancedDeduplicationReport(
            results=results,
            title="Test Deduplication Report",
            output_dir=tmpdirname,
            create_dir=True
        )
        
        assert isinstance(reporter, EnhancedDeduplicationReport)
        assert reporter.title == "Test Deduplication Report"
        assert reporter.output_dir == tmpdirname


def test_enhanced_deduplication_report_html_generation():
    """Test that the HTML report can be generated."""
    # Create test dataset
    df = create_test_dataset()
    
    # Create results dictionary
    results = {
        'dataframe': df,
        'flag_column': 'predicted_duplicate',
        'truth_column': 'is_duplicate',
        'metrics': calculate_deduplication_metrics(
            df=df,
            prediction_column='predicted_duplicate',
            truth_column='is_duplicate'
        )
    }
    
    # Generate HTML report
    with tempfile.TemporaryDirectory() as tmpdirname:
        reporter = EnhancedDeduplicationReport(
            results=results,
            title="Test Deduplication Report",
            output_dir=tmpdirname
        )
        
        output_path = os.path.join(tmpdirname, "test_report.html")
        html_path = reporter.generate_html_report(
            output_path=output_path,
            include_pairs=True,
            max_pairs=10
        )
        
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0
        
        # Verify the file contains expected content
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test Deduplication Report" in content
            assert "Deduplication Metrics" in content


def test_enhanced_deduplication_report_markdown_generation():
    """Test that the Markdown report can be generated."""
    # Create test dataset
    df = create_test_dataset()
    
    # Create results dictionary
    results = {
        'dataframe': df,
        'flag_column': 'predicted_duplicate',
        'truth_column': 'is_duplicate',
        'metrics': calculate_deduplication_metrics(
            df=df,
            prediction_column='predicted_duplicate',
            truth_column='is_duplicate'
        )
    }
    
    # Generate Markdown report
    with tempfile.TemporaryDirectory() as tmpdirname:
        reporter = EnhancedDeduplicationReport(
            results=results,
            title="Test Deduplication Report",
            output_dir=tmpdirname
        )
        
        output_path = os.path.join(tmpdirname, "test_report.md")
        md_path = reporter.generate_markdown_report(
            output_path=output_path,
            include_pairs=True,
            max_pairs=10
        )
        
        assert os.path.exists(md_path)
        assert os.path.getsize(md_path) > 0
        
        # Verify the file contains expected content
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Test Deduplication Report" in content
            assert "## Deduplication Metrics" in content


def test_generate_enhanced_report_function():
    """Test the high-level generate_enhanced_report function."""
    # Create test dataset
    df = create_test_dataset()
    
    # Run flag_and_evaluate to get complete results
    results = flag_and_evaluate(
        df=df,
        columns=['name', 'email'],
        known_duplicate_column='is_duplicate',
        threshold=0.8,
        method='auto',
        flag_column='test_predicted',
        generate_report=True,
        report_format='text',
        include_plots=True,
        auto_mode=True,
        show_progress=False
    )
    
    # Generate reports
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Only test markdown to avoid dependencies
        report_paths = generate_enhanced_report(
            results=results,
            formats=['markdown'],
            output_dir=tmpdirname,
            title="Test Enhanced Report",
            include_pairs=True,
            max_pairs=10,
            filename_prefix="test_enhanced"
        )
        
        assert 'markdown' in report_paths
        assert os.path.exists(report_paths['markdown'])
        assert os.path.getsize(report_paths['markdown']) > 0