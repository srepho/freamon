"""
Tests for the display_detection_report method of DataTypeDetector class.
"""
import pandas as pd
import pytest

from freamon.utils.datatype_detector import DataTypeDetector


def test_display_detection_report():
    """Test that the display_detection_report method works correctly."""
    # Create a simple test dataframe
    df = pd.DataFrame({
        'id': range(100),
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'date_str': ['2023-01-01', '2023-02-01'] * 50,
        'month_year': ['Jan-23', 'Feb-23'] * 50,
        'amount': [float(i) for i in range(100)],
        'with_nulls': [None if i % 5 == 0 else i for i in range(100)]
    })
    
    # Create detector and run detection
    detector = DataTypeDetector(df)
    detector.detect_all_types()
    
    # Get the styled report
    report = detector.display_detection_report()
    
    # Basic checks - we can't check styling directly, but we can verify the DataFrame structure
    assert isinstance(report, pd.io.formats.style.Styler)
    
    # Unwrap the DataFrame
    report_df = report.data
    
    # Check column presence
    expected_columns = [
        'Column', 'Storage Type', 'Logical Type', 'Semantic Type', 
        'Null Count', 'Null %', 'Unique Values', 'Suggested Conversion'
    ]
    for col in expected_columns:
        assert col in report_df.columns
    
    # Check that all original columns are included
    df_columns = set(df.columns)
    report_columns = set(report_df['Column'])
    assert df_columns.issubset(report_columns)
    
    # Check specific detections
    id_row = report_df[report_df['Column'] == 'id'].iloc[0]
    assert 'id' in id_row['Semantic Type'].lower()
    
    # Check month_year detection
    month_year_row = report_df[report_df['Column'] == 'month_year'].iloc[0]
    assert 'month_year' in month_year_row['Semantic Type'].lower()
    
    # Check that the report works with include_stats=False too
    simple_report = detector.display_detection_report(include_stats=False)
    assert isinstance(simple_report, pd.io.formats.style.Styler)


def test_display_detection_report_edge_cases():
    """Test edge cases for the display_detection_report method."""
    # Empty dataframe
    empty_df = pd.DataFrame()
    detector = DataTypeDetector(empty_df)
    report = detector.display_detection_report()
    # It could return either a styled dataframe or a regular one
    assert isinstance(report, (pd.DataFrame, pd.io.formats.style.Styler))
    
    # DataFrame with only null values
    null_df = pd.DataFrame({
        'all_nulls': [None, None, None],
        'more_nulls': [float('nan'), None, float('nan')]
    })
    detector = DataTypeDetector(null_df)
    report = detector.display_detection_report()
    assert isinstance(report, (pd.DataFrame, pd.io.formats.style.Styler))
    
    # Check that all columns from the original dataframe are in the report
    if isinstance(report, pd.io.formats.style.Styler):
        report_df = report.data
    else:
        report_df = report
    assert set(null_df.columns).issubset(set(report_df['Column']))