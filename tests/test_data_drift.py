"""Tests for the data drift detection module."""

import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from freamon.data_quality import DataDriftDetector, detect_drift


class TestDataDriftDetector:
    """Tests for the DataDriftDetector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample reference and current datasets."""
        # Create reference data
        np.random.seed(42)
        n_rows = 1000
        
        ref_data = pd.DataFrame({
            'numeric1': np.random.normal(10, 2, n_rows),
            'numeric2': np.random.exponential(2, n_rows),
            'categorical1': np.random.choice(['A', 'B', 'C'], n_rows, p=[0.6, 0.3, 0.1]),
            'categorical2': np.random.choice(['X', 'Y', 'Z', 'W'], n_rows),
            'datetime1': [datetime(2023, 1, 1) + timedelta(days=x) for x in range(n_rows)],
        })
        
        # Create current data with some drift
        np.random.seed(43)  # Different seed for different distribution
        
        # Numeric drift: shift mean and variance
        cur_data = pd.DataFrame({
            'numeric1': np.random.normal(12, 3, n_rows),  # Mean and std changed
            'numeric2': np.random.exponential(2, n_rows),  # No significant change
            'categorical1': np.random.choice(['A', 'B', 'C'], n_rows, p=[0.3, 0.6, 0.1]),  # Proportions changed
            'categorical2': np.random.choice(['X', 'Y', 'Z', 'W'], n_rows),  # No significant change
            'datetime1': [datetime(2023, 6, 1) + timedelta(days=x) for x in range(n_rows)],  # Shifted by 5 months
        })
        
        return ref_data, cur_data
    
    def test_data_drift_detector_initialization(self, sample_data):
        """Test initialization of DataDriftDetector."""
        ref_data, cur_data = sample_data
        
        # Test successful initialization
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Check feature type inference
        assert set(detector.cat_features) == {'categorical1', 'categorical2'}
        assert set(detector.num_features) == {'numeric1', 'numeric2'}
        assert set(detector.datetime_features) == {'datetime1'}
    
    def test_numeric_drift_detection(self, sample_data):
        """Test drift detection for numeric features."""
        ref_data, cur_data = sample_data
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Test detecting drift in specific numeric features
        results = detector.detect_numeric_drift(features=['numeric1', 'numeric2'])
        
        # numeric1 should show drift
        assert 'numeric1' in results
        assert results['numeric1']['is_drift'] is True
        assert results['numeric1']['ref_mean'] < results['numeric1']['cur_mean']
        
        # numeric2 might not show significant drift
        assert 'numeric2' in results
        
        # Check output structure
        for feature, result in results.items():
            assert 'type' in result
            assert 'p_value' in result
            assert 'is_drift' in result
            assert 'ref_mean' in result
            assert 'cur_mean' in result
            assert 'drift_plot' in result
    
    def test_categorical_drift_detection(self, sample_data):
        """Test drift detection for categorical features."""
        ref_data, cur_data = sample_data
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Test detecting drift in specific categorical features
        results = detector.detect_categorical_drift(features=['categorical1', 'categorical2'])
        
        # categorical1 should show drift
        assert 'categorical1' in results
        assert results['categorical1']['is_drift'] is True
        
        # Check output structure
        for feature, result in results.items():
            assert 'type' in result
            assert 'chi2' in result
            assert 'p_value' in result
            assert 'is_drift' in result
            assert 'psi' in result
            assert 'drift_plot' in result
    
    def test_datetime_drift_detection(self, sample_data):
        """Test drift detection for datetime features."""
        ref_data, cur_data = sample_data
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Test detecting drift in datetime features
        results = detector.detect_datetime_drift()
        
        # datetime1 should show drift (5 months difference)
        assert 'datetime1' in results
        assert results['datetime1']['is_drift'] is True
        assert results['datetime1']['time_diff_days'] > 0
        
        # Check output structure
        for feature, result in results.items():
            assert 'type' in result
            assert 'p_value' in result
            assert 'is_drift' in result
            assert 'ref_mean' in result
            assert 'cur_mean' in result
            assert 'time_diff_days' in result
            assert 'drift_plot' in result
    
    def test_all_drift_detection(self, sample_data):
        """Test detecting all types of drift at once."""
        ref_data, cur_data = sample_data
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Run all drift detection methods
        results = detector.detect_all_drift()
        
        # Check structure of results
        assert 'numeric' in results
        assert 'categorical' in results
        assert 'datetime' in results
        assert 'summary' in results
        
        # Check summary statistics
        summary = results['summary']
        assert 'dataset_summary' in summary
        assert 'drifting_features' in summary
        
        # Ensure the most drifting features are identified
        assert len(summary['drifting_features']) > 0
        
        # Check that drifting_features are sorted by p-value
        if len(summary['drifting_features']) > 1:
            p_values = [f['p_value'] for f in summary['drifting_features'] if f['p_value'] is not None]
            assert all(p_values[i] <= p_values[i+1] for i in range(len(p_values)-1))
    
    def test_generate_drift_report(self, sample_data):
        """Test generating a drift report."""
        ref_data, cur_data = sample_data
        detector = DataDriftDetector(ref_data, cur_data)
        
        # Run drift detection
        detector.detect_all_drift()
        
        # Generate HTML report
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "drift_report.html"
            detector.generate_drift_report(str(report_path))
            
            # Check that the report was generated
            assert report_path.exists()
            
            # Check content of the report
            with open(report_path, 'r') as f:
                content = f.read()
                assert "<!DOCTYPE html>" in content
                assert "Data Drift Report" in content
                assert "Drift Summary" in content
                # Check for specific features
                assert "numeric1" in content
                assert "categorical1" in content
                assert "datetime1" in content
    
    def test_convenience_function(self, sample_data):
        """Test the detect_drift convenience function."""
        ref_data, cur_data = sample_data
        
        # Use the convenience function
        results = detect_drift(ref_data, cur_data)
        
        # Check structure of results
        assert 'numeric' in results
        assert 'categorical' in results
        assert 'datetime' in results
        assert 'summary' in results