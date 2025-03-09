"""
Module for detecting data drift between datasets.

This module provides functions and classes for detecting and quantifying
data drift between two datasets, which is essential for monitoring model performance
and ensuring data quality over time.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import BytesIO
import base64

from freamon.utils import check_dataframe_type, convert_dataframe


class DataDriftDetector:
    """
    Class for detecting data drift between reference and current datasets.
    
    This class provides methods to calculate statistical metrics that quantify
    how much the distribution of features has changed between two datasets.
    
    Parameters
    ----------
    reference_data : Any
        The reference dataframe (baseline). Supports pandas and other dataframe types.
    current_data : Any
        The current dataframe to compare against the reference. Supports pandas and other dataframe types.
    cat_features : Optional[List[str]], default=None
        List of categorical feature names. If None, will be inferred.
    num_features : Optional[List[str]], default=None
        List of numerical feature names. If None, will be inferred.
    datetime_features : Optional[List[str]], default=None
        List of datetime feature names. If None, will be inferred.
    """
    
    def __init__(
        self,
        reference_data: Any,
        current_data: Any,
        cat_features: Optional[List[str]] = None,
        num_features: Optional[List[str]] = None, 
        datetime_features: Optional[List[str]] = None
    ):
        """
        Initialize the DataDriftDetector with reference and current datasets.
        
        Parameters
        ----------
        reference_data : Any
            The reference dataframe (baseline). Supports pandas and other dataframe types.
        current_data : Any
            The current dataframe to compare against the reference. Supports pandas and other dataframe types.
        cat_features : Optional[List[str]], default=None
            List of categorical feature names. If None, will be inferred.
        num_features : Optional[List[str]], default=None
            List of numerical feature names. If None, will be inferred.
        datetime_features : Optional[List[str]], default=None
            List of datetime feature names. If None, will be inferred.
        """
        # Convert to pandas if needed
        self.ref_df_type = check_dataframe_type(reference_data)
        self.cur_df_type = check_dataframe_type(current_data)
        
        if self.ref_df_type != 'pandas':
            self.reference_data = convert_dataframe(reference_data, 'pandas')
        else:
            self.reference_data = reference_data.copy()
            
        if self.cur_df_type != 'pandas':
            self.current_data = convert_dataframe(current_data, 'pandas')
        else:
            self.current_data = current_data.copy()
        
        # Validate dataframes
        self._validate_dataframes()
        
        # Determine feature types if not provided
        self.cat_features = cat_features or self._infer_categorical_features()
        self.num_features = num_features or self._infer_numerical_features()
        self.datetime_features = datetime_features or self._infer_datetime_features()
        
        # Store results
        self.drift_results = {}
        
    def _validate_dataframes(self) -> None:
        """
        Validate that both dataframes have compatible structures.
        
        Raises
        ------
        ValueError
            If the dataframes are incompatible or have issues.
        """
        # Check if dataframes are empty
        if self.reference_data.empty:
            raise ValueError("Reference dataframe is empty")
        if self.current_data.empty:
            raise ValueError("Current dataframe is empty")
            
        # Check common columns
        ref_cols = set(self.reference_data.columns)
        cur_cols = set(self.current_data.columns)
        common_cols = ref_cols.intersection(cur_cols)
        
        if len(common_cols) == 0:
            raise ValueError("No common columns found between reference and current dataframes")
        
        # Warn about columns only in one dataframe
        only_in_ref = ref_cols - cur_cols
        only_in_cur = cur_cols - ref_cols
        
        if only_in_ref:
            warnings.warn(f"Columns only in reference data: {only_in_ref}")
        if only_in_cur:
            warnings.warn(f"Columns only in current data: {only_in_cur}")
    
    def _infer_categorical_features(self) -> List[str]:
        """
        Infer categorical features from both dataframes.
        
        Returns
        -------
        List[str]
            List of inferred categorical feature names.
        """
        # Get common columns
        common_cols = set(self.reference_data.columns).intersection(
            set(self.current_data.columns)
        )
        
        # Identify categorical columns (object, category, or few unique values)
        cat_cols = []
        for col in common_cols:
            # Check if column is already of categorical type
            ref_dtypes = self.reference_data.select_dtypes(include=['category', 'object'])
            cur_dtypes = self.current_data.select_dtypes(include=['category', 'object'])
            
            if col in ref_dtypes.columns or col in cur_dtypes.columns:
                cat_cols.append(col)
            # Check for numerical columns with few unique values (less than 5% of rows)
            elif col in self.reference_data.select_dtypes(include=['number']).columns and \
                 col in self.current_data.select_dtypes(include=['number']).columns:
                n_unique_ref = self.reference_data[col].nunique()
                n_unique_cur = self.current_data[col].nunique()
                
                ref_threshold = max(10, int(0.05 * len(self.reference_data)))
                cur_threshold = max(10, int(0.05 * len(self.current_data)))
                
                if n_unique_ref <= ref_threshold and n_unique_cur <= cur_threshold:
                    cat_cols.append(col)
        
        return cat_cols
    
    def _infer_numerical_features(self) -> List[str]:
        """
        Infer numerical features from both dataframes.
        
        Returns
        -------
        List[str]
            List of inferred numerical feature names.
        """
        # Get common columns
        common_cols = set(self.reference_data.columns).intersection(
            set(self.current_data.columns)
        )
        
        # Get numerical columns from both dataframes
        ref_num_cols = set(self.reference_data.select_dtypes(include=['number']).columns)
        cur_num_cols = set(self.current_data.select_dtypes(include=['number']).columns)
        
        # Identify numerical columns that are not categorical
        num_cols = []
        for col in common_cols:
            if col in ref_num_cols and col in cur_num_cols:
                # Skip if already identified as categorical
                if col in self.cat_features:
                    continue
                
                num_cols.append(col)
        
        return num_cols
    
    def _infer_datetime_features(self) -> List[str]:
        """
        Infer datetime features from both dataframes.
        
        Returns
        -------
        List[str]
            List of inferred datetime feature names.
        """
        # Get common columns
        common_cols = set(self.reference_data.columns).intersection(
            set(self.current_data.columns)
        )
        
        # Get datetime columns from both dataframes
        ref_dt_cols = set(self.reference_data.select_dtypes(include=['datetime']).columns)
        cur_dt_cols = set(self.current_data.select_dtypes(include=['datetime']).columns)
        
        # Identify datetime columns (in either dataframe)
        datetime_cols = []
        for col in common_cols:
            if col in ref_dt_cols or col in cur_dt_cols:
                datetime_cols.append(col)
        
        return datetime_cols
    
    def detect_numeric_drift(
        self,
        features: Optional[List[str]] = None,
        threshold: float = 0.05,
        method: str = 'ks'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in numerical features.
        
        Parameters
        ----------
        features : Optional[List[str]], default=None
            List of numerical features to check. If None, uses all inferred numerical features.
        threshold : float, default=0.05
            P-value threshold for statistical tests. Lower values indicate stronger evidence of drift.
        method : str, default='ks'
            Statistical test to use: 'ks' (Kolmogorov-Smirnov), 'anderson' (Anderson-Darling),
            or 'wasserstein' (Wasserstein distance).
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with drift detection results for each numerical feature.
        """
        features = features or self.num_features
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in self.current_data.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue
                
            # Get non-null values for both datasets
            ref_values = self.reference_data[feature].dropna().values
            cur_values = self.current_data[feature].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                warnings.warn(f"Feature {feature} has no non-null values in one of the datasets")
                continue
            
            # Calculate statistics
            ref_mean = np.mean(ref_values)
            cur_mean = np.mean(cur_values)
            ref_std = np.std(ref_values)
            cur_std = np.std(cur_values)
            
            # Calculate statistical test based on method
            if method == 'ks':
                statistic, p_value = stats.ks_2samp(ref_values, cur_values)
                test_name = "Kolmogorov-Smirnov"
            elif method == 'anderson':
                statistic = stats.anderson_ksamp([ref_values, cur_values])
                p_value = statistic.significance_level / 100  # Convert percentage to probability
                test_name = "Anderson-Darling"
            elif method == 'wasserstein':
                p_value = None  # Not a p-value based test
                statistic = stats.wasserstein_distance(ref_values, cur_values)
                test_name = "Wasserstein Distance"
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Determine if drift is detected
            if p_value is not None:
                is_drift = p_value < threshold
            else:
                # For Wasserstein, higher distance = more drift
                # Use a heuristic threshold based on the range of the data
                value_range = max(np.max(ref_values) - np.min(ref_values),
                                np.max(cur_values) - np.min(cur_values))
                is_drift = statistic > 0.1 * value_range  # 10% of range as heuristic threshold
            
            # Calculate statistical distance measures
            psi = self._calculate_psi_numeric(ref_values, cur_values)
            
            # Calculate visual drift plots
            plot_img = self._create_drift_plot(
                ref_values, cur_values, feature, 'numeric',
                p_value, statistic, test_name
            )
            
            # Store results
            results[feature] = {
                'type': 'numeric',
                'test_method': method,
                'statistic': float(statistic),
                'p_value': float(p_value) if p_value is not None else None,
                'threshold': threshold,
                'is_drift': bool(is_drift),
                'ref_mean': float(ref_mean),
                'cur_mean': float(cur_mean),
                'ref_std': float(ref_std),
                'cur_std': float(cur_std),
                'mean_diff': float(cur_mean - ref_mean),
                'mean_diff_pct': float((cur_mean - ref_mean) / ref_mean) if ref_mean != 0 else float('inf'),
                'std_diff': float(cur_std - ref_std),
                'std_diff_pct': float((cur_std - ref_std) / ref_std) if ref_std != 0 else float('inf'),
                'psi': float(psi),
                'ref_size': len(ref_values),
                'cur_size': len(cur_values),
                'drift_plot': plot_img
            }
        
        # Store in overall results
        self.drift_results['numeric'] = results
        
        return results
    
    def detect_categorical_drift(
        self,
        features: Optional[List[str]] = None,
        threshold: float = 0.05,
        max_categories: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in categorical features.
        
        Parameters
        ----------
        features : Optional[List[str]], default=None
            List of categorical features to check. If None, uses all inferred categorical features.
        threshold : float, default=0.05
            P-value threshold for chi-square test. Lower values indicate stronger evidence of drift.
        max_categories : int, default=20
            Maximum number of categories to consider. Features with more categories will be skipped.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with drift detection results for each categorical feature.
        """
        features = features or self.cat_features
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in self.current_data.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue
                
            # Get non-null values and frequencies for both datasets
            ref_values = self.reference_data[feature].value_counts(normalize=True, dropna=True)
            cur_values = self.current_data[feature].value_counts(normalize=True, dropna=True)
            
            # Get all unique categories
            all_categories = set(ref_values.index).union(set(cur_values.index))
            
            # Skip if too many categories
            if len(all_categories) > max_categories:
                warnings.warn(f"Feature {feature} has too many categories ({len(all_categories)}), skipping")
                continue
            
            # Fill missing categories with zeros
            for cat in all_categories:
                if cat not in ref_values:
                    ref_values[cat] = 0
                if cat not in cur_values:
                    cur_values[cat] = 0
            
            # Sort by category
            ref_values = ref_values.sort_index()
            cur_values = cur_values.sort_index()
            
            # Calculate chi-square test
            # Convert to counts for chi-square test
            ref_counts = (ref_values * len(self.reference_data[feature].dropna())).round().astype(int)
            cur_counts = (cur_values * len(self.current_data[feature].dropna())).round().astype(int)
            
            # Ensure counts are at least 1 for chi-square test
            ref_counts = ref_counts.replace(0, 1)
            cur_counts = cur_counts.replace(0, 1)
            
            observed = np.array([ref_counts.values, cur_counts.values])
            
            try:
                chi2, p_value, _, _ = stats.chi2_contingency(observed)
                is_drift = p_value < threshold
            except Exception as e:
                warnings.warn(f"Chi-square test failed for feature {feature}: {e}")
                chi2, p_value, is_drift = None, None, False
            
            # Calculate PSI
            psi = self._calculate_psi_categorical(ref_values, cur_values)
            
            # Calculate visual drift plots
            plot_img = self._create_drift_plot(
                ref_values, cur_values, feature, 'categorical',
                p_value, chi2, "Chi-square"
            )
            
            # Calculate Jensen-Shannon divergence (symmetric version of KL divergence)
            js_divergence = self._calculate_js_divergence(ref_values, cur_values)
            
            # Store results
            results[feature] = {
                'type': 'categorical',
                'chi2': float(chi2) if chi2 is not None else None,
                'p_value': float(p_value) if p_value is not None else None,
                'threshold': threshold,
                'is_drift': bool(is_drift),
                'psi': float(psi),
                'js_divergence': float(js_divergence),
                'categories': len(all_categories),
                'ref_size': len(self.reference_data[feature].dropna()),
                'cur_size': len(self.current_data[feature].dropna()),
                'drift_plot': plot_img
            }
        
        # Store in overall results
        self.drift_results['categorical'] = results
        
        return results
    
    def detect_datetime_drift(
        self, 
        features: Optional[List[str]] = None,
        threshold: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in datetime features.
        
        Parameters
        ----------
        features : Optional[List[str]], default=None
            List of datetime features to check. If None, uses all inferred datetime features.
        threshold : float, default=0.05
            P-value threshold for statistical tests. Lower values indicate stronger evidence of drift.
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary with drift detection results for each datetime feature.
        """
        features = features or self.datetime_features
        results = {}
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in self.current_data.columns:
                warnings.warn(f"Feature {feature} not found in both datasets")
                continue
                
            # Get non-null values for both datasets
            ref_series = self.reference_data[feature].dropna()
            cur_series = self.current_data[feature].dropna()
            
            if len(ref_series) == 0 or len(cur_series) == 0:
                warnings.warn(f"Feature {feature} has no non-null values in one of the datasets")
                continue
            
            # Convert to unix timestamps for statistical tests
            ref_values = pd.to_numeric(ref_series) / 10**9  # nanoseconds to seconds
            cur_values = pd.to_numeric(cur_series) / 10**9
            
            # Calculate statistics
            ref_mean = np.mean(ref_values)
            cur_mean = np.mean(cur_values)
            ref_min = np.min(ref_values)
            ref_max = np.max(ref_values)
            cur_min = np.min(cur_values)
            cur_max = np.max(cur_values)
            
            # Calculate statistical test
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            is_drift = p_value < threshold
            
            # Calculate visual drift plots - convert timestamps to years for plotting
            plot_img = self._create_drift_plot(
                pd.to_datetime(ref_series).map(lambda x: x.year),
                pd.to_datetime(cur_series).map(lambda x: x.year),
                feature, 'datetime', p_value, statistic, "Kolmogorov-Smirnov"
            )
            
            # Convert timestamps to human-readable dates for results
            ref_mean_date = pd.to_datetime(ref_mean * 10**9)
            cur_mean_date = pd.to_datetime(cur_mean * 10**9)
            ref_min_date = pd.to_datetime(ref_min * 10**9)
            ref_max_date = pd.to_datetime(ref_max * 10**9)
            cur_min_date = pd.to_datetime(cur_min * 10**9)
            cur_max_date = pd.to_datetime(cur_max * 10**9)
            
            # Store results
            results[feature] = {
                'type': 'datetime',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'threshold': threshold,
                'is_drift': bool(is_drift),
                'ref_mean': str(ref_mean_date),
                'cur_mean': str(cur_mean_date),
                'ref_min': str(ref_min_date),
                'ref_max': str(ref_max_date),
                'cur_min': str(cur_min_date),
                'cur_max': str(cur_max_date),
                'time_diff_seconds': float(cur_mean - ref_mean),
                'time_diff_days': float((cur_mean - ref_mean) / (24 * 3600)),
                'ref_size': len(ref_series),
                'cur_size': len(cur_series),
                'drift_plot': plot_img
            }
        
        # Store in overall results
        self.drift_results['datetime'] = results
        
        return results
    
    def detect_all_drift(
        self,
        numeric_threshold: float = 0.05,
        categorical_threshold: float = 0.05,
        datetime_threshold: float = 0.05
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Run all drift detection methods.
        
        Parameters
        ----------
        numeric_threshold : float, default=0.05
            P-value threshold for numerical drift detection.
        categorical_threshold : float, default=0.05
            P-value threshold for categorical drift detection.
        datetime_threshold : float, default=0.05
            P-value threshold for datetime drift detection.
            
        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Dictionary with all drift detection results.
        """
        # Run all drift detection methods
        numeric_results = self.detect_numeric_drift(threshold=numeric_threshold)
        categorical_results = self.detect_categorical_drift(threshold=categorical_threshold)
        datetime_results = self.detect_datetime_drift(threshold=datetime_threshold)
        
        # Calculate overall drift statistics
        num_drifting = sum(1 for f in numeric_results.values() if f['is_drift'])
        cat_drifting = sum(1 for f in categorical_results.values() if f['is_drift'])
        dt_drifting = sum(1 for f in datetime_results.values() if f['is_drift'])
        
        total_features = len(numeric_results) + len(categorical_results) + len(datetime_results)
        total_drifting = num_drifting + cat_drifting + dt_drifting
        
        # Create dataset summary
        summary = {
            'dataset_summary': {
                'reference_rows': len(self.reference_data),
                'current_rows': len(self.current_data),
                'row_count_change': len(self.current_data) - len(self.reference_data),
                'row_count_change_pct': (len(self.current_data) - len(self.reference_data)) / len(self.reference_data) if len(self.reference_data) > 0 else float('inf'),
                'numeric_features': len(numeric_results),
                'categorical_features': len(categorical_results),
                'datetime_features': len(datetime_results),
                'total_features': total_features,
                'drifting_numeric': num_drifting,
                'drifting_categorical': cat_drifting,
                'drifting_datetime': dt_drifting,
                'total_drifting': total_drifting,
                'drift_percentage': (total_drifting / total_features) * 100 if total_features > 0 else 0
            }
        }
        
        # Create list of drifting features
        drifting_features = []
        for feature_type, results in [
            ('numeric', numeric_results),
            ('categorical', categorical_results),
            ('datetime', datetime_results)
        ]:
            for feature, result in results.items():
                if result['is_drift']:
                    drifting_features.append({
                        'feature': feature,
                        'type': feature_type,
                        'p_value': result.get('p_value'),
                        'statistic': result.get('statistic', result.get('chi2'))
                    })
        
        # Sort drifting features by p-value
        drifting_features.sort(key=lambda x: x['p_value'] if x['p_value'] is not None else 1.0)
        summary['drifting_features'] = drifting_features
        
        # Add summary to results
        self.drift_results['summary'] = summary
        
        return self.drift_results
    
    def _calculate_psi_numeric(self, ref_values: np.ndarray, cur_values: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for numeric features.
        
        Parameters
        ----------
        ref_values : np.ndarray
            Reference values.
        cur_values : np.ndarray
            Current values.
        bins : int, default=10
            Number of bins for PSI calculation.
            
        Returns
        -------
        float
            PSI value. PSI < 0.1 indicates no significant change,
            0.1 <= PSI < 0.2 indicates moderate change,
            PSI >= 0.2 indicates significant change.
        """
        # Determine bin edges based on reference data
        bin_edges = np.histogram_bin_edges(ref_values, bins=bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_values, bins=bin_edges)
        cur_hist, _ = np.histogram(cur_values, bins=bin_edges)
        
        # Convert to percentages and add small epsilon to avoid division by zero
        epsilon = 1e-6
        ref_pct = ref_hist / (len(ref_values) + epsilon) + epsilon
        cur_pct = cur_hist / (len(cur_values) + epsilon) + epsilon
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _calculate_psi_categorical(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """
        Calculate Population Stability Index (PSI) for categorical features.
        
        Parameters
        ----------
        ref_values : pd.Series
            Reference value frequencies (normalized).
        cur_values : pd.Series
            Current value frequencies (normalized).
            
        Returns
        -------
        float
            PSI value. PSI < 0.1 indicates no significant change,
            0.1 <= PSI < 0.2 indicates moderate change,
            PSI >= 0.2 indicates significant change.
        """
        # Align both series to have the same categories
        ref_values, cur_values = ref_values.align(cur_values, fill_value=0)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        ref_pct = ref_values + epsilon
        cur_pct = cur_values + epsilon
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _calculate_js_divergence(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """
        Calculate Jensen-Shannon Divergence between reference and current distributions.
        
        Parameters
        ----------
        ref_values : pd.Series
            Reference value frequencies (normalized).
        cur_values : pd.Series
            Current value frequencies (normalized).
            
        Returns
        -------
        float
            Jensen-Shannon Divergence. 0 indicates identical distributions,
            higher values indicate more divergence.
        """
        # Align both series to have the same categories
        ref_values, cur_values = ref_values.align(cur_values, fill_value=0)
        
        # Add small epsilon to avoid numerical issues
        epsilon = 1e-10
        p = np.array(ref_values) + epsilon
        q = np.array(cur_values) + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate midpoint distribution
        m = (p + q) / 2
        
        # Calculate JS divergence
        js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        
        return float(js_div)
    
    def _create_drift_plot(
        self,
        ref_values: Union[np.ndarray, pd.Series],
        cur_values: Union[np.ndarray, pd.Series],
        feature_name: str,
        feature_type: str,
        p_value: Optional[float] = None,
        statistic: Optional[float] = None,
        test_name: Optional[str] = None
    ) -> str:
        """
        Create a visual representation of drift.
        
        Parameters
        ----------
        ref_values : Union[np.ndarray, pd.Series]
            Reference values.
        cur_values : Union[np.ndarray, pd.Series]
            Current values.
        feature_name : str
            Name of the feature.
        feature_type : str
            Type of the feature ('numeric', 'categorical', or 'datetime').
        p_value : Optional[float], default=None
            P-value from statistical test.
        statistic : Optional[float], default=None
            Test statistic value.
        test_name : Optional[str], default=None
            Name of the statistical test.
            
        Returns
        -------
        str
            Base64-encoded image string.
        """
        plt.figure(figsize=(10, 6))
        
        if feature_type == 'numeric':
            # Create histogram plot
            plt.hist(ref_values, bins=30, alpha=0.5, label='Reference', density=True)
            plt.hist(cur_values, bins=30, alpha=0.5, label='Current', density=True)
            plt.xlabel(feature_name)
            plt.ylabel('Density')
            
        elif feature_type == 'categorical':
            # Create bar plot (ref_values and cur_values are already aligned)
            categories = ref_values.index
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, ref_values, width, label='Reference')
            plt.bar(x + width/2, cur_values, width, label='Current')
            
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.xticks(x, categories, rotation=45, ha='right')
            
        elif feature_type == 'datetime':
            # Create histogram for years
            plt.hist(ref_values, bins=20, alpha=0.5, label='Reference', density=True)
            plt.hist(cur_values, bins=20, alpha=0.5, label='Current', density=True)
            plt.xlabel(f"{feature_name} (Year)")
            plt.ylabel('Density')
            
        # Add drift statistics
        if p_value is not None and test_name is not None:
            if statistic is not None:
                plt.title(f"Drift Analysis for {feature_name}\n{test_name}: {statistic:.4f}, p-value: {p_value:.4f}")
            else:
                plt.title(f"Drift Analysis for {feature_name}\np-value: {p_value:.4f}")
        else:
            plt.title(f"Drift Analysis for {feature_name}")
            
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def generate_drift_report(self, output_path: str, title: str = "Data Drift Report") -> None:
        """
        Generate a comprehensive HTML data drift report.
        
        Parameters
        ----------
        output_path : str
            Path where the HTML report will be saved.
        title : str, default="Data Drift Report"
            Title for the report.
        """
        # If no analyses have been run, run them all
        if not self.drift_results or 'summary' not in self.drift_results:
            self.detect_all_drift()
        
        # Import Jinja2 here to avoid dependency issues
        import jinja2
        import os
        
        # Get the Jinja2 template
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .section { margin-bottom: 30px; }
                .plot-img { max-width: 100%; height: auto; }
                .table-sm td, .table-sm th { padding: 0.25rem; }
                .drift-critical { background-color: #f8d7da; }
                .drift-warning { background-color: #fff3cd; }
                .drift-safe { background-color: #d1e7dd; }
                .card { margin-bottom: 20px; }
                .summary-metric { font-size: 24px; font-weight: bold; }
                .summary-label { font-size: 14px; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-4">{{ title }}</h1>
                
                <!-- Summary Section -->
                <div class="section">
                    <h2>Drift Summary</h2>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <div class="summary-metric">{{ results.summary.dataset_summary.drift_percentage | round(1) }}%</div>
                                    <div class="summary-label">Features with Drift</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <div class="summary-metric">{{ results.summary.dataset_summary.total_drifting }}/{{ results.summary.dataset_summary.total_features }}</div>
                                    <div class="summary-label">Drifting Features</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <div class="summary-metric">{{ results.summary.dataset_summary.reference_rows }}</div>
                                    <div class="summary-label">Reference Rows</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <div class="summary-metric">{{ results.summary.dataset_summary.current_rows }}</div>
                                    <div class="summary-label">Current Rows</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {% if results.summary.drifting_features %}
                    <div class="card mt-4">
                        <div class="card-header">
                            <h5 class="card-title">Top Drifting Features</h5>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Type</th>
                                        <th>P-Value</th>
                                        <th>Test Statistic</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for feature in results.summary.drifting_features[:10] %}
                                    <tr>
                                        <td>{{ feature.feature }}</td>
                                        <td>{{ feature.type }}</td>
                                        <td>{{ "%.4f"|format(feature.p_value) if feature.p_value is not none else 'N/A' }}</td>
                                        <td>{{ "%.4f"|format(feature.statistic) if feature.statistic is not none else 'N/A' }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    {% endif %}
                </div>
                
                <!-- Numeric Features Section -->
                {% if results.numeric %}
                <div class="section">
                    <h2>Numeric Features Drift</h2>
                    <div class="row">
                        {% for feature, analysis in results.numeric.items() %}
                        <div class="col-md-12">
                            <div class="card {{ 'drift-critical' if analysis.is_drift else 'drift-safe' }}">
                                <div class="card-header">
                                    <h5 class="card-title">{{ feature }} 
                                        {% if analysis.is_drift %}
                                        <span class="badge bg-danger">Drift Detected</span>
                                        {% else %}
                                        <span class="badge bg-success">No Drift</span>
                                        {% endif %}
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-7">
                                            {% if analysis.drift_plot %}
                                            <img src="{{ analysis.drift_plot }}" class="plot-img" alt="Drift Plot for {{ feature }}">
                                            {% endif %}
                                        </div>
                                        <div class="col-md-5">
                                            <table class="table table-sm">
                                                <tbody>
                                                    <tr>
                                                        <th>Test</th>
                                                        <td>{{ analysis.test_method }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Statistic</th>
                                                        <td>{{ "%.4f"|format(analysis.statistic) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>P-Value</th>
                                                        <td>{{ "%.4f"|format(analysis.p_value) if analysis.p_value is not none else 'N/A' }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>PSI</th>
                                                        <td>{{ "%.4f"|format(analysis.psi) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Reference Mean</th>
                                                        <td>{{ "%.4f"|format(analysis.ref_mean) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Current Mean</th>
                                                        <td>{{ "%.4f"|format(analysis.cur_mean) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Mean Difference</th>
                                                        <td>{{ "%.4f"|format(analysis.mean_diff) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Mean Difference %</th>
                                                        <td>{{ "%.2f"|format(analysis.mean_diff_pct * 100) }}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Standard Deviation (Ref)</th>
                                                        <td>{{ "%.4f"|format(analysis.ref_std) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Standard Deviation (Cur)</th>
                                                        <td>{{ "%.4f"|format(analysis.cur_std) }}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <!-- Categorical Features Section -->
                {% if results.categorical %}
                <div class="section">
                    <h2>Categorical Features Drift</h2>
                    <div class="row">
                        {% for feature, analysis in results.categorical.items() %}
                        <div class="col-md-12">
                            <div class="card {{ 'drift-critical' if analysis.is_drift else 'drift-safe' }}">
                                <div class="card-header">
                                    <h5 class="card-title">{{ feature }} 
                                        {% if analysis.is_drift %}
                                        <span class="badge bg-danger">Drift Detected</span>
                                        {% else %}
                                        <span class="badge bg-success">No Drift</span>
                                        {% endif %}
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-7">
                                            {% if analysis.drift_plot %}
                                            <img src="{{ analysis.drift_plot }}" class="plot-img" alt="Drift Plot for {{ feature }}">
                                            {% endif %}
                                        </div>
                                        <div class="col-md-5">
                                            <table class="table table-sm">
                                                <tbody>
                                                    <tr>
                                                        <th>Test</th>
                                                        <td>Chi-square</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Chi-square</th>
                                                        <td>{{ "%.4f"|format(analysis.chi2) if analysis.chi2 is not none else 'N/A' }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>P-Value</th>
                                                        <td>{{ "%.4f"|format(analysis.p_value) if analysis.p_value is not none else 'N/A' }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>PSI</th>
                                                        <td>{{ "%.4f"|format(analysis.psi) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>JS Divergence</th>
                                                        <td>{{ "%.4f"|format(analysis.js_divergence) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Categories</th>
                                                        <td>{{ analysis.categories }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Reference Sample Size</th>
                                                        <td>{{ analysis.ref_size }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Current Sample Size</th>
                                                        <td>{{ analysis.cur_size }}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                <!-- Datetime Features Section -->
                {% if results.datetime %}
                <div class="section">
                    <h2>Datetime Features Drift</h2>
                    <div class="row">
                        {% for feature, analysis in results.datetime.items() %}
                        <div class="col-md-12">
                            <div class="card {{ 'drift-critical' if analysis.is_drift else 'drift-safe' }}">
                                <div class="card-header">
                                    <h5 class="card-title">{{ feature }} 
                                        {% if analysis.is_drift %}
                                        <span class="badge bg-danger">Drift Detected</span>
                                        {% else %}
                                        <span class="badge bg-success">No Drift</span>
                                        {% endif %}
                                    </h5>
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-7">
                                            {% if analysis.drift_plot %}
                                            <img src="{{ analysis.drift_plot }}" class="plot-img" alt="Drift Plot for {{ feature }}">
                                            {% endif %}
                                        </div>
                                        <div class="col-md-5">
                                            <table class="table table-sm">
                                                <tbody>
                                                    <tr>
                                                        <th>Test</th>
                                                        <td>Kolmogorov-Smirnov</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Statistic</th>
                                                        <td>{{ "%.4f"|format(analysis.statistic) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>P-Value</th>
                                                        <td>{{ "%.4f"|format(analysis.p_value) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Reference Mean</th>
                                                        <td>{{ analysis.ref_mean }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Current Mean</th>
                                                        <td>{{ analysis.cur_mean }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Time Difference (days)</th>
                                                        <td>{{ "%.2f"|format(analysis.time_diff_days) }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Reference Range</th>
                                                        <td>{{ analysis.ref_min }} to {{ analysis.ref_max }}</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Current Range</th>
                                                        <td>{{ analysis.cur_min }} to {{ analysis.cur_max }}</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Render the template
        template = jinja2.Template(template_str)
        html_content = template.render(
            title=title,
            results=self.drift_results,
        )
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write the HTML to a file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Drift report saved to {output_path}")


def detect_drift(
    reference_data: Any,
    current_data: Any,
    cat_features: Optional[List[str]] = None,
    num_features: Optional[List[str]] = None,
    datetime_features: Optional[List[str]] = None,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect data drift between reference and current datasets.
    
    This is a convenience function that creates a DataDriftDetector and runs all drift detection methods.
    
    Parameters
    ----------
    reference_data : Any
        The reference dataframe (baseline). Supports pandas and other dataframe types.
    current_data : Any
        The current dataframe to compare against the reference. Supports pandas and other dataframe types.
    cat_features : Optional[List[str]], default=None
        List of categorical feature names. If None, will be inferred.
    num_features : Optional[List[str]], default=None
        List of numerical feature names. If None, will be inferred.
    datetime_features : Optional[List[str]], default=None
        List of datetime feature names. If None, will be inferred.
    threshold : float, default=0.05
        P-value threshold for statistical tests. Lower values indicate stronger evidence of drift.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with drift detection results.
    """
    detector = DataDriftDetector(
        reference_data, current_data,
        cat_features=cat_features,
        num_features=num_features,
        datetime_features=datetime_features
    )
    
    return detector.detect_all_drift(
        numeric_threshold=threshold,
        categorical_threshold=threshold,
        datetime_threshold=threshold
    )