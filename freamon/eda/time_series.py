"""
Time series analysis functions for EDA.

This module provides a comprehensive set of functions for analyzing time series data,
including trend detection, seasonality analysis, stationarity testing, and forecasting evaluation.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def analyze_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    freq : Optional[str], default=None
        The frequency to use for resampling. If None, no resampling is done.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with time series analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for time series analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    if len(df_clean) == 0:
        return {"error": "No data available after removing missing values"}
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Resample if frequency is provided
    if freq is not None:
        ts_df = ts_df.resample(freq).mean()
    
    # Basic statistics
    stats = {
        "count": int(ts_df[value_col].count()),
        "missing": int(ts_df[value_col].isna().sum()),
        "missing_pct": float(ts_df[value_col].isna().mean() * 100),
        "mean": float(ts_df[value_col].mean()),
        "std": float(ts_df[value_col].std()),
        "min": float(ts_df[value_col].min()),
        "max": float(ts_df[value_col].max()),
        "start_date": ts_df.index.min().isoformat(),
        "end_date": ts_df.index.max().isoformat(),
        "duration_days": int((ts_df.index.max() - ts_df.index.min()).days),
    }
    
    # Calculate trends
    # First value, middle value, last value
    if len(ts_df) >= 3:
        first_val = float(ts_df[value_col].iloc[0])
        mid_idx = len(ts_df) // 2
        mid_val = float(ts_df[value_col].iloc[mid_idx])
        last_val = float(ts_df[value_col].iloc[-1])
        
        stats["first_value"] = first_val
        stats["mid_value"] = mid_val
        stats["last_value"] = last_val
        
        # Calculate absolute and percentage changes
        abs_change = last_val - first_val
        pct_change = (abs_change / first_val) * 100 if first_val != 0 else np.nan
        
        stats["absolute_change"] = float(abs_change)
        stats["percent_change"] = float(pct_change) if not np.isnan(pct_change) else None
        
        # Determine trend direction
        if abs_change > 0:
            stats["trend"] = "increasing"
        elif abs_change < 0:
            stats["trend"] = "decreasing"
        else:
            stats["trend"] = "stable"
    
    # Analyze autocorrelation
    if len(ts_df) >= 3:
        # Calculate autocorrelation
        try:
            acf_values = acf(ts_df[value_col].dropna(), nlags=min(10, len(ts_df) - 1))
            stats["autocorrelation"] = {f"lag_{i}": float(val) for i, val in enumerate(acf_values)}
        except:
            # ACF calculation might fail for various reasons
            pass
    
    # Create plot if requested
    if include_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        ts_df[value_col].plot(ax=ax)
        
        # Customize plot
        ax.set_title(f'Time Series: {value_col} over time')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.grid(True)
        
        # Add trend line if we have enough data
        if len(ts_df) >= 3:
            # Simple linear trend
            x = np.arange(len(ts_df))
            y = ts_df[value_col].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(ts_df.index, p(x), "r--", alpha=0.7, label=f"Trend: {stats['trend']}")
            ax.legend()
        
        # Save plot to BytesIO
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        stats["plot"] = f"data:image/png;base64,{img_str}"
    
    return stats


def analyze_autocorrelation(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    lags: int = 20,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze autocorrelation in a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    lags : int, default=20
        The number of lags to include in the autocorrelation analysis.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with autocorrelation analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for autocorrelation analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    if len(df_clean) < 3:
        return {"error": "Insufficient data for autocorrelation analysis"}
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Limit lags to at most n-1
    max_lags = min(lags, len(ts_df) - 1)
    
    # Calculate autocorrelation
    try:
        acf_values = acf(ts_df[value_col], nlags=max_lags)
        
        # Create result dictionary
        result = {
            "lags": max_lags,
            "values": {f"lag_{i}": float(val) for i, val in enumerate(acf_values)},
        }
        
        # Find significant lags (rule of thumb: > 2/sqrt(n))
        n = len(ts_df)
        threshold = 2 / np.sqrt(n)
        
        significant_lags = []
        for i in range(1, len(acf_values)):  # Skip lag 0 (always 1.0)
            if abs(acf_values[i]) > threshold:
                significant_lags.append({
                    "lag": i,
                    "value": float(acf_values[i]),
                    "significant": "positive" if acf_values[i] > 0 else "negative",
                })
        
        result["significant_lags"] = significant_lags
        result["significance_threshold"] = float(threshold)
        
        # Create plot if requested
        if include_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot ACF values
            lags = range(len(acf_values))
            ax.vlines(x=lags, ymin=0, ymax=acf_values, colors='tab:blue')
            ax.axhline(y=0, color='black', linestyle='-')
            ax.axhline(y=threshold, color='tab:red', linestyle='--', label='Significance Threshold')
            ax.axhline(y=-threshold, color='tab:red', linestyle='--')
            
            # Customize plot
            ax.set_title(f'Autocorrelation Function for {value_col}')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set x-ticks
            ax.set_xticks(range(0, len(acf_values), max(1, len(acf_values) // 10)))
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            result["plot"] = f"data:image/png;base64,{img_str}"
        
        return result
    
    except Exception as e:
        return {"error": f"Failed to calculate autocorrelation: {str(e)}"}


def analyze_seasonality(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    decomposition_method: str = 'classical',
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze seasonality in a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    freq : Optional[str], default=None
        The frequency to use for resampling. If None, the function will try to infer it.
    decomposition_method : str, default='classical'
        Method to use for seasonal decomposition. Options:
        - 'classical': Use statsmodels seasonal_decompose
        - 'stl': Use Seasonal-Trend decomposition using LOESS (STL)
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with seasonality analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for seasonality analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    if len(df_clean) < 4:
        return {"error": "Insufficient data for seasonality analysis"}
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Determine frequency if not provided
    if freq is None:
        # Check if the index has a regular frequency
        inferred_freq = pd.infer_freq(ts_df.index)
        if inferred_freq is not None:
            freq = inferred_freq
        else:
            # Try common frequencies
            # Daily data is often a good default
            freq = "D"
    
    # Resample if needed to ensure regular frequency
    try:
        ts_df = ts_df.resample(freq).mean().dropna()
    except Exception:
        # If resampling fails, continue with the original data
        pass
    
    if len(ts_df) < 4:
        return {"error": "Insufficient data after resampling"}
    
    # Auto-detect seasonal periods based on frequency and data length
    detected_periods = []
    
    # Try to infer periods from the data frequency
    if freq in ['M', 'MS', 'ME']:
        # Monthly data - annual seasonality
        if len(ts_df) >= 24:
            detected_periods.append(12)
    elif freq in ['Q', 'QS', 'QE']:
        # Quarterly data - annual seasonality
        if len(ts_df) >= 12:
            detected_periods.append(4)
    elif freq in ['D', 'B']:
        # Daily data - weekly and annual seasonality
        if len(ts_df) >= 14:
            detected_periods.append(7)  # Weekly
        if len(ts_df) >= 730:  # ~2 years
            detected_periods.append(365)  # Annual
    elif freq in ['W', 'W-MON', 'W-SUN']:
        # Weekly data - might have quarterly and annual patterns
        if len(ts_df) >= 12:
            detected_periods.append(4)  # ~Monthly
        if len(ts_df) >= 104:  # ~2 years
            detected_periods.append(52)  # Annual
    elif freq in ['H']:
        # Hourly data - daily, weekly, and annual seasonality
        if len(ts_df) >= 48:
            detected_periods.append(24)  # Daily
        if len(ts_df) >= 168:
            detected_periods.append(24 * 7)  # Weekly
    
    # If no periods detected and we have enough data, use defaults
    if not detected_periods:
        if len(ts_df) >= 24:
            detected_periods.append(12)  # Default to annual
        elif len(ts_df) >= 14:
            detected_periods.append(7)   # Default to weekly
        else:
            detected_periods.append(min(len(ts_df) // 2, 2))  # Very small data
    
    # Use the first detected period for decomposition
    period = detected_periods[0]
    
    # Make sure period is not too large for the data
    period = min(period, len(ts_df) // 2)
    
    # Try to determine seasonality using the requested decomposition method
    try:
        seasonal_analysis = {
            "freq": freq,
            "period": period,
            "detected_periods": detected_periods,
        }
        
        if decomposition_method == 'stl' and len(ts_df) >= 2 * period + 1:
            # Use STL decomposition (Seasonal-Trend decomposition using LOESS)
            # Better for complex seasonality patterns and robust to outliers
            stl_result = STL(
                ts_df[value_col],
                period=period,
                robust=True,
            ).fit()
            
            # Extract components
            seasonal_analysis.update({
                "method": "STL",
                "trend": {
                    "min": float(stl_result.trend.min()),
                    "max": float(stl_result.trend.max()),
                    "mean": float(stl_result.trend.mean()),
                },
                "seasonal": {
                    "min": float(stl_result.seasonal.min()),
                    "max": float(stl_result.seasonal.max()),
                    "mean": float(stl_result.seasonal.mean()),
                    "strength": float(
                        stl_result.seasonal.std() / 
                        (stl_result.trend.std() + stl_result.seasonal.std())
                    ),
                },
                "residual": {
                    "min": float(stl_result.resid.min()),
                    "max": float(stl_result.resid.max()),
                    "mean": float(stl_result.resid.mean()),
                    "std": float(stl_result.resid.std()),
                },
            })
            
            # Add seasonality strength measure
            seasonal_analysis["seasonal"]["strength_robust"] = float(
                1 - stl_result.resid.var() / (stl_result.seasonal + stl_result.resid).var()
            )
            
            # Create plot if requested
            if include_plot:
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                
                # Plot observed, trend, seasonal, and residual components
                ts_df[value_col].plot(ax=axes[0], label='Observed')
                axes[0].set_ylabel('Observed')
                axes[0].set_title(f'STL Decomposition (period={period})')
                axes[0].legend()
                
                stl_result.trend.plot(ax=axes[1], label='Trend')
                axes[1].set_ylabel('Trend')
                axes[1].legend()
                
                stl_result.seasonal.plot(ax=axes[2], label='Seasonal')
                axes[2].set_ylabel('Seasonal')
                axes[2].legend()
                
                stl_result.resid.plot(ax=axes[3], label='Residual')
                axes[3].set_ylabel('Residual')
                axes[3].set_xlabel('Date')
                axes[3].legend()
                
                # Save plot to BytesIO
                buf = BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                
                # Convert to base64
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                seasonal_analysis["plot"] = f"data:image/png;base64,{img_str}"
        
        else:
            # Use classical decomposition
            # Determine model based on data characteristics
            model = "additive"  # default
            
            # If the data has a pronounced upward or downward trend, multiplicative might be better
            if len(ts_df) >= 10 and ts_df[value_col].min() > 0:
                # Simple trend check
                first_half_mean = ts_df[value_col].iloc[:len(ts_df)//2].mean()
                second_half_mean = ts_df[value_col].iloc[len(ts_df)//2:].mean()
                
                # If there's a strong trend, use multiplicative (if all values are positive)
                if abs(second_half_mean - first_half_mean) > 0.5 * first_half_mean:
                    model = "multiplicative"
            
            # Perform seasonal decomposition
            result = seasonal_decompose(
                ts_df[value_col],
                model=model,
                period=period,
            )
            
            # Extract components
            seasonal_analysis.update({
                "method": "classical",
                "model": model,
                "trend": {
                    "min": float(result.trend.dropna().min()),
                    "max": float(result.trend.dropna().max()),
                    "mean": float(result.trend.dropna().mean()),
                },
                "seasonal": {
                    "min": float(result.seasonal.dropna().min()),
                    "max": float(result.seasonal.dropna().max()),
                    "mean": float(result.seasonal.dropna().mean()),
                    "strength": float(
                        result.seasonal.dropna().std() / 
                        (result.trend.dropna().std() + result.seasonal.dropna().std())
                    ),
                },
                "residual": {
                    "min": float(result.resid.dropna().min()),
                    "max": float(result.resid.dropna().max()),
                    "mean": float(result.resid.dropna().mean()),
                    "std": float(result.resid.dropna().std()),
                },
            })
            
            # Create plot if requested
            if include_plot:
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                
                # Plot observed, trend, seasonal, and residual components
                result.observed.plot(ax=axes[0], label='Observed')
                axes[0].set_ylabel('Observed')
                axes[0].set_title(f'Seasonal Decomposition ({model}, period={period})')
                axes[0].legend()
                
                result.trend.plot(ax=axes[1], label='Trend')
                axes[1].set_ylabel('Trend')
                axes[1].legend()
                
                result.seasonal.plot(ax=axes[2], label='Seasonal')
                axes[2].set_ylabel('Seasonal')
                axes[2].legend()
                
                result.resid.plot(ax=axes[3], label='Residual')
                axes[3].set_ylabel('Residual')
                axes[3].set_xlabel('Date')
                axes[3].legend()
                
                # Save plot to BytesIO
                buf = BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=100)
                plt.close(fig)
                
                # Convert to base64
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                seasonal_analysis["plot"] = f"data:image/png;base64,{img_str}"
        
        # Add seasonal pattern analysis
        if len(detected_periods) > 0 and len(ts_df) >= 2 * period:
            # Look for patterns in the seasonal component
            seasonal_patterns = {}
            
            # For each detected period, analyze the seasonal pattern
            for p in detected_periods:
                if p <= len(ts_df) // 2:
                    try:
                        if decomposition_method == 'stl':
                            seasonal_data = stl_result.seasonal
                        else:
                            seasonal_data = result.seasonal
                        
                        # Group by seasonal position
                        if freq == 'M':
                            # Group by month
                            seasonal_by_pos = pd.DataFrame({
                                'month': ts_df.index.month,
                                'value': seasonal_data
                            }).groupby('month').mean()
                            peaks = seasonal_by_pos['value'].nlargest(2).index.tolist()
                            troughs = seasonal_by_pos['value'].nsmallest(2).index.tolist()
                            
                            seasonal_patterns[f"period_{p}"] = {
                                "peak_months": peaks,
                                "trough_months": troughs
                            }
                        
                        elif freq in ['D', 'B']:
                            # Group by day of week or month
                            if p == 7:  # Weekly pattern
                                seasonal_by_pos = pd.DataFrame({
                                    'dayofweek': ts_df.index.dayofweek,
                                    'value': seasonal_data
                                }).groupby('dayofweek').mean()
                                peaks = seasonal_by_pos['value'].nlargest(2).index.tolist()
                                troughs = seasonal_by_pos['value'].nsmallest(2).index.tolist()
                                
                                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                        'Friday', 'Saturday', 'Sunday']
                                seasonal_patterns["weekly"] = {
                                    "peak_days": [days[i] for i in peaks],
                                    "trough_days": [days[i] for i in troughs]
                                }
                            
                            elif p > 28:  # ~Monthly pattern
                                seasonal_by_pos = pd.DataFrame({
                                    'month': ts_df.index.month,
                                    'value': seasonal_data
                                }).groupby('month').mean()
                                peaks = seasonal_by_pos['value'].nlargest(2).index.tolist()
                                troughs = seasonal_by_pos['value'].nsmallest(2).index.tolist()
                                
                                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                seasonal_patterns["annual"] = {
                                    "peak_months": [months[i-1] for i in peaks],
                                    "trough_months": [months[i-1] for i in troughs]
                                }
                        
                        elif freq == 'H':
                            # Group by hour
                            if p == 24:  # Daily pattern
                                seasonal_by_pos = pd.DataFrame({
                                    'hour': ts_df.index.hour,
                                    'value': seasonal_data
                                }).groupby('hour').mean()
                                peaks = seasonal_by_pos['value'].nlargest(3).index.tolist()
                                troughs = seasonal_by_pos['value'].nsmallest(3).index.tolist()
                                
                                seasonal_patterns["daily"] = {
                                    "peak_hours": peaks,
                                    "trough_hours": troughs
                                }
                    except:
                        pass
            
            if seasonal_patterns:
                seasonal_analysis["patterns"] = seasonal_patterns
        
        return seasonal_analysis
    
    except Exception as e:
        return {"error": f"Failed to perform seasonal decomposition: {str(e)}"}


def analyze_stationarity(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze stationarity of a time series using statistical tests.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    include_plot : bool, default=True
        Whether to include plots in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with stationarity analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for stationarity analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    if len(df_clean) < 8:
        return {"error": "Insufficient data for stationarity analysis"}
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Extract time series
    series = ts_df[value_col]
    
    results = {
        "is_stationary": False,
        "augmented_dickey_fuller": {},
        "kpss_test": {},
        "recommendations": []
    }
    
    # Run Augmented Dickey-Fuller test
    try:
        adf_result = adfuller(series, autolag='AIC')
        
        results["augmented_dickey_fuller"] = {
            "test_statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {str(key): float(value) for key, value in adf_result[4].items()},
            "is_stationary": adf_result[1] < 0.05
        }
    except:
        results["augmented_dickey_fuller"] = {"error": "ADF test failed"}
    
    # Run KPSS test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        
        results["kpss_test"] = {
            "test_statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": {str(key): float(value) for key, value in kpss_result[3].items()},
            "is_stationary": kpss_result[1] >= 0.05
        }
    except:
        results["kpss_test"] = {"error": "KPSS test failed"}
    
    # Combine test results to determine stationarity
    adf_stationary = results["augmented_dickey_fuller"].get("is_stationary", False)
    kpss_stationary = results["kpss_test"].get("is_stationary", False)
    
    if adf_stationary and kpss_stationary:
        results["is_stationary"] = True
        results["stationarity_status"] = "Stationary"
    elif adf_stationary and not kpss_stationary:
        results["is_stationary"] = False
        results["stationarity_status"] = "Trend stationary"
        results["recommendations"].append("Consider detrending the series")
    elif not adf_stationary and kpss_stationary:
        results["is_stationary"] = False
        results["stationarity_status"] = "Difference stationary"
        results["recommendations"].append("Consider differencing the series")
    else:
        results["is_stationary"] = False
        results["stationarity_status"] = "Non-stationary"
        results["recommendations"].append("Consider both detrending and differencing")
    
    # Test differenced series
    try:
        diff_series = series.diff().dropna()
        adf_diff_result = adfuller(diff_series, autolag='AIC')
        
        results["differenced_series"] = {
            "adf_test_statistic": float(adf_diff_result[0]),
            "adf_p_value": float(adf_diff_result[1]),
            "is_stationary": adf_diff_result[1] < 0.05
        }
        
        if adf_diff_result[1] < 0.05:
            results["recommendations"].append("First-order differencing is sufficient")
            results["differencing_order"] = 1
        else:
            # Test second-order differencing
            diff2_series = diff_series.diff().dropna()
            adf_diff2_result = adfuller(diff2_series, autolag='AIC')
            
            if adf_diff2_result[1] < 0.05:
                results["recommendations"].append("Second-order differencing is recommended")
                results["differencing_order"] = 2
            else:
                results["recommendations"].append("Higher-order differencing may be needed")
                results["differencing_order"] = "3+"
    except:
        results["differenced_series"] = {"error": "Differencing test failed"}
    
    # Calculate rolling statistics
    if len(series) >= 10:
        window_size = min(20, len(series) // 5)
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        # Calculate variance ratio (ratio of std in first half vs second half)
        half_point = len(series) // 2
        first_half_std = series.iloc[:half_point].std()
        second_half_std = series.iloc[half_point:].std()
        
        if first_half_std > 0 and second_half_std > 0:
            variance_ratio = max(first_half_std, second_half_std) / min(first_half_std, second_half_std)
            
            results["variance_ratio"] = float(variance_ratio)
            
            if variance_ratio > 2.0:
                results["recommendations"].append("Consider variance stabilizing transformation (e.g., log)")
        
        # Create plot if requested
        if include_plot:
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot original series with rolling statistics
            axes[0].plot(series.index, series.values, label='Original')
            axes[0].plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean (window={window_size})')
            axes[0].plot(rolling_std.index, rolling_std.values, label=f'Rolling Std (window={window_size})')
            axes[0].set_title('Rolling Statistics')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Format x-axis dates
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            axes[0].xaxis.set_major_locator(locator)
            axes[0].xaxis.set_major_formatter(formatter)
            
            # Plot differenced series
            if "differenced_series" in results and "error" not in results["differenced_series"]:
                axes[1].plot(diff_series.index, diff_series.values)
                axes[1].set_title(f'Differenced Series (Order 1) - ADF p-value: {results["differenced_series"]["adf_p_value"]:.4f}')
                axes[1].grid(True, alpha=0.3)
                
                # Format x-axis dates
                axes[1].xaxis.set_major_locator(locator)
                axes[1].xaxis.set_major_formatter(formatter)
                
                # Plot histogram of differenced series
                axes[2].hist(diff_series.dropna(), bins=30, alpha=0.7)
                axes[2].set_title('Histogram of Differenced Series')
                
                # Plot normal distribution overlay
                x = np.linspace(diff_series.min(), diff_series.max(), 100)
                mu, std = diff_series.mean(), diff_series.std()
                pdf = stats.norm.pdf(x, mu, std)
                axes[2].plot(x, pdf * len(diff_series) * (diff_series.max() - diff_series.min()) / 30, 
                           'r-', lw=2, label='Normal Distribution')
                axes[2].legend()
            else:
                # If differencing failed, just plot ACF and PACF
                plot_acf(series, lags=min(30, len(series) // 3), ax=axes[1])
                axes[1].set_title('Autocorrelation Function (ACF)')
                
                plot_pacf(series, lags=min(30, len(series) // 3), ax=axes[2])
                axes[2].set_title('Partial Autocorrelation Function (PACF)')
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            results["plot"] = f"data:image/png;base64,{img_str}"
    
    return results


def analyze_multiple_seasonality(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: Optional[str] = None,
    max_period: Optional[int] = None,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze multiple seasonality patterns in a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    freq : Optional[str], default=None
        The frequency to use for resampling. If None, the function will try to infer it.
    max_period : Optional[int], default=None
        Maximum seasonality period to consider (to limit computation).
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with multiple seasonality analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for seasonality analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    if len(df_clean) < 4:
        return {"error": "Insufficient data for seasonality analysis"}
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Determine frequency if not provided
    if freq is None:
        # Check if the index has a regular frequency
        inferred_freq = pd.infer_freq(ts_df.index)
        if inferred_freq is not None:
            freq = inferred_freq
        else:
            # Try common frequencies
            # Daily data is often a good default
            freq = "D"
    
    # Resample if needed to ensure regular frequency
    try:
        ts_df = ts_df.resample(freq).mean().dropna()
    except Exception:
        # If resampling fails, continue with the original data
        pass
    
    if len(ts_df) < 10:
        return {"error": "Insufficient data for multiple seasonality analysis"}
    
    # Determine seasonality candidates based on data characteristics
    series = ts_df[value_col]
    n = len(series)
    
    # Set max_period to 1/3 of data length if not specified
    if max_period is None:
        max_period = n // 3
    else:
        max_period = min(max_period, n // 2)
    
    # Define common seasonality periods based on frequency
    candidates = []
    
    if freq in ['D', 'B']:
        # Daily data
        candidates = [
            {"period": 7, "name": "Weekly"},
            {"period": 14, "name": "Bi-weekly"},
            {"period": 30, "name": "Monthly"},
            {"period": 90, "name": "Quarterly"},
            {"period": 365, "name": "Annual"}
        ]
    elif freq in ['H', '1h']:
        # Hourly data
        candidates = [
            {"period": 24, "name": "Daily"},
            {"period": 24*7, "name": "Weekly"},
            {"period": 24*30, "name": "Monthly"}
        ]
    elif freq in ['M', 'MS', 'ME']:
        # Monthly data
        candidates = [
            {"period": 3, "name": "Quarterly"},
            {"period": 6, "name": "Semi-annual"},
            {"period": 12, "name": "Annual"}
        ]
    elif freq in ['min', '1min', '5min', '10min', '15min', '30min']:
        # Minute data
        minutes_per_day = 24 * 60
        period_min = int(freq.strip('min')) if freq.strip('min').isdigit() else 1
        
        candidates = [
            {"period": 60 // period_min, "name": "Hourly"},
            {"period": (24 * 60) // period_min, "name": "Daily"},
            {"period": (24 * 60 * 7) // period_min, "name": "Weekly"}
        ]
    else:
        # Generic defaults
        candidates = [
            {"period": 2, "name": "Period 2"},
            {"period": 3, "name": "Period 3"},
            {"period": 4, "name": "Period 4"},
            {"period": 7, "name": "Period 7"},
            {"period": 12, "name": "Period 12"}
        ]
    
    # Filter candidates by max_period and data length
    candidates = [c for c in candidates if c["period"] <= max_period and c["period"] <= n//2]
    
    results = {
        "freq": freq,
        "seasonality_periods": []
    }
    
    # Compute autocorrelation for detection
    try:
        acf_values = acf(series, nlags=min(max_period + 10, n-1), fft=True)
        
        # Determine significance threshold
        significance_threshold = 2 / np.sqrt(n)
        
        # Store ACF values for plotting
        acf_result = {
            "lags": list(range(len(acf_values))),
            "values": acf_values.tolist(),
            "threshold": significance_threshold
        }
        
        results["acf"] = acf_result
        
        # Analyze each candidate period
        for candidate in candidates:
            period = candidate["period"]
            name = candidate["name"]
            
            if period >= len(acf_values):
                continue
                
            # Check autocorrelation at the period lag
            ac_at_period = acf_values[period]
            
            # Check if significant
            is_significant = abs(ac_at_period) > significance_threshold
            
            # Look for peaks around the period
            if period > 2:
                window = min(5, period // 3)
                lower_idx = max(1, period - window)
                upper_idx = min(len(acf_values) - 1, period + window)
                
                # Check if there's a local peak
                local_values = acf_values[lower_idx:upper_idx+1]
                is_peak = (ac_at_period == max(local_values)) and is_significant
            else:
                is_peak = is_significant
            
            # Add to results if significant
            if is_significant:
                period_result = {
                    "period": period,
                    "name": name,
                    "autocorrelation": float(ac_at_period),
                    "is_significant": True,
                    "is_peak": is_peak
                }
                results["seasonality_periods"].append(period_result)
        
        # Sort by autocorrelation strength
        results["seasonality_periods"] = sorted(
            results["seasonality_periods"], 
            key=lambda x: abs(x["autocorrelation"]), 
            reverse=True
        )
        
        # Add STL analysis for top periods if enough data
        if results["seasonality_periods"] and n > 10:
            top_periods = results["seasonality_periods"][:min(3, len(results["seasonality_periods"]))]
            stl_results = []
            
            for period_info in top_periods:
                period = period_info["period"]
                
                if n >= 2 * period + 1:  # Minimum data requirement for STL
                    try:
                        # Run STL decomposition
                        stl_result = STL(
                            series,
                            period=period,
                            robust=True
                        ).fit()
                        
                        # Calculate variance explained
                        seasonal_var = stl_result.seasonal.var()
                        total_var = series.var()
                        variance_explained = seasonal_var / total_var if total_var > 0 else 0
                        
                        # Calculate strength
                        resid_var = stl_result.resid.var()
                        seasonal_strength = max(0, 1 - resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0
                        
                        stl_analysis = {
                            "period": period,
                            "name": period_info["name"],
                            "variance_explained": float(variance_explained),
                            "seasonal_strength": float(seasonal_strength),
                            "is_strong": seasonal_strength > 0.3
                        }
                        stl_results.append(stl_analysis)
                    except:
                        pass
            
            if stl_results:
                results["stl_analysis"] = stl_results
        
        # Create plot if requested
        if include_plot and len(acf_values) > 5:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot time series
            axes[0].plot(series.index, series.values)
            axes[0].set_title('Time Series')
            axes[0].grid(True, alpha=0.3)
            
            # Format x-axis dates
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            axes[0].xaxis.set_major_locator(locator)
            axes[0].xaxis.set_major_formatter(formatter)
            
            # Plot ACF
            lags = range(len(acf_values))
            axes[1].vlines(x=lags, ymin=0, ymax=acf_values, colors='tab:blue')
            axes[1].axhline(y=0, color='black', linestyle='-')
            axes[1].axhline(y=significance_threshold, color='tab:red', 
                           linestyle='--', label='Significance Threshold')
            axes[1].axhline(y=-significance_threshold, color='tab:red', linestyle='--')
            
            # Highlight detected periods
            for period_info in results["seasonality_periods"]:
                period = period_info["period"]
                name = period_info["name"]
                if period < len(lags):
                    axes[1].plot(period, acf_values[period], 'ro', markersize=8, 
                               label=f'{name} (lag={period})' if period == results["seasonality_periods"][0]["period"] else None)
            
            # Customize plot
            axes[1].set_title('Autocorrelation Function (ACF)')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Autocorrelation')
            axes[1].grid(True, alpha=0.3)
            if results["seasonality_periods"]:
                axes[1].legend()
            
            # Set x-axis limits
            axes[1].set_xlim(0, min(max_period + 5, len(acf_values) - 1))
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            results["plot"] = f"data:image/png;base64,{img_str}"
        
        return results
    
    except Exception as e:
        return {"error": f"Failed to analyze multiple seasonality: {str(e)}"}


def analyze_forecast_performance(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    test_size: Union[int, float] = 0.2,
    forecast_periods: Optional[int] = None,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze forecast performance on a time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    date_col : str
        The name of the datetime column.
    value_col : str
        The name of the value column.
    test_size : Union[int, float], default=0.2
        Size of the test set. If float, interpreted as a fraction of the data.
        If int, interpreted as the number of periods.
    forecast_periods : Optional[int], default=None
        Number of periods to forecast. If None, uses test_size.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with forecast performance analysis results.
    """
    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")
    
    # Make sure date_col is a datetime column
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ValueError(f"Column '{date_col}' cannot be converted to datetime")
    
    # Make sure value_col is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"Column '{value_col}' must be numeric for forecast analysis")
    
    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Remove missing values
    df_clean = df[[date_col, value_col]].dropna()
    
    # Set date as index for time series operations
    ts_df = df_clean.set_index(date_col)
    
    # Calculate test size
    if isinstance(test_size, float):
        test_size = int(len(ts_df) * test_size)
    
    # Minimum 5 data points for testing
    test_size = min(max(5, test_size), len(ts_df) // 2)
    
    # Set forecast periods
    if forecast_periods is None:
        forecast_periods = test_size
    
    # Split into train and test
    train = ts_df.iloc[:-test_size]
    test = ts_df.iloc[-test_size:]
    
    if len(train) < 10:
        return {"error": "Insufficient training data"}
    
    results = {
        "train_size": len(train),
        "test_size": len(test),
        "forecast_periods": forecast_periods,
        "models": {}
    }
    
    # Define models to evaluate
    models = {
        "naive": "Naive",
        "mean": "Mean",
        "last_value": "Last Value",
        "drift": "Drift",
        "ses": "Simple Exponential Smoothing",
        "arima": "ARIMA"
    }
    
    model_forecasts = {}
    
    # Generate forecasts for each model
    try:
        # Naive model (use the last value)
        last_value = train.iloc[-1][value_col]
        naive_forecast = pd.Series(
            [last_value] * forecast_periods, 
            index=test.index[:forecast_periods]
        )
        model_forecasts["naive"] = naive_forecast
        
        # Mean model
        mean_value = train[value_col].mean()
        mean_forecast = pd.Series(
            [mean_value] * forecast_periods, 
            index=test.index[:forecast_periods]
        )
        model_forecasts["mean"] = mean_forecast
        
        # Last value model (simple persistence)
        last_value_forecast = pd.Series(
            train[value_col].iloc[-1:].values * forecast_periods, 
            index=test.index[:forecast_periods]
        )
        model_forecasts["last_value"] = last_value_forecast
        
        # Drift model
        if len(train) >= 2:
            first = train[value_col].iloc[0]
            last = train[value_col].iloc[-1]
            slope = (last - first) / (len(train) - 1)
            drift_forecast = pd.Series(
                [last + (i+1) * slope for i in range(forecast_periods)], 
                index=test.index[:forecast_periods]
            )
            model_forecasts["drift"] = drift_forecast
        
        # Simple Exponential Smoothing
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        
        try:
            ses_model = SimpleExpSmoothing(train[value_col]).fit(optimized=True)
            ses_forecast = ses_model.forecast(forecast_periods)
            model_forecasts["ses"] = ses_forecast
        except:
            pass
        
        # ARIMA
        try:
            # Find optimal order based on AIC
            from statsmodels.tsa.arima.model import ARIMA
            
            best_aic = float('inf')
            best_order = (1, 0, 0)
            
            # Try a few simple ARIMA models
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(train[value_col], order=(p, d, q))
                            model_fit = model.fit()
                            aic = model_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit best model
            best_model = ARIMA(train[value_col], order=best_order)
            arima_model = best_model.fit()
            arima_forecast = arima_model.forecast(steps=forecast_periods)
            model_forecasts["arima"] = arima_forecast
            
            results["arima_model_info"] = {
                "order": best_order,
                "aic": float(best_aic)
            }
        except:
            pass
    except Exception as e:
        return {"error": f"Error generating forecasts: {str(e)}"}
    
    # Evaluate model performance
    actual = test[value_col][:forecast_periods]
    
    for model_name, forecast in model_forecasts.items():
        # Make sure forecast index matches actual index
        forecast = forecast[:len(actual)]
        if len(forecast) < len(actual):
            forecast = forecast.reindex(actual.index)
        
        # Calculate metrics
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        
        # Mean absolute percentage error
        non_zero_mask = actual != 0
        if sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((actual[non_zero_mask] - forecast[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        # R-squared
        r2 = r2_score(actual, forecast)
        
        model_metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape) if not np.isnan(mape) else None,
            "r2": float(r2),
        }
        
        results["models"][model_name] = {
            "name": models[model_name],
            "metrics": model_metrics,
        }
    
    # Rank models by RMSE
    model_ranks = sorted(
        [(model_name, results["models"][model_name]["metrics"]["rmse"]) 
         for model_name in results["models"]],
        key=lambda x: x[1]
    )
    
    best_model = model_ranks[0][0]
    results["best_model"] = {
        "name": models[best_model],
        "model_key": best_model
    }
    
    # Determine forecast difficulty
    if len(model_ranks) > 1:
        baseline_rmse = results["models"]["naive"]["metrics"]["rmse"]
        best_rmse = results["models"][best_model]["metrics"]["rmse"]
        
        if best_model == "naive":
            forecast_difficulty = "Very hard to improve over naive"
        elif best_rmse > baseline_rmse * 0.95:
            forecast_difficulty = "Hard to forecast (minimal improvement over baseline)"
        elif best_rmse > baseline_rmse * 0.8:
            forecast_difficulty = "Moderately hard to forecast"
        elif best_rmse > baseline_rmse * 0.5:
            forecast_difficulty = "Moderate forecastability"
        else:
            forecast_difficulty = "Relatively easy to forecast"
        
        results["forecast_difficulty"] = forecast_difficulty
        results["improvement_over_naive"] = (1 - best_rmse / baseline_rmse) * 100 if baseline_rmse > 0 else 0
    
    # Create plot if requested
    if include_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot train and test data
        train[value_col].plot(ax=ax, label='Training Data', color='black')
        actual.plot(ax=ax, label='Actual (Test)', color='blue', linestyle=':')
        
        # Plot forecasts
        colors = ['red', 'green', 'purple', 'orange', 'cyan', 'magenta']
        for i, (model_name, forecast) in enumerate(model_forecasts.items()):
            forecast[:len(actual)].plot(
                ax=ax, 
                label=f"{models[model_name]} Forecast", 
                color=colors[i % len(colors)]
            )
        
        # Customize plot
        ax.set_title('Forecast Performance Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        # Add vertical line to separate train and test
        ax.axvline(x=train.index[-1], color='gray', linestyle='--')
        
        # Save plot to BytesIO
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        results["plot"] = f"data:image/png;base64,{img_str}"
    
    return results