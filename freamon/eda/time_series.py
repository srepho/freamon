"""
Time series analysis functions for EDA.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


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
    
    # Try to determine seasonality using seasonal_decompose
    try:
        # Choose model based on data characteristics
        model = "additive"  # default
        
        # If the data has a pronounced upward or downward trend, multiplicative might be better
        if len(ts_df) >= 10:
            # Simple trend check
            first_half_mean = ts_df[value_col].iloc[:len(ts_df)//2].mean()
            second_half_mean = ts_df[value_col].iloc[len(ts_df)//2:].mean()
            
            # If there's a strong trend, use multiplicative (if all values are positive)
            if abs(second_half_mean - first_half_mean) > 0.5 * first_half_mean and ts_df[value_col].min() > 0:
                model = "multiplicative"
        
        # Determine period
        period = None
        
        # Try to infer period from the data frequency
        if freq in ['M', 'MS', 'ME']:
            # Monthly data - annual seasonality
            period = 12
        elif freq in ['Q', 'QS', 'QE']:
            # Quarterly data - annual seasonality
            period = 4
        elif freq in ['D', 'B']:
            # Daily data - weekly seasonality
            period = 7
        elif freq in ['W', 'W-MON', 'W-SUN']:
            # Weekly data - might have monthly or quarterly patterns
            period = 4
        elif freq in ['H']:
            # Hourly data - daily seasonality
            period = 24
        
        # If we couldn't determine period and we have enough data, try 12 (annual)
        if period is None and len(ts_df) >= 24:
            period = 12
        # For shorter series, try weekly
        elif period is None and len(ts_df) >= 14:
            period = 7
        # For very short series, just use 2
        elif period is None:
            period = 2
        
        # Make sure period is not too large for the data
        period = min(period, len(ts_df) // 2)
        
        # Perform seasonal decomposition
        result = seasonal_decompose(
            ts_df[value_col],
            model=model,
            period=period,
        )
        
        # Extract components
        seasonal_analysis = {
            "model": model,
            "period": period,
            "freq": freq,
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
        }
        
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
        
        return seasonal_analysis
    
    except Exception as e:
        return {"error": f"Failed to perform seasonal decomposition: {str(e)}"}