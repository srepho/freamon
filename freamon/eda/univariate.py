"""
Univariate analysis functions for EDA.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


def analyze_numeric(
    df: pd.DataFrame,
    column: str,
    bins: Optional[int] = None,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a numeric column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    column : str
        The name of the column to analyze.
    bins : Optional[int], default=None
        The number of bins for histograms. If None, a suitable number is chosen.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with analysis results.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Check if column is numeric using select_dtypes
    numeric_cols = df.select_dtypes(include=['number']).columns
    if column not in numeric_cols:
        raise ValueError(f"Column '{column}' is not numeric")
    
    # Get the series
    series = df[column]
    
    # Basic statistics
    stats = {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "range": float(series.max() - series.min()),
    }
    
    # Percentiles
    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for p in percentiles:
        stats[f"percentile_{int(p*100)}"] = float(series.quantile(p))
    
    # Check for zeros
    stats["zero_count"] = int((series == 0).sum())
    stats["zero_pct"] = float((series == 0).mean() * 100)
    
    # Check for negative values
    stats["negative_count"] = int((series < 0).sum())
    stats["negative_pct"] = float((series < 0).mean() * 100)
    
    # Create plot if requested
    if include_plot:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        if bins is None:
            # Use Sturges' rule for bin calculation
            bins = int(np.ceil(np.log2(len(series.dropna())) + 1))
            bins = min(bins, 50)  # Cap at 50 bins
        
        sns.histplot(series.dropna(), kde=True, bins=bins, ax=ax1)
        ax1.set_title(f"Distribution of {column}")
        ax1.set_xlabel(column)
        ax1.set_ylabel("Frequency")
        
        # Box plot
        sns.boxplot(x=series.dropna(), ax=ax2)
        ax2.set_title(f"Box Plot of {column}")
        ax2.set_xlabel(column)
        
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


def analyze_categorical(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 20,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    column : str
        The name of the column to analyze.
    max_categories : int, default=20
        The maximum number of categories to display.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with analysis results.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Get the series
    series = df[column]
    
    # Check if boolean - handle as special case
    is_boolean = column in df.select_dtypes(include=['bool']).columns or (
        series.dropna().nunique() == 2 and 
        set(series.dropna().unique()).issubset({0, 1, True, False})
    )
    
    # Basic statistics
    stats = {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
        "unique": int(series.nunique()),
        "is_boolean": is_boolean,
    }
    
    # Value counts
    value_counts = series.value_counts(dropna=False)
    
    # Handle too many categories
    if len(value_counts) > max_categories:
        # Keep top categories, group others
        top_categories = value_counts.iloc[:max_categories]
        other_count = value_counts.iloc[max_categories:].sum()
        
        # Create new value counts with "Other" category
        value_counts = pd.concat([
            top_categories,
            pd.Series({"Other": other_count})
        ])
        
        stats["categories_limited"] = True
        stats["total_categories"] = int(series.nunique())
    else:
        stats["categories_limited"] = False
    
    # Convert to dictionary and add percentages
    value_dict = {}
    for val, count in value_counts.items():
        # Handle missing values
        if pd.isna(val):
            val_str = "Missing"
        else:
            val_str = str(val)
        
        value_dict[val_str] = {
            "count": int(count),
            "percentage": float(count / len(series) * 100)
        }
    
    stats["value_counts"] = value_dict
    
    # Create plot if requested
    if include_plot:
        plt.figure(figsize=(10, 6))
        
        # Use a horizontal bar chart if many categories
        if len(value_counts) > 10:
            ax = value_counts.plot(kind="barh")
            ax.set_xlabel("Count")
            ax.set_ylabel(column)
        else:
            ax = value_counts.plot(kind="bar")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
        
        ax.set_title(f"Value Counts for {column}")
        
        # Add count labels
        for i, v in enumerate(value_counts):
            if len(value_counts) > 10:
                ax.text(v + 0.1, i, str(v), va="center")
            else:
                ax.text(i, v + 0.1, str(v), ha="center")
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        stats["plot"] = f"data:image/png;base64,{img_str}"
    
    return stats


def analyze_datetime(
    df: pd.DataFrame,
    column: str,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a datetime column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    column : str
        The name of the column to analyze.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with analysis results.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Get the series and ensure it's datetime
    series = df[column]
    datetime_cols = df.select_dtypes(include=['datetime']).columns
    
    if column not in datetime_cols:
        try:
            series = pd.to_datetime(series)
        except (ValueError, TypeError):
            raise ValueError(f"Column '{column}' cannot be converted to datetime")
    
    # Basic statistics
    stats = {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "missing_pct": float(series.isna().mean() * 100),
        "min": series.min().isoformat() if not pd.isna(series.min()) else None,
        "max": series.max().isoformat() if not pd.isna(series.max()) else None,
    }
    
    # Calculate date range
    if not pd.isna(series.min()) and not pd.isna(series.max()):
        date_range = series.max() - series.min()
        stats["range_days"] = date_range.days
    
    # Time components
    date_parts = {
        "year": series.dt.year,
        "month": series.dt.month,
        "day": series.dt.day,
        "weekday": series.dt.dayofweek,
        "hour": series.dt.hour if series.dt.hour.nunique() > 1 else None,
        "minute": series.dt.minute if series.dt.minute.nunique() > 1 else None,
    }
    
    # Remove components with no variation and build date_components
    date_components = {}
    for part, values in date_parts.items():
        if values is not None:
            # Always keep year, month, day, weekday even if no variation
            if part in ["year", "month", "day", "weekday"] or values.nunique() > 1:
                date_components[part] = {
                    "unique": int(values.nunique()),
                    "min": int(values.min()) if not pd.isna(values.min()) else None,
                    "max": int(values.max()) if not pd.isna(values.max()) else None,
                }
    
    stats["components"] = date_components
    
    # Distribution by year and month
    if "year" in date_components:
        year_counts = series.dt.year.value_counts().sort_index()
        year_dict = {str(year): int(count) for year, count in year_counts.items()}
        stats["year_counts"] = year_dict
    
    if "month" in date_components:
        month_counts = series.dt.month.value_counts().sort_index()
        
        # Convert to month names
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        month_dict = {month_names[month]: int(count) for month, count in month_counts.items()}
        stats["month_counts"] = month_dict
    
    if "weekday" in date_components:
        weekday_counts = series.dt.dayofweek.value_counts().sort_index()
        
        # Convert to weekday names
        weekday_names = {
            0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
        }
        weekday_dict = {weekday_names[day]: int(count) for day, count in weekday_counts.items()}
        stats["weekday_counts"] = weekday_dict
    
    # Create plot if requested
    if include_plot:
        # Create figure with multiple subplots
        n_plots = sum(1 for x in ["year_counts", "month_counts", "weekday_counts"] if x in stats)
        fig, axes = plt.subplots(1, n_plots, figsize=(12, 5))
        
        # Ensure axes is always an array
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot year distribution if available
        if "year_counts" in stats:
            ax = axes[plot_idx]
            years = list(map(int, stats["year_counts"].keys()))
            counts = list(stats["year_counts"].values())
            
            ax.bar(years, counts)
            ax.set_title("Distribution by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            
            # Set x-ticks to show all years
            ax.set_xticks(years)
            
            plot_idx += 1
        
        # Plot month distribution if available
        if "month_counts" in stats:
            ax = axes[plot_idx]
            months = list(stats["month_counts"].keys())
            counts = list(stats["month_counts"].values())
            
            ax.bar(months, counts)
            ax.set_title("Distribution by Month")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            
            plot_idx += 1
        
        # Plot weekday distribution if available
        if "weekday_counts" in stats:
            ax = axes[plot_idx]
            weekdays = list(stats["weekday_counts"].keys())
            counts = list(stats["weekday_counts"].values())
            
            ax.bar(weekdays, counts)
            ax.set_title("Distribution by Weekday")
            ax.set_xlabel("Weekday")
            ax.set_ylabel("Count")
        
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