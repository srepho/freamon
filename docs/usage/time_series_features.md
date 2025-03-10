# Time Series Features

Freamon provides advanced capabilities for time series data analysis and feature engineering, making it easy to extract insights and create predictive features for time-dependent data.

## Enhanced Time Series Analysis

The `time_series` module in `freamon.eda` offers comprehensive tools for analyzing temporal patterns in your data:

```python
from freamon.eda.time_series import (
    analyze_timeseries,  # Basic time series analysis
    analyze_autocorrelation,  # Autocorrelation analysis
    analyze_seasonality,  # Decomposition and seasonality detection
    analyze_stationarity,  # Stationarity testing with ADF and KPSS
    analyze_multiple_seasonality,  # Detect multiple seasonal patterns
    analyze_forecast_performance  # Evaluate forecasting difficulty
)
```

### Basic Time Series Analysis

The `analyze_timeseries` function provides a quick overview of your time series:

```python
result = analyze_timeseries(df, date_col='date', value_col='sales')
```

This returns basic statistics, trend identification, and plots.

### Seasonality Analysis

Detect and decompose seasonal patterns with advanced methods:

```python
# Classical decomposition
seasonality = analyze_seasonality(
    df, 
    date_col='date', 
    value_col='sales',
    decomposition_method='classical'
)

# STL decomposition (more robust to outliers)
seasonality = analyze_seasonality(
    df, 
    date_col='date', 
    value_col='sales',
    decomposition_method='stl'
)
```

The function automatically detects appropriate seasonal periods based on your data's frequency and outputs detailed information about trend, seasonal, and residual components.

### Stationarity Analysis

Test if your time series is stationary and get recommendations for transformations:

```python
stationarity = analyze_stationarity(df, date_col='date', value_col='sales')

# Check stationarity status
print(stationarity['stationarity_status'])  # e.g., "Trend stationary"

# Get recommendations
for recommendation in stationarity['recommendations']:
    print(recommendation)  # e.g., "Consider differencing the series"
```

This function runs both Augmented Dickey-Fuller (ADF) and KPSS tests and provides actionable insights.

### Multiple Seasonality Detection

For complex time series with multiple seasonal patterns:

```python
seasonality = analyze_multiple_seasonality(
    df, 
    date_col='date', 
    value_col='sales'
)

# Access detected periods
for period in seasonality['seasonality_periods']:
    print(f"{period['name']} (period={period['period']}): " +
          f"strength={period['autocorrelation']:.3f}")
```

This function can detect daily, weekly, monthly, and annual patterns in your data.

### Forecast Performance Analysis

Evaluate how predictable your time series is:

```python
forecast = analyze_forecast_performance(
    df, 
    date_col='date', 
    value_col='sales',
    test_size=0.2  # Use 20% of data for testing
)

# See which model performs best
print(f"Best model: {forecast['best_model']['name']}")

# Check forecast difficulty
print(f"Forecast difficulty: {forecast['forecast_difficulty']}")
```

This function compares several baseline models (Naive, Mean, Drift, SES, ARIMA) to assess predictability.

## Automated Time Series Feature Engineering

The `TimeSeriesFeatureEngineer` class in `freamon.features.time_series_engineer` provides automated feature generation for time series data.

```python
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer
```

### Basic Usage

```python
# Initialize the engineer
ts_engineer = TimeSeriesFeatureEngineer(
    df,                # DataFrame with time series data
    date_col='date',   # Column containing dates
    target_cols='sales'  # Column(s) to create features for
)

# Add feature creation steps with method chaining
result_df = (ts_engineer
    .create_lag_features(max_lags=7)
    .create_rolling_features(metrics=['mean', 'std'])
    .create_differential_features()
    .transform()
)
```

### Automated Lag Detection

The engineer can automatically detect optimal lag values based on autocorrelation, partial autocorrelation, and mutual information:

```python
# Automatically detect important lags
result_df = (ts_engineer
    .create_lag_features(
        max_lags=14,
        strategy='auto'  # Auto-detect optimal lags
    )
    .transform()
)
```

You can also use predefined lag strategies:
- `'all'`: Use all lags up to max_lags
- `'fibonacci'`: Use Fibonacci sequence lags (1, 2, 3, 5, 8, ...)
- `'exponential'`: Use exponentially increasing lags (1, 2, 4, 8, ...)

### Rolling Window Features

Create features using rolling windows with automatic window size detection:

```python
result_df = (ts_engineer
    .create_rolling_features(
        metrics=['mean', 'std', 'min', 'max'],
        auto_detect=True,  # Auto-detect optimal window sizes
        max_window=30
    )
    .transform()
)
```

You can also specify window sizes manually:

```python
result_df = (ts_engineer
    .create_rolling_features(
        windows=[7, 14, 30],
        metrics=['mean', 'std']
    )
    .transform()
)
```

### Differencing Features

Create differencing features to capture changes over time:

```python
result_df = (ts_engineer
    .create_differential_features(
        periods=[1, 7, 30]  # Daily, weekly, monthly changes
    )
    .transform()
)
```

This creates both raw differences and percentage changes.

### Handling Multiple Time Series

The engineer supports panel data with multiple time series:

```python
# Initialize with a group column
ts_engineer = TimeSeriesFeatureEngineer(
    panel_df,
    date_col='date',
    target_cols='value',
    group_col='id'  # Column that identifies different series
)

# Features will be created within each group
result_df = (ts_engineer
    .create_lag_features(max_lags=5)
    .transform()
)
```

This ensures that features are properly created within each separate time series.

## Example Workflow

A complete example of time series analysis and feature engineering:

```python
import pandas as pd
from freamon.eda.time_series import analyze_seasonality, analyze_stationarity
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer

# Load your time series data
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 1. Analyze seasonality
seasonality = analyze_seasonality(df, 'date', 'sales')
print(f"Detected periods: {seasonality['detected_periods']}")

# 2. Check stationarity
stationarity = analyze_stationarity(df, 'date', 'sales')
if not stationarity['is_stationary']:
    print("Series is not stationary. Recommendations:")
    for rec in stationarity['recommendations']:
        print(f"- {rec}")
    
    # Apply differencing if recommended
    if "differencing" in "".join(stationarity['recommendations']):
        order = stationarity.get('differencing_order', 1)
        print(f"Applying order {order} differencing")

# 3. Create time series features
ts_engineer = TimeSeriesFeatureEngineer(df, 'date', 'sales')
feature_df = (ts_engineer
    .create_lag_features(strategy='auto')
    .create_rolling_features(auto_detect=True)
    .create_differential_features()
    .transform()
)

# 4. Use the features for modeling
# ...
```

## Further Resources

- Check the example file `examples/time_series_enhanced_example.py` for a complete demonstration
- See `freamon.features.time_series_engineer` module documentation for all available parameters