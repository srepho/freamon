"""
Enhanced Time Series Analysis and Automated Feature Engineering Example
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from freamon.eda.time_series import (
    analyze_timeseries, 
    analyze_autocorrelation, 
    analyze_seasonality,
    analyze_stationarity,
    analyze_multiple_seasonality,
    analyze_forecast_performance
)
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer
from freamon.features.engineer import FeatureEngineer

# Create a synthetic time series dataset with daily data
def create_sample_data(n_days=365*2):
    # Start date
    start_date = datetime(2021, 1, 1)
    
    # Create date range
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Create trend component
    trend = np.linspace(100, 200, n_days)
    
    # Weekly seasonality
    weekly_seasonality = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Annual seasonality
    annual_seasonality = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi/4)
    
    # Random noise
    noise = np.random.normal(0, 10, n_days)
    
    # Combine components
    values = trend + weekly_seasonality + annual_seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    return df

# Create random dataset with multiple time series
def create_multiple_time_series(n_days=365, n_series=3):
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    df = pd.DataFrame({'date': dates})
    
    for i in range(n_series):
        # Different seasonality and trend for each series
        trend = np.linspace(100 * (i+1), 150 * (i+1), n_days)
        seasonality = (20 * (i+1)) * np.sin(2 * np.pi * np.arange(n_days) / (30 * (i+1)))
        noise = np.random.normal(0, 5 * (i+1), n_days)
        
        df[f'series_{i+1}'] = trend + seasonality + noise
    
    # Add id column for grouping
    df['id'] = np.repeat(range(1, n_series+1), n_days // n_series + 1)[:n_days]
    
    # Reshape to long format
    df_long = pd.melt(
        df, 
        id_vars=['date', 'id'], 
        value_vars=[f'series_{i+1}' for i in range(n_series)],
        var_name='series_name', 
        value_name='value'
    )
    
    return df_long

def main():
    print("Enhanced Time Series Analysis Example")
    print("=====================================")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample time series data with {len(df)} observations")
    
    # Basic time series analysis
    print("\n1. Basic Time Series Analysis")
    ts_analysis = analyze_timeseries(df, 'date', 'value')
    print(f"Trend detected: {ts_analysis.get('trend', 'Unknown')}")
    print(f"Time range: {ts_analysis.get('start_date', '')} to {ts_analysis.get('end_date', '')}")
    print(f"Percent change: {ts_analysis.get('percent_change', 0):.2f}%")
    
    # Seasonality analysis
    print("\n2. Enhanced Seasonality Analysis")
    seasonality = analyze_seasonality(df, 'date', 'value', decomposition_method='stl')
    print(f"Decomposition method: {seasonality.get('method', 'Unknown')}")
    print(f"Detected periods: {seasonality.get('detected_periods', [])}")
    print(f"Period used: {seasonality.get('period', 0)}")
    
    # If seasonal patterns were detected, print them
    if 'patterns' in seasonality:
        print("\nSeasonal Patterns:")
        for pattern_name, pattern_info in seasonality['patterns'].items():
            print(f"  {pattern_name}:", end=" ")
            for key, value in pattern_info.items():
                print(f"{key}: {value}", end=", ")
            print()
    
    # Stationarity analysis
    print("\n3. Stationarity Analysis")
    stationarity = analyze_stationarity(df, 'date', 'value')
    print(f"Stationarity status: {stationarity.get('stationarity_status', 'Unknown')}")
    print(f"Is stationary: {stationarity.get('is_stationary', False)}")
    
    # Print ADF test results
    adf = stationarity.get('augmented_dickey_fuller', {})
    if adf and 'error' not in adf:
        print(f"  ADF test p-value: {adf.get('p_value', 0):.4f}")
    
    # Print recommendations
    if 'recommendations' in stationarity:
        print("Recommendations:")
        for rec in stationarity['recommendations']:
            print(f"  - {rec}")
    
    # Multiple seasonality detection
    print("\n4. Multiple Seasonality Detection")
    multiple_seasonality = analyze_multiple_seasonality(df, 'date', 'value')
    
    # Print detected seasonal periods
    seasonal_periods = multiple_seasonality.get('seasonality_periods', [])
    if seasonal_periods:
        print("Detected seasonal periods:")
        for period in seasonal_periods:
            print(f"  {period['name']} (period={period['period']}): " +
                  f"autocorrelation={period['autocorrelation']:.3f}")
    
    # Forecast performance analysis
    print("\n5. Forecast Performance Analysis")
    forecast = analyze_forecast_performance(df, 'date', 'value', test_size=0.2)
    
    # Print forecast results
    print(f"Train size: {forecast.get('train_size', 0)}, Test size: {forecast.get('test_size', 0)}")
    
    if 'best_model' in forecast:
        print(f"Best model: {forecast['best_model']['name']}")
        
    if 'forecast_difficulty' in forecast:
        print(f"Forecast difficulty: {forecast['forecast_difficulty']}")
        print(f"Improvement over naive: {forecast.get('improvement_over_naive', 0):.2f}%")
    
    print("\n6. Automated Time Series Feature Engineering")
    
    # Create TimeSeriesFeatureEngineer
    ts_engineer = TimeSeriesFeatureEngineer(df, 'date', 'value')
    
    # Add feature creation steps
    result_df = (ts_engineer
        .create_lag_features(max_lags=7, strategy='auto')
        .create_rolling_features(windows=[3, 7, 14], metrics=['mean', 'std'])
        .create_differential_features(periods=[1, 7])
        .transform()
    )
    
    # Display resulting features
    print(f"Original dataframe shape: {df.shape}")
    print(f"After automatic feature engineering: {result_df.shape}")
    print("\nGenerated features:")
    new_columns = [col for col in result_df.columns if col not in df.columns]
    for col in new_columns[:10]:  # Show first 10 features
        print(f"  - {col}")
    
    if len(new_columns) > 10:
        print(f"  ... and {len(new_columns) - 10} more features")
    
    print("\n7. Multiple Time Series Feature Engineering")
    
    # Create multiple time series dataset
    multi_df = create_multiple_time_series(n_days=365, n_series=3)
    print(f"Created multiple time series dataset with shape: {multi_df.shape}")
    
    # Create TimeSeriesFeatureEngineer for multiple series
    multi_engineer = TimeSeriesFeatureEngineer(
        multi_df, 'date', 'value', group_col='id'
    )
    
    # Add feature creation steps with grouping
    multi_result = (multi_engineer
        .create_lag_features(max_lags=3)
        .create_rolling_features(windows=[3, 7], metrics=['mean', 'std'])
        .transform()
    )
    
    # Display results
    print(f"Original multi-series dataframe shape: {multi_df.shape}")
    print(f"After feature engineering: {multi_result.shape}")
    
    # Show a sample of the resulting dataframe
    print("\nSample of resulting dataframe:")
    sample_id = multi_result['id'].iloc[0]
    print(multi_result[multi_result['id'] == sample_id].head(3))
    
    print("\nDone!")

if __name__ == "__main__":
    main()