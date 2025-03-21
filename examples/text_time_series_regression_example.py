"""
Example demonstrating text processing and time series cross-validation for regression.

This example shows how to:
1. Process a text field using spaCy integration
2. Create advanced text features
3. Combine text features with time series features
4. Use time series cross-validation for regression prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from freamon.utils.text_utils import TextProcessor
from freamon.features.time_series_engineer import TimeSeriesFeatureEngineer
from freamon.model_selection.cross_validation import time_series_cross_validate
from freamon.modeling import (
    create_lightgbm_regressor, LightGBMModel,
    plot_cv_metrics, plot_feature_importance, 
    plot_importance_by_groups, plot_time_series_predictions, 
    plot_cv_predictions_over_time, TEXT_FEATURE_GROUPS, TIME_SERIES_FEATURE_GROUPS
)


# Helper functions for modeling
def get_lightgbm_model(problem_type='regression', **kwargs):
    """Utility function to create a LightGBM model with sensible defaults.
    
    This demonstrates how to create a helper function to simplify model creation.
    
    Parameters
    ----------
    problem_type : str, default='regression'
        The type of problem ('regression' or 'classification')
    **kwargs : dict
        Additional parameters to pass to the model constructor
        
    Returns
    -------
    Model
        A configured LightGBM model ready to use
    """
    from freamon.modeling import create_lightgbm_regressor, create_lightgbm_classifier
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42,
    }
    
    # Update defaults with any user-provided parameters
    params = {**default_params, **kwargs}
    
    if problem_type == 'regression':
        return create_lightgbm_regressor(**params)
    else:
        return create_lightgbm_classifier(**params)


# Create a synthetic dataset with date, text, and target regression value
def create_sample_data(n_samples=500):
    # Generate dates (daily data for past n_samples days)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=n_samples-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate product review text with varying sentiment
    products = ['smartphone', 'laptop', 'headphones', 'smartwatch', 'camera', 'tablet']
    adjectives_positive = ['excellent', 'amazing', 'fantastic', 'great', 'wonderful', 'superb']
    adjectives_neutral = ['decent', 'acceptable', 'average', 'okay', 'fine', 'standard']
    adjectives_negative = ['poor', 'terrible', 'disappointing', 'bad', 'awful', 'horrible']
    
    # Trend component (increasing over time)
    trend = np.linspace(50, 100, n_samples)
    
    # Weekly seasonality (higher ratings on weekends)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_effect = 5 * (day_of_week >= 5)  # Weekend boost
    
    # Create reviews and ratings with noise
    texts = []
    sentiment_scores = []
    
    for i in range(n_samples):
        # Determine sentiment direction (with time trend and weekly pattern)
        base_sentiment = (trend[i] / 100) + (weekly_effect[i] / 100)
        # Add noise
        sentiment_with_noise = base_sentiment + np.random.normal(0, 0.15)
        sentiment_with_noise = max(0.1, min(1.0, sentiment_with_noise))
        sentiment_scores.append(sentiment_with_noise)
        
        # Generate text based on sentiment
        product = np.random.choice(products)
        
        if sentiment_with_noise > 0.7:  # Positive review
            adj = np.random.choice(adjectives_positive)
            text = f"This {product} is {adj}. I'm very happy with my purchase. The performance is great and the design is beautiful."
            if sentiment_with_noise > 0.9:
                text += " Would definitely recommend to everyone!"
        elif sentiment_with_noise > 0.4:  # Neutral review
            adj = np.random.choice(adjectives_neutral)
            text = f"This {product} is {adj}. It works as expected. The performance is acceptable but could be better."
        else:  # Negative review
            adj = np.random.choice(adjectives_negative)
            text = f"This {product} is {adj}. I'm not satisfied with my purchase. The performance is lacking and the build quality is questionable."
            if sentiment_with_noise < 0.2:
                text += " Would not recommend to anyone."
        
        # Add some randomness to text length
        if np.random.rand() > 0.7:
            features = ["battery life", "display", "speed", "camera", "storage", "connectivity"]
            feature = np.random.choice(features)
            if sentiment_with_noise > 0.6:
                text += f" The {feature} is particularly impressive."
            elif sentiment_with_noise > 0.4:
                text += f" The {feature} is adequate."
            else:
                text += f" The {feature} is disappointing."
        
        texts.append(text)
    
    # Create target variable (sales) based on sentiment with noise and lag effect
    # Higher sentiment leads to better sales with a delay
    lagged_sentiment = pd.Series(sentiment_scores).shift(7).fillna(0.5).values
    sales = 100 * lagged_sentiment + 50 * np.array(sentiment_scores) + np.random.normal(0, 10, n_samples)
    
    # Combine into a dataframe
    df = pd.DataFrame({
        'date': dates,
        'review_text': texts,
        'sentiment': sentiment_scores,
        'sales': sales
    })
    
    return df


def main():
    print("Text Processing with Time Series Regression Example")
    print("==================================================")
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head(2))
    
    # Initialize TextProcessor with spaCy
    processor = TextProcessor(use_spacy=True)
    
    print("\n1. Basic Text Cleaning")
    # Apply text preprocessing with automatic backend selection
    df = processor.process_dataframe_column(
        df, 
        'review_text',
        result_column='cleaned_text',
        backend='auto',  # Automatically select the best backend
        batch_size=100,  # Process in batches of 100 (for spaCy)
        use_parallel=False,  # Set to True for parallel processing
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True
    )
    
    # You can also benchmark different backends
    print("\nBenchmarking different backends:")
    if len(df) > 20:  # Only run benchmark if we have enough data
        # Create a smaller sample for benchmarking
        benchmark_sample = df.head(min(100, len(df)))
        benchmark_results = processor.benchmark_text_processing(
            benchmark_sample,
            'review_text',
            iterations=2,  # Run 2 iterations for each backend
            lowercase=True,
            remove_punctuation=True
        )
    
    # Show example of cleaning
    print("\nOriginal vs Cleaned Text:")
    for i in range(2):
        print(f"\nOriginal: {df['review_text'].iloc[i]}")
        print(f"Cleaned:  {df['cleaned_text'].iloc[i]}")
    
    print("\n2. Extract Named Entities with spaCy")
    # Extract entities from a sample text
    sample_text = df['review_text'].iloc[0]
    entities = processor.extract_entities(sample_text)
    print(f"\nText: {sample_text}")
    print("Entities found:")
    for entity_type, entity_list in entities.items():
        print(f"  {entity_type}: {entity_list}")
    
    print("\n3. Generate Text Features")
    # Create various text features
    text_features = processor.create_text_features(
        df, 
        'review_text',
        include_stats=True,
        include_readability=True,
        include_sentiment=True
    )
    
    print(f"Generated {text_features.shape[1]} text features")
    print("Feature examples:")
    for col in sorted(text_features.columns)[:5]:
        print(f"  {col}: {text_features[col].iloc[0]:.3f}")
    
    # Create bag-of-words and TF-IDF features
    bow_features = processor.create_bow_features(
        df, 
        'cleaned_text',
        max_features=20,
        binary=False
    )
    
    tfidf_features = processor.create_tfidf_features(
        df, 
        'cleaned_text',
        max_features=20
    )
    
    print(f"\nGenerated {bow_features.shape[1]} bag-of-words features")
    print(f"Generated {tfidf_features.shape[1]} TF-IDF features")
    
    print("\n4. Time Series Feature Engineering")
    # Create time series features manually instead
    from freamon.features.time_series_engineer import create_auto_lag_features, create_auto_rolling_features
    
    # Add lag features
    ts_features = create_auto_lag_features(
        df, 
        target_cols='sales',
        date_col='date',
        max_lags=7, 
        strategy='all'  # Use all lags instead of auto for reliability
    )
    
    # Add rolling features
    ts_features = create_auto_rolling_features(
        ts_features,
        target_cols='sales',
        date_col='date',
        windows=[3, 7, 14],
        metrics=['mean', 'std'],
        auto_detect=False
    )
    
    # Count time series features
    ts_feature_columns = [col for col in ts_features.columns 
                         if col not in df.columns and col.startswith('sales_')]
    print(f"Generated {len(ts_feature_columns)} time series features")
    print("Sample time series features:")
    for col in ts_feature_columns[:3]:
        print(f"  {col}")
    
    print("\n5. Combine Features and Prepare for Modeling")
    # Combine all features
    # Remove rows with NaN values (from lag features)
    X_combined = pd.concat([
        ts_features,
        text_features,
        bow_features.iloc[:, :10],  # Use only top 10 features for simplicity
        tfidf_features.iloc[:, :10]  # Use only top 10 features for simplicity
    ], axis=1)
    X_combined = X_combined.dropna()
    
    # Keep only rows with complete data
    valid_indices = X_combined.index
    y = df.loc[valid_indices, 'sales']
    
    print(f"Final dataset shape: {X_combined.shape}")
    
    # Select features for modeling (exclude original columns)
    feature_columns = [col for col in X_combined.columns 
                      if col not in ['date', 'review_text', 'cleaned_text', 'sentiment', 'sales']]
    
    print("\n6. Time Series Cross-Validation")
    # Define model creation function for cross-validation
    def create_lightgbm_model(**kwargs):
        # Using our helper function for simplicity
        # The time_series_cross_validate function expects a model instance, not a wrapped model
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            **kwargs
        )
    
    # Perform time series cross-validation with prediction saving
    cv_results = time_series_cross_validate(
        X_combined.reset_index(drop=True),  # Reset index after dropping NaN rows
        target_column='sales',
        date_column='date',
        model_fn=create_lightgbm_model,
        n_splits=5,
        problem_type='regression',
        feature_columns=feature_columns,
        expanding_window=True,
        save_predictions=True  # Save predictions for visualization
    )
    
    print("\nCross-validation results:")
    for metric, values in cv_results.items():
        if metric not in ['fold', 'train_size', 'test_size', 'train_start_date', 
                         'train_end_date', 'test_start_date', 'test_end_date',
                         'predictions', 'test_targets', 'test_dates']:
            print(f"  {metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")
    
    # Visualize CV metrics
    fig_metrics = plot_cv_metrics(cv_results)
    plt.savefig('cv_metrics.png')
    plt.close()
    
    # Visualize time series predictions
    fig_ts = plot_cv_predictions_over_time(cv_results)
    plt.savefig('time_series_predictions.png')
    plt.close()
    
    print("\n7. Feature Importance Analysis")
    # Get complete dataset (after dropping NaN values)
    X = X_combined[feature_columns]
    
    # Train a model on all data for feature importance using our helper function
    model_lgb = get_lightgbm_model(
        problem_type='regression',
        n_estimators=100,
        max_depth=3
    )
    
    # If you prefer to use LightGBM directly, you could use this instead:
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        objective='regression',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    
    # Use direct LightGBM model for simplicity
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=feature_columns).sort_values(ascending=False)
    
    # Visualize feature importance
    fig_importance = plot_feature_importance(model, feature_names=feature_columns, top_n=15)
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 important features:")
    for feature, importance in feature_importance.head(10).items():
        print(f"  {feature}: {importance:.4f}")
    
    print("\n8. Feature Type Analysis")
    
    # Define our feature groups for this analysis
    custom_feature_groups = {
        'Text Statistics': ['text_stat_'],
        'Text Readability': ['text_read_'],
        'Text Sentiment': ['text_sent_'],
        'Bag-of-Words': ['bow_'],
        'TF-IDF': ['tfidf_'],
        'Time Series Lag': ['sales_lag_'],
        'Time Series Rolling': ['sales_rolling_'],
    }
    
    # Define custom function for summarizing feature importance by groups
    def sum_importance_by_group(importance_series, groups):
        results = {}
        for group_name, patterns in groups.items():
            group_features = []
            for pattern in patterns:
                matches = [f for f in importance_series.index if pattern in f]
                group_features.extend(matches)
            
            if group_features:
                results[group_name] = importance_series[group_features].sum()
            else:
                results[group_name] = 0
        
        return pd.Series(results).sort_values(ascending=False)
    
    # Calculate importance by group
    group_importance = sum_importance_by_group(feature_importance, custom_feature_groups)
    
    # Create group importance bar plot
    plt.figure(figsize=(10, 6))
    group_importance.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Group Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_groups_bar.png')
    plt.close()
    
    # Create group importance pie chart
    plt.figure(figsize=(10, 6))
    group_importance[group_importance > 0].plot(kind='pie', autopct='%1.1f%%')
    plt.title('Feature Group Importance')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('feature_groups_pie.png')
    plt.close()
    
    print("\nFeature importance by type:")
    for group, importance in group_importance.items():
        print(f"  {group}: {importance:.4f}")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()