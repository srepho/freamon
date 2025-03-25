"""
Example demonstrating the Automated Modeling Workflow.

This example shows how to use the AutoModelFlow class to build a complete
machine learning pipeline that handles text data, time series features, and
cross-validation automatically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from freamon.modeling.autoflow import AutoModelFlow, auto_model

# Example 1: Text classification with 20 Newsgroups dataset
def run_text_classification_example():
    """Run automated modeling on text classification dataset."""
    print("\n" + "="*80)
    print("EXAMPLE 1: TEXT CLASSIFICATION WITH 20 NEWSGROUPS")
    print("="*80)
    
    # Load a subset of the 20 newsgroups dataset
    categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Create a dataframe
    df = pd.DataFrame({
        'text': newsgroups.data,
        'category': [newsgroups.target_names[i] for i in newsgroups.target],
        'target': newsgroups.target
    })
    
    # Take a small sample for this example
    df = df.sample(300, random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Split data for later testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # Create AutoModelFlow 
    model_flow = AutoModelFlow(
        model_type="lightgbm",
        problem_type="classification",
        text_processing=True,
        time_series_features=False,
        hyperparameter_tuning=True
    )
    
    # Fit model
    print("\nFitting model...")
    model_flow.fit(
        df=train_df,
        target_column='target',
        text_columns=['text'],  # Explicitly specify text column
        cv_folds=3,  # Use fewer folds for example
        tuning_options={
            'n_trials': 15,  # Fewer trials for example
            'eval_metric': 'auc'
        }
    )
    
    # Print top features
    print("\nTop 10 Features:")
    print(model_flow.feature_importance.head(10))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds = model_flow.predict(test_df)
    probs = model_flow.predict_proba(test_df)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(test_df['target'], preds)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_df['target'], preds, 
                               target_names=newsgroups.target_names))
    
    # Display topic model results
    print("\nTopic Model Results for 'text' column:")
    text_topics = model_flow.get_topic_terms('text', n_terms=7)
    for topic_id, terms in text_topics.items():
        print(f"Topic {topic_id+1}: {', '.join(terms)}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    model_flow.plot_importance(top_n=15)
    plt.tight_layout()
    plt.savefig("text_model_feature_importance.png")
    print("Feature importance plot saved to 'text_model_feature_importance.png'")
    
    return model_flow

# Example 2: Time series regression with auto-generated features
def run_time_series_example():
    """Run automated modeling on time series dataset."""
    print("\n" + "="*80)
    print("EXAMPLE 2: TIME SERIES REGRESSION")
    print("="*80)
    
    # Create synthetic time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    
    # Create features
    df = pd.DataFrame({
        'date': dates,
        'temperature': np.sin(np.arange(365)/30) * 10 + 20 + np.random.normal(0, 1, 365),
        'humidity': np.cos(np.arange(365)/30) * 10 + 60 + np.random.normal(0, 2, 365),
        'pressure': np.random.normal(1013, 5, 365),
        'textual_notes': [
            f"Weather observation for day {i+1}: {'sunny' if np.random.rand() > 0.5 else 'cloudy'}, " +
            f"{'windy' if np.random.rand() > 0.7 else 'calm'}, " +
            f"{'precipitation' if np.random.rand() > 0.8 else 'dry'}"
            for i in range(365)
        ]
    })
    
    # Target: temperature tomorrow (with some noise and dependency on other features)
    df['target'] = df['temperature'].shift(-1) + df['humidity'] * 0.05 - df['pressure'] * 0.01 + np.random.normal(0, 0.5, 365)
    df = df.dropna()  # Drop the last row with NaN target
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {', '.join(df.columns)}")
    print(f"Target statistics: min={df['target'].min():.2f}, max={df['target'].max():.2f}, mean={df['target'].mean():.2f}")
    
    # Split data for testing
    train_idx = int(len(df) * 0.8)
    train_df = df.iloc[:train_idx].copy()
    test_df = df.iloc[train_idx:].copy()
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Use the simplified auto_model function
    print("\nFitting model using auto_model function...")
    results = auto_model(
        df=train_df,
        target_column='target',
        date_column='date',
        model_type='lightgbm',
        problem_type='regression',
        text_columns=['textual_notes'],
        cv_folds=3,  # Use fewer folds for example
        metrics=['rmse', 'mae', 'r2'],
        tuning=True,
        tuning_options={'n_trials': 10},  # Fewer trials for example
        time_options={
            'create_target_lags': True,
            'lag_periods': [1, 7, 14],
            'rolling_windows': [7, 14]
        },
        verbose=True
    )
    
    # Extract components from results
    model = results['model']
    feature_importance = results['feature_importance']
    metrics = results['metrics']
    autoflow = results['autoflow']  # Get the AutoModelFlow instance
    
    # Print metrics
    print("\nCross-validation metrics:")
    for name, value in metrics.items():
        if name.endswith('_mean'):
            metric = name.replace('_mean', '')
            mean = value
            std = metrics.get(f"{metric}_std", 0)
            print(f"{metric}: {mean:.4f} ± {std:.4f}")
    
    # Print top features
    print("\nTop 10 Features:")
    print(feature_importance.head(10))
    
    # Make predictions on test set
    preds = autoflow.predict(test_df)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(test_df['target'], preds))
    mae = mean_absolute_error(test_df['target'], preds)
    r2 = r2_score(test_df['target'], preds)
    
    print("\nTest set metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions over time
    plt.figure(figsize=(15, 6))
    plt.plot(test_df['date'], test_df['target'], label='Actual', color='blue')
    plt.plot(test_df['date'], preds, label='Predicted', color='red', linestyle='--')
    plt.title('Temperature Predictions')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("time_series_predictions.png")
    print("Time series predictions plot saved to 'time_series_predictions.png'")
    
    # Examine topic model for textual notes
    if 'textual_notes' in results['text_topics']:
        print("\nTopic Model for Weather Notes:")
        text_topics = results['text_topics']['textual_notes']
        for topic_idx, words in text_topics['topics']:
            print(f"Topic {topic_idx+1}: {', '.join(words[:7])}")
    
    return autoflow

if __name__ == "__main__":
    # Run the examples
    text_model = run_text_classification_example()
    time_series_model = run_time_series_example()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
    print("The examples demonstrate how to use AutoModelFlow for:")
    print("1. Automatic text classification with topic modeling")
    print("2. Time series regression with text features and auto-generated time features")
    print("\nBoth examples show the end-to-end workflow from raw data to predictions.")