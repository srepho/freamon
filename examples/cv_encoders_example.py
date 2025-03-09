"""
Example demonstrating cross-validated target encoding and time-series cross-validation.

This example shows:
1. How to use cross-validated target encoding to prevent target leakage
2. How to use time-series walk-forward validation for model evaluation
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

from freamon.utils.encoders import TargetEncoderWrapper
from freamon.model_selection.cross_validation import (
    cross_validate, 
    time_series_cross_validate,
    walk_forward_validation
)

# ===============================
# Synthetic Dataset Generation
# ===============================

def generate_categorical_dataset(n_samples=1000, n_categories=5, random_state=42):
    """Generate a synthetic dataset with categorical features."""
    np.random.seed(random_state)
    
    # Generate categorical features
    categories = [f'cat_{i}' for i in range(n_categories)]
    cat_probs = np.random.dirichlet(np.ones(n_categories), 1)[0]
    
    category_feature = np.random.choice(categories, size=n_samples, p=cat_probs)
    
    # Generate additional features
    num_feature1 = np.random.normal(0, 1, n_samples)
    num_feature2 = np.random.normal(0, 1, n_samples)
    
    # Create category-specific effects (this creates the relationship for target encoding)
    category_effects = {cat: np.random.normal(i, 0.5) for i, cat in enumerate(categories)}
    
    # Generate target with category effect and some noise
    target = np.array([category_effects[cat] for cat in category_feature]) + \
             0.5 * num_feature1 - 0.3 * num_feature2 + \
             np.random.normal(0, 0.5, n_samples)
    
    # Convert to binary for classification examples
    binary_target = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'category': category_feature,
        'numeric1': num_feature1,
        'numeric2': num_feature2,
        'target_reg': target,
        'target_cls': binary_target
    })
    
    return df

def generate_time_series_dataset(n_samples=365, n_categories=3, random_state=42):
    """Generate a synthetic time series dataset with categorical features."""
    np.random.seed(random_state)
    
    # Generate dates
    dates = pd.date_range(start='2022-01-01', periods=n_samples)
    
    # Generate categorical features
    categories = [f'cat_{i}' for i in range(n_categories)]
    cat_probs = np.random.dirichlet(np.ones(n_categories), 1)[0]
    
    category_feature = np.random.choice(categories, size=n_samples, p=cat_probs)
    
    # Generate numeric features with time dependency
    time_trend = np.linspace(0, 5, n_samples)  # Increasing trend
    seasonality = 2 * np.sin(np.linspace(0, 12*np.pi, n_samples))  # Yearly seasonality
    
    num_feature1 = time_trend + seasonality + np.random.normal(0, 0.5, n_samples)
    num_feature2 = 0.5 * time_trend - 0.3 * seasonality + np.random.normal(0, 0.5, n_samples)
    
    # Create category-specific effects
    category_effects = {cat: np.random.normal(i, 0.5) for i, cat in enumerate(categories)}
    
    # Generate target with time dependency, category effect, and features
    target = time_trend + 0.8 * seasonality + \
             np.array([category_effects[cat] for cat in category_feature]) + \
             0.3 * num_feature1 - 0.2 * num_feature2 + \
             np.random.normal(0, 0.7, n_samples)
    
    # Convert to binary for classification examples
    binary_target = (target > np.median(target)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'category': category_feature,
        'numeric1': num_feature1,
        'numeric2': num_feature2,
        'target_reg': target,
        'target_cls': binary_target
    })
    
    return df

# ===============================
# Target Encoding Example
# ===============================

def demonstrate_target_encoding():
    """Demonstrate the impact of cross-validated target encoding."""
    print("\n== Target Encoding Demonstration ==")
    
    # Generate dataset
    df = generate_categorical_dataset(n_samples=1000, n_categories=10)
    print(f"Dataset shape: {df.shape}")
    
    # Split into train and test
    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Evaluate with different target encoding methods
    results = {}
    
    # 1. Without cross-validation (potential leakage)
    encoder_without_cv = TargetEncoderWrapper(
        columns=['category'],
        smoothing=10.0,
        cv=0  # Disable CV
    )
    train_encoded = encoder_without_cv.fit_transform(train_df, target='target_reg')
    test_encoded = encoder_without_cv.transform(test_df)
    
    # Train model
    model = LinearRegression()
    model.fit(train_encoded[['category', 'numeric1', 'numeric2']], train_encoded['target_reg'])
    
    # Evaluate
    test_pred = model.predict(test_encoded[['category', 'numeric1', 'numeric2']])
    mse_without_cv = ((test_encoded['target_reg'] - test_pred) ** 2).mean()
    results['Without CV'] = mse_without_cv
    
    # 2. With cross-validation (prevents leakage)
    encoder_with_cv = TargetEncoderWrapper(
        columns=['category'],
        smoothing=10.0,
        cv=5  # 5-fold CV
    )
    train_encoded_cv = encoder_with_cv.fit_transform(train_df, target='target_reg')
    test_encoded_cv = encoder_with_cv.transform(test_df)
    
    # Train model
    model_cv = LinearRegression()
    model_cv.fit(train_encoded_cv[['category', 'numeric1', 'numeric2']], train_encoded_cv['target_reg'])
    
    # Evaluate
    test_pred_cv = model_cv.predict(test_encoded_cv[['category', 'numeric1', 'numeric2']])
    mse_with_cv = ((test_encoded_cv['target_reg'] - test_pred_cv) ** 2).mean()
    results['With CV'] = mse_with_cv
    
    # 3. Without encoding (baseline)
    # For this, we'll use one-hot encoding (implicit in pandas get_dummies)
    train_dummies = pd.get_dummies(train_df, columns=['category'], drop_first=True)
    test_dummies = pd.get_dummies(test_df, columns=['category'], drop_first=True)
    
    # Ensure test has all columns from train
    for col in train_dummies.columns:
        if col not in test_dummies.columns and col != 'target_reg' and col != 'target_cls':
            test_dummies[col] = 0
    
    # Use only columns present in training data
    feature_cols = [col for col in train_dummies.columns 
                   if col != 'target_reg' and col != 'target_cls']
    
    # Train model
    model_baseline = LinearRegression()
    model_baseline.fit(train_dummies[feature_cols], train_dummies['target_reg'])
    
    # Evaluate
    test_pred_baseline = model_baseline.predict(test_dummies[feature_cols])
    mse_baseline = ((test_dummies['target_reg'] - test_pred_baseline) ** 2).mean()
    results['One-Hot Encoding'] = mse_baseline
    
    # Compare results
    print("\nMean Squared Error (MSE) Comparison:")
    for method, mse in results.items():
        print(f"{method}: {mse:.4f}")
    
    # Calculate improvement percentage
    if mse_without_cv > mse_with_cv:
        improvement = (mse_without_cv - mse_with_cv) / mse_without_cv * 100
        print(f"\nCross-validated target encoding improved MSE by {improvement:.2f}% "
              f"compared to non-CV target encoding")
    else:
        print("\nIn this random seed, CV didn't improve performance. Try another seed.")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Mean Squared Error by Encoding Method')
    plt.ylabel('MSE (lower is better)')
    plt.tight_layout()
    plt.show()

# ===============================
# Time-Series Cross-Validation Example
# ===============================

def demonstrate_time_series_cv():
    """Demonstrate time-series cross-validation and walk-forward validation."""
    print("\n== Time-Series Cross-Validation Demonstration ==")
    
    # Generate time-series dataset
    df = generate_time_series_dataset(n_samples=730)  # 2 years of data
    print(f"Time series dataset shape: {df.shape}")
    
    # Define model creation function
    def create_model(**kwargs):
        return LinearRegression(**kwargs)
    
    # 1. Standard K-Fold Cross-Validation (incorrect for time series)
    std_cv_results = cross_validate(
        df=df,
        target_column='target_reg',
        model_fn=create_model,
        n_splits=5,
        shuffle=True,  # This is wrong for time series!
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    # 2. Time-Series Cross-Validation
    ts_cv_results = time_series_cross_validate(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=create_model,
        n_splits=5,
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    # 3. Time-Series CV with Expanding Window
    ts_cv_expanding_results = time_series_cross_validate(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=create_model,
        n_splits=5,
        expanding_window=True,  # Use expanding window
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    # 4. Walk-Forward Validation (most realistic for time series)
    wf_results = walk_forward_validation(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=create_model,
        initial_train_size='6M',  # Use first 6 months as initial training
        test_size='1M',           # Test on 1 month at a time
        step_size='1M',           # Move forward 1 month each iteration
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    # Compare results
    print("\nMean Squared Error (MSE) Comparison:")
    print(f"Standard CV (incorrect for time series): {np.mean(std_cv_results['mse']):.4f}")
    print(f"Time-Series CV: {np.mean(ts_cv_results['mse']):.4f}")
    print(f"Time-Series CV with Expanding Window: {np.mean(ts_cv_expanding_results['mse']):.4f}")
    print(f"Walk-Forward Validation: {np.mean(wf_results['mse']):.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    methods = ['Standard CV\n(incorrect)', 'Time-Series CV', 'TS CV with\nExpanding Window', 'Walk-Forward\nValidation']
    mse_values = [
        np.mean(std_cv_results['mse']),
        np.mean(ts_cv_results['mse']),
        np.mean(ts_cv_expanding_results['mse']),
        np.mean(wf_results['mse'])
    ]
    plt.bar(methods, mse_values)
    plt.title('Mean MSE by Validation Method')
    plt.ylabel('MSE (lower is better)')
    plt.xticks(rotation=45)
    
    # Visualize walk-forward results over time
    plt.subplot(1, 2, 2)
    plt.plot(wf_results['test_end_date'], wf_results['mse'], marker='o')
    plt.title('Walk-Forward Validation MSE Over Time')
    plt.xlabel('Test End Date')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ===============================
# Combined Example
# ===============================

def combined_time_series_with_encoding():
    """Demonstrate time-series cross-validation with target encoding."""
    print("\n== Combined Time-Series and Target Encoding Example ==")
    
    # Generate time-series dataset
    df = generate_time_series_dataset(n_samples=730, n_categories=10)
    print(f"Time series dataset shape: {df.shape}")
    
    # Function to create a model with target encoding
    def create_model_with_encoding(cv_for_encoding=5, **kwargs):
        """Create a pipeline with target encoding and a linear model."""
        def model_fn(X_train, y_train, X_test):
            # Apply target encoding
            encoder = TargetEncoderWrapper(
                columns=['category'],
                smoothing=10.0, 
                cv=cv_for_encoding
            )
            
            # Encode training data
            X_train_encoded = encoder.fit_transform(
                pd.DataFrame({'category': X_train['category'], 'target': y_train}),
                target='target'
            )
            
            # Encode test data
            X_test_encoded = encoder.transform(
                pd.DataFrame({'category': X_test['category']})
            )
            
            # Combine with other features
            X_train_final = pd.concat([
                X_train_encoded[['category']],
                X_train[['numeric1', 'numeric2']]
            ], axis=1)
            
            X_test_final = pd.concat([
                X_test_encoded[['category']],
                X_test[['numeric1', 'numeric2']]
            ], axis=1)
            
            # Train model
            model = LinearRegression(**kwargs)
            model.fit(X_train_final, y_train)
            
            return model, X_test_final
        
        return model_fn
    
    # Run walk-forward validation with different encoding approaches
    results = {}
    
    # 1. With cross-validated target encoding
    def model_with_cv_encoding(**kwargs):
        return create_model_with_encoding(cv_for_encoding=5, **kwargs)
    
    cv_encoding_results = walk_forward_validation(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=model_with_cv_encoding,
        initial_train_size='6M',
        test_size='1M',
        step_size='1M',
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    results['With CV Encoding'] = np.mean(cv_encoding_results['mse'])
    
    # 2. Without cross-validation in target encoding (potential leakage)
    def model_without_cv_encoding(**kwargs):
        return create_model_with_encoding(cv_for_encoding=0, **kwargs)
    
    no_cv_encoding_results = walk_forward_validation(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=model_without_cv_encoding,
        initial_train_size='6M',
        test_size='1M',
        step_size='1M',
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    results['Without CV Encoding'] = np.mean(no_cv_encoding_results['mse'])
    
    # 3. One-hot encoding baseline
    def model_with_onehot(**kwargs):
        def model_fn(X_train, y_train, X_test):
            # Apply one-hot encoding
            X_train_encoded = pd.get_dummies(X_train, columns=['category'], drop_first=True)
            X_test_encoded = pd.get_dummies(X_test, columns=['category'], drop_first=True)
            
            # Ensure test has all columns from train
            for col in X_train_encoded.columns:
                if col not in X_test_encoded.columns:
                    X_test_encoded[col] = 0
                    
            # Use only columns present in both
            common_cols = [c for c in X_train_encoded.columns if c in X_test_encoded.columns]
            
            # Train model
            model = LinearRegression(**kwargs)
            model.fit(X_train_encoded[common_cols], y_train)
            
            return model, X_test_encoded[common_cols]
        
        return model_fn
    
    onehot_results = walk_forward_validation(
        df=df,
        target_column='target_reg',
        date_column='date',
        model_fn=model_with_onehot,
        initial_train_size='6M',
        test_size='1M',
        step_size='1M',
        problem_type='regression',
        feature_columns=['category', 'numeric1', 'numeric2']
    )
    
    results['One-Hot Encoding'] = np.mean(onehot_results['mse'])
    
    # Compare results
    print("\nMean Squared Error (MSE) Comparison:")
    for method, mse in results.items():
        print(f"{method}: {mse:.4f}")
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), results.values())
    plt.title('Mean MSE by Encoding Method')
    plt.ylabel('MSE (lower is better)')
    plt.xticks(rotation=45)
    
    # Plot MSE over time for the CV encoding approach
    plt.subplot(1, 2, 2)
    plt.plot(cv_encoding_results['test_end_date'], cv_encoding_results['mse'], 
             marker='o', label='With CV Encoding')
    plt.plot(no_cv_encoding_results['test_end_date'], no_cv_encoding_results['mse'], 
             marker='s', label='Without CV Encoding')
    plt.title('MSE Over Time')
    plt.xlabel('Test End Date')
    plt.ylabel('MSE')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# ===============================
# Main
# ===============================

if __name__ == "__main__":
    demonstrate_target_encoding()
    demonstrate_time_series_cv()
    combined_time_series_with_encoding()