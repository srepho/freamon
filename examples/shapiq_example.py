"""
Example of using ShapIQ for feature importance and interaction detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from freamon.explainability import ShapExplainer, ShapIQExplainer
from freamon.modeling import ModelTrainer

# Create a sample dataset with interactions
def create_interaction_data(n_samples=500):
    np.random.seed(42)
    
    # Create four features
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    x4 = np.random.uniform(-1, 1, n_samples)
    
    # Create a target with strong interaction between x1 and x2
    y = (
        x1 * x2 +          # Interaction term
        0.3 * x3 +         # Main effect term
        0.1 * x4 +         # Weak main effect
        0.1 * np.random.normal(0, 1, n_samples)  # Noise
    )
    
    # Create a DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'target': y
    })
    
    return df

def run_shap_example():
    print("\n===== SHAP Example =====")
    
    # Create data
    df = create_interaction_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = ShapExplainer(model, model_type='tree')
    explainer.fit(X_train)
    
    # Calculate SHAP values
    shap_values = explainer.explain(X_test.iloc[:10])  # for first 10 test instances
    
    print("SHAP values for the first test instance:")
    print(shap_values.iloc[0].sort_values(ascending=False))
    
    # Generate summary plot
    plt.figure(figsize=(10, 6))
    explainer.summary_plot(shap_values, X_test.iloc[:10], plot_type='bar')
    
    print("\nSHAP summary plot generated. Check your display for the visualization.")

def run_shapiq_example():
    print("\n===== ShapIQ Example =====")
    
    # Create data
    df = create_interaction_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create ShapIQ explainer
    explainer = ShapIQExplainer(model, max_order=2)
    explainer.fit(X_train)
    
    # Calculate interaction values
    interactions = explainer.explain(X_test.iloc[:5])  # for first 5 test instances
    
    # Plot main effects
    plt.figure(figsize=(10, 6))
    explainer.plot_main_effects(instance_idx=0)
    
    print("\nShapIQ main effects plot generated. Check your display for the visualization.")
    
    # Plot interaction effects
    plt.figure(figsize=(10, 6))
    explainer.plot_interaction_effects(instance_idx=0)
    
    print("\nShapIQ interaction effects plot generated. Check your display for the visualization.")

def run_model_trainer_example():
    print("\n===== Model Trainer with ShapIQ Example =====")
    
    # Create data
    df = create_interaction_data()
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Define features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Create a model trainer
    trainer = ModelTrainer(
        model_type='sklearn',
        model_name='RandomForestRegressor',
        problem_type='regression',
        params={'n_estimators': 100},
        random_state=42
    )
    
    # Train the model
    trainer.train(X_train, y_train)
    
    # Evaluate the model
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Model metrics: {metrics}")
    
    # Get native feature importance
    native_importance = trainer.get_feature_importance()
    print("\nNative feature importance:")
    print(native_importance)
    
    # Get SHAP-based feature importance
    try:
        shap_importance = trainer.get_feature_importance(method='shap', X=X_train)
        print("\nSHAP-based feature importance:")
        print(shap_importance)
    except ImportError:
        print("\nSHAP package not available, skipping SHAP-based importance")
    
    # Get ShapIQ-based feature importance
    try:
        shapiq_importance = trainer.get_feature_importance(method='shapiq', X=X_train)
        print("\nShapIQ-based feature importance:")
        print(shapiq_importance)
    except ImportError:
        print("\nShapIQ package not available, skipping ShapIQ-based importance")

if __name__ == "__main__":
    # Run the examples
    run_shap_example()
    run_shapiq_example()
    run_model_trainer_example()