"""
Example script demonstrating the full ShapIQ integration in Freamon.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt

# Import Freamon components
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
from freamon.explainability import ShapIQExplainer
from freamon.eda.explainability_report import generate_interaction_report

# Set random seed for reproducibility
np.random.seed(42)

def diabetes_example():
    """Demonstrate ShapIQ with regression on diabetes dataset."""
    print("\n=== ShapIQ with Diabetes Dataset (Regression) ===")
    
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    print("Training baseline model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate baseline model
    baseline_score = model.score(X_test, y_test)
    print(f"Baseline model R² score: {baseline_score:.4f}")
    
    # Initialize ShapIQ feature engineer
    print("\nUsing ShapIQ to detect feature interactions...")
    shapiq_engineer = ShapIQFeatureEngineer(
        model=model,
        X=X_train,
        y=y_train,
        threshold=0.01,
        max_interactions=5
    )
    
    # Detect interactions
    interactions = shapiq_engineer.detect_interactions()
    
    # Print detected interactions
    print(f"Detected {len(interactions)} significant interactions:")
    for feature1, feature2 in interactions:
        strength = shapiq_engineer.interaction_strengths[(feature1, feature2)]
        print(f"  {feature1} × {feature2}: {strength:.4f}")
    
    # Create enhanced features
    print("\nCreating engineered features based on interactions...")
    X_train_enhanced, interaction_report = shapiq_engineer.pipeline(
        operations=['multiply', 'ratio']
    )
    X_test_enhanced = shapiq_engineer.create_features(X_test)
    
    # Print feature information
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")
    print(f"New features: {list(set(X_train_enhanced.columns) - set(X_train.columns))}")
    
    # Train an enhanced model
    print("\nTraining enhanced model with interaction features...")
    enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
    enhanced_model.fit(X_train_enhanced, y_train)
    
    # Evaluate enhanced model
    enhanced_score = enhanced_model.score(X_test_enhanced, y_test)
    print(f"Enhanced model R² score: {enhanced_score:.4f}")
    print(f"Improvement: {(enhanced_score - baseline_score) * 100:.2f}%")
    
    # Generate HTML report
    print("\nGenerating interaction analysis report...")
    report_path = "diabetes_interactions.html"
    generate_interaction_report(
        model=model,
        X=X_train,
        y=y_train,
        output_path=report_path,
        threshold=0.01,
        max_interactions=5
    )
    print(f"Report saved to {report_path}")

def wine_example():
    """Demonstrate ShapIQ with classification on wine dataset."""
    print("\n=== ShapIQ with Wine Dataset (Classification) ===")
    
    # Load the wine dataset
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    print("Training baseline model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate baseline model
    baseline_score = model.score(X_test, y_test)
    print(f"Baseline model accuracy: {baseline_score:.4f}")
    
    # Initialize ShapIQ feature engineer
    print("\nUsing ShapIQ to detect feature interactions...")
    shapiq_engineer = ShapIQFeatureEngineer(
        model=model,
        X=X_train,
        y=y_train,
        threshold=0.03,
        max_interactions=8
    )
    
    # Detect interactions
    interactions = shapiq_engineer.detect_interactions()
    
    # Print detected interactions
    print(f"Detected {len(interactions)} significant interactions:")
    for feature1, feature2 in interactions:
        strength = shapiq_engineer.interaction_strengths[(feature1, feature2)]
        print(f"  {feature1} × {feature2}: {strength:.4f}")
    
    # Create enhanced features
    print("\nCreating engineered features based on interactions...")
    X_train_enhanced, _ = shapiq_engineer.pipeline(
        operations=['multiply', 'ratio', 'add']
    )
    X_test_enhanced = shapiq_engineer.create_features(X_test)
    
    # Print feature information
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")
    
    # Train an enhanced model
    print("\nTraining enhanced model with interaction features...")
    enhanced_model = RandomForestClassifier(n_estimators=100, random_state=42)
    enhanced_model.fit(X_train_enhanced, y_train)
    
    # Evaluate enhanced model
    enhanced_score = enhanced_model.score(X_test_enhanced, y_test)
    print(f"Enhanced model accuracy: {enhanced_score:.4f}")
    print(f"Improvement: {(enhanced_score - baseline_score) * 100:.2f}%")
    
    # Use ShapIQExplainer for additional analysis
    print("\nUsing ShapIQExplainer for advanced feature interaction analysis...")
    explainer = ShapIQExplainer(model, max_order=2)
    explainer.fit(X_train)
    
    # Explain a few instances
    print("Explaining 5 test instances...")
    interactions = explainer.explain(X_test.iloc[:5])
    
    # Plot main effects for first instance
    plt.figure(figsize=(12, 6))
    explainer.plot_main_effects(instance_idx=0, top_k=8)
    plt.tight_layout()
    plt.savefig("wine_main_effects.png")
    print("Main effects plot saved as wine_main_effects.png")
    
    # Plot interaction effects for first instance
    plt.figure(figsize=(12, 6))
    explainer.plot_interaction_effects(instance_idx=0, top_k=8)
    plt.tight_layout()
    plt.savefig("wine_interaction_effects.png")
    print("Interaction effects plot saved as wine_interaction_effects.png")

if __name__ == "__main__":
    # Run the examples
    diabetes_example()
    wine_example()
    
    print("\nCompleted all ShapIQ examples!")