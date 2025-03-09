"""
Example of using ShapIQ for feature engineering and reporting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Import Freamon components
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
from freamon.eda.explainability_report import generate_interaction_report

# Set random seed for reproducibility
np.random.seed(42)

def create_interaction_data(n_samples=1000, noise_level=0.5):
    """
    Create a synthetic dataset with feature interactions.
    """
    # Generate 5 features
    X = pd.DataFrame({
        'feature1': np.random.uniform(-1, 1, n_samples),
        'feature2': np.random.uniform(-1, 1, n_samples),
        'feature3': np.random.uniform(-1, 1, n_samples),
        'feature4': np.random.uniform(-1, 1, n_samples),
        'feature5': np.random.uniform(-1, 1, n_samples),
    })
    
    # Create target with interactions
    y = (
        X['feature1'] * X['feature2'] +               # Strong interaction
        0.5 * X['feature2'] * X['feature4'] +         # Medium interaction
        0.3 * X['feature1'] +                         # Main effect
        0.2 * X['feature5'] +                         # Weak main effect
        0.1 * X['feature1'] * X['feature3'] +         # Weak interaction
        noise_level * np.random.normal(0, 1, n_samples)  # Noise
    )
    
    return X, y

def plot_scatter(X, y, title):
    """
    Create a scatter plot showing the feature interaction.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        X['feature1'], X['feature2'], c=y, cmap='viridis', 
        alpha=0.7, s=50, edgecolors='k', linewidths=0.5
    )
    plt.colorbar(label='Target Value')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    print("Generating synthetic data with feature interactions...")
    X, y = create_interaction_data()
    
    print("Data shape:", X.shape)
    print("Features:", X.columns.tolist())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a basic model without interaction features
    print("\nTraining base model without interaction features...")
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    base_score = base_model.score(X_test, y_test)
    print(f"Base model R² score: {base_score:.4f}")
    
    # Show the interacting features
    plot_scatter(X, y, 'Feature Interaction in Synthetic Data')
    
    # Use ShapIQ to detect interactions
    print("\nUsing ShapIQ to detect feature interactions...")
    shapiq_engineer = ShapIQFeatureEngineer(
        model=base_model,
        X=X_train,
        y=y_train,
        threshold=0.02,  # Lower threshold to capture more interactions
        max_interactions=5
    )
    
    # Detect interactions
    interactions = shapiq_engineer.detect_interactions()
    print(f"Detected {len(interactions)} significant interactions:")
    
    for feature1, feature2 in interactions:
        strength = shapiq_engineer.interaction_strengths[(feature1, feature2)]
        print(f"  {feature1} × {feature2}: {strength:.4f}")
    
    # Create features using detected interactions
    print("\nCreating engineered features based on detected interactions...")
    X_train_enhanced, interaction_report = shapiq_engineer.pipeline(X_train)
    X_test_enhanced = shapiq_engineer.create_features(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")
    print(f"New features: {X_train_enhanced.columns.tolist()[X_train.shape[1]:]}")
    
    # Train a model with interaction features
    print("\nTraining enhanced model with interaction features...")
    enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
    enhanced_model.fit(X_train_enhanced, y_train)
    enhanced_score = enhanced_model.score(X_test_enhanced, y_test)
    print(f"Enhanced model R² score: {enhanced_score:.4f}")
    print(f"Improvement: {(enhanced_score - base_score) * 100:.2f}%")
    
    # Generate HTML report
    print("\nGenerating interaction analysis report...")
    report_path = "interaction_report.html"
    generate_interaction_report(
        model=base_model,
        X=X_train,
        y=y_train,
        output_path=report_path,
        threshold=0.02,
        max_interactions=5
    )
    print(f"Report saved to {report_path}")
    
    print("\nFeature importance from enhanced model:")
    feature_importances = pd.Series(
        enhanced_model.feature_importances_,
        index=X_train_enhanced.columns
    ).sort_values(ascending=False)
    
    print(feature_importances.head(10))

if __name__ == "__main__":
    main()