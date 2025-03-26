"""
Simplified test script for auto_model functionality in a conda environment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from freamon.modeling.autoflow import auto_model

def test_simple_classification():
    """Test auto_model with a simple classification problem."""
    print("\n" + "="*80)
    print("TESTING AUTO_MODEL CLASSIFICATION")
    print("="*80)
    
    # Create a smaller synthetic classification dataset
    X, y = make_classification(
        n_samples=100,  # Smaller dataset
        n_features=5,   # Fewer features
        n_informative=3,
        random_state=42
    )
    
    # Convert to DataFrame
    columns = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Run auto_model with minimal settings
    print("\nRunning auto_model for classification...")
    results = auto_model(
        df=df,
        target_column='target',
        problem_type='classification',
        model_type='lgbm_classifier',
        metrics=['accuracy'],
        tuning=False,  # Disable tuning for speed
        cv_folds=2,    # Minimal CV
        verbose=True,
        random_state=42
    )
    
    # Print test metrics
    print("\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(8, 5))
    results['autoflow'].plot_importance()
    plt.tight_layout()
    plt.savefig("classification_feature_importance.png")
    print("Feature importance plot saved to 'classification_feature_importance.png'")
    
    return results

if __name__ == "__main__":
    print(f"Python version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    # Run simplified test
    classification_results = test_simple_classification()
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("The test verified:")
    print("1. Basic classification functionality")
    print("2. Feature importance visualization")
    print("\nPlease check the generated PNG file for visual confirmation.")