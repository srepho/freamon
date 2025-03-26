"""
Test script to run the classification example from README_AUTO_MODEL.md
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from freamon.modeling.autoflow import auto_model

# Create synthetic dataset for classification
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Sample data:\n{df.head()}")

# Run auto_model for classification per README example
print("\nRunning auto_model for classification...")
try:
    results = auto_model(
        df=df,
        target_column='target',
        problem_type='classification',
        model_type='lgbm_classifier',
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        tuning=False,  # Set to True to enable hyperparameter tuning
        random_state=42
    )

    # Access results
    model = results['model']  # The trained model
    metrics = results['test_metrics']  # Evaluation metrics
    importance = results['feature_importance']  # Feature importance

    # Print test metrics
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    print(importance.head(10))

    # Plot feature importance manually using the importance DataFrame
    plt.figure(figsize=(10, 6))
    sorted_importance = importance.sort_values(by='importance', ascending=False).head(10)
    plt.barh(sorted_importance['feature'], sorted_importance['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig("readme_example_feature_importance.png")
    print("\nFeature importance plot saved to 'readme_example_feature_importance.png'")
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()