"""
Test script to demonstrate how to use a trained auto_model for predictions.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_classification
from freamon.modeling.autoflow import auto_model

# 1. Create and train a model
print("Creating and training a model...")
X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    random_state=42
)

# Create training data
train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
train_df['target'] = y

print(f"Training data shape: {train_df.shape}")

# Train the model
results = auto_model(
    df=train_df,
    target_column='target',
    problem_type='classification',
    model_type='lgbm_classifier',
    cv_folds=2,  # Use minimal CV for speed
    tuning=False,  # No tuning for speed
    random_state=42
)

# Print model results
print("\nTraining completed. Test metrics:")
for metric, value in results['test_metrics'].items():
    print(f"{metric}: {value:.4f}")

# 2. Create new prediction data
print("\nCreating new data for predictions...")
X_new, _ = make_classification(
    n_samples=10,
    n_features=5,
    n_informative=3,
    random_state=100  # Different random state for new data
)

# Create prediction data (without target)
new_df = pd.DataFrame(X_new, columns=[f'feature_{i}' for i in range(5)])
print(f"New data shape: {new_df.shape}")
print(f"New data sample:\n{new_df.head(3)}")

# 3. Method 1: Make predictions using the model directly
print("\nMETHOD 1: Using model directly")
model = results['model']
predictions = model.predict(new_df)
probabilities = model.predict_proba(new_df)

print("Predictions:", predictions[:5])
print("Probabilities:\n", probabilities[:5])

# 4. Method 2: Make predictions using the AutoModelFlow instance
print("\nMETHOD 2: Using AutoModelFlow instance")
autoflow = results['autoflow']
predictions_2 = autoflow.predict(new_df)
probabilities_2 = autoflow.predict_proba(new_df)

print("Predictions:", predictions_2[:5])
print("Probabilities:\n", probabilities_2[:5])

# 5. Save and reload the model
print("\nSaving and reloading the model...")
with open('temp_autoflow_model.pkl', 'wb') as f:
    pickle.dump(autoflow, f)

with open('temp_autoflow_model.pkl', 'rb') as f:
    loaded_autoflow = pickle.load(f)

# Make predictions with the loaded model
print("\nMaking predictions with loaded model:")
predictions_3 = loaded_autoflow.predict(new_df)
print("Predictions:", predictions_3[:5])

# Clean up
import os
os.remove('temp_autoflow_model.pkl')
print("\nTest completed successfully!")