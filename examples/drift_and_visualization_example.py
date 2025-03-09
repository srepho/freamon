"""Example demonstrating data drift detection and pipeline visualization.

This example shows how to:
1. Detect data drift between two datasets
2. Create and visualize an ML pipeline
3. Generate interactive HTML reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from freamon.data_quality import DataDriftDetector
from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    EvaluationStep,
    visualize_pipeline,
    generate_interactive_html
)


def generate_sample_data(n_rows=1000, seed=42, drift=False):
    """Generate sample data for demonstration.
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed
        drift: Whether to add drift to the data
    
    Returns:
        DataFrame with sample data
    """
    np.random.seed(seed)
    
    if drift:
        # Data with drift (different distributions)
        mean_age = 45
        std_age = 18
        income_shape = 2.2
        purchase_probs = [0.4, 0.4, 0.2]
        start_date = datetime(2023, 6, 1)
    else:
        # Reference data
        mean_age = 35
        std_age = 15
        income_shape = 1.8
        purchase_probs = [0.7, 0.2, 0.1]
        start_date = datetime(2023, 1, 1)
    
    data = pd.DataFrame({
        'age': np.random.normal(mean_age, std_age, n_rows).clip(18, 90).astype(int),
        'income': np.random.exponential(income_shape, n_rows) * 10000 + 20000,
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_rows),
        'purchase_frequency': np.random.choice(['Low', 'Medium', 'High'], n_rows, p=purchase_probs),
        'signup_date': [start_date + timedelta(days=x % 180) for x in range(n_rows)],
        'last_purchase_amount': np.random.gamma(2, 50, n_rows),
    })
    
    # Add some calculated features
    data['income_per_age'] = data['income'] / data['age']
    
    # Add target variable
    purchase_prob = (
        (data['age'] - 18) / 50 * 0.3 +
        (data['income'] / 100000) * 0.5 +
        (data['education'].map({'High School': 0, 'Bachelor': 0.3, 'Master': 0.6, 'PhD': 1})) * 0.2
    )
    
    if drift:
        # Add noise to make the relationship different
        purchase_prob = purchase_prob * 0.7 + np.random.random(n_rows) * 0.3
    
    data['purchase_probability'] = purchase_prob.clip(0, 1)
    data['purchased'] = (np.random.random(n_rows) < data['purchase_probability']).astype(int)
    
    return data


def detect_and_report_drift():
    """Detect data drift and generate a report."""
    print("Generating sample data...")
    reference_data = generate_sample_data(seed=42, drift=False)
    current_data = generate_sample_data(seed=43, drift=True)
    
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Current data shape: {current_data.shape}")
    
    # Initialize drift detector
    print("\nInitializing drift detector...")
    detector = DataDriftDetector(reference_data, current_data)
    
    # Detect drift in all features
    print("Detecting data drift...")
    results = detector.detect_all_drift()
    
    # Print summary
    summary = results['summary']['dataset_summary']
    print(f"\nDrift Summary:")
    print(f"- Total features analyzed: {summary['total_features']}")
    print(f"- Features with drift: {summary['total_drifting']} ({summary['drift_percentage']:.1f}%)")
    print(f"- Numeric features with drift: {summary['drifting_numeric']}/{summary['numeric_features']}")
    print(f"- Categorical features with drift: {summary['drifting_categorical']}/{summary['categorical_features']}")
    print(f"- Datetime features with drift: {summary['drifting_datetime']}/{summary['datetime_features']}")
    
    # Print top drifting features
    print("\nTop drifting features:")
    for i, feature in enumerate(results['summary']['drifting_features'][:5]):
        p_value = feature.get('p_value')
        if p_value is not None:
            print(f"{i+1}. {feature['feature']} ({feature['type']}): p-value = {p_value:.4f}")
        else:
            print(f"{i+1}. {feature['feature']} ({feature['type']})")
    
    # Generate HTML report
    print("\nGenerating drift report...")
    os.makedirs("reports", exist_ok=True)
    detector.generate_drift_report("reports/drift_report.html", "Data Drift Analysis Report")
    print("Drift report saved to reports/drift_report.html")


def create_and_visualize_pipeline():
    """Create an ML pipeline and visualize it."""
    print("\nCreating example pipeline...")
    
    # Create feature engineering step
    feature_step = FeatureEngineeringStep(name="feature_engineering")
    feature_step.add_operation(
        method="add_polynomial_features",
        columns=["age", "income"],
        degree=2,
        interaction_only=True
    )
    feature_step.add_operation(
        method="add_binned_features",
        columns=["income"],
        n_bins=5,
        strategy="quantile"
    )
    
    # Create feature selection step
    select_step = FeatureSelectionStep(
        name="feature_selection",
        method="model_based",
        n_features=10,
        features_to_keep=["age", "income"]
    )
    
    # Create model training step
    model_step = ModelTrainingStep(
        name="model_training",
        model_type="lightgbm",
        problem_type="classification",
        eval_metric="auc",
        hyperparameters={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100
        }
    )
    
    # Create evaluation step
    eval_step = EvaluationStep(
        name="evaluation",
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        problem_type="classification"
    )
    
    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_step(feature_step)
    pipeline.add_step(select_step)
    pipeline.add_step(model_step)
    pipeline.add_step(eval_step)
    
    # Generate basic visualization
    print("Generating pipeline visualization...")
    os.makedirs("reports", exist_ok=True)
    visualize_pipeline(pipeline, output_path="reports/pipeline_viz", format="png")
    print("Pipeline visualization saved to reports/pipeline_viz.png")
    
    # Generate interactive HTML report
    print("Generating interactive HTML visualization...")
    generate_interactive_html(pipeline, output_path="reports/pipeline_report.html")
    print("Interactive visualization saved to reports/pipeline_report.html")


def main():
    """Run the example."""
    print("=== Data Drift Detection Example ===")
    detect_and_report_drift()
    
    print("\n=== Pipeline Visualization Example ===")
    create_and_visualize_pipeline()
    
    print("\nExample completed. Check the reports directory for output files.")


if __name__ == "__main__":
    main()