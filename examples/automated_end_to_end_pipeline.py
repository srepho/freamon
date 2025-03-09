"""
Automated End-to-End Machine Learning Pipeline with Comprehensive Reporting

This example demonstrates a complete automated ML workflow with Freamon that includes:
1. Data quality analysis and preprocessing
2. Drift detection between datasets
3. Feature engineering with ShapIQ integration
4. Hyperparameter optimization
5. Model training and evaluation
6. Comprehensive reporting at each stage
7. Final HTML dashboard combining all results

The pipeline is designed to be fully automated while providing detailed insights
at every step of the process.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgb
from datetime import datetime

# Freamon imports
from freamon.data_quality.analyzer import DataQualityAnalyzer
from freamon.data_quality.drift import DataDriftDetector
from freamon.eda.analyzer import EDAAnalyzer
from freamon.eda.explainability_report import generate_interaction_report
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    ShapIQFeatureEngineeringStep,
    FeatureSelectionStep, 
    HyperparameterTuningStep,
    ModelTrainingStep,
    EvaluationStep
)
from freamon.pipeline.visualization import visualize_pipeline, generate_interactive_html
from freamon.features.engineer import FeatureEngineer
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer
from freamon.features.selector import FeatureSelector
from freamon.modeling.lightgbm import LightGBMTrainer
from freamon.modeling.tuning import HyperparameterTuner
from freamon.modeling.early_stopping import EarlyStopping
from freamon.modeling.importance import PermutationImportance
from freamon.modeling.calibration import ModelCalibrator
from freamon.utils.encoders import EncoderManager
from freamon.model_selection.cross_validation import CrossValidator

# Create output directory for reports
output_dir = "automated_pipeline_output"
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare_data():
    """Load sample dataset and prepare it for analysis."""
    # Load California housing dataset
    housing = fetch_california_housing()
    
    # Create pandas DataFrame
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseValue")
    
    # Add some synthetic datetime features for time series analysis
    dates = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
    X['date'] = dates
    X['month'] = dates.month
    X['year'] = dates.year
    X['day_of_week'] = dates.dayofweek
    
    # Add some categorical features
    X['Location'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(X))
    X['Size'] = pd.qcut(X['AveRooms'], 4, labels=['Small', 'Medium', 'Large', 'XLarge'])
    
    # Introduce some missing values for data quality testing
    for col in X.columns[:3]:
        mask = np.random.random(len(X)) < 0.05
        X.loc[mask, col] = np.nan
    
    # Split data for training/testing and create a "drift" version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create "drifted" data by slightly modifying distributions
    X_drift = X_test.copy()
    for col in X_drift.select_dtypes(include=np.number).columns:
        if col not in ['date', 'month', 'year', 'day_of_week']:
            # Add small shift to numeric columns to simulate drift
            X_drift[col] = X_drift[col] * (1 + np.random.normal(0, 0.1, len(X_drift)))
    
    return X_train, X_test, X_drift, y_train, y_test

def run_data_quality_analysis(X_train, X_test, output_dir):
    """Run data quality analysis and generate reports."""
    print("Running data quality analysis...")
    
    # Analyze training data quality
    analyzer = DataQualityAnalyzer(X_train)
    quality_results = analyzer.run_all_analyses()
    
    # Generate comprehensive HTML report
    analyzer.generate_report(os.path.join(output_dir, "data_quality_report.html"))
    
    # Return cleaned data
    X_train_clean = analyzer.get_clean_data()
    
    return X_train_clean, quality_results

def detect_data_drift(X_train, X_test, X_drift, output_dir):
    """Detect drift between datasets and generate reports."""
    print("Analyzing data drift...")
    
    # Compare test data to training data
    test_detector = DataDriftDetector(X_train, X_test)
    test_drift_results = test_detector.detect_all_drift()
    test_detector.generate_drift_report(os.path.join(output_dir, "test_drift_report.html"))
    
    # Compare drifted data to training data
    drift_detector = DataDriftDetector(X_train, X_drift)
    drift_results = drift_detector.detect_all_drift()
    drift_detector.generate_drift_report(os.path.join(output_dir, "synthetic_drift_report.html"))
    
    return test_drift_results, drift_results

def run_exploratory_analysis(X_train, y_train, output_dir):
    """Run exploratory data analysis and generate reports."""
    print("Running exploratory data analysis...")
    
    # Create EDA analyzer
    eda = EDAAnalyzer(X_train, y_train)
    
    # Run full analysis
    eda_results = eda.run_full_analysis()
    
    # Generate comprehensive HTML report
    eda.generate_report(os.path.join(output_dir, "eda_report.html"))
    
    return eda_results

def build_automated_pipeline(X_train, y_train, X_test, y_test, output_dir):
    """Build and run the automated end-to-end pipeline."""
    print("Building automated pipeline...")
    
    # Initialize encoder manager for handling categorical features
    encoder_manager = EncoderManager()
    
    # Define feature engineering step
    feature_engineer = FeatureEngineer(
        categorical_encoding="target",
        datetime_encoding="cyclic",
        handle_missing=True,
        missing_strategy="median",
        outlier_detection=True,
        outlier_method="iqr",
        encoder_manager=encoder_manager
    )
    
    # Define ShapIQ feature engineering for interaction detection
    shapiq_engineer = ShapIQFeatureEngineer(
        max_interactions=5,
        min_interaction_strength=0.01,
        sample_size=1000,
        interaction_type="FSII"
    )
    
    # Define feature selection
    feature_selector = FeatureSelector(
        method="importance",
        threshold=0.01,
        max_features=10
    )
    
    # Define hyperparameter tuning
    tuner = HyperparameterTuner(
        model_type="lightgbm",
        param_grid={
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "max_depth": [5, 7, 9],
            "min_data_in_leaf": [20, 50, 100]
        },
        cv=5,
        scoring="neg_mean_squared_error",
        n_trials=10
    )
    
    # Define model training
    trainer = LightGBMTrainer(
        objective="regression",
        metric="rmse",
        early_stopping=EarlyStopping(patience=10, metric="rmse"),
        n_estimators=1000,
        verbose=0
    )
    
    # Define evaluation metrics
    evaluator = PermutationImportance(
        metrics=["r2", "rmse", "mae"], 
        n_repeats=5
    )
    
    # Define calibration
    calibrator = ModelCalibrator(method="isotonic")
    
    # Create pipeline with all steps
    pipeline = Pipeline()
    
    # Add all steps to pipeline
    pipeline.add_step(FeatureEngineeringStep(feature_engineer, name="Feature Engineering"))
    pipeline.add_step(ShapIQFeatureEngineeringStep(shapiq_engineer, name="Interaction Detection"))
    pipeline.add_step(FeatureSelectionStep(feature_selector, name="Feature Selection"))
    pipeline.add_step(HyperparameterTuningStep(tuner, name="Hyperparameter Tuning"))
    pipeline.add_step(ModelTrainingStep(trainer, name="LightGBM Training"))
    pipeline.add_step(EvaluationStep(evaluator, name="Model Evaluation"))
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = pipeline.predict(X_test)
    
    # Generate pipeline visualization
    visualize_pipeline(
        pipeline, 
        output_file=os.path.join(output_dir, "pipeline_visualization.png"),
        show_details=True
    )
    
    # Generate interactive HTML report for pipeline
    generate_interactive_html(
        pipeline,
        output_file=os.path.join(output_dir, "interactive_pipeline.html"),
        include_code=True,
        include_data_samples=True
    )
    
    # Generate feature interaction report
    generate_interaction_report(
        model=pipeline.get_step("LightGBM Training").get_model(),
        X=X_test,
        feature_names=X_test.columns,
        output_file=os.path.join(output_dir, "feature_interactions.html")
    )
    
    return pipeline, y_pred

def generate_integrated_dashboard(output_dir):
    """Generate a master dashboard that links to all reports."""
    print("Generating integrated dashboard...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Freamon Automated Pipeline Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .report-card {{ margin-bottom: 20px; transition: transform 0.3s; }}
            .report-card:hover {{ transform: translateY(-5px); }}
            .pipeline-diagram {{ max-width: 100%; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row mb-4">
                <div class="col-12">
                    <h1 class="display-4">Freamon Automated Pipeline Dashboard</h1>
                    <p class="lead">Generated on {timestamp}</p>
                    <hr/>
                </div>
            </div>
            
            <div class="row mb-4">
                <div class="col-12">
                    <h2>Pipeline Visualization</h2>
                    <img src="pipeline_visualization.png" class="img-fluid pipeline-diagram" alt="Pipeline Visualization">
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Data Quality Analysis</h5>
                            <p class="card-text">Comprehensive analysis of data quality issues including missing values, outliers, and duplicates.</p>
                            <a href="data_quality_report.html" class="btn btn-primary">View Report</a>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Exploratory Data Analysis</h5>
                            <p class="card-text">In-depth exploration of features and target relationships with visualizations.</p>
                            <a href="eda_report.html" class="btn btn-primary">View Report</a>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Data Drift Analysis</h5>
                            <p class="card-text">Analysis of distribution drift between training and test datasets.</p>
                            <a href="test_drift_report.html" class="btn btn-primary">View Test Drift</a>
                            <a href="synthetic_drift_report.html" class="btn btn-secondary mt-2">View Synthetic Drift</a>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Feature Interactions</h5>
                            <p class="card-text">Analysis of important feature interactions using ShapIQ.</p>
                            <a href="feature_interactions.html" class="btn btn-primary">View Interactions</a>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Interactive Pipeline</h5>
                            <p class="card-text">Interactive visualization of pipeline steps with detailed information.</p>
                            <a href="interactive_pipeline.html" class="btn btn-primary">Explore Pipeline</a>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card report-card">
                        <div class="card-body">
                            <h5 class="card-title">Download Reports</h5>
                            <p class="card-text">Download all reports as a single package for sharing.</p>
                            <button class="btn btn-primary">Download All (ZIP)</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "dashboard.html"), "w") as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {os.path.join(output_dir, 'dashboard.html')}")

def main():
    """Run the full automated pipeline workflow."""
    print("Starting automated end-to-end pipeline...")
    
    # Load and prepare data
    X_train, X_test, X_drift, y_train, y_test = load_and_prepare_data()
    
    # Run data quality analysis
    X_train_clean, quality_results = run_data_quality_analysis(X_train, X_test, output_dir)
    
    # Detect data drift
    test_drift_results, drift_results = detect_data_drift(X_train_clean, X_test, X_drift, output_dir)
    
    # Run exploratory data analysis
    eda_results = run_exploratory_analysis(X_train_clean, y_train, output_dir)
    
    # Build and run automated pipeline
    pipeline, y_pred = build_automated_pipeline(X_train_clean, y_train, X_test, y_test, output_dir)
    
    # Generate integrated dashboard
    generate_integrated_dashboard(output_dir)
    
    print(f"\nAutomated pipeline complete! All reports are available in the '{output_dir}' directory.")
    print(f"Open '{os.path.join(output_dir, 'dashboard.html')}' to view the integrated dashboard.")

if __name__ == "__main__":
    main()