"""
Example demonstrating cross-validation training capabilities in freamon.

This example shows how to:
1. Use CrossValidatedTrainer as a standalone class
2. Use CrossValidationTrainingStep in a pipeline
3. Compare different cross-validation strategies
4. Compare different ensemble methods
5. Visualize cross-validation results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from freamon.model_selection.cv_trainer import CrossValidatedTrainer, CrossValidationTrainingStep
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import DataFrameStep, FeatureSelectionStep
from freamon.modeling.metrics import calculate_metrics
from freamon.eda import EDAAnalyzer


def load_datasets():
    """Load example datasets for demonstration."""
    # Load Iris for classification
    iris = load_iris()
    iris_df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )
    iris_df['target'] = iris.target
    
    # Load California Housing for regression
    housing = fetch_california_housing()
    housing_df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )
    housing_df['target'] = housing.target
    
    return {
        "iris": {
            "data": iris_df,
            "type": "classification"
        },
        "housing": {
            "data": housing_df,
            "type": "regression"
        }
    }


def example_standalone_trainer():
    """Example of using CrossValidatedTrainer as a standalone class."""
    print("\n===== Using CrossValidatedTrainer as a standalone class =====")
    
    # Load and prepare data
    print("Loading and preparing Iris dataset...")
    datasets = load_datasets()
    iris_df = datasets["iris"]["data"]
    
    X = iris_df.drop(columns=['target'])
    y = iris_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit trainer with standard k-fold CV
    print("\nTraining LightGBM with standard 5-fold cross-validation...")
    trainer = CrossValidatedTrainer(
        model_type="lightgbm",
        problem_type="classification",
        cv_strategy="kfold",
        n_splits=5,
        ensemble_method="best",
        eval_metric="accuracy",
        random_state=42
    )
    
    trainer.fit(X_train, y_train)
    
    # Make predictions
    y_pred = trainer.predict(X_test)
    y_prob = trainer.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        problem_type="classification"
    )
    
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Get CV results
    cv_results = trainer.get_cv_results()
    print("\nCross-validation results:")
    for metric, values in cv_results.items():
        print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    # Get feature importances
    importances = trainer.get_feature_importances()
    print("\nFeature importances:")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.4f}")


def example_pipeline_integration():
    """Example of using CrossValidationTrainingStep in a pipeline."""
    print("\n===== Using CrossValidationTrainingStep in a pipeline =====")
    
    # Load and prepare data
    print("Loading and preparing Housing dataset...")
    datasets = load_datasets()
    housing_df = datasets["housing"]["data"]
    
    X = housing_df.drop(columns=['target'])
    y = housing_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with CV training step
    print("\nCreating pipeline with cross-validation training step...")
    pipeline = Pipeline()
    
    # Add steps to pipeline
    pipeline.add_step(
        DataFrameStep(name="data_preparation")
    )
    
    # Add feature selection step
    pipeline.add_step(
        FeatureSelectionStep(
            name="feature_selection",
            method="variance",
            threshold=0.0,  # Keep all features in this example
        )
    )
    
    # Add cross-validation training step
    pipeline.add_step(
        CrossValidationTrainingStep(
            name="model_training",
            model_type="lightgbm",
            problem_type="regression",
            cv_strategy="kfold",
            n_splits=5,
            ensemble_method="average",  # Use average ensemble
            eval_metric="r2",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5
            },
            random_state=42
        )
    )
    
    # Fit pipeline
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Access the CV training step
    cv_step = pipeline.get_step("model_training")
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        problem_type="regression"
    )
    
    print(f"Test R²: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    
    # Get CV results
    cv_results = cv_step.get_cv_results()
    print("\nCross-validation results:")
    for metric, values in cv_results.items():
        print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    # Get feature importances
    importances = cv_step.get_feature_importances()
    print("\nFeature importances (top 5):")
    for feature, importance in importances.head(5).items():
        print(f"  {feature}: {importance:.4f}")


def example_compare_cv_strategies():
    """Example comparing different cross-validation strategies."""
    print("\n===== Comparing Different CV Strategies =====")
    
    # Load and prepare data
    datasets = load_datasets()
    housing_df = datasets["housing"]["data"]
    
    # Add a date column for time series CV
    n_samples = len(housing_df)
    start_date = pd.Timestamp('2010-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_samples)]
    housing_df['date'] = dates
    
    X = housing_df.drop(columns=['target', 'date'])
    y = housing_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Make date column available for time series CV
    X_train_with_date = housing_df.loc[X_train.index, X.columns.tolist() + ['date']]
    
    # Define CV strategies to compare
    cv_strategies = [
        {
            "name": "K-Fold CV",
            "strategy": "kfold",
            "kwargs": {}
        },
        {
            "name": "Time Series CV",
            "strategy": "timeseries",
            "kwargs": {"date_column": "date", "expanding_window": True}
        }
    ]
    
    results = {}
    
    # Train models with different CV strategies
    for cv_config in cv_strategies:
        print(f"\nTraining with {cv_config['name']}...")
        
        trainer = CrossValidatedTrainer(
            model_type="lightgbm",
            problem_type="regression",
            cv_strategy=cv_config["strategy"],
            n_splits=5,
            ensemble_method="best",
            eval_metric="r2",
            random_state=42,
            **cv_config["kwargs"]
        )
        
        trainer.fit(X_train_with_date, y_train)
        
        # Make predictions
        y_pred = trainer.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            problem_type="regression"
        )
        
        # Store results
        results[cv_config["name"]] = {
            "test_metrics": metrics,
            "cv_results": trainer.get_cv_results()
        }
        
        print(f"Test R²: {metrics['r2']:.4f}")
        print(f"Test RMSE: {metrics['rmse']:.4f}")
    
    # Plot comparison
    cv_metrics = {}
    for name, result in results.items():
        cv_metrics[name] = {
            "mean": np.mean(result["cv_results"]["r2"]),
            "std": np.std(result["cv_results"]["r2"]),
            "test": result["test_metrics"]["r2"]
        }
    
    # Create bar chart
    names = list(cv_metrics.keys())
    cv_means = [cv_metrics[name]["mean"] for name in names]
    cv_stds = [cv_metrics[name]["std"] for name in names]
    test_scores = [cv_metrics[name]["test"] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_bars = ax.bar(x - width/2, cv_means, width, label='CV Mean R²', yerr=cv_stds, capsize=5)
    test_bars = ax.bar(x + width/2, test_scores, width, label='Test R²')
    
    ax.set_xlabel('CV Strategy')
    ax.set_ylabel('R² Score')
    ax.set_title('Comparison of CV Strategies: CV vs Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(cv_bars)
    add_labels(test_bars)
    
    plt.tight_layout()
    plt.savefig('cv_strategy_comparison.png')
    print("\nComparison chart saved to 'cv_strategy_comparison.png'")


def example_compare_ensemble_methods():
    """Example comparing different ensemble methods."""
    print("\n===== Comparing Different Ensemble Methods =====")
    
    # Load and prepare data
    datasets = load_datasets()
    iris_df = datasets["iris"]["data"]
    
    X = iris_df.drop(columns=['target'])
    y = iris_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define ensemble methods to compare
    ensemble_methods = ["best", "average", "weighted", "stacking"]
    
    results = {}
    
    # Train models with different ensemble methods
    for method in ensemble_methods:
        print(f"\nTraining with {method} ensemble method...")
        
        trainer = CrossValidatedTrainer(
            model_type="lightgbm",
            problem_type="classification",
            cv_strategy="stratified",
            n_splits=5,
            ensemble_method=method,
            eval_metric="accuracy",
            stratify_by="target",
            random_state=42
        )
        
        trainer.fit(X_train, y_train)
        
        # Make predictions
        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            problem_type="classification"
        )
        
        # Store results
        results[method] = {
            "test_metrics": metrics,
            "cv_results": trainer.get_cv_results()
        }
        
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Plot comparison
    metrics_to_plot = ["accuracy", "balanced_accuracy", "roc_auc"]
    test_metrics = {name: [] for name in ensemble_methods}
    
    for name in ensemble_methods:
        for metric in metrics_to_plot:
            test_metrics[name].append(results[name]["test_metrics"][metric])
    
    # Create grouped bar chart
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, method in enumerate(ensemble_methods):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, test_metrics[method], width, label=method.capitalize())
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Ensemble Methods: Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ensemble_method_comparison.png')
    print("\nComparison chart saved to 'ensemble_method_comparison.png'")


def example_eda_integration():
    """Example of integrating cross-validation with EDA."""
    print("\n===== Integrating Cross-Validation with EDA =====")
    
    # Load and prepare data
    datasets = load_datasets()
    housing_df = datasets["housing"]["data"]
    
    # Perform EDA
    print("Performing exploratory data analysis...")
    analyzer = EDAAnalyzer(housing_df, target_column='target')
    
    # Analyze data
    analyzer.analyze_basic_stats()
    analyzer.analyze_univariate()
    analyzer.analyze_bivariate()
    
    # Split data
    X = housing_df.drop(columns=['target'])
    y = housing_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use feature importance from EDA for feature selection
    df_corr = housing_df.corr()
    target_corrs = df_corr['target'].abs().sort_values(ascending=False)
    target_corrs = target_corrs.drop('target')
    
    # Select top 5 features
    top_features = target_corrs.head(5).index.tolist()
    print(f"\nTop 5 features based on correlation with target: {top_features}")
    
    # Train model with cross-validation using selected features
    print("\nTraining model with cross-validation using selected features...")
    trainer = CrossValidatedTrainer(
        model_type="lightgbm",
        problem_type="regression",
        cv_strategy="kfold",
        n_splits=5,
        ensemble_method="average",
        eval_metric="r2",
        random_state=42
    )
    
    # Train on selected features
    trainer.fit(X_train[top_features], y_train)
    
    # Make predictions
    y_pred = trainer.predict(X_test[top_features])
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        problem_type="regression"
    )
    
    print(f"Test R² with selected features: {metrics['r2']:.4f}")
    print(f"Test RMSE with selected features: {metrics['rmse']:.4f}")
    
    # Compare with all features
    print("\nTraining model with all features for comparison...")
    trainer_all = CrossValidatedTrainer(
        model_type="lightgbm",
        problem_type="regression",
        cv_strategy="kfold",
        n_splits=5,
        ensemble_method="average",
        eval_metric="r2",
        random_state=42
    )
    
    trainer_all.fit(X_train, y_train)
    y_pred_all = trainer_all.predict(X_test)
    
    metrics_all = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred_all,
        problem_type="regression"
    )
    
    print(f"Test R² with all features: {metrics_all['r2']:.4f}")
    print(f"Test RMSE with all features: {metrics_all['rmse']:.4f}")
    
    # Show cross-validation results
    cv_results = trainer.get_cv_results()
    cv_results_all = trainer_all.get_cv_results()
    
    print("\nCross-validation R² with selected features:")
    print(f"  Mean: {np.mean(cv_results['r2']):.4f}")
    print(f"  Std: {np.std(cv_results['r2']):.4f}")
    
    print("\nCross-validation R² with all features:")
    print(f"  Mean: {np.mean(cv_results_all['r2']):.4f}")
    print(f"  Std: {np.std(cv_results_all['r2']):.4f}")


if __name__ == "__main__":
    print("Cross-Validation Training Examples")
    print("=================================")
    
    # Run examples
    example_standalone_trainer()
    example_pipeline_integration()
    example_compare_cv_strategies()
    example_compare_ensemble_methods()
    example_eda_integration()
    
    print("\nAll examples completed successfully!")