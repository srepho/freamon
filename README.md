# freamon

<p align="center">
  <img src="package_logo.webp" alt="Freamon Logo" width="250"/>
</p>

A package to make data science projects on tabular data easier. Named after the great character from The Wire played by Clarke Peters.

## Features

- **Data Quality Assessment:** Missing values, outliers, data types, duplicates
- **Exploratory Data Analysis (EDA):** Statistical analysis and visualizations
- **Feature Engineering:** 
  - **Standard Features:** Polynomial, interaction, datetime, binned features
  - **Automatic Interaction Detection:** ShapIQ-based automatic feature engineering
- **Categorical Encoding:** 
  - **Basic Encoders:** One-hot, ordinal, target encoding
  - **Advanced Encoders:** Binary, hashing, weight of evidence (WOE) encoding
- **Text Processing:** Basic NLP with optional spaCy integration
- **Model Selection:** Train/test splitting with time-series awareness
- **Modeling:** Training, evaluation, and validation
  - **Support for Multiple Libraries:** scikit-learn, LightGBM, XGBoost, CatBoost
  - **Intelligent Hyperparameter Tuning:** Parameter-importance aware tuning for LightGBM
  - **Cross-Validation:** Both standard and time series-aware cross-validation
- **Explainability:** 
  - **SHAP Support:** Feature importance and explanations
  - **ShapIQ Integration:** Feature interactions detection and visualization
  - **Interactive Reports:** HTML reports for explainability findings
  - **Permutation Importance:** Better feature importance for black-box models
- **Pipeline System:**
  - **Integrated Workflow:** Connect feature engineering, selection, and modeling
  - **Modular Design:** Mix and match steps for custom workflows
  - **Persistence:** Save and load complete pipelines
  - **Visualization:** Pipeline visualization with multiple backends
- **Multiple DataFrame Backends:** 
  - **Pandas:** Standard interface
  - **Polars:** High-performance alternative
  - **Dask:** Out-of-core processing for large datasets

## Installation

```bash
# Basic installation
pip install freamon

# With all optional dependencies
pip install freamon[all]

# With specific optional dependencies
pip install freamon[lightgbm]        # For LightGBM support
pip install freamon[xgboost]         # For XGBoost support
pip install freamon[catboost]        # For CatBoost support
pip install freamon[nlp]             # For NLP capabilities with spaCy
pip install freamon[polars]          # For Polars support
pip install freamon[dask]            # For Dask support
pip install freamon[explainability]  # For SHAP and ShapIQ integration
pip install freamon[visualization]   # For pipeline visualization with Graphviz
pip install freamon[tuning]          # For hyperparameter tuning support

# Development installation
git clone https://github.com/yourusername/freamon.git
cd freamon
pip install -e ".[dev,all]"
```

## Quick Start

### LightGBM with Intelligent Hyperparameter Tuning (New!)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from freamon import LightGBMModel

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Add a categorical feature
X['category'] = pd.qcut(X['mean radius'], 4, labels=['A', 'B', 'C', 'D'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model with automatic hyperparameter tuning
model = LightGBMModel(
    problem_type='classification',
    metric='auc',
    tuning_trials=50,  # Number of hyperparameter trials
    random_state=42
)

# Fit the model with automatic hyperparameter tuning
model.fit(
    X_train, y_train,
    categorical_features=['category'],  # List categorical features
    validation_size=0.2,  # Create validation set from training data
    tune_hyperparameters=True  # Enable intelligent tuning
)

# Get feature importance
importance = model.get_feature_importance(method='native')
print("Top 5 features:", importance.head(5))

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test AUC: {auc:.4f}")

# Save model for later use
model.save("breast_cancer_model.joblib")

# Load the saved model
loaded_model = LightGBMModel.load("breast_cancer_model.joblib")
```

### Pipeline Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    EvaluationStep
)

# Load and split your data
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline steps
feature_step = FeatureEngineeringStep(name="feature_engineering")
feature_step.add_operation(
    method="add_polynomial_features",
    columns=["feature1", "feature2"],
    degree=2
)
feature_step.add_operation(
    method="add_binned_features",
    columns=["feature3"],
    n_bins=5
)

model_step = ModelTrainingStep(
    name="model",
    model_type="lightgbm",
    problem_type="classification",
    hyperparameters={"num_leaves": 31, "learning_rate": 0.05}
)

eval_step = EvaluationStep(
    name="evaluation",
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

# Create and fit pipeline
pipeline = Pipeline()
pipeline.add_step(feature_step)
pipeline.add_step(model_step)
pipeline.add_step(eval_step)
pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipeline.predict(X_test)
metrics = eval_step.evaluate(y_test, y_pred, model_step.predict_proba(X_test))
print(f"Evaluation metrics: {metrics}")

# Save pipeline for later use
pipeline.save("my_pipeline")
```

### Traditional Workflow

```python
import pandas as pd
from freamon.data_quality import DataQualityAnalyzer
from freamon.modeling import ModelTrainer
from freamon.model_selection import train_test_split
from freamon.utils import OneHotEncoderWrapper
from freamon.utils.dataframe_utils import detect_datetime_columns

# Load your data
df = pd.read_csv("your_data.csv")

# Automatically detect and convert datetime columns
df = detect_datetime_columns(df)

# Analyze data quality
analyzer = DataQualityAnalyzer(df)
analyzer.generate_report("data_quality_report.html")

# Handle missing values
from freamon.data_quality import handle_missing_values
df_clean = handle_missing_values(df, strategy="mean")

# Encode categorical features
encoder = OneHotEncoderWrapper()
df_encoded = encoder.fit_transform(df_clean)

# Split data
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

# Train a model
feature_cols = [col for col in train_df.columns if col != "target"]
trainer = ModelTrainer(
    model_type="lightgbm",
    model_name="LGBMClassifier",
    problem_type="classification",
)
metrics = trainer.train(
    train_df[feature_cols],
    train_df["target"],
    X_val=test_df[feature_cols],
    y_val=test_df["target"],
)

# View the results
print(f"Validation metrics: {metrics}")
```

## Module Overview

- **data_quality:** Tools for assessing and improving data quality
  - **drift:** Data drift detection and monitoring
  - **outliers:** Outlier detection and handling
  - **missing_values:** Missing value analysis and imputation
- **utils:** Utility functions for working with dataframes and encoders
  - **dataframe_utils:** Tools for different dataframe backends and date detection
  - **encoders:** Categorical variable encoding tools with cross-validation support
  - **text_utils:** Text processing utilities
- **model_selection:** Methods for splitting data and cross-validation
  - **cross_validation:** Standard and time series cross-validation tools
  - **splitter:** Train/test splitting with special modes for time series
- **modeling:** Model training, evaluation, and comparison
  - **model:** Base model class with consistent interface
  - **factory:** Model creation utilities for multiple libraries
  - **trainer:** Training and evaluation tools
  - **lightgbm:** High-level LightGBM interface with intelligent tuning
  - **tuning:** Hyperparameter optimization with parameter importance awareness
  - **importance:** Permutation-based feature importance
  - **calibration:** Probability calibration for classification models
- **pipeline:** Integrated workflow system connecting feature engineering with model training
  - **pipeline:** Core Pipeline interface
  - **steps:** Reusable pipeline steps for different tasks
  - **visualization:** Pipeline visualization tools

Check out the [ROADMAP.md](ROADMAP.md) file for information on planned features and development phases.

## Development

To contribute to freamon, install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=freamon
```

## License

MIT License