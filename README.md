# freamon

<p align="center">
  <img src="package_logo.webp" alt="Freamon Logo" width="250"/>
</p>

A package to make data science projects on tabular data easier. Named after the great character from The Wire played by Clarke Peters.

## Features

- **Data Quality Assessment:** Missing values, outliers, data types, duplicates
- **Exploratory Data Analysis (EDA):** Statistical analysis and visualizations
- **Feature Engineering:** Automated and manual feature engineering
- **Categorical Encoding:** One-hot, ordinal, target encoding
- **Text Processing:** Basic NLP with optional spaCy integration
- **Model Selection:** Train/test splitting with time-series awareness
- **Modeling:** Training, evaluation, and validation
  - **Support for Multiple Libraries:** scikit-learn, LightGBM, XGBoost, CatBoost
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
pip install freamon[lightgbm]  # For LightGBM support
pip install freamon[xgboost]   # For XGBoost support
pip install freamon[catboost]  # For CatBoost support
pip install freamon[nlp]       # For NLP capabilities with spaCy
pip install freamon[polars]    # For Polars support
pip install freamon[dask]      # For Dask support

# Development installation
git clone https://github.com/yourusername/freamon.git
cd freamon
pip install -e ".[dev,all]"
```

## Quick Start

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

### Using with Polars

```python
import polars as pl
from freamon.utils.dataframe_utils import detect_datetime_columns, convert_dataframe

# Load data with Polars
df = pl.read_csv("your_data.csv")

# Detect and convert datetime columns
df = detect_datetime_columns(df)

# Convert to pandas for operations that require it
pandas_df = convert_dataframe(df, "pandas")

# ... perform operations ...

# Convert back to polars
result = convert_dataframe(pandas_df, "polars")
```

## Module Overview

- **data_quality:** Tools for assessing and improving data quality
- **utils:** Utility functions for working with dataframes and encoders
  - **dataframe_utils:** Tools for different dataframe backends and date detection
  - **encoders:** Categorical variable encoding tools
  - **text_utils:** Text processing utilities
- **model_selection:** Methods for splitting data and cross-validation
- **modeling:** Model training, evaluation, and comparison

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