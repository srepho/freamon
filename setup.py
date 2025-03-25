"""
Setup script for the freamon package.
"""
from setuptools import setup, find_packages

setup(
    name="freamon",
    version="0.3.30",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "category_encoders",
        "networkx",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "mypy",
            "isort",
            "flake8",
        ],
        "all": [
            "lightgbm",
            "xgboost",
            "catboost",
            "spacy",
            "polars",
            "pyarrow>=12.0.0,<15.0.0",
            "dask[dataframe]",
            "shapiq",
            "shap",
            "optuna>=3.0.0",
            "plotly",
        ],
        "full": [
            "lightgbm",
            "xgboost",
            "catboost",
            "spacy",
            "polars",
            "pyarrow>=12.0.0,<15.0.0",
            "dask[dataframe]",
            "shapiq",
            "shap",
            "optuna>=3.0.0",
            "plotly",
            "pytest",
            "pytest-cov",
            "black",
            "mypy",
            "isort",
            "flake8",
        ],
        "lightgbm": [
            "lightgbm",
            "optuna>=3.0.0",
        ],
        "tuning": [
            "optuna>=3.0.0",
            "plotly",
        ],
        "xgboost": [
            "xgboost",
        ],
        "catboost": [
            "catboost",
        ],
        "nlp": [
            "spacy",
        ],
        "polars": [
            "polars",
        ],
        "dask": [
            "dask[dataframe]",
        ],
        "explainability": [
            "shapiq",
            "shap",
        ],
        "visualization": [
            "plotly",
            "graphviz",
        ],
        "performance": [
            "pyarrow>=12.0.0,<15.0.0",
        ],
    },
    python_requires=">=3.10",
)