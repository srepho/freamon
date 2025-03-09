"""
Setup script for the freamon package.
"""
from setuptools import setup, find_packages

setup(
    name="freamon",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "category_encoders",
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
            "dask[dataframe]",
            "shapiq",
            "shap",
        ],
        "lightgbm": [
            "lightgbm",
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
    },
    python_requires=">=3.10",
)