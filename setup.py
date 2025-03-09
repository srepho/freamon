"""
Setup script for the freamon package.
"""
from setuptools import setup, find_packages

setup(
    name="freamon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
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
    },
    python_requires=">=3.10",
)