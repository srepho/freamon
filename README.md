# freamon

A package to make data science projects on tabular data easier. Named after the great character from The Wire played by Clarke Peters.

## Features

- **Data Quality Assessment:** Missing values, outliers, data types, etc.
- **Exploratory Data Analysis (EDA):** Statistical analysis and visualizations
- **Feature Engineering:** Automated and manual feature engineering
- **Model Selection:** Train/test splitting with time-series awareness
- **Modeling:** Training, evaluation, and validation

## Installation

```bash
# From PyPI (not yet available)
# pip install freamon

# Development installation
git clone https://github.com/yourusername/freamon.git
cd freamon
pip install -e .
```

## Quick Start

```python
import pandas as pd
from freamon.data_quality import DataQualityAnalyzer

# Load your data
df = pd.read_csv("your_data.csv")

# Analyze data quality
analyzer = DataQualityAnalyzer(df)
analyzer.generate_report("data_quality_report.html")

# Handle missing values
from freamon.data_quality import handle_missing_values
df_clean = handle_missing_values(df, strategy="mean")
```

## Development

To contribute to freamon, install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT License