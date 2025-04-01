# Jupyter Notebook Integration for EDA Reports

Freamon now provides enhanced Jupyter notebook integration with the new `display_eda_report()` method in the `EDAAnalyzer` class. This feature allows you to quickly visualize and analyze your dataset directly in a Jupyter notebook with rich, interactive displays.

## Quick Start

```python
import pandas as pd
from freamon.eda.analyzer import EDAAnalyzer

# Create an analyzer with your dataframe
analyzer = EDAAnalyzer(df, target_column='target')

# Run the analysis
analyzer.run_full_analysis()

# Display the interactive report in your notebook
analyzer.display_eda_report()
```

## Features

The interactive EDA report includes:

- **Column Analysis Table**: A color-coded table showing:
  - Column types (numeric, categorical, datetime, text)
  - Missing value percentages with highlighting for concerning levels
  - Cardinality analysis (number and percentage of unique values)
  - Statistical relationship with target variable (if provided)

- **Correlation Analysis**:
  - Interactive correlation heatmap for numeric variables
  - Identification of top correlated pairs
  - Strength and direction interpretation for correlations

- **Missing Value Analysis**:
  - Correlation heatmap for missing values
  - Identification of missing value patterns
  - Highlighted strong relationships in missing data

## Example

For a complete example, see the `examples/jupyter_display_example.py` file or run the following in a Jupyter notebook:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from freamon.eda.analyzer import EDAAnalyzer

# Load a sample dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Create some missing values for demonstration
df.loc[np.random.choice(df.index, size=20), 'age'] = np.nan
df.loc[np.random.choice(df.index, size=15), 'bmi'] = np.nan
df.loc[np.random.choice(df.index, size=10), ['age', 'bp']] = np.nan

# Add a datetime column for more variety
df['date'] = pd.date_range(start='2023-01-01', periods=len(df))

# Add a categorical column
df['category'] = pd.cut(df['target'], bins=5, 
                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Create the EDAAnalyzer
analyzer = EDAAnalyzer(df, target_column='target', date_column='date')

# Display the interactive report
analyzer.display_eda_report()
```

## Benefits

The `display_eda_report()` method offers several advantages over traditional report generation:

1. **Interactivity**: View and interact with data directly in the notebook
2. **Quick Insights**: Get immediate visual feedback without generating full reports
3. **Statistical Testing**: Automatic selection of appropriate statistical tests based on variable types
4. **Visual Emphasis**: Color-coded displays highlight important patterns and relationships
5. **Context-Aware**: Adapts the visualization based on the data characteristics

## Related Documentation

For more comprehensive reporting options, see:
- [Advanced EDA Reports](README_ADVANCED_EDA.md)
- [Markdown Reports](README_MARKDOWN_REPORTS.md)
- [Export Capabilities](README_EXPORT.md)