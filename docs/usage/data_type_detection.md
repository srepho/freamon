# Advanced Data Type Detection

The Freamon library includes a powerful data type detection system that goes beyond standard pandas data types. It can identify logical data types, categorical vs. continuous numeric features, and semantic data types.

## Basic Usage

```python
import pandas as pd
from freamon.eda import detect_column_types, optimize_dataframe_types

# Load your data
df = pd.read_csv("your_data.csv")

# Detect column types
types = detect_column_types(df)

# Print detected types
for column, info in types.items():
    print(f"{column}: {info['logical_type']}")
    
    # Print semantic type if detected
    if 'semantic_type' in info:
        print(f"  Semantic type: {info['semantic_type']}")
        
    # Print conversion suggestions
    if 'suggested_conversion' in info:
        print(f"  Suggested conversion: {info['suggested_conversion']['convert_to']}")

# Optimize dataframe types automatically
optimized_df = optimize_dataframe_types(df)
```

## Using Custom Semantic Types

You can define your own domain-specific semantic type patterns:

```python
from freamon.eda import detect_column_types

# Define custom patterns
custom_patterns = {
    'account_number': r'^ACCT-\d{5}$',
    'tax_id': r'^\d{2}-\d{7}$',
    'product_sku': r'^SKU-[A-Z]\d{4}$',
    'medical_record': r'^MRN-\d{6}$',
    'device_id': r'^DEV-\w+-\d{4}$'
}

# Detect types with custom patterns
types = detect_column_types(df, custom_patterns=custom_patterns)

# Check results
for column, info in types.items():
    if 'semantic_type' in info:
        print(f"{column}: {info['semantic_type']}")
```

## Using with EDAAnalyzer

The data type detection is automatically used when you create an `EDAAnalyzer` instance:

```python
from freamon.eda import EDAAnalyzer

analyzer = EDAAnalyzer(df)

# Run analysis
analyzer.run_full_analysis()

# The detected types are available in the basic_stats results
detected_types = analyzer.analysis_results["basic_stats"]["detected_types"]
```

## Available Features

### Logical Type Detection

The detector identifies the following logical types:

- `integer` - Integer numeric values
- `float` - Floating-point numeric values
- `datetime` - Date and time values
- `categorical` - Category values
- `categorical_integer` - Integer values that represent categories
- `categorical_float` - Float values that represent categories
- `continuous_integer` - Integer values that represent a continuous scale
- `continuous_float` - Float values that represent a continuous scale
- `boolean` - Boolean values
- `string` - Text values

### Semantic Type Detection

For string and numeric columns, the detector can identify the following semantic types:

- `id` - Identifier values
- `email` - Email addresses
- `url` - Web URLs
- `ip_address` - IP addresses
- `phone_number` - Phone numbers
- `zip_code` - ZIP/Postal codes
- `credit_card` - Credit card numbers
- `ssn` - Social security numbers
- `uuid` - UUID strings
- `path` - File system paths
- `currency` - Currency values
- `isbn` - ISBN book identifiers
- `latitude` - Geographic latitude values
- `longitude` - Geographic longitude values
- `name` - Personal names
- `address` - Address strings
- `excel_date` - Numbers representing Excel dates (days since 1899-12-30)
- `mixed_date_formats` - Date columns with multiple different date formats
- `scientific_notation` - Numeric values in scientific notation (e.g., 1.23e-10)

#### Australian-specific types:
- `au_postcode` - Australian postal codes (4 digits, may have leading zeros)
- `au_phone` - Australian phone numbers
- `au_mobile` - Australian mobile numbers  
- `au_abn` - Australian Business Numbers
- `au_acn` - Australian Company Numbers
- `au_tfn` - Australian Tax File Numbers

### Conversion Suggestions

The detector provides suggestions for optimizing column types, including:

- Converting string columns to more efficient categorical types
- Converting string date columns to datetime
- Converting integer timestamp columns to datetime
- Converting Excel date numbers to proper datetime objects
- Converting columns with mixed date formats using multi-pass parsing
- Downcasting integers and floats to smaller types
- Converting numeric categorical columns to categorical type
- Properly zero-padding Australian postcodes stored as integers
- Maintaining scientific notation when appropriate for numeric columns

## Using the DataTypeDetector Class

For more control over the detection process, you can use the `DataTypeDetector` class directly:

```python
from freamon.eda import DataTypeDetector

detector = DataTypeDetector(
    df,
    sample_size=2000,          # Sample size for pattern detection
    threshold=0.85,            # Threshold for pattern matching
    detect_semantic_types=True, # Whether to detect semantic types
    categorize_numeric=True     # Whether to categorize numeric columns
)

# Detect all types
types = detector.detect_all_types()

# Get a detailed report
report = detector.get_column_report()

# Convert types based on suggestions
converted_df = detector.convert_types()
```

## Excel Date Detection

The detector can identify numeric columns that likely represent Excel dates, which are stored as days since December 30, 1899:

```python
import pandas as pd
from freamon.utils.datatype_detector import detect_column_types, optimize_dataframe_types

# Sample data with Excel dates
df = pd.DataFrame({
    'date_col': [43831, 44196, 44562],  # Jan 1 for 2020, 2021, and 2022
    'description': ['New Year 2020', 'New Year 2021', 'New Year 2022']
})

# Detect types - will identify the Excel dates
types = detect_column_types(df)
print(types['date_col'])
# Output: {'storage_type': 'int64', 'logical_type': 'datetime', 
#          'semantic_type': 'excel_date', 
#          'suggested_conversion': {'convert_to': 'datetime', 
#                                  'method': 'pd.to_datetime(unit="D", origin="1899-12-30")'}}

# Automatically convert the Excel dates to proper datetime objects
df_converted = optimize_dataframe_types(df)
print(df_converted['date_col'])
# Output: 
# 0   2020-01-01
# 1   2021-01-01
# 2   2022-01-01
# Name: date_col, dtype: datetime64[ns]
```

This is particularly useful when processing data exported from Excel to CSV, where date columns often lose their formatting and are saved as numbers.

## Mixed Date Format Detection

The detector can identify and handle columns that contain dates in multiple different formats:

```python
import pandas as pd
import numpy as np
from freamon.utils.datatype_detector import detect_column_types, optimize_dataframe_types

# Sample data with mixed date formats
df = pd.DataFrame({
    'mixed_dates': [
        '2022-01-01',       # ISO format
        '01/15/2022',       # US format
        'January 20, 2022', # Full text format
        '2022.01.25',       # Dot separated
        '2022/01/30'        # Slash separated ISO
    ]
})

# Detect types - will identify mixed date formats
types = detect_column_types(df)
print(types['mixed_dates'])
# Output will indicate mixed_date_formats as the semantic type

# Convert using multi-pass date parsing
df_converted = optimize_dataframe_types(df)
print(df_converted['mixed_dates'])
# All dates will be properly converted to datetime
```

The multi-pass date conversion works by:
1. First attempting automatic parsing with pandas
2. Then identifying specific formats for dates that failed to parse
3. Applying different format strings to different rows based on the detected patterns
4. Maintaining already parsed dates to avoid overwriting them

This is particularly useful for real-world datasets where dates may be entered inconsistently.

## Scientific Notation Detection

The detector can identify numeric columns that contain values in scientific notation:

```python
import pandas as pd
from freamon.utils.datatype_detector import detect_column_types, optimize_dataframe_types

# Sample data with scientific notation
df = pd.DataFrame({
    'small_values': [1.23e-10, 4.56e-12, 7.89e-15],
    'large_values': [1.23e+10, 4.56e+12, 7.89e+15]
})

# Detect types
types = detect_column_types(df)
print(types['small_values'])
# Output will indicate scientific_notation as the semantic type

# Convert types (will maintain scientific notation where appropriate)
df_converted = optimize_dataframe_types(df)
```

Scientific notation detection is valuable for:
- Preserving very small or very large values that would otherwise lose precision
- Understanding when to use logarithmic scales for visualization
- Identifying features that may need special handling in modeling

## Integration with ML Pipelines

You can integrate the type detection into your machine learning pipeline:

```python
from freamon.eda import detect_column_types
from freamon.pipeline import Pipeline
from freamon.modeling import LightGBMModel

# Detect column types
types = detect_column_types(df)

# Extract feature groups based on detected types
categorical_features = [col for col, info in types.items() 
                       if 'categorical' in info.get('logical_type', '')]
numeric_features = [col for col, info in types.items() 
                   if any(t in info.get('logical_type', '') for t in ['continuous', 'float', 'integer']) 
                   and 'categorical' not in info.get('logical_type', '')]
datetime_features = [col for col, info in types.items() 
                    if info.get('logical_type', '') == 'datetime']

# Create pipeline with feature groups
pipeline = Pipeline()
pipeline.add_categorical_features(categorical_features)
pipeline.add_numeric_features(numeric_features)
pipeline.add_datetime_features(datetime_features)

# Add model and fit
pipeline.add_model(LightGBMModel())
pipeline.fit(df, target_column='target')
```