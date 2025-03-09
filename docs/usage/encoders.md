# Categorical Encoders

Freamon provides various encoders for categorical variables. These include basic encoders built on top of scikit-learn as well as advanced encoders from the `category_encoders` package.

## Basic Encoders

### One-Hot Encoder

```python
from freamon.utils.encoders import OneHotEncoderWrapper

# Create and fit the encoder
encoder = OneHotEncoderWrapper(columns=['category_column'])
transformed_df = encoder.fit_transform(df)
```

### Ordinal Encoder

```python
from freamon.utils.encoders import OrdinalEncoderWrapper

# Create and fit the encoder
encoder = OrdinalEncoderWrapper(
    columns=['category_column'], 
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
transformed_df = encoder.fit_transform(df)
```

### Target Encoder

```python
from freamon.utils.encoders import TargetEncoderWrapper

# Create and fit the encoder
encoder = TargetEncoderWrapper(
    columns=['category_column'],
    smoothing=10.0
)
transformed_df = encoder.fit_transform(df, target='target_column')
```

## Advanced Encoders

These encoders require the `category_encoders` package to be installed:

```bash
pip install category_encoders
```

### Binary Encoder

Binary encoding converts categories into a binary code representation, which can be more efficient than one-hot encoding for high-cardinality columns.

```python
from freamon.utils.encoders import BinaryEncoderWrapper

# Create and fit the encoder
encoder = BinaryEncoderWrapper(columns=['high_cardinality_column'])
transformed_df = encoder.fit_transform(df)
```

### Hashing Encoder

Hashing encoder applies a hash function to the categories, which is useful for very high-cardinality columns.

```python
from freamon.utils.encoders import HashingEncoderWrapper

# Create and fit the encoder
encoder = HashingEncoderWrapper(
    columns=['very_high_cardinality_column'],
    n_components=8  # Number of features to create
)
transformed_df = encoder.fit_transform(df)
```

### Weight of Evidence (WOE) Encoder

WOE encoding is particularly useful for classification tasks, as it incorporates target information.

```python
from freamon.utils.encoders import WOEEncoderWrapper

# Create and fit the encoder
encoder = WOEEncoderWrapper(columns=['category_column'])
transformed_df = encoder.fit_transform(df, target='target_column')
```