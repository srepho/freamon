# ShapIQ Integration

Freamon provides integrated support for ShapIQ, enabling powerful feature interaction detection and explainability capabilities. This documentation covers how to use ShapIQ with Freamon to:

1. Detect and visualize feature interactions
2. Create engineered features based on detected interactions
3. Generate interactive HTML reports with interaction analysis
4. Use ShapIQ's advanced explainability features

## Installation

ShapIQ is an optional dependency of Freamon. To use ShapIQ functionality, install it with:

```bash
# Install freamon with ShapIQ
pip install "freamon[explainability]"

# Or add ShapIQ directly
pip install shapiq shap
```

## Feature Interaction Detection

The `ShapIQFeatureEngineer` class automatically detects significant interactions between features:

```python
from freamon.features import ShapIQFeatureEngineer
from sklearn.ensemble import RandomForestRegressor

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Initialize ShapIQ feature engineer
shapiq_engineer = ShapIQFeatureEngineer(
    model=model,
    X=X_train,
    y=y_train,
    threshold=0.05,  # Minimum interaction strength 
    max_interactions=10  # Max number of interactions to consider
)

# Detect significant interactions
interactions = shapiq_engineer.detect_interactions()

# Print detected interactions
for feature1, feature2 in interactions:
    strength = shapiq_engineer.interaction_strengths[(feature1, feature2)]
    print(f"{feature1} × {feature2}: {strength:.4f}")
```

## Creating Interaction Features

Once you've detected interactions, you can automatically create engineered features based on them:

```python
# Create interaction features
X_train_enhanced = shapiq_engineer.create_features(
    operations=['multiply'],  # Available: multiply, divide, add, subtract, ratio 
    prefix='shapiq'  # Prefix for new feature names
)

# Apply the same transformations to test data
X_test_enhanced = shapiq_engineer.create_features(X_test)

# Train a model with enhanced features
enhanced_model = RandomForestRegressor(n_estimators=100, random_state=42)
enhanced_model.fit(X_train_enhanced, y_train)
enhanced_score = enhanced_model.score(X_test_enhanced, y_test)
print(f"Enhanced model R² score: {enhanced_score:.4f}")
```

## Complete ShapIQ Pipeline

For convenience, you can use the pipeline method to detect interactions and create features in one step:

```python
# Run the complete pipeline
X_train_enhanced, interaction_report = shapiq_engineer.pipeline(
    operations=['multiply', 'ratio']
)

# The interaction_report contains details about detected interactions
print(f"Detected {interaction_report['num_interactions']} significant interactions")
```

## Generating HTML Interaction Reports

Freamon can generate comprehensive HTML reports that visualize feature interactions:

```python
from freamon.eda.explainability_report import generate_interaction_report

# Generate an HTML report
report_path = "interactions_report.html"
generate_interaction_report(
    model=model,
    X=X_train,
    y=y_train,
    output_path=report_path,
    threshold=0.05,
    max_interactions=10
)
print(f"Report saved to {report_path}")
```

The generated HTML report includes:
- Bar charts of interaction strengths
- Detailed tables of feature interactions
- SHAP summary plots (if SHAP is installed)
- Feature importance analysis

## Advanced Explainability with ShapIQ

The `ShapIQExplainer` class provides direct access to ShapIQ's explainability capabilities:

```python
from freamon.explainability import ShapIQExplainer

# Create a ShapIQ explainer
explainer = ShapIQExplainer(model, max_order=2)
explainer.fit(X_train)

# Compute interaction values
interactions = explainer.explain(X_test.iloc[:5])  # Explain first 5 instances

# Plot main effects for the first instance
explainer.plot_main_effects(instance_idx=0)

# Plot interaction effects for the first instance
explainer.plot_interaction_effects(instance_idx=0)
```

## Integration with Model Training

ShapIQ seamlessly integrates with Freamon's model training workflow:

```python
from freamon.modeling import ModelTrainer

# Create and train a model
trainer = ModelTrainer(
    model_type='lightgbm',
    problem_type='regression'
)
trainer.train(X_train, y_train)

# Get ShapIQ-based feature importance
importance = trainer.get_feature_importance(method='shapiq', X=X_train)
print(importance)
```

## Performance Considerations

Computing ShapIQ interactions can be computationally intensive. Consider these tips:

1. Start with a small subset of your data for initial interaction discovery
2. Use a lower-complexity model for faster computation 
3. Limit the `max_order` parameter to 2 (pairwise interactions) for most use cases
4. Set a reasonable `threshold` to filter out weaker interactions

## Additional Resources

For more examples and advanced usage, refer to the example scripts:
- `examples/shapiq_example.py` - Basic ShapIQ usage
- `examples/shapiq_feature_engineering_example.py` - Complete feature engineering workflow