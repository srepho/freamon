# Explainability

Freamon provides tools for explaining model predictions using SHAP and ShapIQ integrations.

## SHAP Explainer

The `ShapExplainer` provides a wrapper around SHAP (SHapley Additive exPlanations) to explain model predictions.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from freamon.explainability import ShapExplainer

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create and fit the explainer
explainer = ShapExplainer(model, model_type='tree')
explainer.fit(X_train)

# Generate SHAP values for predictions
shap_values = explainer.explain(X_test)

# Create summary plot
explainer.summary_plot(shap_values, X_test)
```

## ShapIQ Explainer

The `ShapIQExplainer` extends model interpretability by detecting feature interactions using ShapIQ.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from freamon.explainability import ShapIQExplainer

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create and fit the ShapIQ explainer
explainer = ShapIQExplainer(model, max_order=2)
explainer.fit(X_train)

# Generate interaction values
interactions = explainer.explain(X_test)

# Analyze main effects
explainer.plot_main_effects(instance_idx=0)

# Analyze pairwise interactions
explainer.plot_interaction_effects(instance_idx=0)
```

## Automatic Feature Engineering with ShapIQ

The `ShapIQFeatureEngineer` class can automatically detect important feature interactions and create engineered features.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a feature engineer
engineer = ShapIQFeatureEngineer(
    model=model,
    X=X_train,
    y=y_train,
    threshold=0.05,
    max_interactions=10
)

# Detect interactions and create features
df_with_features, report = engineer.pipeline(operations=['multiply'])

# Train a new model with the interaction features
model_enhanced = RandomForestRegressor(n_estimators=100, random_state=42)
model_enhanced.fit(df_with_features, y_train)
```

## Explainability Reports

Freamon can generate interactive HTML reports to visualize interaction analysis:

```python
from freamon.eda.explainability_report import generate_interaction_report

# Generate an HTML report of interactions
generate_interaction_report(
    model=model,
    X=X_train,
    y=y_train,
    output_path="interaction_report.html",
    threshold=0.05,
    max_interactions=10
)
```