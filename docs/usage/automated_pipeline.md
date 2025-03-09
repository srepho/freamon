# Automated End-to-End Pipeline

This guide explains how to use Freamon's automated pipeline capabilities to create complete machine learning workflows with comprehensive reporting.

## Overview

The automated pipeline functionality in Freamon combines several key components:

1. **Data Quality Analysis** - Detect and handle issues in your data
2. **Drift Detection** - Monitor changes in data distributions
3. **Exploratory Data Analysis** - Understand feature relationships and patterns
4. **Feature Engineering** - Create new features and handle existing ones
5. **Model Training & Evaluation** - Build and assess machine learning models
6. **Comprehensive Reporting** - Generate detailed HTML reports at each stage
7. **Integrated Dashboard** - Access all reports through a single interface

## Basic Usage

```python
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    ModelTrainingStep,
    EvaluationStep
)
from freamon.data_quality.analyzer import DataQualityAnalyzer

# Analyze data quality first
analyzer = DataQualityAnalyzer(X_train)
quality_results = analyzer.run_all_analyses()
analyzer.generate_report("data_quality_report.html")
X_train_clean = analyzer.get_clean_data()

# Create and run a basic pipeline
pipeline = Pipeline()
pipeline.add_step(FeatureEngineeringStep(feature_engineer))
pipeline.add_step(ModelTrainingStep(model_trainer))
pipeline.add_step(EvaluationStep(evaluator))

# Fit the pipeline
pipeline.fit(X_train_clean, y_train)

# Generate reports
from freamon.pipeline.visualization import generate_interactive_html
generate_interactive_html(pipeline, "pipeline_report.html")
```

## Complete Automated Workflow

For a complete end-to-end workflow with all reporting features, see the `automated_end_to_end_pipeline.py` example in the examples directory. This example demonstrates:

1. Loading and preparing data
2. Running data quality checks
3. Detecting data drift
4. Conducting exploratory analysis
5. Building a pipeline with feature engineering, interaction detection, and model training
6. Generating comprehensive reports
7. Creating an integrated dashboard

## Key Components

### Data Quality Analysis

```python
from freamon.data_quality.analyzer import DataQualityAnalyzer

analyzer = DataQualityAnalyzer(X_train)
quality_results = analyzer.run_all_analyses()
analyzer.generate_report("data_quality_report.html")
X_train_clean = analyzer.get_clean_data()
```

### Drift Detection

```python
from freamon.data_quality.drift import DataDriftDetector

detector = DataDriftDetector(reference_data, current_data)
drift_results = detector.detect_all_drift()
detector.generate_drift_report("drift_report.html")
```

### Exploratory Analysis

```python
from freamon.eda.analyzer import EDAAnalyzer

eda = EDAAnalyzer(X, y)
results = eda.run_full_analysis()
eda.generate_report("eda_report.html")
```

### Feature Engineering

```python
from freamon.features.engineer import FeatureEngineer
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer

# Basic feature engineering
feature_engineer = FeatureEngineer(
    categorical_encoding="target",
    datetime_encoding="cyclic",
    handle_missing=True
)

# Interaction detection with ShapIQ
shapiq_engineer = ShapIQFeatureEngineer(
    max_interactions=5,
    interaction_type="FSII"
)
```

### Pipeline Construction

```python
from freamon.pipeline import Pipeline
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    ShapIQFeatureEngineeringStep,
    ModelTrainingStep
)

pipeline = Pipeline()
pipeline.add_step(FeatureEngineeringStep(feature_engineer))
pipeline.add_step(ShapIQFeatureEngineeringStep(shapiq_engineer))
pipeline.add_step(ModelTrainingStep(model_trainer))
```

### Reporting

```python
from freamon.pipeline.visualization import (
    visualize_pipeline,
    generate_interactive_html
)
from freamon.eda.explainability_report import generate_interaction_report

# Generate static pipeline visualization
visualize_pipeline(pipeline, "pipeline_visualization.png")

# Generate interactive HTML report
generate_interactive_html(
    pipeline,
    "interactive_pipeline.html",
    include_code=True
)

# Generate feature interaction report
generate_interaction_report(
    model=pipeline.get_step("Model Training").get_model(),
    X=X_test,
    feature_names=X_test.columns,
    output_file="feature_interactions.html"
)
```

## Creating an Integrated Dashboard

The integrated dashboard provides a single entry point to access all reports generated during the pipeline process. You can create a custom dashboard using standard HTML/CSS and link to all the generated reports:

```python
def generate_dashboard(output_dir):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Pipeline Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container">
            <h1>Machine Learning Pipeline Dashboard</h1>
            <div class="row">
                <div class="col-md-4">
                    <a href="data_quality_report.html" class="btn btn-primary">Data Quality Report</a>
                </div>
                <div class="col-md-4">
                    <a href="drift_report.html" class="btn btn-primary">Drift Detection Report</a>
                </div>
                <div class="col-md-4">
                    <a href="interactive_pipeline.html" class="btn btn-primary">Pipeline Report</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "dashboard.html"), "w") as f:
        f.write(html_content)
```

## Customizing the Pipeline

The automated pipeline is highly customizable. You can:

- Add or remove specific steps as needed
- Configure each component with custom parameters
- Add custom steps by extending the `PipelineStep` class
- Modify the reporting outputs by customizing the HTML templates

## Best Practices

1. **Always check data quality first** - Data quality issues can cascade into model performance problems
2. **Monitor for drift regularly** - Data distributions can change over time
3. **Save all pipeline artifacts** - Use `pipeline.save("path/to/save")` to preserve the entire workflow
4. **Include all reports in your documentation** - Comprehensive reporting helps with model governance
5. **Use interactive HTML reports for sharing** - These reports can be viewed in any browser without additional software

## Conclusion

The automated pipeline functionality in Freamon provides a powerful framework for creating end-to-end machine learning workflows with comprehensive reporting. By combining data quality checks, feature engineering, model training, and detailed visualization, you can create robust and reproducible machine learning pipelines.