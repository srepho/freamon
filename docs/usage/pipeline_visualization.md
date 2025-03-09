# Pipeline Visualization

Freamon provides powerful tools for visualizing machine learning pipelines, making it easier to understand, document, and share complex workflows.

## Overview

The pipeline visualization module offers:

- Static flowchart generation of pipeline structure
- Interactive HTML reports with detailed pipeline information
- Support for multiple output formats (PNG, SVG, PDF)
- Code generation for recreating pipelines
- Detailed visualization of each pipeline step

## Basic Usage

### Creating a Simple Visualization

```python
from freamon.pipeline import Pipeline, visualize_pipeline

# Create a pipeline
pipeline = Pipeline([step1, step2, step3])

# Generate a visualization
visualize_pipeline(pipeline, output_path="pipeline_visualization.png")
```

### In-Memory Visualization

For notebooks or web applications, you can get a base64-encoded image:

```python
from freamon.pipeline import visualize_pipeline

# Get base64-encoded image
img_str = visualize_pipeline(pipeline, format="png")

# For display in a Jupyter notebook
from IPython.display import Image, display
import base64
import io

img_data = base64.b64decode(img_str.split(",")[1])
display(Image(data=img_data))
```

## Interactive HTML Reports

Generate rich interactive reports with detailed information about each step:

```python
from freamon.pipeline import generate_interactive_html

# Generate interactive HTML report
generate_interactive_html(
    pipeline, 
    output_path="pipeline_report.html",
    include_details=True,
    include_code_example=True
)
```

The interactive report includes:
- Visual pipeline flowchart
- Detailed information about each step
- Color-coding for different step types
- Code example for recreating the pipeline
- Responsive design for various screen sizes

## Customization Options

### Output Formats

```python
# Generate PNG (default)
visualize_pipeline(pipeline, output_path="pipeline.png", format="png")

# Generate SVG for vector graphics
visualize_pipeline(pipeline, output_path="pipeline.svg", format="svg")

# Generate PDF
visualize_pipeline(pipeline, output_path="pipeline.pdf", format="pdf")
```

### Visualization Detail Level

```python
# Simplified visualization without details
visualize_pipeline(pipeline, show_details=False)

# Full detailed visualization
visualize_pipeline(pipeline, show_details=True)
```

### Image Resolution

```python
# Higher resolution image
visualize_pipeline(pipeline, dpi=300)
```

## Backend Options

The visualization system automatically uses the best available backend:

1. **Graphviz**: Produces cleaner, more professional visualizations (preferred)
2. **Matplotlib**: Fallback option if Graphviz is not installed

To install Graphviz for better visualizations:

```bash
# Install the Python library
pip install graphviz

# On Ubuntu/Debian
apt-get install graphviz

# On macOS using Homebrew
brew install graphviz

# On Windows using conda
conda install graphviz
```

## Integrating with Documentation

Pipeline visualizations are excellent for documentation:

```python
from freamon.pipeline import visualize_pipeline
import os

# Generate visualizations for documentation
os.makedirs("docs/images", exist_ok=True)

# Create visualizations for different pipelines
for name, pipeline in pipelines.items():
    visualize_pipeline(
        pipeline,
        output_path=f"docs/images/{name}_pipeline",
        format="png"
    )
```

## Example: Complete Visualization Workflow

```python
import os
from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    ModelTrainingStep,
    visualize_pipeline,
    generate_interactive_html
)

# Create a simple pipeline
feature_step = FeatureEngineeringStep(name="feature_engineering")
feature_step.add_operation(
    method="add_polynomial_features",
    columns=["feature1", "feature2"],
    degree=2
)

model_step = ModelTrainingStep(
    name="model_training",
    model_type="lightgbm",
    problem_type="classification"
)

# Create pipeline
pipeline = Pipeline()
pipeline.add_step(feature_step)
pipeline.add_step(model_step)

# Create output directory
os.makedirs("docs/pipeline", exist_ok=True)

# Generate static visualization
visualize_pipeline(
    pipeline,
    output_path="docs/pipeline/flowchart",
    format="png",
    dpi=150
)

# Generate interactive HTML report
generate_interactive_html(
    pipeline,
    output_path="docs/pipeline/interactive_report.html"
)

print("Pipeline visualizations created in docs/pipeline/")
```

## Sharing and Collaboration

The interactive HTML reports are particularly useful for sharing with team members:

1. Generate the HTML report
2. Share the HTML file via email, Slack, or internal documentation
3. Recipients can open the file in any web browser without installing additional software

This enables clear communication about pipeline structure and improves collaboration between data scientists, engineers, and stakeholders.

## Tips for Effective Visualizations

- Use descriptive names for each step in your pipeline
- Keep pipelines modular with specific steps for each task
- For complex pipelines, consider creating separate visualizations for logical subsets
- Include the visualization in your project documentation
- Use the interactive HTML report for sharing detailed information
- Regenerate visualizations when pipeline structure changes