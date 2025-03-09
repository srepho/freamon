"""
Module for visualizing machine learning pipelines.

This module provides functions for visualizing the structure and flow 
of ML pipelines, enabling better understanding and communication of 
complex workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path
import base64
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

from freamon.pipeline.pipeline import Pipeline


def visualize_pipeline(
    pipeline: Pipeline, 
    output_path: Optional[str] = None,
    format: str = 'png',
    show_details: bool = True,
    dpi: int = 150
) -> Union[str, None]:
    """
    Visualize a pipeline as a flowchart.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to visualize.
    output_path : Optional[str], default=None
        Path to save the visualization. If None, will display the visualization.
    format : str, default='png'
        Output format ('png', 'svg', 'pdf')
    show_details : bool, default=True
        Whether to show detailed information about each step.
    dpi : int, default=150
        Resolution of the output image.
        
    Returns
    -------
    Union[str, None]
        If output_path is None, returns the visualization as a base64-encoded string.
        Otherwise, saves the visualization to the specified path and returns None.
    """
    if GRAPHVIZ_AVAILABLE:
        return _visualize_pipeline_graphviz(pipeline, output_path, format, show_details)
    else:
        return _visualize_pipeline_matplotlib(pipeline, output_path, format, show_details, dpi)


def _visualize_pipeline_graphviz(
    pipeline: Pipeline, 
    output_path: Optional[str] = None,
    format: str = 'png',
    show_details: bool = True
) -> Union[str, None]:
    """
    Visualize a pipeline using Graphviz.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to visualize.
    output_path : Optional[str], default=None
        Path to save the visualization. If None, will display the visualization.
    format : str, default='png'
        Output format ('png', 'svg', 'pdf')
    show_details : bool, default=True
        Whether to show detailed information about each step.
        
    Returns
    -------
    Union[str, None]
        If output_path is None, returns the visualization as a base64-encoded string.
        Otherwise, saves the visualization to the specified path and returns None.
    """
    if not GRAPHVIZ_AVAILABLE:
        raise ImportError("Graphviz is not installed. Please install it with 'pip install graphviz'.")
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='ML Pipeline Visualization')
    dot.attr(rankdir='TB', size='8,8')  # Top to bottom layout
    
    # Add nodes for each step
    for i, step in enumerate(pipeline.steps):
        step_info = f"Step {i+1}: {step.name}"
        step_type = step.__class__.__name__
        
        # Add more details if requested
        if show_details:
            if hasattr(step, 'operations') and isinstance(step.operations, list):
                operations = []
                for op in step.operations:
                    if isinstance(op, dict) and 'method' in op:
                        operations.append(op['method'])
                if operations:
                    step_info += f"\nOperations: {', '.join(operations)}"
            
            if hasattr(step, 'model_type'):
                step_info += f"\nModel: {step.model_type}"
                
            if hasattr(step, 'n_features'):
                step_info += f"\nFeatures: {step.n_features or 'All'}"
        
        # Set node style based on step type
        if "FeatureEngineering" in step_type:
            color = "lightblue"
        elif "FeatureSelection" in step_type:
            color = "lightgreen"
        elif "Model" in step_type:
            color = "lightcoral"
        elif "Evaluation" in step_type:
            color = "gold"
        else:
            color = "white"
        
        # Add node
        dot.node(f'step_{i}', step_info, style='filled', fillcolor=color)
    
    # Add edges
    for i in range(len(pipeline.steps) - 1):
        dot.edge(f'step_{i}', f'step_{i+1}')
    
    # Add input and output nodes
    dot.node('input', 'Input Data', shape='parallelogram', style='filled', fillcolor='lightgrey')
    dot.edge('input', 'step_0')
    
    # Check if any step is a model step
    has_model_step = any("Model" in step.__class__.__name__ for step in pipeline.steps)
    if has_model_step:
        dot.node('output', 'Predictions', shape='parallelogram', style='filled', fillcolor='lightgrey')
        dot.edge(f'step_{len(pipeline.steps)-1}', 'output')
    
    # Render the graph
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        dot.render(output_path, format=format, cleanup=True)
        return None
    else:
        img = dot.pipe(format=format)
        b64_img = base64.b64encode(img).decode('utf-8')
        return f"data:image/{format};base64,{b64_img}"


def _visualize_pipeline_matplotlib(
    pipeline: Pipeline, 
    output_path: Optional[str] = None,
    format: str = 'png',
    show_details: bool = True,
    dpi: int = 150
) -> Union[str, None]:
    """
    Visualize a pipeline using Matplotlib (fallback for when Graphviz is not available).
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to visualize.
    output_path : Optional[str], default=None
        Path to save the visualization. If None, will display the visualization.
    format : str, default='png'
        Output format ('png', 'svg', 'pdf')
    show_details : bool, default=True
        Whether to show detailed information about each step.
    dpi : int, default=150
        Resolution of the output image.
        
    Returns
    -------
    Union[str, None]
        If output_path is None, returns the visualization as a base64-encoded string.
        Otherwise, saves the visualization to the specified path and returns None.
    """
    # Calculate the number of steps
    n_steps = len(pipeline.steps) + 2  # +2 for input and output
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, n_steps*0.8), dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_steps)
    
    # Turn off axis
    ax.axis('off')
    
    # Draw boxes for each step
    box_width = 8
    box_height = 0.6
    x_start = 1
    
    # Add input box
    y_pos = n_steps - 0.8
    rect = patches.Rectangle((x_start, y_pos), box_width, box_height, 
                            linewidth=1, edgecolor='black', facecolor='lightgrey')
    ax.add_patch(rect)
    ax.text(x_start + box_width/2, y_pos + box_height/2, 'Input Data', 
           ha='center', va='center')
    
    # Add arrow
    y_arrow_end = y_pos
    y_arrow_start = y_pos - 0.2
    ax.arrow(x_start + box_width/2, y_arrow_start, 0, -0.3, 
             head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add steps
    for i, step in enumerate(pipeline.steps):
        y_pos = n_steps - 1.5 - i
        
        # Determine box color based on step type
        step_type = step.__class__.__name__
        if "FeatureEngineering" in step_type:
            color = "lightblue"
        elif "FeatureSelection" in step_type:
            color = "lightgreen"
        elif "Model" in step_type:
            color = "lightcoral"
        elif "Evaluation" in step_type:
            color = "gold"
        else:
            color = "white"
        
        # Draw box
        rect = patches.Rectangle((x_start, y_pos), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Add step info
        step_info = f"Step {i+1}: {step.name}"
        
        # Add more details if requested
        additional_info = []
        if show_details:
            if hasattr(step, 'operations') and isinstance(step.operations, list):
                operations = []
                for op in step.operations:
                    if isinstance(op, dict) and 'method' in op:
                        operations.append(op['method'].replace('add_', ''))
                if operations:
                    additional_info.append(f"Ops: {', '.join(operations)}")
            
            if hasattr(step, 'model_type'):
                additional_info.append(f"Model: {step.model_type}")
                
            if hasattr(step, 'n_features') and step.n_features is not None:
                additional_info.append(f"Features: {step.n_features}")
        
        # Add text
        ax.text(x_start + box_width/2, y_pos + box_height*0.7, step_info, 
               ha='center', va='center', fontsize=10)
        
        if additional_info:
            ax.text(x_start + box_width/2, y_pos + box_height*0.3, 
                   ' | '.join(additional_info), 
                   ha='center', va='center', fontsize=8)
        
        # Add arrow (except for the last step)
        if i < len(pipeline.steps) - 1:
            y_arrow_end = y_pos
            y_arrow_start = y_pos - 0.2
            ax.arrow(x_start + box_width/2, y_arrow_start, 0, -0.3, 
                     head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add output box for models
    if any("Model" in s.__class__.__name__ for s in pipeline.steps):
        y_pos = 0.8
        rect = patches.Rectangle((x_start, y_pos), box_width, box_height, 
                                linewidth=1, edgecolor='black', facecolor='lightgrey')
        ax.add_patch(rect)
        ax.text(x_start + box_width/2, y_pos + box_height/2, 'Predictions', 
               ha='center', va='center')
        
        # Add arrow to output
        y_arrow_end = n_steps - 1.5 - (len(pipeline.steps) - 1) - 0.2
        ax.arrow(x_start + box_width/2, y_arrow_end, 0, -0.3, 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Add title
    ax.text(5, n_steps - 0.2, 'Pipeline Visualization', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Save or return
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.tight_layout()
        plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/{format};base64,{img_str}"


def generate_interactive_html(
    pipeline: Pipeline,
    output_path: str,
    include_details: bool = True,
    include_code_example: bool = True
) -> None:
    """
    Generate an interactive HTML visualization of a pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to visualize.
    output_path : str
        Path to save the HTML file.
    include_details : bool, default=True
        Whether to include detailed information about each step.
    include_code_example : bool, default=True
        Whether to include code example for recreating the pipeline.
    """
    # Import Jinja2 here to avoid dependency issues if not needed
    import jinja2
    
    # Get pipeline summary
    summary = pipeline.summary()
    
    # Generate pipeline visualization
    pipeline_img = visualize_pipeline(pipeline, format='png', show_details=include_details)
    
    # Generate code example if requested
    code_example = None
    if include_code_example:
        code_example = _generate_code_example(pipeline)
    
    # Create a template for the HTML
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pipeline Visualization</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            .section { margin-bottom: 30px; }
            .pipeline-img { max-width: 100%; height: auto; }
            .step-card { margin-bottom: 15px; }
            .step-card.feature-engineering { border-left: 5px solid #9ec5fe; }
            .step-card.feature-selection { border-left: 5px solid #a3cfbb; }
            .step-card.model-training { border-left: 5px solid #f1aeb5; }
            .step-card.evaluation { border-left: 5px solid #ffe69c; }
            .step-card.custom { border-left: 5px solid #d3d3d3; }
            .card-header { font-weight: bold; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
            .card-img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">Pipeline Visualization</h1>
            
            <!-- Pipeline Diagram -->
            <div class="section">
                <div class="card">
                    <div class="card-header">
                        <h2>Pipeline Flow</h2>
                    </div>
                    <div class="card-body">
                        <img src="{{ pipeline_img }}" class="pipeline-img" alt="Pipeline Visualization">
                    </div>
                </div>
            </div>
            
            <!-- Pipeline Steps -->
            <div class="section">
                <div class="card">
                    <div class="card-header">
                        <h2>Pipeline Steps</h2>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            {% for step in summary.steps %}
                            <div class="col-md-12">
                                {% if "FeatureEngineering" in step.type %}
                                <div class="card step-card feature-engineering">
                                {% elif "FeatureSelection" in step.type %}
                                <div class="card step-card feature-selection">
                                {% elif "Model" in step.type %}
                                <div class="card step-card model-training">
                                {% elif "Evaluation" in step.type %}
                                <div class="card step-card evaluation">
                                {% else %}
                                <div class="card step-card custom">
                                {% endif %}
                                    <div class="card-header">
                                        {{ loop.index }}. {{ step.name }} ({{ step.type }})
                                    </div>
                                    <div class="card-body">
                                        <p><strong>Type:</strong> {{ step.type }}</p>
                                        <p><strong>Is Fitted:</strong> {{ step.is_fitted }}</p>
                                        
                                        {% if step.properties %}
                                        <h6>Properties:</h6>
                                        <ul>
                                            {% for key, value in step.properties.items() %}
                                            <li><strong>{{ key }}:</strong> {{ value }}</li>
                                            {% endfor %}
                                        </ul>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
            
            {% if code_example %}
            <!-- Code Example -->
            <div class="section">
                <div class="card">
                    <div class="card-header">
                        <h2>Code Example</h2>
                    </div>
                    <div class="card-body">
                        <pre><code>{{ code_example }}</code></pre>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Extend the summary with step properties
    for i, step in enumerate(summary["steps"]):
        # Add properties based on step type
        step["properties"] = {}
        
        pipeline_step = pipeline.steps[i]
        
        if "FeatureEngineering" in step["type"]:
            if hasattr(pipeline_step, "operations"):
                operations = []
                for op in pipeline_step.operations:
                    if isinstance(op, dict) and "method" in op:
                        method = op["method"]
                        params = op.get("params", {})
                        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                        operations.append(f"{method}({params_str})")
                step["properties"]["Operations"] = operations
        
        elif "FeatureSelection" in step["type"]:
            if hasattr(pipeline_step, "method"):
                step["properties"]["Method"] = pipeline_step.method
            if hasattr(pipeline_step, "n_features"):
                step["properties"]["Number of Features"] = pipeline_step.n_features
        
        elif "Model" in step["type"]:
            if hasattr(pipeline_step, "model_type"):
                step["properties"]["Model Type"] = pipeline_step.model_type
            if hasattr(pipeline_step, "problem_type"):
                step["properties"]["Problem Type"] = pipeline_step.problem_type
            if hasattr(pipeline_step, "hyperparameters"):
                step["properties"]["Hyperparameters"] = str(pipeline_step.hyperparameters)
        
        elif "Evaluation" in step["type"]:
            if hasattr(pipeline_step, "metrics"):
                step["properties"]["Metrics"] = ", ".join(pipeline_step.metrics)
            if hasattr(pipeline_step, "problem_type"):
                step["properties"]["Problem Type"] = pipeline_step.problem_type
    
    # Render the template
    template = jinja2.Template(template_str)
    html_content = template.render(
        pipeline_img=pipeline_img,
        summary=summary,
        code_example=code_example
    )
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the HTML to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive HTML visualization saved to {output_path}")


def _generate_code_example(pipeline: Pipeline) -> str:
    """
    Generate Python code example for recreating the pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to generate code for.
        
    Returns
    -------
    str
        Python code example.
    """
    code_lines = [
        "from freamon.pipeline import (",
        "    Pipeline,",
        "    FeatureEngineeringStep,",
        "    FeatureSelectionStep,",
        "    ModelTrainingStep,",
        "    EvaluationStep",
        ")",
        "",
        "# Create pipeline steps"
    ]
    
    # Generate code for each step
    for i, step in enumerate(pipeline.steps):
        step_type = step.__class__.__name__
        step_var = f"{step.name.lower().replace(' ', '_')}_step"
        
        if "FeatureEngineering" in step_type and not "ShapIQ" in step_type:
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            code_lines.append(f"{step_var} = FeatureEngineeringStep(name=\"{step.name}\")")
            
            # Add operations
            if hasattr(step, "operations"):
                for op in step.operations:
                    if isinstance(op, dict) and "method" in op:
                        method = op["method"]
                        params = op.get("params", {})
                        params_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                        code_lines.append(f"{step_var}.add_operation(\n    method=\"{method}\",\n    {params_str}\n)")
        
        elif "ShapIQ" in step_type:
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            params = []
            if hasattr(step, "model_type"):
                params.append(f"model_type=\"{step.model_type}\"")
            if hasattr(step, "n_interactions"):
                params.append(f"n_interactions={step.n_interactions}")
            if hasattr(step, "max_interaction_size"):
                params.append(f"max_interaction_size={step.max_interaction_size}")
            if hasattr(step, "categorical_features") and step.categorical_features:
                params.append(f"categorical_features={repr(step.categorical_features)}")
                
            params_str = ",\n    ".join(params)
            code_lines.append(f"{step_var} = ShapIQFeatureEngineeringStep(\n    name=\"{step.name}\",\n    {params_str}\n)")
            
        elif "FeatureSelection" in step_type:
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            params = []
            if hasattr(step, "method"):
                params.append(f"method=\"{step.method}\"")
            if hasattr(step, "n_features"):
                params.append(f"n_features={step.n_features}")
            if hasattr(step, "threshold") and step.threshold is not None:
                params.append(f"threshold={step.threshold}")
            if hasattr(step, "features_to_keep") and step.features_to_keep:
                params.append(f"features_to_keep={repr(step.features_to_keep)}")
                
            params_str = ",\n    ".join(params)
            code_lines.append(f"{step_var} = FeatureSelectionStep(\n    name=\"{step.name}\",\n    {params_str}\n)")
            
        elif "Model" in step_type:
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            params = []
            if hasattr(step, "model_type"):
                params.append(f"model_type=\"{step.model_type}\"")
            if hasattr(step, "problem_type"):
                params.append(f"problem_type=\"{step.problem_type}\"")
            if hasattr(step, "eval_metric") and step.eval_metric:
                params.append(f"eval_metric=\"{step.eval_metric}\"")
            if hasattr(step, "hyperparameters") and step.hyperparameters:
                hyperparams_str = "{\n        " + ",\n        ".join(f'"{k}": {repr(v)}' for k, v in step.hyperparameters.items()) + "\n    }"
                params.append(f"hyperparameters={hyperparams_str}")
            if hasattr(step, "cv_folds") and step.cv_folds > 0:
                params.append(f"cv_folds={step.cv_folds}")
                
            params_str = ",\n    ".join(params)
            code_lines.append(f"{step_var} = ModelTrainingStep(\n    name=\"{step.name}\",\n    {params_str}\n)")
            
        elif "Evaluation" in step_type:
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            params = []
            if hasattr(step, "metrics") and step.metrics:
                metrics_str = repr(step.metrics)
                params.append(f"metrics={metrics_str}")
            if hasattr(step, "problem_type"):
                params.append(f"problem_type=\"{step.problem_type}\"")
                
            params_str = ",\n    ".join(params)
            code_lines.append(f"{step_var} = EvaluationStep(\n    name=\"{step.name}\",\n    {params_str}\n)")
            
        else:
            # Generic step for custom steps
            code_lines.append(f"\n# Step {i+1}: {step.name}")
            code_lines.append(f"# Custom step of type {step_type}")
            code_lines.append(f"{step_var} = {step_type}(name=\"{step.name}\")")
    
    # Create pipeline
    code_lines.append("\n# Create pipeline")
    code_lines.append("pipeline = Pipeline()")
    
    # Add steps to pipeline
    for step in pipeline.steps:
        step_var = f"{step.name.lower().replace(' ', '_')}_step"
        code_lines.append(f"pipeline.add_step({step_var})")
    
    # Usage example
    code_lines.extend([
        "",
        "# Example usage",
        "# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)",
        "# pipeline.fit(X_train, y_train)",
        "# predictions = pipeline.predict(X_test)"
    ])
    
    return "\n".join(code_lines)