"""
Generate HTML reports for model explainability.
"""
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap package is not installed. Some reporting features will be limited.")

try:
    import shapiq
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False
    warnings.warn("shapiq package is not installed. ShapIQ reporting features will be limited.")

from freamon.explainability import ShapExplainer, ShapIQExplainer
from freamon.features.shapiq_engineer import ShapIQFeatureEngineer


def generate_interaction_report(
    model: Any,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    output_path: str,
    title: str = "Feature Interactions Report",
    theme: str = "cosmo",
    max_order: int = 2,
    threshold: float = 0.05,
    max_interactions: int = 10,
) -> Dict[str, Any]:
    """
    Generate an HTML report of feature interactions using ShapIQ.
    
    Parameters
    ----------
    model : Any
        The model to explain. Should have a predict method.
    X : pd.DataFrame
        The data to use for explaining the model.
    y : Union[pd.Series, np.ndarray]
        The target variable.
    output_path : str
        The path to save the HTML report.
    title : str, default="Feature Interactions Report"
        The title of the report.
    theme : str, default="cosmo"
        The Bootstrap theme to use.
    max_order : int, default=2
        Maximum interaction order to compute.
    threshold : float, default=0.05
        Minimum interaction strength threshold.
    max_interactions : int, default=10
        Maximum number of top interactions to show.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing information about detected interactions.
    """
    # Check for shapiq
    if not SHAPIQ_AVAILABLE:
        raise ImportError("shapiq package is required for interaction analysis.")
    
    # Validate theme
    valid_themes = [
        'cosmo', 'flatly', 'journal', 'lumen', 'sandstone',
        'simplex', 'spacelab', 'united', 'yeti'
    ]
    if theme not in valid_themes:
        print(f"Warning: Invalid theme '{theme}'. Using 'cosmo' instead.")
        theme = 'cosmo'
    
    # Create the ShapIQ feature engineer to detect interactions
    engineer = ShapIQFeatureEngineer(
        model=model,
        X=X,
        y=y,
        max_order=max_order,
        threshold=threshold,
        max_interactions=max_interactions,
    )
    
    # Run the pipeline
    _, interaction_report = engineer.pipeline()
    
    # Generate SHAP summary plots
    shap_plots = {}
    if SHAP_AVAILABLE:
        # Create model type
        try:
            model_type = 'tree' if hasattr(model, 'feature_importances_') else 'kernel'
            explainer = ShapExplainer(model, model_type=model_type)
            explainer.fit(X)
            
            # Calculate SHAP values for a sample of the data
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            sample_X = X.iloc[sample_indices]
            
            shap_values = explainer.explain(sample_X)
            
            # Create a summary plot
            plt.figure(figsize=(10, 8))
            explainer.summary_plot(shap_values, sample_X, plot_type='bar', show=False)
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            shap_plots["summary"] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
            plt.close()
            
            # Create a summary beeswarm plot
            plt.figure(figsize=(10, 8))
            explainer.summary_plot(shap_values, sample_X, plot_type='dot', show=False)
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            shap_plots["beeswarm"] = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate SHAP plots. Error: {e}")
    
    # Start building HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootswatch@5.2.3/dist/{theme}/bootstrap.min.css">
        <style>
            body {{ padding-top: 20px; padding-bottom: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .card {{ margin-bottom: 20px; }}
            .table-responsive {{ margin-bottom: 20px; }}
            .plot-img {{ max-width: 100%; height: auto; }}
            .nav-pills .nav-link.active {{ background-color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">{title}</h1>
            <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="alert alert-info" role="alert">
                This report provides an analysis of feature interactions and importance detected by ShapIQ and SHAP.
            </div>
            
            <ul class="nav nav-pills mb-4" id="explainability-tabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="interaction-tab" data-bs-toggle="pill" data-bs-target="#interaction" 
                    type="button" role="tab" aria-controls="interaction" aria-selected="true">Feature Interactions</button>
                </li>
    """
    
    if shap_plots:
        html += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="importance-tab" data-bs-toggle="pill" data-bs-target="#importance" 
                    type="button" role="tab" aria-controls="importance" aria-selected="false">Feature Importance</button>
                </li>
        """
    
    html += """
            </ul>
            
            <div class="tab-content" id="explainability-tab-content">
    """
    
    # Feature Interactions tab
    html += """
                <div class="tab-pane fade show active" id="interaction" role="tabpanel" aria-labelledby="interaction-tab">
                    <div class="section">
                        <h2>Feature Interactions Analysis</h2>
                        <p>
                            This section shows the most significant feature interactions detected by ShapIQ,
                            which can be used to create new engineered features to improve model performance.
                        </p>
    """
    
    # Add interaction plot if available
    if interaction_report["plot"]:
        html += f"""
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">Top Feature Interactions</h5>
                            </div>
                            <div class="card-body">
                                <div class="text-center">
                                    <img src="{interaction_report["plot"]}" class="plot-img" alt="Feature Interactions">
                                </div>
                            </div>
                        </div>
        """
    
    # Add interaction details
    if interaction_report["interactions"]:
        html += """
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">Interaction Details</h5>
                            </div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Feature 1</th>
                                            <th>Feature 2</th>
                                            <th>Interaction Strength</th>
                                        </tr>
                                    </thead>
                                    <tbody>
        """
        
        # Add rows for each interaction
        for interaction in interaction_report["interactions"]:
            html += f"""
                                        <tr>
                                            <td>{interaction["feature1"]}</td>
                                            <td>{interaction["feature2"]}</td>
                                            <td>{interaction["strength"]:.4f}</td>
                                        </tr>
            """
        
        html += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
        """
    else:
        html += """
                        <div class="alert alert-warning" role="alert">
                            No significant interactions were detected with the current threshold.
                            Try lowering the threshold or using a different model.
                        </div>
        """
    
    html += """
                    </div>
                </div>
    """
    
    # Feature Importance tab (SHAP)
    if shap_plots:
        html += """
                <div class="tab-pane fade" id="importance" role="tabpanel" aria-labelledby="importance-tab">
                    <div class="section">
                        <h2>Feature Importance Analysis</h2>
                        <p>
                            This section shows the global feature importance and value impact using SHAP.
                        </p>
        """
        
        if "summary" in shap_plots:
            html += f"""
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">SHAP Feature Importance</h5>
                            </div>
                            <div class="card-body">
                                <div class="text-center">
                                    <img src="{shap_plots["summary"]}" class="plot-img" alt="SHAP Feature Importance">
                                </div>
                                <p class="mt-3 text-muted">
                                    Features are ranked by the mean absolute SHAP value (average impact on model output).
                                </p>
                            </div>
                        </div>
            """
        
        if "beeswarm" in shap_plots:
            html += f"""
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">SHAP Feature Values</h5>
                            </div>
                            <div class="card-body">
                                <div class="text-center">
                                    <img src="{shap_plots["beeswarm"]}" class="plot-img" alt="SHAP Feature Values">
                                </div>
                                <p class="mt-3 text-muted">
                                    Each dot represents a sample. Position on the x-axis shows the impact on the prediction.
                                    Colors show the feature value (red = high, blue = low).
                                </p>
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
        """
    
    # Close the main container and add scripts
    html += """
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Interaction report saved to {output_path}")
    
    return interaction_report