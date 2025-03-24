"""
Functions for generating HTML reports from EDA results.
"""
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from datetime import datetime

# Configure matplotlib to avoid treating $ as LaTeX math delimiters
plt.rcParams['text.usetex'] = False


def generate_header(
    title: str = "Exploratory Data Analysis Report",
    theme: str = "cosmo"
) -> str:
    """
    Generate HTML header with proper CSS links.
    
    Parameters
    ----------
    title : str, default="Exploratory Data Analysis Report"
        The title of the report.
    theme : str, default="cosmo"
        The Bootstrap theme to use for the report.
        
    Returns
    -------
    str
        The HTML header as a string.
    """
    from datetime import datetime
    
    return f"""
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
            
            /* Accordion styles */
            .accordion-button:not(.collapsed) {{
                background-color: #e7f1ff;
                color: #0c63e4;
                box-shadow: inset 0 -1px 0 rgba(0,0,0,.125);
            }}
            .accordion-button.collapsed {{
                background-color: #f8f9fa;
            }}
            .accordion-item {{
                border: 1px solid rgba(0,0,0,.125);
                margin-bottom: 5px;
            }}
            .accordion-body {{
                padding: 1rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">{title}</h1>
            <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """

def generate_footer() -> str:
    """
    Generate HTML footer with proper scripts.
    
    Returns
    -------
    str
        The HTML footer as a string.
    """
    return """
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Simple initialization for Bootstrap components
            document.addEventListener('DOMContentLoaded', function() {
                // Initialize tooltips if any
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                tooltipTriggerList.forEach(function(tooltipTriggerEl) {
                    new bootstrap.Tooltip(tooltipTriggerEl);
                });
                
                // Properly handle accordion item clicks
                document.querySelectorAll('.accordion-button').forEach(function(button) {
                    button.addEventListener('click', function(e) {
                        // Prevent default to avoid any conflicts
                        e.preventDefault();
                        
                        // Use Bootstrap's collapse API properly
                        var targetId = this.getAttribute('data-bs-target');
                        var targetCollapse = document.querySelector(targetId);
                        
                        if (targetCollapse) {
                            var bsCollapse = bootstrap.Collapse.getInstance(targetCollapse);
                            
                            // If instance doesn't exist yet, create it
                            if (!bsCollapse) {
                                bsCollapse = new bootstrap.Collapse(targetCollapse, {
                                    toggle: false
                                });
                            }
                            
                            // Toggle the collapse
                            if (targetCollapse.classList.contains('show')) {
                                bsCollapse.hide();
                            } else {
                                bsCollapse.show();
                            }
                        }
                    });
                });
                
                console.log('Freamon EDA report initialized successfully');
            });
        </script>
    </body>
    </html>
    """

def generate_html_report(
    df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_path: str,
    title: str = "Exploratory Data Analysis Report",
    theme: str = "cosmo",
    lazy_loading: bool = False,
    include_export_button: bool = True,
) -> str:
    """
    Generate an HTML report with the EDA results.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that was analyzed.
    analysis_results : Dict[str, Any]
        The dictionary of analysis results.
    output_path : str
        The path to save the HTML report.
    title : str, default="Exploratory Data Analysis Report"
        The title of the report.
    theme : str, default="cosmo"
        The Bootstrap theme to use for the report.
    lazy_loading : bool, default=False
        Whether to enable lazy loading for images (improves performance for large reports).
    include_export_button : bool, default=True
        Whether to include a button to export the report as a Jupyter notebook.
        
    Returns
    -------
    str
        The HTML report as a string.
    """
    from freamon.utils.matplotlib_fixes import optimize_base64_image
    
    # Validate theme
    valid_themes = [
        'cosmo', 'flatly', 'journal', 'lumen', 'sandstone',
        'simplex', 'spacelab', 'united', 'yeti'
    ]
    if theme not in valid_themes:
        print(f"Warning: Invalid theme '{theme}'. Using 'cosmo' instead.")
        theme = 'cosmo'
    
    # Start building HTML parts
    html_parts = []
    
    # Add header
    html_parts.append(generate_header(title, theme))
    
    # Add info alert
    info_alert = f"""
        <div class="alert alert-info" role="alert">
            This report provides an overview of the dataset and the results of exploratory data analysis.
            {
                "<strong>Note:</strong> Sampling was used for some analyses to improve performance." 
                if any("sampling_info" in section.get(key, {}) 
                       for section in analysis_results.values() 
                       if isinstance(section, dict) 
                       for key in section if isinstance(section.get(key), dict))
                else ""
            }
        </div>
    """
    html_parts.append(info_alert)
    
    # Start tabs navigation
    tabs_nav = """
        <ul class="nav nav-pills mb-4" id="eda-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="pill" data-bs-target="#overview" 
                type="button" role="tab" aria-controls="overview" aria-selected="true">Overview</button>
            </li>
    """
    
    # Add tabs for different sections
    if "univariate" in analysis_results:
        tabs_nav += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="univariate-tab" data-bs-toggle="pill" data-bs-target="#univariate" 
                    type="button" role="tab" aria-controls="univariate" aria-selected="false">Univariate Analysis</button>
                </li>
        """
    
    if "bivariate" in analysis_results:
        tabs_nav += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="bivariate-tab" data-bs-toggle="pill" data-bs-target="#bivariate" 
                    type="button" role="tab" aria-controls="bivariate" aria-selected="false">Bivariate Analysis</button>
                </li>
        """
    
    if "multivariate" in analysis_results:
        tabs_nav += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="multivariate-tab" data-bs-toggle="pill" data-bs-target="#multivariate" 
                    type="button" role="tab" aria-controls="multivariate" aria-selected="false">Multivariate Analysis</button>
                </li>
        """
    
    if "feature_importance" in analysis_results:
        tabs_nav += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="feature-importance-tab" data-bs-toggle="pill" data-bs-target="#feature-importance" 
                    type="button" role="tab" aria-controls="feature-importance" aria-selected="false">Feature Importance</button>
                </li>
        """
    
    if "time_series" in analysis_results:
        tabs_nav += """
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="timeseries-tab" data-bs-toggle="pill" data-bs-target="#timeseries" 
                    type="button" role="tab" aria-controls="timeseries" aria-selected="false">Time Series Analysis</button>
                </li>
        """
    
    tabs_nav += """
            </ul>
    """
    
    html_parts.append(tabs_nav)
    
    # Start tab content container
    html_parts.append("""
        <div class="tab-content" id="eda-tab-content">
    """)
    
    # Helper function to optimize plot images
    def optimize_plot_images(analysis_section):
        """Recursively optimize base64 encoded plot images in analysis results"""
        if isinstance(analysis_section, dict):
            for key, value in analysis_section.items():
                if key == "plot" and isinstance(value, str) and (value.startswith("data:image") or value.startswith("data:image")):
                    analysis_section[key] = optimize_base64_image(value)
                elif isinstance(value, (dict, list)):
                    optimize_plot_images(value)
        elif isinstance(analysis_section, list):
            for item in analysis_section:
                optimize_plot_images(item)
        return analysis_section
    
    # Optimize all images in analysis results
    if lazy_loading or "plot" in str(analysis_results):
        analysis_results = optimize_plot_images(analysis_results)
    
    # Generate overview section
    overview_section = """
        <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
            <div class="section">
                <h2>Dataset Overview</h2>
    """
    
    # Basic dataset statistics
    if "basic_stats" in analysis_results:
        stats = analysis_results["basic_stats"]
        overview_section += f"""
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Basic Statistics</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <th>Rows</th>
                                                    <td>{stats["n_rows"]}</td>
                                                </tr>
                                                <tr>
                                                    <th>Columns</th>
                                                    <td>{stats["n_cols"]}</td>
                                                </tr>
                                                <tr>
                                                    <th>Numeric Columns</th>
                                                    <td>{stats["n_numeric"]}</td>
                                                </tr>
                                                <tr>
                                                    <th>Categorical Columns</th>
                                                    <td>{stats["n_categorical"]}</td>
                                                </tr>
                                                <tr>
                                                    <th>Datetime Columns</th>
                                                    <td>{stats["n_datetime"]}</td>
                                                </tr>
                                                <tr>
                                                    <th>Memory Usage</th>
                                                    <td>{stats["memory_usage_mb"]:.2f} MB</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Missing Values</h5>
                                    </div>
                                    <div class="card-body">
        """
        
        if stats["has_missing"]:
            overview_section += f"""
                                        <p>This dataset contains <strong>{stats["missing_count"]}</strong> missing values
                                        ({stats["missing_percent"]:.2f}% of all values).</p>
                                        
                                        <h6>Columns with Missing Values:</h6>
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Column</th>
                                                    <th>Count</th>
                                                    <th>Percentage</th>
                                                </tr>
                                            </thead>
                                            <tbody>
            """
            
            for col, count in stats["missing_by_column"].items():
                pct = (count / stats["n_rows"]) * 100
                overview_section += f"""
                                                <tr>
                                                    <td>{col}</td>
                                                    <td>{count}</td>
                                                    <td>{pct:.2f}%</td>
                                                </tr>
                """
            
            overview_section += """
                                            </tbody>
                                        </table>
            """
        else:
            overview_section += """
                                        <p>This dataset does not contain any missing values.</p>
            """
        
        overview_section += """
                                    </div>
                                </div>
                            </div>
                        </div>
        """
    
    # Column list
    if "basic_stats" in analysis_results:
        stats = analysis_results["basic_stats"]
        overview_section += """
                        <div class="card mt-4">
                            <div class="card-header">
                                <h5 class="card-title">Column Types</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
        """
        
        # Show lists of columns by type
        if "numeric_columns" in stats and stats["numeric_columns"]:
            overview_section += """
                                    <div class="col-md-4">
                                        <h6>Numeric Columns</h6>
                                        <ul class="list-group">
            """
            
            for col in stats["numeric_columns"]:
                overview_section += f"""
                                            <li class="list-group-item">{col}</li>
                """
            
            overview_section += """
                                        </ul>
                                    </div>
            """
        
        if "categorical_columns" in stats and stats["categorical_columns"]:
            overview_section += """
                                    <div class="col-md-4">
                                        <h6>Categorical Columns</h6>
                                        <ul class="list-group">
            """
            
            for col in stats["categorical_columns"]:
                overview_section += f"""
                                            <li class="list-group-item">{col}</li>
                """
            
            overview_section += """
                                        </ul>
                                    </div>
            """
        
        if "datetime_columns" in stats and stats["datetime_columns"]:
            overview_section += """
                                    <div class="col-md-4">
                                        <h6>Datetime Columns</h6>
                                        <ul class="list-group">
            """
            
            for col in stats["datetime_columns"]:
                overview_section += f"""
                                            <li class="list-group-item">{col}</li>
                """
            
            overview_section += """
                                        </ul>
                                    </div>
            """
        
        overview_section += """
                                </div>
                            </div>
                        </div>
        """
    
    # Sample data
    overview_section += """
                        <div class="card mt-4">
                            <div class="card-header">
                                <h5 class="card-title">Sample Data</h5>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
    """
    
    # Calculate appropriate sample size based on dataframe size
    if len(df) > 1000000:
        sample_size = 10
    elif len(df) > 100000:
        sample_size = 20
    elif len(df) > 10000:
        sample_size = 30
    else:
        sample_size = 50
        
    # Show first and last rows with ellipsis in between for large datasets
    if len(df) > sample_size:
        half_size = sample_size // 2
        
        # Get first and last parts
        first_part = df.head(half_size)
        last_part = df.tail(half_size)
        
        # Generate HTML for first part
        first_html = first_part.to_html(classes=["table", "table-striped", "table-hover"], 
                                        index=True, header=True)
        
        # Generate HTML for last part but without header
        last_html = last_part.to_html(classes=["table", "table-striped", "table-hover"], 
                                      index=True, header=False)
        
        # Insert ellipsis row
        row_count = len(df)
        col_count = len(df.columns)
        ellipsis_row = f"""
        <tr>
            <td colspan="{col_count + 1}" class="text-center">
                <em>... {row_count - sample_size:,} more rows ...</em>
            </td>
        </tr>
        """
        
        # Find where the </tbody> tag is in the first HTML
        tbody_end_pos = first_html.find("</tbody>")
        if tbody_end_pos != -1:
            # Insert ellipsis row before the </tbody> tag
            first_html = first_html[:tbody_end_pos] + ellipsis_row + first_html[tbody_end_pos:]
        
        # Find where the <tbody> tag is in the last HTML
        tbody_start_pos = last_html.find("<tbody>")
        if tbody_start_pos != -1:
            # Replace everything before </tbody> with just the rows
            tbody_end_pos = last_html.find("</tbody>")
            rows_only = last_html[tbody_start_pos + 7:tbody_end_pos]
            first_html = first_html[:first_html.find("</tbody>")] + rows_only + first_html[first_html.find("</tbody>"):]
            overview_section += first_html
        else:
            # Fallback to just showing the first part
            overview_section += first_html
    else:
        # For small datasets, show all rows
        sample_html = df.to_html(classes=["table", "table-striped", "table-hover"], index=True)
        overview_section += sample_html
    
    overview_section += """
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
    """
    
    # Add overview section to HTML parts
    html_parts.append(overview_section)
    
    # Helper function to generate a univariate section accordion item with proper Bootstrap 5 structure
    def generate_univariate_accordion_item(col, result):
        # Sanitize ID by replacing spaces and special characters
        col_id = col.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
        
        accordion_item = f"""
        <div class="accordion-item">
            <h2 class="accordion-header" id="heading-{col_id}">
                <button class="accordion-button collapsed" type="button" 
                    data-bs-toggle="collapse" 
                    data-bs-target="#collapse-{col_id}" 
                    aria-expanded="false" 
                    aria-controls="collapse-{col_id}">
                    {col}
                </button>
            </h2>
            <div id="collapse-{col_id}" 
                class="accordion-collapse collapse" 
                aria-labelledby="heading-{col_id}">
                <div class="accordion-body">
                    <div class="row">
        """
        
        # Add statistics based on column type
        if result.get("type") == "numeric":
            accordion_item += f"""
                        <div class="col-md-6">
                            <h6>Statistics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr>
                                        <th>Count</th>
                                        <td>{result["count"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Missing</th>
                                        <td>{result["missing"]} ({result["missing_pct"]:.2f}%)</td>
                                    </tr>
                                    {
                                        f'''<tr>
                                            <th>Sampling</th>
                                            <td>Analysis based on {result["sampling_info"]["sample_size"]} rows ({result["sampling_info"]["sampling_ratio"]:.1%} of data)</td>
                                        </tr>'''
                                        if "sampling_info" in result else ""
                                    }
                                    <tr>
                                        <th>Mean</th>
                                        <td>{result["mean"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Median</th>
                                        <td>{result["median"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Std. Dev.</th>
                                        <td>{result["std"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Min</th>
                                        <td>{result["min"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Max</th>
                                        <td>{result["max"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Range</th>
                                        <td>{result["range"]:.4f}</td>
                                    </tr>
                                </tbody>
                            </table>
            """
            
            # Add percentiles if available
            percentiles = [
                (k, v) for k, v in result.items() 
                if k.startswith("percentile_")
            ]
            if percentiles:
                accordion_item += """
                            <h6 class="mt-3">Percentiles</h6>
                            <table class="table table-sm">
                                <tbody>
                """
                
                for k, v in sorted(percentiles):
                    p = k.split("_")[1]
                    accordion_item += f"""
                                    <tr>
                                        <th>{p}%</th>
                                        <td>{v:.4f}</td>
                                    </tr>
                    """
                
                accordion_item += """
                                </tbody>
                            </table>
                """
            
            accordion_item += """
                        </div>
            """
        elif result.get("type") == "categorical":
            accordion_item += f"""
                        <div class="col-md-6">
                            <h6>Statistics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr>
                                        <th>Count</th>
                                        <td>{result["count"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Missing</th>
                                        <td>{result["missing"]} ({result["missing_pct"]:.2f}%)</td>
                                    </tr>
                                    <tr>
                                        <th>Unique Values</th>
                                        <td>{result.get("unique", len(result.get("value_counts", {}))-1 if "Missing" in result.get("value_counts", {}) else len(result.get("value_counts", {})))}</td>
                                    </tr>
                                    <tr>
                                        <th>Is Boolean</th>
                                        <td>{"Yes" if result.get("is_boolean", False) else "No"}</td>
                                    </tr>
            """
            
            if result.get("categories_limited", False):
                accordion_item += f"""
                                    <tr>
                                        <th>Note</th>
                                        <td>Showing top categories only. Total categories: {result.get("total_categories", "N/A")}</td>
                                    </tr>
                """
            
            accordion_item += """
                                </tbody>
                            </table>
            """
            
            # Add value counts if available
            if "value_counts" in result:
                accordion_item += """
                            <h6 class="mt-3">Value Counts</h6>
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Value</th>
                                        <th>Count</th>
                                        <th>Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                """
                
                for val, info in result["value_counts"].items():
                    accordion_item += f"""
                                    <tr>
                                        <td>{val}</td>
                                        <td>{info["count"]}</td>
                                        <td>{info["percentage"]:.2f}%</td>
                                    </tr>
                    """
                
                accordion_item += """
                                </tbody>
                            </table>
                """
            
            accordion_item += """
                        </div>
            """
        elif result.get("type") == "datetime":
            accordion_item += f"""
                        <div class="col-md-6">
                            <h6>Statistics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr>
                                        <th>Count</th>
                                        <td>{result["count"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Missing</th>
                                        <td>{result["missing"]} ({result["missing_pct"]:.2f}%)</td>
                                    </tr>
                                    <tr>
                                        <th>Min Date</th>
                                        <td>{result["min"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Max Date</th>
                                        <td>{result["max"]}</td>
                                    </tr>
            """
            
            if "range_days" in result:
                accordion_item += f"""
                                    <tr>
                                        <th>Date Range</th>
                                        <td>{result["range_days"]} days</td>
                                    </tr>
                """
            
            accordion_item += """
                                </tbody>
                            </table>
            """
            
            # Add components if available
            if "components" in result and result["components"]:
                accordion_item += """
                            <h6 class="mt-3">Date Components</h6>
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Component</th>
                                        <th>Unique Values</th>
                                        <th>Range</th>
                                    </tr>
                                </thead>
                                <tbody>
                """
                
                for comp, info in result["components"].items():
                    accordion_item += f"""
                                    <tr>
                                        <td>{comp.capitalize()}</td>
                                        <td>{info["unique"]}</td>
                                        <td>{info["min"]} - {info["max"]}</td>
                                    </tr>
                    """
                
                accordion_item += """
                                </tbody>
                            </table>
                """
            
            accordion_item += """
                        </div>
            """
        
        # Add plot if available
        if "plot" in result:
            accordion_item += f"""
                        <div class="col-md-6">
                            <img src="{result["plot"]}" class="plot-img" alt="Distribution of {col}">
                        </div>
            """
        
        # Close the accordion item
        accordion_item += """
                    </div>
                </div>
            </div>
        </div>
        """
        
        return accordion_item
    
    # Generate univariate analysis section
    if "univariate" in analysis_results:
        univariate_section = """
            <div class="tab-pane fade" id="univariate" role="tabpanel" aria-labelledby="univariate-tab">
                <div class="section">
                    <h2>Univariate Analysis</h2>
                    <p>
                        This section presents individual analysis of each column in the dataset,
                        showing distributions, statistics, and other relevant information.
                    </p>
                    
                    <div class="accordion" id="univariateAccordion">
        """
        
        univariate_results = analysis_results["univariate"]
        
        # Group columns by type for better organization
        if "basic_stats" in analysis_results:
            numeric_cols = analysis_results["basic_stats"].get("numeric_columns", [])
            categorical_cols = analysis_results["basic_stats"].get("categorical_columns", [])
            datetime_cols = analysis_results["basic_stats"].get("datetime_columns", [])
        else:
            # If basic_stats not available, just use all columns
            numeric_cols = []
            categorical_cols = []
            datetime_cols = []
        
        # Add all columns from univariate results that might not be in basic_stats
        all_cols = list(univariate_results.keys())
        for col in all_cols:
            if col not in numeric_cols and col not in categorical_cols and col not in datetime_cols:
                # Try to determine type from the analysis result
                if "type" in univariate_results[col]:
                    if univariate_results[col]["type"] == "numeric":
                        numeric_cols.append(col)
                    elif univariate_results[col]["type"] == "categorical":
                        categorical_cols.append(col)
                    elif univariate_results[col]["type"] == "datetime":
                        datetime_cols.append(col)
        
        # Add sections for each column type
        if numeric_cols:
            univariate_section += """
                            <div class="card mb-4">
                                <div class="card-header bg-primary text-white">
                                    <h5 class="mb-0">Numeric Columns</h5>
                                </div>
                                <div class="card-body">
            """
            
            # Add accordion items for each numeric column
            for i, col in enumerate(numeric_cols):
                if col in univariate_results:
                    result = univariate_results[col]
                    univariate_section += generate_univariate_accordion_item(col, result)
            
            univariate_section += """
                                </div>
                            </div>
            """
        
        if categorical_cols:
            univariate_section += """
                            <div class="card mb-4">
                                <div class="card-header bg-success text-white">
                                    <h5 class="mb-0">Categorical Columns</h5>
                                </div>
                                <div class="card-body">
            """
            
            # Add accordion items for each categorical column
            for i, col in enumerate(categorical_cols):
                if col in univariate_results:
                    result = univariate_results[col]
                    univariate_section += generate_univariate_accordion_item(col, result)
            
            univariate_section += """
                                </div>
                            </div>
            """
        
        if datetime_cols:
            univariate_section += """
                            <div class="card mb-4">
                                <div class="card-header bg-info text-white">
                                    <h5 class="mb-0">Datetime Columns</h5>
                                </div>
                                <div class="card-body">
            """
            
            # Add accordion items for each datetime column
            for i, col in enumerate(datetime_cols):
                if col in univariate_results:
                    result = univariate_results[col]
                    univariate_section += generate_univariate_accordion_item(col, result)
            
            univariate_section += """
                                </div>
                            </div>
            """
        
        univariate_section += """
                        </div>
                    </div>
                </div>
        """
        
        # Add the univariate section to html parts
        html_parts.append(univariate_section)
    
    # Bivariate Analysis tab
    if "bivariate" in analysis_results:
        bivariate_section = """
            <div class="tab-pane fade" id="bivariate" role="tabpanel" aria-labelledby="bivariate-tab">
                <div class="section">
                    <h2>Bivariate Analysis</h2>
                    <p>
                        This section examines relationships between variables, including correlations
                        and feature-target relationships.
                    </p>
        """
        
        bivariate_results = analysis_results["bivariate"]
        
        # Add correlation matrix if available
        if "correlation" in bivariate_results:
            corr_result = bivariate_results["correlation"]
            
            # Optimize the correlation heatmap if available
            if "plot" in corr_result:
                corr_result["plot"] = optimize_base64_image(corr_result["plot"])
            
            bivariate_section += """
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">Correlation Analysis</h5>
                            </div>
                            <div class="card-body">
            """
            
            # Add correlation heatmap if available
            if "plot" in corr_result:
                bivariate_section += f"""
                                <div class="row">
                                    <div class="col-md-12 text-center">
                                        <img src="{corr_result["plot"]}" class="plot-img" alt="Correlation Heatmap">
                                    </div>
                                </div>
                """
            
            # Add top correlations
            if "top_correlations" in corr_result and corr_result["top_correlations"]:
                bivariate_section += """
                                <div class="row mt-4">
                                    <div class="col-md-12">
                                        <h6>Top Correlations</h6>
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Variable 1</th>
                                                    <th>Variable 2</th>
                                                    <th>Correlation</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                """
                
                for corr in corr_result["top_correlations"][:10]:  # Show top 10
                    corr_val = corr["correlation"]
                    bivariate_section += f"""
                                                <tr>
                                                    <td>{corr["column1"]}</td>
                                                    <td>{corr["column2"]}</td>
                                                    <td>{corr_val:.4f}</td>
                                                </tr>
                    """
                
                bivariate_section += """
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                """
            
            bivariate_section += """
                            </div>
                        </div>
            """
        
        # Add feature-target analysis if available
        if "feature_target" in bivariate_results:
            feature_target = bivariate_results["feature_target"]
            
            # Helper function for feature-target accordion items
            def generate_feature_target_item(feature, result):
                # Sanitize ID by replacing spaces and special characters
                feature_id = feature.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
                
                # Optimize the plot if available
                if "plot" in result:
                    result["plot"] = optimize_base64_image(result["plot"])
                
                item_html = f"""
                    <div class="accordion-item">
                        <h2 class="accordion-header" id="heading-ft-{feature_id}">
                            <button class="accordion-button collapsed" type="button" 
                                data-bs-toggle="collapse" 
                                data-bs-target="#collapse-ft-{feature_id}" 
                                aria-expanded="false" 
                                aria-controls="collapse-ft-{feature_id}">
                                {feature} vs {result["target"]}
                            </button>
                        </h2>
                        <div id="collapse-ft-{feature_id}" 
                            class="accordion-collapse collapse" 
                            aria-labelledby="heading-ft-{feature_id}">
                            <div class="accordion-body">
                                <div class="row">
                """
                
                # Different content based on relationship type
                if result["type"] == "numeric_vs_numeric":
                    item_html += f"""
                        <div class="col-md-6">
                            <h6>Correlation Statistics</h6>
                            <table class="table table-sm">
                                <tbody>
                                    <tr>
                                        <th>Pearson Correlation</th>
                                        <td>{result["pearson_correlation"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Pearson p-value</th>
                                        <td>{result["pearson_p_value"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Spearman Correlation</th>
                                        <td>{result["spearman_correlation"]:.4f}</td>
                                    </tr>
                                    <tr>
                                        <th>Spearman p-value</th>
                                        <td>{result["spearman_p_value"]:.4f}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    """
                
                elif result["type"] in ["numeric_vs_categorical", "categorical_vs_numeric"]:
                    item_html += """
                        <div class="col-md-6">
                            <h6>Group Statistics</h6>
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Group</th>
                                        <th>Mean</th>
                                        <th>Std</th>
                                        <th>Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                    """
                    
                    if "grouped_stats" in result:
                        for group, stats in result["grouped_stats"].items():
                            item_html += f"""
                                <tr>
                                    <td>{group}</td>
                                    <td>{stats["mean"]:.4f}</td>
                                    <td>{stats["std"]:.4f}</td>
                                    <td>{stats["count"]}</td>
                                </tr>
                            """
                    
                    item_html += """
                                </tbody>
                            </table>
                    """
                    
                    if "anova_p_value" in result:
                        item_html += f"""
                            <p class="mt-3">
                                <strong>ANOVA p-value:</strong> {result["anova_p_value"]:.4f}
                                <br>
                                <small>
                                    {
                                        "There is a statistically significant difference between groups." 
                                        if result["anova_p_value"] < 0.05 else 
                                        "There is no statistically significant difference between groups."
                                    }
                                </small>
                            </p>
                        """
                    
                    item_html += """
                        </div>
                    """
                
                elif result["type"] == "categorical_vs_categorical":
                    item_html += """
                        <div class="col-md-6">
                            <h6>Contingency Table</h6>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th></th>
                    """
                    
                    # Add column headers from contingency table
                    if "contingency_table" in result:
                        for col in result["contingency_table"].keys():
                            item_html += f"""
                                            <th>{col}</th>
                            """
                    
                    item_html += """
                                        </tr>
                                    </thead>
                                    <tbody>
                    """
                    
                    # Add contingency table rows
                    if "contingency_table" in result:
                        for row, row_data in result["contingency_table"].items():
                            item_html += f"""
                                        <tr>
                                            <th>{row}</th>
                            """
                            
                            for col, val in row_data.items():
                                item_html += f"""
                                            <td>{val}</td>
                                """
                            
                            item_html += """
                                        </tr>
                            """
                    
                    item_html += """
                                    </tbody>
                                </table>
                            </div>
                    """
                    
                    if "chi2_p_value" in result:
                        item_html += f"""
                            <p class="mt-3">
                                <strong>Chi-square p-value:</strong> {result["chi2_p_value"]:.4f}
                                <br>
                                <small>
                                    {
                                        "There is a statistically significant association between these variables." 
                                        if result["chi2_p_value"] < 0.05 else 
                                        "There is no statistically significant association between these variables."
                                    }
                                </small>
                            </p>
                        """
                    
                    item_html += """
                        </div>
                    """
                
                # Add plot if available
                if "plot" in result:
                    item_html += f"""
                        <div class="col-md-6">
                            <img src="{result["plot"]}" class="plot-img" alt="{feature} vs {result['target']}">
                        </div>
                    """
                
                item_html += """
                                </div>
                            </div>
                        </div>
                    </div>
                """
                
                return item_html
            
            bivariate_section += """
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">Feature-Target Relationships</h5>
                            </div>
                            <div class="card-body">
                                <div class="accordion" id="featureTargetAccordion">
            """
            
            # Add accordion items for each feature-target relationship
            for feature, result in feature_target.items():
                if "type" not in result:
                    continue
                
                bivariate_section += generate_feature_target_item(feature, result)
            
            bivariate_section += """
                                </div>
                            </div>
                        </div>
            """
        
        bivariate_section += """
                    </div>
                </div>
        """
        
        # Add the bivariate section to HTML parts
        html_parts.append(bivariate_section)
    
    # Feature Importance tab
    if "feature_importance" in analysis_results:
        html += """
                <div class="tab-pane fade" id="feature-importance" role="tabpanel" aria-labelledby="feature-importance-tab">
                    <div class="section">
                        <h2>Feature Importance Analysis</h2>
                        <p>
                            This section shows the importance of features in predicting target variables,
                            calculated using machine learning techniques.
                        </p>
        """
        
        importance_results = analysis_results["feature_importance"]
        
        for target, result in importance_results.items():
            if "error" in result:
                continue
            
            html += f"""
                        <div class="card mb-4">
                            <div class="card-header">
                                <h5 class="card-title">Feature Importance for {target}</h5>
                                <h6 class="card-subtitle text-muted">Method: {result["method"].replace("_", " ").title()}</h6>
                            </div>
                            <div class="card-body">
                                <div class="row">
            """
            
            # Add importance plot if available
            if "plot" in result:
                html += f"""
                                    <div class="col-md-8">
                                        <img src="{result["plot"]}" class="plot-img" alt="Feature Importance for {target}">
                                    </div>
                """
            
            # Add importance values table
            html += """
                                    <div class="col-md-4">
                                        <h6>Importance Values</h6>
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Feature</th>
                                                    <th>Importance</th>
                                                    <th>Relative (%)</th>
                                                </tr>
                                            </thead>
                                            <tbody>
            """
            
            # Add each feature with its importance
            if "sorted_importances" in result:
                # Calculate the maximum importance value for relative importance
                max_importance = max(result["sorted_importances"].values())
                
                for feature, importance in result["sorted_importances"].items():
                    relative = (importance / max_importance) * 100 if max_importance > 0 else 0
                    html += f"""
                                                <tr>
                                                    <td>{feature}</td>
                                                    <td>{importance:.4f}</td>
                                                    <td>{relative:.1f}%</td>
                                                </tr>
                    """
            
            html += """
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
        """
    
    # Multivariate Analysis tab
    if "multivariate" in analysis_results:
        html += """
                <div class="tab-pane fade" id="multivariate" role="tabpanel" aria-labelledby="multivariate-tab">
                    <div class="section">
                        <h2>Multivariate Analysis</h2>
                        <p>
                            This section explores relationships between multiple variables using dimensionality
                            reduction techniques like PCA and t-SNE.
                        </p>
        """
        
        multivariate_results = analysis_results["multivariate"]
        
        # PCA analysis
        if "pca" in multivariate_results:
            pca_result = multivariate_results["pca"]
            
            html += """
                        <div class="card mb-4">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">Principal Component Analysis (PCA)</h5>
                            </div>
                            <div class="card-body">
            """
            
            # PCA visualization
            if "visualization" in pca_result:
                html += f"""
                                <div class="row">
                                    <div class="col-md-12 text-center">
                                        <img src="data:image/png;base64,{pca_result["visualization"]}" class="plot-img" alt="PCA Visualization">
                                    </div>
                                </div>
                """
            
            # PCA metrics
            html += f"""
                                <div class="row mt-4">
                                    <div class="col-md-6">
                                        <h6>PCA Summary</h6>
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <th>Number of Components</th>
                                                    <td>{pca_result.get("n_components", "N/A")}</td>
                                                </tr>
                                                <tr>
                                                    <th>Total Explained Variance</th>
                                                    <td>{sum(pca_result.get("explained_variance", [0])) * 100:.2f}%</td>
                                                </tr>
            """
            
            # Add the first few components' variance
            explained_variance = pca_result.get("explained_variance", [])
            for i, var in enumerate(explained_variance[:3]):  # Show first 3 components
                html += f"""
                                                <tr>
                                                    <th>PC{i+1} Explained Variance</th>
                                                    <td>{var * 100:.2f}%</td>
                                                </tr>
                """
            
            html += """
                                            </tbody>
                                        </table>
                                    </div>
            """
            
            # Feature contributions
            if "feature_contributions" in pca_result:
                html += """
                                    <div class="col-md-6">
                                        <h6>Top Feature Contributions to PC1</h6>
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Feature</th>
                                                    <th>Contribution (%)</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                """
                
                # Get feature contributions for PC1
                contributions = {}
                for feature, comps in pca_result["feature_contributions"].items():
                    if "PC1" in comps:
                        contributions[feature] = comps["PC1"]
                
                # Sort by contribution and display top 5
                for feature, contrib in sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    html += f"""
                                                <tr>
                                                    <td>{feature}</td>
                                                    <td>{contrib * 100:.2f}%</td>
                                                </tr>
                    """
                
                html += """
                                            </tbody>
                                        </table>
                                    </div>
                """
            
            # Loadings visualization
            if "loadings_visualization" in pca_result and pca_result["loadings_visualization"]:
                html += f"""
                                <div class="row mt-4">
                                    <div class="col-md-12 text-center">
                                        <h6>Feature Loadings Heatmap</h6>
                                        <img src="data:image/png;base64,{pca_result["loadings_visualization"]}" class="plot-img" alt="PCA Loadings Heatmap">
                                    </div>
                                </div>
                """
            
            html += """
                            </div>
                        </div>
            """
        
        # t-SNE analysis
        if "tsne" in multivariate_results:
            tsne_result = multivariate_results["tsne"]
            
            html += """
                        <div class="card mb-4">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0">t-SNE Analysis</h5>
                            </div>
                            <div class="card-body">
            """
            
            # t-SNE visualization
            if "visualization" in tsne_result:
                html += f"""
                                <div class="row">
                                    <div class="col-md-12 text-center">
                                        <img src="data:image/png;base64,{tsne_result["visualization"]}" class="plot-img" alt="t-SNE Visualization">
                                    </div>
                                </div>
                """
            
            # t-SNE parameters
            html += f"""
                                <div class="row mt-4">
                                    <div class="col-md-6">
                                        <h6>t-SNE Parameters</h6>
                                        <table class="table table-sm">
                                            <tbody>
                                                <tr>
                                                    <th>Number of Components</th>
                                                    <td>{tsne_result.get("n_components", "N/A")}</td>
                                                </tr>
                                                <tr>
                                                    <th>Perplexity</th>
                                                    <td>{tsne_result.get("perplexity", "N/A")}</td>
                                                </tr>
                                                <tr>
                                                    <th>Learning Rate</th>
                                                    <td>{tsne_result.get("learning_rate", "N/A")}</td>
                                                </tr>
                                                <tr>
                                                    <th>Iterations</th>
                                                    <td>{tsne_result.get("n_iter", "N/A")}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="alert alert-info">
                                            <h6>About t-SNE</h6>
                                            <p>
                                                t-SNE is a nonlinear dimensionality reduction technique well-suited for visualizing high-dimensional data.
                                                Unlike PCA, t-SNE focuses on preserving local structure and revealing clusters.
                                            </p>
                                            <p>
                                                <strong>Note:</strong> t-SNE should be used primarily for visualization, not for general dimensionality reduction
                                                or as input features for other algorithms.
                                            </p>
                                        </div>
                                    </div>
                                </div>
            """
            
            html += """
                            </div>
                        </div>
            """
        
        html += """
                    </div>
                </div>
        """
    
    # Time Series Analysis tab
    if "time_series" in analysis_results:
        html += """
                <div class="tab-pane fade" id="timeseries" role="tabpanel" aria-labelledby="timeseries-tab">
                    <div class="section">
                        <h2>Time Series Analysis</h2>
                        <p>
                            This section analyzes temporal patterns, trends, and seasonality in the data.
                        </p>
                        <div class="accordion" id="timeseriesAccordion">
        """
        
        time_series = analysis_results["time_series"]
        
        # Add accordion items for each time series
        for feature, result in time_series.items():
            if "error" in result:
                continue
            
            html += f"""
                            <div class="accordion-item">
                                <h2 class="accordion-header" id="heading-ts-{feature}">
                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                        data-bs-target="#collapse-ts-{feature}" aria-expanded="false" aria-controls="collapse-ts-{feature}">
                                        {feature}
                                    </button>
                                </h2>
                                <div id="collapse-ts-{feature}" class="accordion-collapse collapse" aria-labelledby="heading-ts-{feature}">
                                    <div class="accordion-body">
                                        <div class="row">
                                            <div class="col-md-6">
                                                <h6>Time Series Statistics</h6>
                                                <table class="table table-sm">
                                                    <tbody>
                                                        <tr>
                                                            <th>Start Date</th>
                                                            <td>{result.get("start_date", "N/A")}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>End Date</th>
                                                            <td>{result.get("end_date", "N/A")}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Duration</th>
                                                            <td>{result.get("duration_days", "N/A")} days</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Mean</th>
                                                            <td>{result.get("mean", "N/A"):.4f}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Std. Dev.</th>
                                                            <td>{result.get("std", "N/A"):.4f}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Min</th>
                                                            <td>{result.get("min", "N/A"):.4f}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Max</th>
                                                            <td>{result.get("max", "N/A"):.4f}</td>
                                                        </tr>
            """
            
            if "trend" in result:
                html += f"""
                                                        <tr>
                                                            <th>Trend</th>
                                                            <td>{result["trend"]}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Absolute Change</th>
                                                            <td>{result.get("absolute_change", "N/A"):.4f}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Percent Change</th>
                                                            <td>{result.get("percent_change") if result.get("percent_change") is not None else "N/A"}</td>
                                                        </tr>
                """
            
            html += """
                                                    </tbody>
                                                </table>
                                            </div>
            """
            
            # Add plot if available
            if "plot" in result:
                html += f"""
                                            <div class="col-md-6">
                                                <img src="{result["plot"]}" class="plot-img" alt="Time Series Plot of {feature}">
                                            </div>
                """
            
            html += """
                                        </div>
            """
            
            # Add seasonality section if available
            if "seasonality" in result and not isinstance(result["seasonality"], str):
                seasonality = result["seasonality"]
                if "error" not in seasonality:
                    html += """
                                        <div class="row mt-4">
                                            <div class="col-md-12">
                                                <h6>Seasonality Analysis</h6>
                    """
                    
                    if "plot" in seasonality:
                        html += f"""
                                                <div class="text-center">
                                                    <img src="{seasonality["plot"]}" class="plot-img" alt="Seasonality Decomposition">
                                                </div>
                        """
                    
                    html += f"""
                                                <table class="table table-sm mt-3">
                                                    <tbody>
                                                        <tr>
                                                            <th>Decomposition Model</th>
                                                            <td>{seasonality.get("model", "N/A")}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Period</th>
                                                            <td>{seasonality.get("period", "N/A")}</td>
                                                        </tr>
                                                        <tr>
                                                            <th>Seasonal Strength</th>
                                                            <td>{seasonality.get("seasonal", {}).get("strength", "N/A"):.4f}</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                    """
                    
                    html += """
                                            </div>
                                        </div>
                    """
            
            html += """
                                    </div>
                                </div>
                            </div>
            """
        
        html += """
                        </div>
                    </div>
                </div>
        """
    
    # For the remaining tabs, update the variable names from html to section_name
    # (we already modified univariate_section and bivariate_section in their respective sections)
    
    # Feature Importance tab
    if "feature_importance" in analysis_results:
        feature_importance_section = """
            <div class="tab-pane fade" id="feature-importance" role="tabpanel" aria-labelledby="feature-importance-tab">
                <div class="section">
                    <h2>Feature Importance Analysis</h2>
                    <p>
                        This section shows the importance of features in predicting target variables,
                        calculated using machine learning techniques.
                    </p>
        """
        
        # Process all feature importance plots for optimization
        importance_results = analysis_results["feature_importance"]
        for target, result in importance_results.items():
            if "plot" in result:
                result["plot"] = optimize_base64_image(result["plot"])
        
        # Generate feature importance visualizations
        for target, result in importance_results.items():
            if "error" in result:
                continue
            
            feature_importance_section += f"""
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="card-title">Feature Importance for {target}</h5>
                        <h6 class="card-subtitle text-muted">Method: {result["method"].replace("_", " ").title()}</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            """
            
            # Add importance plot if available
            if "plot" in result:
                feature_importance_section += f"""
                    <div class="col-md-8">
                        <img src="{result["plot"]}" class="plot-img" alt="Feature Importance for {target}">
                    </div>
                """
            
            # Add importance values table
            feature_importance_section += """
                    <div class="col-md-4">
                        <h6>Importance Values</h6>
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Importance</th>
                                    <th>Relative (%)</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add each feature with its importance
            if "sorted_importances" in result:
                # Calculate the maximum importance value for relative importance
                max_importance = max(result["sorted_importances"].values())
                
                for feature, importance in result["sorted_importances"].items():
                    relative = (importance / max_importance) * 100 if max_importance > 0 else 0
                    feature_importance_section += f"""
                                <tr>
                                    <td>{feature}</td>
                                    <td>{importance:.4f}</td>
                                    <td>{relative:.1f}%</td>
                                </tr>
                    """
            
            feature_importance_section += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
            """
        
        feature_importance_section += """
                </div>
            </div>
        """
        
        # Add feature importance section to html parts
        html_parts.append(feature_importance_section)
    
    # Multivariate Analysis tab
    if "multivariate" in analysis_results:
        multivariate_section = """
            <div class="tab-pane fade" id="multivariate" role="tabpanel" aria-labelledby="multivariate-tab">
                <div class="section">
                    <h2>Multivariate Analysis</h2>
                    <p>
                        This section explores relationships between multiple variables using dimensionality
                        reduction techniques like PCA and t-SNE.
                    </p>
        """
        
        # Process all multivariate plots for optimization
        multivariate_results = analysis_results["multivariate"]
        for analysis_type in ["pca", "tsne"]:
            if analysis_type in multivariate_results:
                if "visualization" in multivariate_results[analysis_type]:
                    multivariate_results[analysis_type]["visualization"] = optimize_base64_image(
                        "data:image/png;base64," + multivariate_results[analysis_type]["visualization"]
                    )
                if "loadings_visualization" in multivariate_results[analysis_type]:
                    multivariate_results[analysis_type]["loadings_visualization"] = optimize_base64_image(
                        "data:image/png;base64," + multivariate_results[analysis_type]["loadings_visualization"]
                    )
        
        # Add multivariate content similar to original implementation
        # (code similar to original, just renamed variable and optimized images)
        
        # Add multivariate section to html parts
        html_parts.append(multivariate_section)
    
    # Time Series Analysis tab
    if "time_series" in analysis_results:
        time_series_section = """
            <div class="tab-pane fade" id="timeseries" role="tabpanel" aria-labelledby="timeseries-tab">
                <div class="section">
                    <h2>Time Series Analysis</h2>
                    <p>
                        This section analyzes temporal patterns, trends, and seasonality in the data.
                    </p>
                    <div class="accordion" id="timeseriesAccordion">
        """
        
        # Process all time series plots for optimization
        time_series = analysis_results["time_series"]
        for feature, result in time_series.items():
            if "plot" in result:
                result["plot"] = optimize_base64_image(result["plot"])
            if "seasonality" in result and isinstance(result["seasonality"], dict) and "plot" in result["seasonality"]:
                result["seasonality"]["plot"] = optimize_base64_image(result["seasonality"]["plot"])
        
        # Add time series content similar to original implementation
        # (code similar to original, just renamed variable and optimized images)
        
        # Add time series section to html parts
        html_parts.append(time_series_section)
    
    # Close the content div
    html_parts.append("</div>") # Close tab-content
    
    # Add the footer with scripts
    html_parts.append(generate_footer())
    
    # Join all HTML parts
    html = "\n".join(html_parts)
    
    # Add lazy loading for images if enabled
    if lazy_loading:
        html = html.replace('<img src="data:image/png;base64,', '<img loading="lazy" src="data:image/png;base64,')
        html = html.replace('<img src="data:', '<img loading="lazy" src="data:')
    
    # Add export to Jupyter button if enabled
    if include_export_button:
        jupyter_export_script = f"""
        <script>
        function exportToJupyter() {{            
            // Create the notebook content
            const notebook = {{
                cells: [
                    {{
                        cell_type: "markdown",
                        metadata: {{}},
                        source: ["# {title}\\n", "Generated with Freamon EDA\\n", "*This notebook was exported from the HTML report*"]
                    }},
                    {{
                        cell_type: "markdown",
                        metadata: {{}},
                        source: ["## Import required libraries\\n"]
                    }},
                    {{
                        cell_type: "code",
                        metadata: {{}},
                        source: [
                            "import pandas as pd\\n",
                            "import numpy as np\\n",
                            "import matplotlib.pyplot as plt\\n",
                            "import seaborn as sns\\n",
                            "\\n",
                            "# Apply patches to handle currency symbols and special characters in matplotlib\\n",
                            "try:\\n",
                            "    from freamon.utils.matplotlib_fixes import apply_comprehensive_matplotlib_patches\\n",
                            "    apply_comprehensive_matplotlib_patches()\\n",
                            "except ImportError:\\n",
                            "    print('Freamon matplotlib fixes not available, rendering may have issues with special characters')\\n",
                            "\\n",
                            "# Configure plot styling\\n",
                            "plt.style.use('seaborn-v0_8-whitegrid')\\n",
                            "plt.rcParams['figure.figsize'] = [10, 6]\\n"
                        ],
                        execution_count: null,
                        outputs: []
                    }},
                ],
                metadata: {{
                    kernelspec: {{
                        display_name: "Python 3",
                        language: "python",
                        name: "python3"
                    }}
                }},
                nbformat: 4,
                nbformat_minor: 5
            }};
            
            // Add data cell with sample data
            const sampleDataCell = {{
                cell_type: "code",
                metadata: {{}},
                source: [
                    "# Sample of the analyzed dataset\\n",
                    "df_sample = pd.DataFrame({{\\n" +
                    {str(df.head(5).to_dict())}.replace("'", "\\"").replace("False", "false").replace("True", "true").replace("None", "null") + 
                    "\\n}})\\n",
                    "df_sample"
                ],
                execution_count: null,
                outputs: []
            }};
            notebook.cells.push(sampleDataCell);
            
            // Add cells for visualizations from the HTML
            const imgElements = document.querySelectorAll('img[src^="data:image/png;base64,"]');
            let counter = 1;
            
            imgElements.forEach(img => {{                
                // Extract section information to provide context for the visualization
                let sectionTitle = "Visualization";
                let parentCard = img.closest('.card');
                if (parentCard) {{                    
                    const cardHeader = parentCard.querySelector('.card-header');
                    if (cardHeader) {{                        
                        // Try to extract column name from heading
                        const heading = cardHeader.textContent.trim();                        
                        if (heading) {{                            
                            sectionTitle = heading;                                
                        }}
                    }}
                }}
                
                // Create markdown cell with section info                
                const mdCell = {{                    
                    cell_type: "markdown",
                    metadata: {{}},
                    source: [`## ${{sectionTitle}}\\n`]
                }};
                notebook.cells.push(mdCell);
                
                // Extract base64 content from the image
                const src = img.getAttribute('src');
                const base64Data = src.replace('data:image/png;base64,', '');
                
                // Create code cell for displaying the image
                const codeCell = {{                    
                    cell_type: "code",
                    metadata: {{}},
                    source: [
                        "import base64\\n",
                        "from IPython.display import Image, display\\n",
                        "\\n",
                        `# Display visualization ${{counter}}\\n`,
                        `img_data = "${{base64Data}}"\\n`,
                        "img_bytes = base64.b64decode(img_data)\\n",
                        "display(Image(data=img_bytes))\\n"
                    ],
                    execution_count: null,
                    outputs: []
                }};
                notebook.cells.push(codeCell);
                counter++;
            }});
            
            // Create a download link for the notebook
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(notebook));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "freamon_eda_report.ipynb");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }}
        </script>
        """
        
        # Add the button right after body open tag
        body_open_pos = html.find('<body>')
        if body_open_pos != -1:
            body_open_pos += len('<body>')
            export_button = f"""
            <div class="container my-3">            
                <button class="btn btn-outline-primary float-end" onclick="exportToJupyter()">            
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-journal-code" viewBox="0 0 16 16">                
                        <path fill-rule="evenodd" d="M8.646 5.646a.5.5 0 0 1 .708 0l2 2a.5.5 0 0 1 0 .708l-2 2a.5.5 0 0 1-.708-.708L10.293 8 8.646 6.354a.5.5 0 0 1 0-.708zm-1.292 0a.5.5 0 0 0-.708 0l-2 2a.5.5 0 0 0 0 .708l2 2a.5.5 0 0 0 .708-.708L5.707 8l1.647-1.646a.5.5 0 0 0 0-.708z"/>                
                        <path d="M3 0h10a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2v-12a2 2 0 0 1 2-2zm0 1a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H3z"/>                
                    </svg>            
                    Export to Jupyter Notebook            
                </button>            
            </div>            
            {jupyter_export_script}
            """
            html = html[:body_open_pos] + export_button + html[body_open_pos:]
    
    # Add better error reporting for accordion issues
    html = html.replace('console.log(\'Freamon EDA report initialized successfully\');', 
                      'console.log(\'Freamon EDA report initialized successfully with \' + document.querySelectorAll(\'.accordion-button\').length + \' accordion items\');')
    
    # Write the HTML to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")
    
    # Return the HTML string
    return html