"""
Enhanced deduplication reporting system with comprehensive visualizations.

This module provides advanced reporting capabilities for deduplication results including:
- Comprehensive HTML reports with detailed statistics and visualizations
- Excel reports with detailed duplicate pairs and similarity scores
- PowerPoint report generation for presentations
- Special Jupyter notebook rendering for interactive exploration
- Markdown reports for documentation and version control
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import os
import base64
from datetime import datetime
import logging
import json
import warnings
import textwrap
from IPython.display import display, HTML, Markdown, Javascript
import ipywidgets as widgets
from tqdm.auto import tqdm

from freamon.eda.report import (
    generate_header,
    generate_footer,
    fig_to_base64,
    add_table_styles
)
from freamon.deduplication.report import (
    generate_deduplication_report,
    export_deduplication_report,
    prepare_duplicate_report_data
)
from freamon.deduplication.evaluation import (
    calculate_deduplication_metrics,
    plot_confusion_matrix,
    evaluate_threshold_sensitivity,
    generate_evaluation_report
)

# Set up logging
logger = logging.getLogger(__name__)


class EnhancedDeduplicationReport:
    """
    Comprehensive deduplication reporting with multiple output formats.
    
    This class provides enhanced visualization and reporting capabilities
    for deduplication results, including:
    - Interactive HTML reports with detailed statistics
    - Excel exports with duplicate pairs
    - PowerPoint presentations
    - Jupyter notebook integration
    - Markdown reports for documentation
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing deduplication results
    title : str, default="Deduplication Analysis"
        Title for the report
    output_dir : Optional[str], default="dedup_reports"
        Directory to save reports
    create_dir : bool, default=True
        Whether to create the output directory if it doesn't exist
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        title: str = "Deduplication Analysis",
        output_dir: Optional[str] = "dedup_reports",
        create_dir: bool = True
    ):
        self.results = results
        self.title = title
        self.output_dir = output_dir
        
        # Create output directory if requested
        if output_dir and create_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate results structure
        self._validate_results()
        
        # Store report paths
        self.report_paths = {}
        
        # Initialize custom theme and styling
        self._init_styling()
    
    def _validate_results(self) -> None:
        """Validate the results dictionary structure."""
        required_keys = ['dataframe']
        
        for key in required_keys:
            if key not in self.results:
                raise ValueError(f"Missing required key in results: {key}")
                
        # Ensure we have either metrics or raw data to calculate them
        if 'metrics' not in self.results:
            # Try to extract or calculate metrics
            if 'flag_column' in self.results and 'truth_column' in self.results:
                logger.info("Calculating metrics from dataframe...")
                self.results['metrics'] = calculate_deduplication_metrics(
                    df=self.results['dataframe'],
                    prediction_column=self.results['flag_column'],
                    truth_column=self.results['truth_column']
                )
            else:
                logger.warning("No metrics in results and insufficient data to calculate them")
    
    def _init_styling(self) -> None:
        """Initialize custom styling for reports."""
        # Custom color palette for consistency across reports
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2c3e50',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#1abc9c',
            'light': '#ecf0f1',
            'dark': '#34495e',
            'gray': '#95a5a6'
        }
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([
            self.colors['primary'],
            self.colors['success'],
            self.colors['warning'],
            self.colors['danger'],
            self.colors['info']
        ])
    
    def generate_all_reports(
        self,
        formats: List[str] = ['html', 'excel', 'markdown', 'pptx'],
        include_pairs: bool = True,
        max_pairs: int = 100,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate reports in all specified formats.
        
        Parameters
        ----------
        formats : List[str], default=['html', 'excel', 'markdown', 'pptx']
            List of output formats to generate
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the reports
        max_pairs : int, default=100
            Maximum number of duplicate pairs to include
        filename_prefix : Optional[str], default=None
            Prefix for all output filenames
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping format to output file path
        """
        # Validate formats
        valid_formats = ['html', 'excel', 'markdown', 'pptx', 'jupyter']
        formats = [f.lower() for f in formats]
        
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f"Unsupported format: {fmt}. Valid formats are: {valid_formats}")
        
        # Create filename prefix if not provided
        if filename_prefix is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filename_prefix = f"deduplication_report_{date_str}"
        
        # Generate each requested format
        for fmt in formats:
            if fmt == 'html':
                self.generate_html_report(
                    output_path=os.path.join(self.output_dir, f"{filename_prefix}.html"),
                    include_pairs=include_pairs,
                    max_pairs=max_pairs
                )
            elif fmt == 'excel':
                self.generate_excel_report(
                    output_path=os.path.join(self.output_dir, f"{filename_prefix}.xlsx"),
                    include_pairs=include_pairs,
                    max_pairs=max_pairs
                )
            elif fmt == 'markdown':
                self.generate_markdown_report(
                    output_path=os.path.join(self.output_dir, f"{filename_prefix}.md"),
                    include_pairs=include_pairs,
                    max_pairs=max_pairs
                )
            elif fmt == 'pptx':
                self.generate_powerpoint_report(
                    output_path=os.path.join(self.output_dir, f"{filename_prefix}.pptx"),
                    include_pairs=include_pairs,
                    max_pairs=max_pairs
                )
            elif fmt == 'jupyter':
                # Jupyter format doesn't save a file
                pass
        
        return self.report_paths
    
    def generate_html_report(
        self,
        output_path: str,
        include_pairs: bool = True,
        max_pairs: int = 100,
        theme: str = "cosmo"
    ) -> str:
        """
        Generate comprehensive HTML report for deduplication results.
        
        Parameters
        ----------
        output_path : str
            Path to save the HTML report
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the report
        max_pairs : int, default=100
            Maximum number of duplicate pairs to include
        theme : str, default="cosmo"
            Bootstrap theme for the report
            
        Returns
        -------
        str
            Path to the generated HTML report
        """
        # Prepare results in the format expected by generate_deduplication_report
        report_results = self._prepare_report_data()
        
        # Generate the report
        generate_deduplication_report(
            results=report_results,
            title=self.title,
            format="html",
            output_path=output_path,
            include_pairs=include_pairs,
            max_pairs=max_pairs,
            theme=theme
        )
        
        logger.info(f"HTML report generated: {output_path}")
        self.report_paths['html'] = output_path
        return output_path
    
    def generate_excel_report(
        self,
        output_path: str,
        include_pairs: bool = True,
        max_pairs: int = 100
    ) -> str:
        """
        Generate detailed Excel report with duplicate pairs and metrics.
        
        Parameters
        ----------
        output_path : str
            Path to save the Excel report
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the report
        max_pairs : int, default=100
            Maximum number of duplicate pairs to include
            
        Returns
        -------
        str
            Path to the generated Excel report
        """
        # Prepare results in the format expected by export_deduplication_report
        report_results = self._prepare_report_data()
        
        # Generate the report
        export_deduplication_report(
            results=report_results,
            format="excel",
            output_path=output_path.replace('.xlsx', ''),  # Function will add extension
            include_pairs=include_pairs,
            max_pairs=max_pairs
        )
        
        logger.info(f"Excel report generated: {output_path}")
        self.report_paths['excel'] = output_path
        return output_path
    
    def generate_markdown_report(
        self,
        output_path: str,
        include_pairs: bool = True,
        max_pairs: int = 100
    ) -> str:
        """
        Generate Markdown report for deduplication results.
        
        Parameters
        ----------
        output_path : str
            Path to save the Markdown report
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the report
        max_pairs : int, default=100
            Maximum number of duplicate pairs to include
            
        Returns
        -------
        str
            Path to the generated Markdown report
        """
        # Prepare results in the format expected by generate_deduplication_report
        report_results = self._prepare_report_data()
        
        # Generate the report
        generate_deduplication_report(
            results=report_results,
            title=self.title,
            format="markdown",
            output_path=output_path,
            include_pairs=include_pairs,
            max_pairs=max_pairs
        )
        
        logger.info(f"Markdown report generated: {output_path}")
        self.report_paths['markdown'] = output_path
        return output_path
    
    def generate_powerpoint_report(
        self,
        output_path: str,
        include_pairs: bool = True,
        max_pairs: int = 5  # Fewer pairs for PowerPoint
    ) -> str:
        """
        Generate PowerPoint presentation with deduplication results.
        
        Parameters
        ----------
        output_path : str
            Path to save the PowerPoint presentation
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the presentation
        max_pairs : int, default=5
            Maximum number of duplicate pairs to include
            
        Returns
        -------
        str
            Path to the generated PowerPoint presentation
        """
        # Prepare results in the format expected by export_deduplication_report
        report_results = self._prepare_report_data()
        
        # Generate the report
        export_deduplication_report(
            results=report_results,
            format="pptx",
            output_path=output_path.replace('.pptx', ''),  # Function will add extension
            include_pairs=include_pairs,
            max_pairs=max_pairs
        )
        
        logger.info(f"PowerPoint presentation generated: {output_path}")
        self.report_paths['pptx'] = output_path
        return output_path
    
    def display_jupyter_report(
        self,
        include_pairs: bool = True,
        max_pairs: int = 10,
        interactive: bool = True
    ) -> None:
        """
        Display an interactive report in a Jupyter notebook.
        
        Parameters
        ----------
        include_pairs : bool, default=True
            Whether to include duplicate pairs in the report
        max_pairs : int, default=10
            Maximum number of duplicate pairs to include
        interactive : bool, default=True
            Whether to include interactive widgets
            
        Returns
        -------
        None
            The report is displayed directly in the notebook
        """
        try:
            # Check if running in Jupyter
            get_ipython  # noqa
        except NameError:
            logger.warning("Not running in a Jupyter environment. Cannot display report.")
            return
        
        # Prepare report data
        report_data = self._prepare_report_data()
        
        # Create styled header
        display(HTML(f"""
        <div style='background-color: #2c3e50; color: white; padding: 10px; border-radius: 5px;'>
            <h1>{self.title}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """))
        
        # Display summary metrics
        if 'metrics' in report_data:
            self._display_jupyter_metrics(report_data['metrics'])
        
        # Display visualization tabs if interactive
        if interactive:
            self._display_interactive_visualizations(report_data)
        else:
            # Display static visualizations
            self._display_static_visualizations(report_data)
        
        # Display duplicate pairs if requested
        if include_pairs and 'duplicate_pairs' in report_data:
            self._display_jupyter_duplicate_pairs(report_data, max_pairs)
    
    def _display_jupyter_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display metrics section in Jupyter notebook."""
        # Create styled metrics section
        metrics_html = """
        <div style='margin-top: 20px;'>
            <h2>Deduplication Metrics</h2>
            <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
        """
        
        # Format metrics as cards
        for metric, value in metrics.items():
            if metric.lower() not in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
                # Determine card color based on metric name
                if 'precision' in metric.lower():
                    color = self.colors['primary']
                elif 'recall' in metric.lower():
                    color = self.colors['success']
                elif 'f1' in metric.lower():
                    color = self.colors['info']
                elif 'accuracy' in metric.lower():
                    color = self.colors['warning']
                else:
                    color = self.colors['secondary']
                
                # Format the value
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                    # Convert to percentage if between 0-1
                    if 0 <= value <= 1 and metric.lower() not in ['threshold']:
                        formatted_value = f"{value * 100:.2f}%"
                else:
                    formatted_value = str(value)
                
                # Create card for this metric
                metrics_html += f"""
                <div style='background-color: {color}; color: white; padding: 15px; border-radius: 5px; flex: 1; min-width: 150px;'>
                    <div style='font-size: 0.9em; text-transform: uppercase;'>{metric.replace('_', ' ').title()}</div>
                    <div style='font-size: 1.8em; font-weight: bold;'>{formatted_value}</div>
                </div>
                """
        
        metrics_html += """
            </div>
        </div>
        """
        
        # Add confusion matrix metrics if available
        if all(k in metrics for k in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']):
            tp = metrics['true_positives']
            fp = metrics['false_positives']
            tn = metrics['true_negatives']
            fn = metrics['false_negatives']
            
            metrics_html += f"""
            <div style='margin-top: 20px;'>
                <h3>Confusion Matrix</h3>
                <table style='border-collapse: collapse; width: 50%; min-width: 300px; text-align: center;'>
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 10px;'></td>
                        <td style='border: 1px solid #ddd; padding: 10px; font-weight: bold;'>Predicted Duplicate</td>
                        <td style='border: 1px solid #ddd; padding: 10px; font-weight: bold;'>Predicted Not Duplicate</td>
                    </tr>
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 10px; font-weight: bold;'>Actual Duplicate</td>
                        <td style='border: 1px solid #ddd; padding: 10px; background-color: #d4edda;'>{tp}</td>
                        <td style='border: 1px solid #ddd; padding: 10px; background-color: #fff3cd;'>{fn}</td>
                    </tr>
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 10px; font-weight: bold;'>Actual Not Duplicate</td>
                        <td style='border: 1px solid #ddd; padding: 10px; background-color: #f8d7da;'>{fp}</td>
                        <td style='border: 1px solid #ddd; padding: 10px; background-color: #d1ecf1;'>{tn}</td>
                    </tr>
                </table>
            </div>
            """
        
        display(HTML(metrics_html))
    
    def _display_interactive_visualizations(self, report_data: Dict[str, Any]) -> None:
        """Display interactive visualizations using ipywidgets."""
        try:
            # Create tabs for different visualizations
            tab = widgets.Tab()
            tab_contents = []
            
            # Tab 1: Confusion Matrix
            if 'confusion_matrix_base64' in report_data:
                cm_img = widgets.Image(
                    value=base64.b64decode(report_data['confusion_matrix_base64']),
                    format='png',
                )
                cm_box = widgets.VBox([
                    widgets.HTML(value="<h3>Confusion Matrix Visualization</h3>"),
                    cm_img
                ])
                tab_contents.append(cm_box)
            
            # Tab 2: Feature Importances (if available)
            if 'feature_importances' in report_data and report_data['feature_importances']:
                # Create feature importance chart
                plt.figure(figsize=(10, 6))
                feature_importances = report_data['feature_importances']
                
                # Convert to DataFrame for plotting
                if isinstance(feature_importances, dict):
                    fi_df = pd.DataFrame({
                        'Feature': list(feature_importances.keys()),
                        'Importance': list(feature_importances.values())
                    }).sort_values(by='Importance', ascending=False)
                else:
                    fi_df = feature_importances
                
                # Plot top 15 features
                sns.barplot(x='Importance', y='Feature', data=fi_df.head(15))
                plt.title('Top Features for Duplicate Detection')
                plt.tight_layout()
                
                # Convert to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                
                fi_img = widgets.Image(
                    value=buf.read(),
                    format='png',
                )
                
                fi_box = widgets.VBox([
                    widgets.HTML(value="<h3>Feature Importance</h3>"),
                    fi_img
                ])
                tab_contents.append(fi_box)
            
            # Tab 3: Threshold Evaluation (if available)
            if 'thresholds_data' in report_data and report_data['thresholds_data'] is not None:
                # Create threshold evaluation chart
                plt.figure(figsize=(10, 6))
                thresholds_data = report_data['thresholds_data']
                
                plt.plot(thresholds_data['threshold'], thresholds_data['precision'], 'b-', label='Precision')
                plt.plot(thresholds_data['threshold'], thresholds_data['recall'], 'r-', label='Recall')
                plt.plot(thresholds_data['threshold'], thresholds_data['f1'], 'g-', label='F1 Score')
                plt.xlabel('Probability Threshold')
                plt.ylabel('Score')
                plt.title('Precision, Recall, and F1 at Different Thresholds')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Convert to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                
                th_img = widgets.Image(
                    value=buf.read(),
                    format='png',
                )
                
                # Create slider for interactive threshold exploration
                threshold_slider = widgets.FloatSlider(
                    value=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    description='Threshold:',
                    style={'description_width': 'initial'}
                )
                
                # Create output widget for displaying metrics at selected threshold
                threshold_output = widgets.Output()
                
                # Update function for slider
                def update_threshold_metrics(change):
                    threshold = change['new']
                    threshold_row = thresholds_data[thresholds_data['threshold'].apply(lambda x: abs(x - threshold) < 0.051)].iloc[0]
                    
                    with threshold_output:
                        threshold_output.clear_output()
                        display(HTML(f"""
                        <div style='margin-top: 10px;'>
                            <h4>Metrics at threshold {threshold:.2f}</h4>
                            <table style='border-collapse: collapse; width: 100%;'>
                                <tr>
                                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Precision</th>
                                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Recall</th>
                                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>F1 Score</th>
                                </tr>
                                <tr>
                                    <td style='border: 1px solid #ddd; padding: 8px;'>{threshold_row['precision']:.4f}</td>
                                    <td style='border: 1px solid #ddd; padding: 8px;'>{threshold_row['recall']:.4f}</td>
                                    <td style='border: 1px solid #ddd; padding: 8px;'>{threshold_row['f1']:.4f}</td>
                                </tr>
                            </table>
                        </div>
                        """))
                
                # Register callback
                threshold_slider.observe(update_threshold_metrics, names='value')
                
                # Initial update
                update_threshold_metrics({'new': threshold_slider.value})
                
                th_box = widgets.VBox([
                    widgets.HTML(value="<h3>Threshold Evaluation</h3>"),
                    th_img,
                    threshold_slider,
                    threshold_output
                ])
                tab_contents.append(th_box)
            
            # Configure tab
            if tab_contents:
                tab.children = tab_contents
                if len(tab_contents) >= 1:
                    tab.set_title(0, "Confusion Matrix")
                if len(tab_contents) >= 2:
                    tab.set_title(1, "Feature Importance")
                if len(tab_contents) >= 3:
                    tab.set_title(2, "Threshold Evaluation")
                
                display(tab)
            else:
                display(HTML("<p>No visualizations available.</p>"))
                
        except ImportError:
            logger.warning("ipywidgets not available. Falling back to static visualizations.")
            self._display_static_visualizations(report_data)
    
    def _display_static_visualizations(self, report_data: Dict[str, Any]) -> None:
        """Display static visualizations for environments without ipywidgets."""
        # Display confusion matrix if available
        if 'confusion_matrix_base64' in report_data:
            display(HTML(f"""
            <div style='margin-top: 20px;'>
                <h3>Confusion Matrix</h3>
                <img src="data:image/png;base64,{report_data['confusion_matrix_base64']}" 
                     style="max-width: 600px; border: 1px solid #ddd; border-radius: 5px;">
            </div>
            """))
            
        # Display feature importances if available
        if 'feature_importances' in report_data and report_data['feature_importances']:
            # Create feature importance chart
            plt.figure(figsize=(10, 6))
            feature_importances = report_data['feature_importances']
            
            # Convert to DataFrame for plotting
            if isinstance(feature_importances, dict):
                fi_df = pd.DataFrame({
                    'Feature': list(feature_importances.keys()),
                    'Importance': list(feature_importances.values())
                }).sort_values(by='Importance', ascending=False)
            else:
                fi_df = feature_importances
            
            # Plot top 15 features
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(15))
            plt.title('Top Features for Duplicate Detection')
            plt.tight_layout()
            
            # Display the plot
            display(plt.gcf())
            plt.close()
            
        # Display threshold evaluation if available
        if 'thresholds_data' in report_data and report_data['thresholds_data'] is not None:
            # Create threshold evaluation chart
            plt.figure(figsize=(10, 6))
            thresholds_data = report_data['thresholds_data']
            
            plt.plot(thresholds_data['threshold'], thresholds_data['precision'], 'b-', label='Precision')
            plt.plot(thresholds_data['threshold'], thresholds_data['recall'], 'r-', label='Recall')
            plt.plot(thresholds_data['threshold'], thresholds_data['f1'], 'g-', label='F1 Score')
            plt.xlabel('Probability Threshold')
            plt.ylabel('Score')
            plt.title('Precision, Recall, and F1 at Different Thresholds')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Display the plot
            display(plt.gcf())
            plt.close()
            
            # Also display a table of threshold values
            display(HTML(f"""
            <div style='margin-top: 20px;'>
                <h3>Threshold Evaluation</h3>
                <p>The table below shows performance metrics at different threshold values:</p>
                {thresholds_data.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
            </div>
            """))
    
    def _display_jupyter_duplicate_pairs(self, report_data: Dict[str, Any], max_pairs: int) -> None:
        """Display duplicate pairs section in Jupyter notebook."""
        duplicate_pairs = report_data['duplicate_pairs']
        record_data = report_data.get('record_data', None)
        
        display(HTML(f"""
        <div style='margin-top: 20px;'>
            <h2>Sample Duplicate Pairs</h2>
        </div>
        """))
        
        # Handle different formats of duplicate pairs
        if isinstance(duplicate_pairs, pd.DataFrame):
            # Limit number of pairs
            pairs_df = duplicate_pairs.head(max_pairs)
            display(pairs_df)
            
        elif isinstance(duplicate_pairs, list):
            # Create a simple table
            pairs_html = """
            <table style='border-collapse: collapse; width: 50%; min-width: 300px;'>
                <tr>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Index 1</th>
                    <th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Index 2</th>
                </tr>
            """
            
            for pair in duplicate_pairs[:max_pairs]:
                if isinstance(pair, tuple) and len(pair) == 2:
                    pairs_html += f"""
                    <tr>
                        <td style='border: 1px solid #ddd; padding: 8px;'>{pair[0]}</td>
                        <td style='border: 1px solid #ddd; padding: 8px;'>{pair[1]}</td>
                    </tr>
                    """
            
            pairs_html += "</table>"
            display(HTML(pairs_html))
        
        # Show example records if available
        if record_data is not None and isinstance(record_data, list) and len(record_data) > 0:
            display(HTML("<h3>Example Records</h3>"))
            
            for i, record_pair in enumerate(record_data[:max_pairs]):
                record_html = f"""
                <div style='margin-top: 15px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden;'>
                    <div style='background-color: {self.colors['secondary']}; color: white; padding: 10px;'>
                        <strong>Duplicate Pair {i+1}</strong>
                """
                
                # Add probability if available
                if len(record_pair) > 2 and isinstance(record_pair[2], float):
                    record_html += f" (Similarity: {record_pair[2]:.4f})"
                
                record_html += """
                    </div>
                    <div style='display: flex; flex-wrap: wrap;'>
                """
                
                # Record 1
                record_html += """
                        <div style='flex: 1; min-width: 300px; padding: 10px;'>
                            <h4>Record 1</h4>
                            <table style='border-collapse: collapse; width: 100%;'>
                """
                
                for field, value in record_pair[0].items():
                    record_html += f"""
                                <tr>
                                    <td style='border: 1px solid #ddd; padding: 8px; font-weight: bold; width: 30%;'>{field}</td>
                                    <td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>
                                </tr>
                    """
                
                record_html += """
                            </table>
                        </div>
                """
                
                # Record 2
                record_html += """
                        <div style='flex: 1; min-width: 300px; padding: 10px;'>
                            <h4>Record 2</h4>
                            <table style='border-collapse: collapse; width: 100%;'>
                """
                
                for field, value in record_pair[1].items():
                    record_html += f"""
                                <tr>
                                    <td style='border: 1px solid #ddd; padding: 8px; font-weight: bold; width: 30%;'>{field}</td>
                                    <td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>
                                </tr>
                    """
                
                record_html += """
                            </table>
                        </div>
                    </div>
                </div>
                """
                
                display(HTML(record_html))
    
    def _prepare_report_data(self) -> Dict[str, Any]:
        """
        Prepare data for reporting in the format expected by report functions.
        
        Returns
        -------
        Dict[str, Any]
            Processed report data
        """
        report_data = self.results.copy()
        
        # Create visualization figures
        if 'metrics' in report_data and 'confusion_matrix_base64' not in report_data:
            # Get dataframe and columns for creating confusion matrix
            df = report_data.get('dataframe', None)
            flag_column = report_data.get('flag_column', None)
            truth_column = report_data.get('truth_column', None)
            
            if df is not None and flag_column is not None and truth_column is not None:
                # Generate confusion matrix base64
                try:
                    cm_base64 = plot_confusion_matrix(
                        df=df,
                        prediction_column=flag_column,
                        truth_column=truth_column,
                        as_base64=True
                    )
                    report_data['confusion_matrix_base64'] = cm_base64
                except Exception as e:
                    logger.warning(f"Error generating confusion matrix: {str(e)}")
        
        # Extract duplicate pairs if available
        if 'duplicate_pairs' not in report_data:
            # Try to extract if we have model and dataframe
            df = report_data.get('dataframe', None)
            model = report_data.get('model', None)
            threshold = report_data.get('threshold', 0.8)
            
            if df is not None and model is not None and hasattr(model, 'find_duplicates'):
                try:
                    duplicate_pairs = model.find_duplicates(
                        df=df,
                        threshold=threshold,
                        return_probabilities=True
                    )
                    report_data['duplicate_pairs'] = duplicate_pairs
                except Exception as e:
                    logger.warning(f"Error extracting duplicate pairs: {str(e)}")
        
        # Create records data if available
        if 'record_data' not in report_data and 'duplicate_pairs' in report_data:
            df = report_data.get('dataframe', None)
            duplicate_pairs = report_data.get('duplicate_pairs', None)
            
            if df is not None and duplicate_pairs is not None:
                try:
                    # Create record data for display
                    record_data = []
                    
                    # Handle different formats of duplicate_pairs
                    if isinstance(duplicate_pairs, pd.DataFrame):
                        sample_pairs = duplicate_pairs.head(10).values
                    elif isinstance(duplicate_pairs, list):
                        sample_pairs = duplicate_pairs[:10]
                        # Add dummy probabilities for list format
                        sample_pairs = [(pair[0], pair[1], 1.0) for pair in sample_pairs]
                    else:
                        sample_pairs = []
                    
                    for pair in sample_pairs:
                        if len(pair) >= 2:  # Check if pair has at least two indices
                            idx1, idx2 = int(pair[0]), int(pair[1])
                            prob = float(pair[2]) if len(pair) > 2 else 1.0
                            
                            # Get record data safely
                            if idx1 < len(df) and idx2 < len(df):
                                record1 = df.iloc[idx1].to_dict()
                                record2 = df.iloc[idx2].to_dict()
                                
                                # Sample fields if too many
                                if len(record1) > 10:
                                    record1 = {k: record1[k] for k in list(record1.keys())[:10]}
                                if len(record2) > 10:
                                    record2 = {k: record2[k] for k in list(record2.keys())[:10]}
                                
                                record_data.append((record1, record2, prob))
                    
                    if record_data:
                        report_data['record_data'] = record_data
                except Exception as e:
                    logger.warning(f"Error creating record data: {str(e)}")
        
        return report_data


def generate_enhanced_report(
    results: Dict[str, Any],
    formats: List[str] = ['html'],
    output_dir: str = "dedup_reports",
    title: str = "Deduplication Analysis",
    include_pairs: bool = True,
    max_pairs: int = 100,
    filename_prefix: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate enhanced deduplication reports in multiple formats.
    
    This function provides a simple interface to generate comprehensive
    deduplication reports in various formats.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing deduplication results
    formats : List[str], default=['html']
        List of output formats ('html', 'excel', 'markdown', 'pptx', 'jupyter')
    output_dir : str, default="dedup_reports"
        Directory to save reports
    title : str, default="Deduplication Analysis"
        Title for the report
    include_pairs : bool, default=True
        Whether to include duplicate pairs in the reports
    max_pairs : int, default=100
        Maximum number of duplicate pairs to include
    filename_prefix : Optional[str], default=None
        Prefix for output filenames
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping format to output file path
    """
    # Create reporter instance
    reporter = EnhancedDeduplicationReport(
        results=results,
        title=title,
        output_dir=output_dir,
        create_dir=True
    )
    
    # Generate reports in all requested formats
    report_paths = reporter.generate_all_reports(
        formats=formats,
        include_pairs=include_pairs,
        max_pairs=max_pairs,
        filename_prefix=filename_prefix
    )
    
    # Special handling for Jupyter format
    if 'jupyter' in formats:
        reporter.display_jupyter_report(
            include_pairs=include_pairs,
            max_pairs=max_pairs,
            interactive=True
        )
    
    return report_paths


def display_jupyter_report(
    results: Dict[str, Any],
    title: str = "Deduplication Analysis",
    include_pairs: bool = True,
    max_pairs: int = 10,
    interactive: bool = True
) -> None:
    """
    Display an interactive deduplication report in a Jupyter notebook.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing deduplication results
    title : str, default="Deduplication Analysis"
        Title for the report
    include_pairs : bool, default=True
        Whether to include duplicate pairs in the report
    max_pairs : int, default=10
        Maximum number of duplicate pairs to include
    interactive : bool, default=True
        Whether to include interactive widgets
        
    Returns
    -------
    None
        The report is displayed directly in the notebook
    """
    reporter = EnhancedDeduplicationReport(
        results=results,
        title=title
    )
    
    reporter.display_jupyter_report(
        include_pairs=include_pairs,
        max_pairs=max_pairs,
        interactive=interactive
    )