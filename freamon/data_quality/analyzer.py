"""
Module for data quality analysis and reporting.
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import jinja2

from freamon.utils import check_dataframe_type, convert_dataframe
from freamon.data_quality.duplicates import detect_duplicates
from freamon.data_quality.cardinality import analyze_cardinality


class DataQualityAnalyzer:
    """
    Class for analyzing data quality and generating HTML reports.
    
    Parameters
    ----------
    df : Any
        The dataframe to analyze. Supports pandas and other dataframe types.
    """
    
    def __init__(self, df: Any):
        """
        Initialize the DataQualityAnalyzer with a dataframe.
        
        Parameters
        ----------
        df : Any
            The dataframe to analyze. Supports pandas and other dataframe types.
        """
        self.df_type = check_dataframe_type(df)
        
        # Convert to pandas internally if needed
        if self.df_type != 'pandas':
            self.df = convert_dataframe(df, 'pandas')
        else:
            self.df = df.copy()
        
        self._validate_dataframe()
        
        # Initialize results dictionary
        self.results = {}
    
    def _validate_dataframe(self) -> None:
        """
        Validate that the dataframe is of a supported type and has valid structure.
        
        Raises
        ------
        ValueError
            If the dataframe is not of a supported type or has invalid structure.
        """
        # Check if the dataframe is empty
        if self.df.empty:
            raise ValueError("Dataframe is empty")
        
        # Check if there are any columns
        if len(self.df.columns) == 0:
            raise ValueError("Dataframe has no columns")
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataframe.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with missing value analysis results.
        """
        # Count missing values
        missing_count = self.df.isna().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        
        # Get columns with missing values
        missing_columns = missing_count[missing_count > 0]
        
        # Calculate overall missing metrics
        total_missing = missing_count.sum()
        total_cells = len(self.df) * len(self.df.columns)
        total_percent = (total_missing / total_cells) * 100
        
        # Create plot if there are missing values
        missing_plot = None
        if total_missing > 0:
            # Create visualization of missing values
            plt.figure(figsize=(10, 6))
            
            # Bar chart of missing values by column
            columns = missing_columns.index.tolist()
            counts = missing_columns.values
            y_pos = np.arange(len(columns))
            
            plt.barh(y_pos, counts)
            plt.yticks(y_pos, columns)
            plt.xlabel('Count')
            plt.ylabel('Column')
            plt.title('Missing Values by Column')
            
            # Add count labels
            for i, count in enumerate(counts):
                plt.text(count + 0.5, i, str(count), va='center')
            
            # Save plot to BytesIO
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            missing_plot = f"data:image/png;base64,{img_str}"
        
        result = {
            "missing_count": missing_count.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "missing_columns": missing_columns.to_dict(),
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "total_percent": float(total_percent),
            "has_missing": total_missing > 0,
        }
        
        if missing_plot:
            result["plot"] = missing_plot
        
        # Store result
        self.results["missing_values"] = result
        
        return result
    
    def analyze_data_types(self) -> Dict[str, Any]:
        """
        Analyze data types in the dataframe.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with data type analysis results.
        """
        # Get data types
        dtypes = self.df.dtypes.astype(str).to_dict()
        
        # Categorize columns by type
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime']).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['category']).columns.tolist()
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        # Count type categories
        type_counts = {
            'numeric': len(numeric_cols),
            'datetime': len(datetime_cols),
            'categorical': len(categorical_cols),
            'object': len(object_cols),
            'boolean': len(bool_cols),
        }
        
        # Count value types for each column
        type_consistency = {}
        for col in self.df.columns:
            if col in self.df.select_dtypes(include=['number']).columns:
                # For numeric columns, check if there are strings or other types
                # that have been coerced to NaN
                has_nan = self.df[col].isna().any()
                type_consistency[col] = {
                    "consistent": True,
                    "has_missing": has_nan,
                }
            else:
                # For non-numeric columns, check the unique Python types
                unique_types = {type(val).__name__ for val in self.df[col].dropna()}
                type_consistency[col] = {
                    "consistent": len(unique_types) <= 1,
                    "types": list(unique_types),
                    "has_missing": self.df[col].isna().any(),
                }
        
        # Create plot of data types
        plt.figure(figsize=(10, 6))
        
        # Pie chart of column types
        type_labels = []
        type_values = []
        for type_name, count in type_counts.items():
            if count > 0:
                type_labels.append(f"{type_name} ({count})")
                type_values.append(count)
        
        plt.pie(type_values, labels=type_labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Column Data Types')
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        types_plot = f"data:image/png;base64,{img_str}"
        
        result = {
            "dtypes": dtypes,
            "type_consistency": type_consistency,
            "type_counts": type_counts,
            "numeric_columns": numeric_cols,
            "datetime_columns": datetime_cols,
            "categorical_columns": categorical_cols,
            "object_columns": object_cols,
            "boolean_columns": bool_cols,
            "plot": types_plot,
        }
        
        # Store result
        self.results["data_types"] = result
        
        return result
    
    def analyze_duplicates(
        self,
        subset: Optional[Union[str, List[str]]] = None,
        return_counts: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze duplicate rows in the dataframe.
        
        Parameters
        ----------
        subset : Optional[Union[str, List[str]]], default=None
            Column(s) to consider for identifying duplicates. If None, uses all columns.
        return_counts : bool, default=True
            If True, include duplicate value counts in the results.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with duplicate analysis results.
        """
        # Detect duplicates
        duplicate_info = detect_duplicates(
            self.df, subset=subset, return_counts=return_counts
        )
        
        # Store the results
        self.results["duplicates"] = duplicate_info
        
        return duplicate_info
    
    def analyze_cardinality(
        self,
        columns: Optional[List[str]] = None,
        max_unique_to_list: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the cardinality (number of unique values) of columns in the dataframe.
        
        Parameters
        ----------
        columns : Optional[List[str]], default=None
            List of column names to analyze. If None, analyzes all columns.
        max_unique_to_list : int, default=20
            Maximum number of unique values to include in the result for each column.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary mapping column names to dictionaries of cardinality statistics.
        """
        # Analyze cardinality
        cardinality_info = analyze_cardinality(
            self.df, columns=columns, max_unique_to_list=max_unique_to_list
        )
        
        # Create a summary
        cardinality_summary = {}
        for col, info in cardinality_info.items():
            cardinality_summary[col] = {
                'unique_count': info['unique_count'],
                'cardinality_ratio': info['cardinality_ratio'],
                'cardinality_type': info['cardinality_type'],
            }
        
        # Create a plot of cardinality ratios
        plt.figure(figsize=(12, 6))
        
        # Get columns and ratios, sort by ratio
        cols = []
        ratios = []
        for col, info in sorted(
            cardinality_info.items(),
            key=lambda x: x[1]['cardinality_ratio'],
            reverse=True
        ):
            cols.append(col)
            ratios.append(info['cardinality_ratio'])
        
        # Bar chart of cardinality ratios
        plt.barh(cols, ratios)
        plt.xlabel('Cardinality Ratio')
        plt.title('Cardinality Ratio by Column')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        summary_plot = f"data:image/png;base64,{img_str}"
        
        # Store the results with summary
        self.results["cardinality"] = {
            "details": cardinality_info,
            "summary": cardinality_summary,
            "summary_plot": summary_plot,
        }
        
        return cardinality_info
    
    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Run all available analyses on the dataframe.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with all analysis results.
        """
        # Run all analyses
        self.analyze_missing_values()
        self.analyze_data_types()
        self.analyze_duplicates()
        self.analyze_cardinality()
        
        # Add basic dataframe info
        self.results["basic_info"] = {
            "shape": list(self.df.shape),
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / (1024 * 1024)),
        }
        
        # Add column list for convenience
        self.results["column_list"] = self.df.columns.tolist()
        
        return self.results
    
    def generate_report(self, output_path: str, title: str = "Data Quality Report") -> None:
        """
        Generate a comprehensive HTML data quality report.
        
        Parameters
        ----------
        output_path : str
            Path where the HTML report will be saved.
        title : str, default="Data Quality Report"
            Title for the report.
        """
        # If no analyses have been run, run them all
        if not self.results:
            self.run_all_analyses()
        
        # Get the Jinja2 template
        template_str = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 20px; }
                .section { margin-bottom: 30px; }
                .plot-img { max-width: 100%; height: auto; }
                .table-sm td, .table-sm th { padding: 0.25rem; }
                .card { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-4">{{ title }}</h1>
                
                <!-- Overview Section -->
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Basic Information</h5>
                                </div>
                                <div class="card-body">
                                    <table class="table table-sm">
                                        <tbody>
                                            <tr>
                                                <th>Rows</th>
                                                <td>{{ results.basic_info.rows }}</td>
                                            </tr>
                                            <tr>
                                                <th>Columns</th>
                                                <td>{{ results.basic_info.columns }}</td>
                                            </tr>
                                            <tr>
                                                <th>Memory Usage</th>
                                                <td>{{ "%.2f"|format(results.basic_info.memory_usage_mb) }} MB</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Data Types</h5>
                                </div>
                                <div class="card-body">
                                    {% if results.data_types.plot %}
                                    <img src="{{ results.data_types.plot }}" class="plot-img" alt="Data Types">
                                    {% else %}
                                    <table class="table table-sm">
                                        <tbody>
                                            {% for type_name, count in results.data_types.type_counts.items() %}
                                            {% if count > 0 %}
                                            <tr>
                                                <th>{{ type_name }}</th>
                                                <td>{{ count }}</td>
                                            </tr>
                                            {% endif %}
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Missing Values Section -->
                {% if results.missing_values %}
                <div class="section">
                    <h2>Missing Values</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Missing Values Summary</h5>
                                </div>
                                <div class="card-body">
                                    <table class="table table-sm">
                                        <tbody>
                                            <tr>
                                                <th>Total Missing</th>
                                                <td>{{ results.missing_values.total_missing }}</td>
                                            </tr>
                                            <tr>
                                                <th>Missing Percentage</th>
                                                <td>{{ "%.2f"|format(results.missing_values.total_percent) }}%</td>
                                            </tr>
                                            <tr>
                                                <th>Columns with Missing</th>
                                                <td>{{ results.missing_values.missing_columns|length }}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            {% if results.missing_values.plot %}
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Missing Values by Column</h5>
                                </div>
                                <div class="card-body">
                                    <img src="{{ results.missing_values.plot }}" class="plot-img" alt="Missing Values by Column">
                                </div>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    
                    {% if results.missing_values.missing_columns %}
                    <div class="card mt-4">
                        <div class="card-header">
                            <h5 class="card-title">Columns with Missing Values</h5>
                        </div>
                        <div class="card-body">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Column</th>
                                        <th>Missing Count</th>
                                        <th>Missing Percentage</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for column, count in results.missing_values.missing_columns.items() %}
                                    <tr>
                                        <td>{{ column }}</td>
                                        <td>{{ count }}</td>
                                        <td>{{ "%.2f"|format(results.missing_values.missing_percent[column]) }}%</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
                
                <!-- Duplicates Section -->
                {% if results.duplicates %}
                <div class="section">
                    <h2>Duplicate Rows</h2>
                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Duplicates Summary</h5>
                                </div>
                                <div class="card-body">
                                    {% if results.duplicates.has_duplicates %}
                                    <p>
                                        Found <strong>{{ results.duplicates.duplicate_count }}</strong> 
                                        duplicate rows ({{ "%.2f"|format(results.duplicates.duplicate_percentage) }}% of all rows).
                                    </p>
                                    <table class="table table-sm">
                                        <tbody>
                                            <tr>
                                                <th>Total Rows</th>
                                                <td>{{ results.duplicates.total_rows }}</td>
                                            </tr>
                                            <tr>
                                                <th>Unique Rows</th>
                                                <td>{{ results.duplicates.unique_rows }}</td>
                                            </tr>
                                            <tr>
                                                <th>Duplicate Rows</th>
                                                <td>{{ results.duplicates.duplicate_count }}</td>
                                            </tr>
                                            <tr>
                                                <th>Duplicate Percentage</th>
                                                <td>{{ "%.2f"|format(results.duplicates.duplicate_percentage) }}%</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                    
                                    {% if results.duplicates.value_counts %}
                                    <h6 class="mt-3">Most Common Duplicate Values:</h6>
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Values</th>
                                                <th>Count</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for item in results.duplicates.value_counts[:10] %}
                                            <tr>
                                                <td>
                                                {% for key, value in item.values.items() %}
                                                    <small>{{ key }}: {{ value }}</small><br>
                                                {% endfor %}
                                                </td>
                                                <td>{{ item.count }}</td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                    {% endif %}
                                    
                                    {% else %}
                                    <p>No duplicate rows found in the dataset.</p>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Cardinality Section -->
                {% if results.cardinality %}
                <div class="section">
                    <h2>Cardinality Analysis</h2>
                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Cardinality Overview</h5>
                                </div>
                                <div class="card-body">
                                    {% if results.cardinality.summary_plot %}
                                    <img src="{{ results.cardinality.summary_plot }}" class="plot-img" alt="Cardinality Ratios">
                                    {% endif %}
                                    
                                    <h6 class="mt-3">Column Cardinality Summary:</h6>
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Column</th>
                                                <th>Unique Values</th>
                                                <th>Cardinality Ratio</th>
                                                <th>Type</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for column, info in results.cardinality.summary.items() %}
                                            <tr>
                                                <td>{{ column }}</td>
                                                <td>{{ info.unique_count }}</td>
                                                <td>{{ "%.4f"|format(info.cardinality_ratio) }}</td>
                                                <td>{{ info.cardinality_type }}</td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="accordion" id="cardinalityAccordion">
                                {% for column, info in results.cardinality.details.items() %}
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="heading-{{ column }}">
                                        <button class="accordion-button collapsed" type="button" 
                                                data-bs-toggle="collapse" data-bs-target="#collapse-{{ column }}" 
                                                aria-expanded="false" aria-controls="collapse-{{ column }}">
                                            {{ column }} ({{ info.unique_count }} unique values)
                                        </button>
                                    </h2>
                                    <div id="collapse-{{ column }}" class="accordion-collapse collapse" 
                                         aria-labelledby="heading-{{ column }}" data-bs-parent="#cardinalityAccordion">
                                        <div class="accordion-body">
                                            <div class="row">
                                                <div class="col-md-6">
                                                    <h6>Value Distribution:</h6>
                                                    <table class="table table-sm">
                                                        <thead>
                                                            <tr>
                                                                <th>Value</th>
                                                                <th>Count</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {% for value, count in info.value_counts.items() %}
                                                            <tr>
                                                                <td>{{ value }}</td>
                                                                <td>{{ count }}</td>
                                                            </tr>
                                                            {% endfor %}
                                                        </tbody>
                                                    </table>
                                                    {% if info.value_counts_limited %}
                                                    <small class="text-muted">
                                                        Showing limited values. The column has more unique values than displayed.
                                                    </small>
                                                    {% endif %}
                                                </div>
                                                <div class="col-md-6">
                                                    {% if info.plot %}
                                                    <img src="{{ info.plot }}" class="plot-img" alt="Distribution for {{ column }}">
                                                    {% endif %}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Data Types Section -->
                {% if results.data_types %}
                <div class="section">
                    <h2>Data Types Analysis</h2>
                    <div class="row">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    <h5 class="card-title">Column Data Types</h5>
                                </div>
                                <div class="card-body">
                                    <table class="table table-sm">
                                        <thead>
                                            <tr>
                                                <th>Column</th>
                                                <th>Data Type</th>
                                                <th>Consistent Types</th>
                                                <th>Has Missing</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for column, dtype in results.data_types.dtypes.items() %}
                                            <tr>
                                                <td>{{ column }}</td>
                                                <td>{{ dtype }}</td>
                                                <td>
                                                    {% if results.data_types.type_consistency[column].consistent %}
                                                    <span class="badge bg-success">Yes</span>
                                                    {% else %}
                                                    <span class="badge bg-warning">No</span>
                                                    {% endif %}
                                                </td>
                                                <td>
                                                    {% if results.data_types.type_consistency[column].has_missing %}
                                                    <span class="badge bg-warning">Yes</span>
                                                    {% else %}
                                                    <span class="badge bg-success">No</span>
                                                    {% endif %}
                                                </td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Render the template
        template = jinja2.Template(template_str)
        html_content = template.render(
            title=title,
            results=self.results,
        )
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write the HTML to a file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")