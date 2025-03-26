"""
Evaluation metrics and visualization for deduplication.

This module provides tools for evaluating deduplication performance against
known duplicate flags and generating reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import logging
import io
import base64

logger = logging.getLogger(__name__)


def calculate_deduplication_metrics(
    df: Any,
    prediction_column: str,
    truth_column: str,
    positive_label: Any = True
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for deduplication results.
    
    Parameters
    ----------
    df : Any
        Dataframe containing prediction and ground truth columns
    prediction_column : str
        Column name containing predicted duplicate flags
    truth_column : str
        Column name containing ground truth duplicate flags
    positive_label : Any, default=True
        Value in the columns that indicates a duplicate
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics including precision, recall, and F1 score
    """
    # Handle potential missing values
    df_eval = df[[prediction_column, truth_column]].dropna()
    
    if len(df_eval) == 0:
        logger.warning("No valid data for evaluation after removing missing values")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
    
    # Get predictions and ground truth
    y_pred = df_eval[prediction_column].values
    y_true = df_eval[truth_column].values
    
    # Convert non-boolean types to boolean if necessary
    if not pd.api.types.is_bool_dtype(y_pred):
        y_pred = y_pred == positive_label
    if not pd.api.types.is_bool_dtype(y_true):
        y_true = y_true == positive_label
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def plot_confusion_matrix(
    df: Any,
    prediction_column: str,
    truth_column: str,
    positive_label: Any = True,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Deduplication Confusion Matrix",
    as_base64: bool = False
) -> Optional[str]:
    """
    Create a confusion matrix visualization for deduplication results.
    
    Parameters
    ----------
    df : Any
        Dataframe containing prediction and ground truth columns
    prediction_column : str
        Column name containing predicted duplicate flags
    truth_column : str
        Column name containing ground truth duplicate flags
    positive_label : Any, default=True
        Value in the columns that indicates a duplicate
    figsize : Tuple[int, int], default=(8, 6)
        Figure size as (width, height) in inches
    title : str, default="Deduplication Confusion Matrix"
        Plot title
    as_base64 : bool, default=False
        If True, return the plot as a base64-encoded string
        
    Returns
    -------
    Optional[str]
        If as_base64 is True, returns a base64-encoded string of the plot
    """
    # Handle potential missing values
    df_eval = df[[prediction_column, truth_column]].dropna()
    
    if len(df_eval) == 0:
        logger.warning("No valid data for evaluation after removing missing values")
        return None
    
    # Get predictions and ground truth
    y_pred = df_eval[prediction_column].values
    y_true = df_eval[truth_column].values
    
    # Convert non-boolean types to boolean if necessary
    if not pd.api.types.is_bool_dtype(y_pred):
        y_pred = y_pred == positive_label
    if not pd.api.types.is_bool_dtype(y_true):
        y_true = y_true == positive_label
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Use a blue color palette
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["Not Duplicate", "Duplicate"],
        yticklabels=["Not Duplicate", "Duplicate"]
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # If we want a base64 encoded string
    if as_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str
    else:
        plt.tight_layout()
        plt.show()
        return None


def evaluate_threshold_sensitivity(
    df: Any,
    columns: List[str],
    truth_column: str,
    weights: Optional[Dict[str, float]] = None,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    method: str = 'composite',
    show_plot: bool = True,
    as_base64: bool = False
) -> Dict[str, Any]:
    """
    Evaluate deduplication performance across different similarity thresholds.
    
    Parameters
    ----------
    df : Any
        Dataframe containing data and ground truth
    columns : List[str]
        Columns to use for deduplication
    truth_column : str
        Column name containing ground truth duplicate flags
    weights : Optional[Dict[str, float]], default=None
        Weight for each column in similarity calculation
    thresholds : List[float], default=[0.5, 0.6, 0.7, 0.8, 0.9]
        Threshold values to evaluate
    method : str, default='composite'
        Similarity calculation method
    show_plot : bool, default=True
        Whether to show the plot
    as_base64 : bool, default=False
        If True, return the plot as a base64-encoded string
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation metrics for each threshold
        and optionally the base64-encoded plot
    """
    from freamon.deduplication import flag_similar_records
    
    results = {'thresholds': thresholds, 'metrics': []}
    precision_values = []
    recall_values = []
    f1_values = []
    
    for threshold in thresholds:
        # Run deduplication with current threshold
        result_df = flag_similar_records(
            df=df,
            columns=columns,
            weights=weights,
            threshold=threshold,
            method=method,
            flag_column='_predicted_duplicate',
            auto_mode=True,  # Use auto mode for best performance
            show_progress=False  # Don't show progress to reduce noise
        )
        
        # Calculate metrics
        metrics = calculate_deduplication_metrics(
            df=result_df,
            prediction_column='_predicted_duplicate',
            truth_column=truth_column
        )
        
        results['metrics'].append(metrics)
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        f1_values.append(metrics['f1'])
    
    # Create threshold sensitivity plot
    if show_plot or as_base64:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision_values, 'b-', marker='o', label='Precision')
        plt.plot(thresholds, recall_values, 'r-', marker='s', label='Recall')
        plt.plot(thresholds, f1_values, 'g-', marker='^', label='F1 Score')
        
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Score')
        plt.title('Deduplication Performance vs. Threshold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Find optimal threshold for F1
        optimal_idx = np.argmax(f1_values)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='gray', linestyle='--', alpha=0.7)
        plt.text(
            optimal_threshold, 0.5, 
            f'Optimal: {optimal_threshold:.2f}', 
            rotation=90, 
            verticalalignment='center'
        )
        
        if as_base64:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            results['plot'] = img_str
        elif show_plot:
            plt.tight_layout()
            plt.show()
    
    # Determine optimal threshold based on F1 score
    optimal_idx = np.argmax(f1_values)
    results['optimal_threshold'] = thresholds[optimal_idx]
    results['optimal_metrics'] = results['metrics'][optimal_idx]
    
    return results


def generate_evaluation_report(
    df: Any,
    prediction_column: str,
    truth_column: str,
    format: str = 'text',
    include_plots: bool = True
) -> Any:
    """
    Generate a comprehensive evaluation report for deduplication results.
    
    Parameters
    ----------
    df : Any
        Dataframe containing prediction and ground truth columns
    prediction_column : str
        Column name containing predicted duplicate flags
    truth_column : str
        Column name containing ground truth duplicate flags
    format : str, default='text'
        Report format: 'text', 'html', or 'markdown'
    include_plots : bool, default=True
        Whether to include visualizations in the report
        
    Returns
    -------
    Any
        Report in the specified format
    """
    # Calculate metrics
    metrics = calculate_deduplication_metrics(
        df=df,
        prediction_column=prediction_column,
        truth_column=truth_column
    )
    
    # Format precision, recall and F1 as percentages
    precision_pct = metrics['precision'] * 100
    recall_pct = metrics['recall'] * 100
    f1_pct = metrics['f1'] * 100
    accuracy_pct = metrics['accuracy'] * 100
    
    # Calculate distributions
    total_records = len(df)
    
    # Handle categorical or boolean column types safely
    if pd.api.types.is_categorical_dtype(df[prediction_column]):
        pred_col = df[prediction_column].astype(int)
    else:
        pred_col = df[prediction_column]
        
    if pd.api.types.is_categorical_dtype(df[truth_column]):
        truth_col = df[truth_column].astype(int)
    else:
        truth_col = df[truth_column]
    
    predicted_duplicates = pred_col.sum()
    actual_duplicates = truth_col.sum()
    predicted_pct = (predicted_duplicates / total_records) * 100
    actual_pct = (actual_duplicates / total_records) * 100
    
    # Generate confusion matrix base64 if needed
    cm_base64 = None
    if include_plots and format in ['html', 'markdown']:
        cm_base64 = plot_confusion_matrix(
            df=df,
            prediction_column=prediction_column,
            truth_column=truth_column,
            as_base64=True
        )
    
    # Create the report based on format
    if format == 'text':
        report = f"""
        DEDUPLICATION EVALUATION REPORT
        ------------------------------
        Total Records: {total_records}
        
        PREDICTIONS vs ACTUAL:
        - Predicted Duplicates: {predicted_duplicates} ({predicted_pct:.2f}%)
        - Actual Duplicates: {actual_duplicates} ({actual_pct:.2f}%)
        
        PERFORMANCE METRICS:
        - Precision: {precision_pct:.2f}%
        - Recall: {recall_pct:.2f}%
        - F1 Score: {f1_pct:.2f}%
        - Accuracy: {accuracy_pct:.2f}%
        
        CONFUSION MATRIX:
        - True Positives: {metrics['true_positives']}
        - False Positives: {metrics['false_positives']}
        - True Negatives: {metrics['true_negatives']}
        - False Negatives: {metrics['false_negatives']}
        """
        return report.strip()
    
    elif format == 'markdown':
        report = f"""
        # Deduplication Evaluation Report
        
        ## Summary
        - **Total Records**: {total_records}
        - **Predicted Duplicates**: {predicted_duplicates} ({predicted_pct:.2f}%)
        - **Actual Duplicates**: {actual_duplicates} ({actual_pct:.2f}%)
        
        ## Performance Metrics
        | Metric | Value |
        |--------|-------|
        | Precision | {precision_pct:.2f}% |
        | Recall | {recall_pct:.2f}% |
        | F1 Score | {f1_pct:.2f}% |
        | Accuracy | {accuracy_pct:.2f}% |
        
        ## Confusion Matrix
        | | | 
        |---|---|
        | True Positives: {metrics['true_positives']} | False Positives: {metrics['false_positives']} |
        | False Negatives: {metrics['false_negatives']} | True Negatives: {metrics['true_negatives']} |
        """
        
        if include_plots and cm_base64:
            report += f"\n\n## Visualization\n\n![Confusion Matrix](data:image/png;base64,{cm_base64})\n"
        
        return report.strip()
    
    elif format == 'html':
        html = f"""
        <div class="deduplication-report">
            <h1>Deduplication Evaluation Report</h1>
            
            <div class="summary-section">
                <h2>Summary</h2>
                <p><strong>Total Records</strong>: {total_records}</p>
                <p><strong>Predicted Duplicates</strong>: {predicted_duplicates} ({predicted_pct:.2f}%)</p>
                <p><strong>Actual Duplicates</strong>: {actual_duplicates} ({actual_pct:.2f}%)</p>
            </div>
            
            <div class="metrics-section">
                <h2>Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{precision_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{recall_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>{f1_pct:.2f}%</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{accuracy_pct:.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="confusion-section">
                <h2>Confusion Matrix</h2>
                <table class="confusion-table">
                    <tr>
                        <td></td>
                        <td colspan="2"><strong>Predicted</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Actual</strong></td>
                        <td><strong>Duplicate</strong></td>
                        <td><strong>Not Duplicate</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Duplicate</strong></td>
                        <td class="true-positive">{metrics['true_positives']}</td>
                        <td class="false-negative">{metrics['false_negatives']}</td>
                    </tr>
                    <tr>
                        <td><strong>Not Duplicate</strong></td>
                        <td class="false-positive">{metrics['false_positives']}</td>
                        <td class="true-negative">{metrics['true_negatives']}</td>
                    </tr>
                </table>
            </div>
        """
        
        if include_plots and cm_base64:
            html += f"""
            <div class="visualization-section">
                <h2>Visualization</h2>
                <img src="data:image/png;base64,{cm_base64}" alt="Confusion Matrix" />
            </div>
            """
        
        html += "</div>"
        
        # Add CSS styling
        html += """
        <style>
        .deduplication-report {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .deduplication-report h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
        }
        .deduplication-report h2 {
            color: #3498db;
            margin-top: 20px;
        }
        .metrics-table, .confusion-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        .metrics-table th, .metrics-table td,
        .confusion-table th, .confusion-table td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        .metrics-table th, .confusion-table th {
            background-color: #3498db;
            color: white;
        }
        .true-positive {
            background-color: #d4edda;
        }
        .false-positive {
            background-color: #f8d7da;
        }
        .false-negative {
            background-color: #fff3cd;
        }
        .true-negative {
            background-color: #d1ecf1;
        }
        .visualization-section img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """
        
        return html
    
    else:
        raise ValueError(f"Unsupported report format: {format}")


def flag_and_evaluate(
    df: Any,
    columns: List[str],
    known_duplicate_column: str,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.8,
    method: str = 'composite',
    flag_column: str = 'is_duplicate',
    generate_report: bool = False,
    report_format: str = 'text',
    include_plots: bool = True,
    auto_mode: bool = True,
    show_progress: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run deduplication and evaluate against known duplicate flags.
    
    Parameters
    ----------
    df : Any
        Dataframe containing data and ground truth
    columns : List[str]
        Columns to use for deduplication
    known_duplicate_column : str
        Column name containing ground truth duplicate flags
    weights : Optional[Dict[str, float]], default=None
        Weight for each column in similarity calculation
    threshold : float, default=0.8
        Similarity threshold
    method : str, default='composite'
        Similarity calculation method
    flag_column : str, default='is_duplicate'
        Column name for the output duplicate flags
    generate_report : bool, default=False
        Whether to generate an evaluation report
    report_format : str, default='text'
        Report format: 'text', 'html', or 'markdown'
    include_plots : bool, default=True
        Whether to include visualizations in the report
    auto_mode : bool, default=True
        Whether to use automatic parameter selection
    show_progress : bool, default=True
        Whether to show progress during deduplication
    **kwargs : dict
        Additional parameters to pass to flag_similar_records
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing results, metrics, and optional report
    """
    from freamon.deduplication import flag_similar_records
    
    # Run deduplication
    result_df = flag_similar_records(
        df=df,
        columns=columns,
        weights=weights,
        threshold=threshold,
        method=method,
        flag_column=flag_column,
        auto_mode=auto_mode,
        show_progress=show_progress,
        **kwargs
    )
    
    # Calculate evaluation metrics
    metrics = calculate_deduplication_metrics(
        df=result_df,
        prediction_column=flag_column,
        truth_column=known_duplicate_column
    )
    
    # Create confusion matrix if plots are requested
    cm_base64 = None
    if include_plots:
        cm_base64 = plot_confusion_matrix(
            df=result_df,
            prediction_column=flag_column,
            truth_column=known_duplicate_column,
            as_base64=True
        )
    
    # Generate evaluation report if requested
    report = None
    if generate_report:
        report = generate_evaluation_report(
            df=result_df,
            prediction_column=flag_column,
            truth_column=known_duplicate_column,
            format=report_format,
            include_plots=include_plots
        )
    
    # Return results
    results = {
        'dataframe': result_df,
        'metrics': metrics,
        'confusion_matrix_base64': cm_base64
    }
    
    if report:
        results['report'] = report
    
    return results