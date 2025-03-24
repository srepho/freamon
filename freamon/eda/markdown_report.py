"""
Functions for generating Markdown reports from EDA results.
"""
from typing import Any, Dict, List, Optional, Union, Callable
import os
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure matplotlib to avoid treating $ as LaTeX math delimiters
plt.rcParams['text.usetex'] = False


def generate_markdown_report(
    df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_path: str,
    title: str = "Exploratory Data Analysis Report",
    convert_to_html: bool = False,
    include_export_button: bool = True,
) -> str:
    """
    Generate a Markdown report with the EDA results.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that was analyzed.
    analysis_results : Dict[str, Any]
        The dictionary of analysis results.
    output_path : str
        The path to save the Markdown report.
    title : str, default="Exploratory Data Analysis Report"
        The title of the report.
    convert_to_html : bool, default=False
        Whether to convert the Markdown to HTML after generating it.
    include_export_button : bool, default=True
        Whether to include a button to export the report as a Jupyter notebook (only if convert_to_html=True).
        
    Returns
    -------
    str
        The Markdown report as a string.
    """
    from freamon.utils.matplotlib_fixes import optimize_base64_image
    
    # Start building Markdown parts
    md_parts = []
    
    # Add title and date
    md_parts.append(f"# {title}")
    md_parts.append(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_parts.append("")  # Empty line

    # Add info alert
    sampling_used = any("sampling_info" in section.get(key, {}) 
                        for section in analysis_results.values() 
                        if isinstance(section, dict) 
                        for key in section if isinstance(section.get(key), dict))
    
    md_parts.append("> **Info**: This report provides an overview of the dataset and the results of exploratory data analysis.")
    if sampling_used:
        md_parts.append("> **Note**: Sampling was used for some analyses to improve performance.")
    md_parts.append("")  # Empty line
    
    # Generate overview section
    md_parts.append("## Dataset Overview")
    
    # Basic dataset statistics
    if "basic_stats" in analysis_results:
        stats = analysis_results["basic_stats"]
        md_parts.append("### Basic Statistics")
        md_parts.append("")
        md_parts.append("| Metric | Value |")
        md_parts.append("|--------|-------|")
        md_parts.append(f"| Rows | {stats['n_rows']} |")
        md_parts.append(f"| Columns | {stats['n_cols']} |")
        md_parts.append(f"| Numeric Columns | {stats['n_numeric']} |")
        md_parts.append(f"| Categorical Columns | {stats['n_categorical']} |")
        md_parts.append(f"| Datetime Columns | {stats['n_datetime']} |")
        md_parts.append(f"| Memory Usage | {stats['memory_usage_mb']:.2f} MB |")
        md_parts.append("")  # Empty line
        
        # Missing values
        md_parts.append("### Missing Values")
        md_parts.append("")
        
        if stats["has_missing"]:
            md_parts.append(f"This dataset contains **{stats['missing_count']}** missing values ({stats['missing_percent']:.2f}% of all values).")
            md_parts.append("")
            md_parts.append("#### Columns with Missing Values:")
            md_parts.append("")
            md_parts.append("| Column | Count | Percentage |")
            md_parts.append("|--------|-------|------------|")
            
            for col, count in stats["missing_by_column"].items():
                pct = (count / stats["n_rows"]) * 100
                md_parts.append(f"| {col} | {count} | {pct:.2f}% |")
        else:
            md_parts.append("This dataset does not contain any missing values.")
        
        md_parts.append("")  # Empty line
    
    # Column list
    if "basic_stats" in analysis_results:
        stats = analysis_results["basic_stats"]
        md_parts.append("### Column Types")
        md_parts.append("")
        
        # Show lists of columns by type
        if "numeric_columns" in stats and stats["numeric_columns"]:
            md_parts.append("#### Numeric Columns:")
            md_parts.append("")
            for col in stats["numeric_columns"]:
                md_parts.append(f"- {col}")
            md_parts.append("")
        
        if "categorical_columns" in stats and stats["categorical_columns"]:
            md_parts.append("#### Categorical Columns:")
            md_parts.append("")
            for col in stats["categorical_columns"]:
                md_parts.append(f"- {col}")
            md_parts.append("")
        
        if "datetime_columns" in stats and stats["datetime_columns"]:
            md_parts.append("#### Datetime Columns:")
            md_parts.append("")
            for col in stats["datetime_columns"]:
                md_parts.append(f"- {col}")
            md_parts.append("")
    
    # Sample data
    md_parts.append("### Sample Data")
    md_parts.append("")
    
    # Calculate appropriate sample size based on dataframe size
    if len(df) > 1000000:
        sample_size = 10
    elif len(df) > 100000:
        sample_size = 20
    elif len(df) > 10000:
        sample_size = 30
    else:
        sample_size = 50
    
    # Show sample data with markdown table format
    if len(df) > sample_size:
        half_size = sample_size // 2
        
        # Get first and last parts
        first_part = df.head(half_size)
        last_part = df.tail(half_size)
        
        # Convert DataFrame to markdown table
        md_parts.extend(_dataframe_to_markdown(first_part))
        
        # Add ellipsis
        row_count = len(df)
        md_parts.append(f"*... {row_count - sample_size:,} more rows ...*")
        md_parts.append("")
        
        # Add last part
        md_parts.extend(_dataframe_to_markdown(last_part, include_header=False))
    else:
        # For small datasets, show all rows
        md_parts.extend(_dataframe_to_markdown(df))
    
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
    if "plot" in str(analysis_results):
        analysis_results = optimize_plot_images(analysis_results)
    
    # Generate univariate analysis section
    if "univariate" in analysis_results:
        md_parts.append("## Univariate Analysis")
        md_parts.append("")
        md_parts.append("This section presents individual analysis of each column in the dataset, "
                     "showing distributions, statistics, and other relevant information.")
        md_parts.append("")
        
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
        
        # Helper function to generate a univariate section for a column
        def generate_univariate_content(col, result):
            content = []
            content.append(f"### {col}")
            content.append("")
            
            # Add statistics based on column type
            if result.get("type") == "numeric":
                content.append("#### Statistics")
                content.append("")
                content.append("| Metric | Value |")
                content.append("|--------|-------|")
                content.append(f"| Count | {result['count']} |")
                content.append(f"| Missing | {result['missing']} ({result['missing_pct']:.2f}%) |")
                
                if "sampling_info" in result:
                    content.append(f"| Sampling | Analysis based on {result['sampling_info']['sample_size']} rows ({result['sampling_info']['sampling_ratio']:.1%} of data) |")
                
                content.append(f"| Mean | {result['mean']:.4f} |")
                content.append(f"| Median | {result['median']:.4f} |")
                content.append(f"| Std. Dev. | {result['std']:.4f} |")
                content.append(f"| Min | {result['min']:.4f} |")
                content.append(f"| Max | {result['max']:.4f} |")
                content.append(f"| Range | {result['range']:.4f} |")
                content.append("")
                
                # Add percentiles if available
                percentiles = [
                    (k, v) for k, v in result.items() 
                    if k.startswith("percentile_")
                ]
                if percentiles:
                    content.append("#### Percentiles")
                    content.append("")
                    content.append("| Percentile | Value |")
                    content.append("|------------|-------|")
                    
                    for k, v in sorted(percentiles):
                        p = k.split("_")[1]
                        content.append(f"| {p}% | {v:.4f} |")
                    content.append("")
            
            elif result.get("type") == "categorical":
                content.append("#### Statistics")
                content.append("")
                content.append("| Metric | Value |")
                content.append("|--------|-------|")
                content.append(f"| Count | {result['count']} |")
                content.append(f"| Missing | {result['missing']} ({result['missing_pct']:.2f}%) |")
                content.append(f"| Unique Values | {result.get('unique', len(result.get('value_counts', {}))-1 if 'Missing' in result.get('value_counts', {}) else len(result.get('value_counts', {})))} |")
                content.append(f"| Is Boolean | {'Yes' if result.get('is_boolean', False) else 'No'} |")
                
                if result.get("categories_limited", False):
                    content.append(f"| Note | Showing top categories only. Total categories: {result.get('total_categories', 'N/A')} |")
                
                content.append("")
                
                # Add value counts if available
                if "value_counts" in result:
                    content.append("#### Value Counts")
                    content.append("")
                    content.append("| Value | Count | Percentage |")
                    content.append("|-------|-------|------------|")
                    
                    for val, info in result["value_counts"].items():
                        content.append(f"| {val} | {info['count']} | {info['percentage']:.2f}% |")
                    content.append("")
            
            elif result.get("type") == "datetime":
                content.append("#### Statistics")
                content.append("")
                content.append("| Metric | Value |")
                content.append("|--------|-------|")
                content.append(f"| Count | {result['count']} |")
                content.append(f"| Missing | {result['missing']} ({result['missing_pct']:.2f}%) |")
                content.append(f"| Min Date | {result['min']} |")
                content.append(f"| Max Date | {result['max']} |")
                
                if "range_days" in result:
                    content.append(f"| Date Range | {result['range_days']} days |")
                
                content.append("")
                
                # Add components if available
                if "components" in result and result["components"]:
                    content.append("#### Date Components")
                    content.append("")
                    content.append("| Component | Unique Values | Range |")
                    content.append("|-----------|---------------|-------|")
                    
                    for comp, info in result["components"].items():
                        content.append(f"| {comp.capitalize()} | {info['unique']} | {info['min']} - {info['max']} |")
                    content.append("")
            
            # Add plot if available
            if "plot" in result:
                content.append("#### Distribution")
                content.append("")
                content.append(f"![Distribution of {col}]({result['plot']})")
                content.append("")
            
            return content
        
        # Add sections for each column type
        if numeric_cols:
            md_parts.append("### Numeric Columns")
            md_parts.append("")
            
            # Add details for each numeric column
            for col in numeric_cols:
                if col in univariate_results:
                    result = univariate_results[col]
                    md_parts.extend(generate_univariate_content(col, result))
        
        if categorical_cols:
            md_parts.append("### Categorical Columns")
            md_parts.append("")
            
            # Add details for each categorical column
            for col in categorical_cols:
                if col in univariate_results:
                    result = univariate_results[col]
                    md_parts.extend(generate_univariate_content(col, result))
        
        if datetime_cols:
            md_parts.append("### Datetime Columns")
            md_parts.append("")
            
            # Add details for each datetime column
            for col in datetime_cols:
                if col in univariate_results:
                    result = univariate_results[col]
                    md_parts.extend(generate_univariate_content(col, result))
    
    # Bivariate Analysis section
    if "bivariate" in analysis_results:
        md_parts.append("## Bivariate Analysis")
        md_parts.append("")
        md_parts.append("This section examines relationships between variables, including correlations "
                      "and feature-target relationships.")
        md_parts.append("")
        
        bivariate_results = analysis_results["bivariate"]
        
        # Add correlation matrix if available
        if "correlation" in bivariate_results:
            corr_result = bivariate_results["correlation"]
            
            md_parts.append("### Correlation Analysis")
            md_parts.append("")
            
            # Add correlation heatmap if available
            if "plot" in corr_result:
                md_parts.append("#### Correlation Heatmap")
                md_parts.append("")
                md_parts.append(f"![Correlation Heatmap]({corr_result['plot']})")
                md_parts.append("")
            
            # Add top correlations
            if "top_correlations" in corr_result and corr_result["top_correlations"]:
                md_parts.append("#### Top Correlations")
                md_parts.append("")
                md_parts.append("| Variable 1 | Variable 2 | Correlation |")
                md_parts.append("|-----------|-----------|-------------|")
                
                for corr in corr_result["top_correlations"][:10]:  # Show top 10
                    corr_val = corr["correlation"]
                    md_parts.append(f"| {corr['column1']} | {corr['column2']} | {corr_val:.4f} |")
                md_parts.append("")
        
        # Add feature-target analysis if available
        if "feature_target" in bivariate_results:
            feature_target = bivariate_results["feature_target"]
            
            # Helper function for feature-target analysis
            def generate_feature_target_content(feature, result):
                content = []
                content.append(f"### {feature} vs {result['target']}")
                content.append("")
                
                # Different content based on relationship type
                if result["type"] == "numeric_vs_numeric":
                    content.append("#### Correlation Statistics")
                    content.append("")
                    content.append("| Metric | Value |")
                    content.append("|--------|-------|")
                    content.append(f"| Pearson Correlation | {result['pearson_correlation']:.4f} |")
                    content.append(f"| Pearson p-value | {result['pearson_p_value']:.4f} |")
                    content.append(f"| Spearman Correlation | {result['spearman_correlation']:.4f} |")
                    content.append(f"| Spearman p-value | {result['spearman_p_value']:.4f} |")
                    content.append("")
                
                elif result["type"] in ["numeric_vs_categorical", "categorical_vs_numeric"]:
                    content.append("#### Group Statistics")
                    content.append("")
                    content.append("| Group | Mean | Std | Count |")
                    content.append("|-------|------|-----|-------|")
                    
                    if "grouped_stats" in result:
                        for group, stats in result["grouped_stats"].items():
                            content.append(f"| {group} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} |")
                    content.append("")
                    
                    if "anova_p_value" in result:
                        content.append(f"**ANOVA p-value:** {result['anova_p_value']:.4f}")
                        content.append("")
                        if result["anova_p_value"] < 0.05:
                            content.append("> There is a statistically significant difference between groups.")
                        else:
                            content.append("> There is no statistically significant difference between groups.")
                        content.append("")
                
                elif result["type"] == "categorical_vs_categorical":
                    content.append("#### Contingency Table")
                    content.append("")
                    
                    # Generate contingency table in markdown format
                    if "contingency_table" in result:
                        # Create header
                        header = "| |"
                        for col in result["contingency_table"].keys():
                            header += f" {col} |"
                        content.append(header)
                        
                        # Create separator
                        separator = "|---|"
                        for _ in range(len(result["contingency_table"].keys())):
                            separator += "---|"
                        content.append(separator)
                        
                        # Create rows
                        for row, row_data in result["contingency_table"].items():
                            row_str = f"| {row} |"
                            for col, val in row_data.items():
                                row_str += f" {val} |"
                            content.append(row_str)
                        content.append("")
                    
                    if "chi2_p_value" in result:
                        content.append(f"**Chi-square p-value:** {result['chi2_p_value']:.4f}")
                        content.append("")
                        if result["chi2_p_value"] < 0.05:
                            content.append("> There is a statistically significant association between these variables.")
                        else:
                            content.append("> There is no statistically significant association between these variables.")
                        content.append("")
                
                # Add plot if available
                if "plot" in result:
                    content.append("#### Relationship Visualization")
                    content.append("")
                    content.append(f"![{feature} vs {result['target']}]({result['plot']})")
                    content.append("")
                
                return content
            
            md_parts.append("### Feature-Target Relationships")
            md_parts.append("")
            
            # Add content for each feature-target relationship
            for feature, result in feature_target.items():
                if "type" not in result:
                    continue
                
                md_parts.extend(generate_feature_target_content(feature, result))
    
    # Feature Importance section
    if "feature_importance" in analysis_results:
        md_parts.append("## Feature Importance Analysis")
        md_parts.append("")
        md_parts.append("This section shows the importance of features in predicting target variables, "
                      "calculated using machine learning techniques.")
        md_parts.append("")
        
        importance_results = analysis_results["feature_importance"]
        
        for target, result in importance_results.items():
            if "error" in result:
                continue
            
            md_parts.append(f"### Feature Importance for {target}")
            md_parts.append(f"**Method:** {result['method'].replace('_', ' ').title()}")
            md_parts.append("")
            
            # Add importance plot if available
            if "plot" in result:
                md_parts.append("#### Importance Visualization")
                md_parts.append("")
                md_parts.append(f"![Feature Importance for {target}]({result['plot']})")
                md_parts.append("")
            
            # Add importance values table
            md_parts.append("#### Importance Values")
            md_parts.append("")
            md_parts.append("| Feature | Importance | Relative (%) |")
            md_parts.append("|---------|------------|--------------|")
            
            # Add each feature with its importance
            if "sorted_importances" in result:
                # Calculate the maximum importance value for relative importance
                max_importance = max(result["sorted_importances"].values())
                
                for feature, importance in result["sorted_importances"].items():
                    relative = (importance / max_importance) * 100 if max_importance > 0 else 0
                    md_parts.append(f"| {feature} | {importance:.4f} | {relative:.1f}% |")
            md_parts.append("")
    
    # Multivariate Analysis section
    if "multivariate" in analysis_results:
        md_parts.append("## Multivariate Analysis")
        md_parts.append("")
        md_parts.append("This section explores relationships between multiple variables using dimensionality "
                      "reduction techniques like PCA and t-SNE.")
        md_parts.append("")
        
        multivariate_results = analysis_results["multivariate"]
        
        # PCA analysis
        if "pca" in multivariate_results:
            pca_result = multivariate_results["pca"]
            
            md_parts.append("### Principal Component Analysis (PCA)")
            md_parts.append("")
            
            # PCA visualization
            if "visualization" in pca_result:
                pca_viz = pca_result["visualization"]
                # If it's base64 without the data prefix, add it
                if not pca_viz.startswith("data:"):
                    pca_viz = f"data:image/png;base64,{pca_viz}"
                
                md_parts.append("#### PCA Visualization")
                md_parts.append("")
                md_parts.append(f"![PCA Visualization]({pca_viz})")
                md_parts.append("")
            
            # PCA metrics
            md_parts.append("#### PCA Summary")
            md_parts.append("")
            md_parts.append("| Metric | Value |")
            md_parts.append("|--------|-------|")
            md_parts.append(f"| Number of Components | {pca_result.get('n_components', 'N/A')} |")
            md_parts.append(f"| Total Explained Variance | {sum(pca_result.get('explained_variance', [0])) * 100:.2f}% |")
            
            # Add the first few components' variance
            explained_variance = pca_result.get("explained_variance", [])
            for i, var in enumerate(explained_variance[:3]):  # Show first 3 components
                md_parts.append(f"| PC{i+1} Explained Variance | {var * 100:.2f}% |")
            md_parts.append("")
            
            # Feature contributions
            if "feature_contributions" in pca_result:
                md_parts.append("#### Top Feature Contributions to PC1")
                md_parts.append("")
                md_parts.append("| Feature | Contribution (%) |")
                md_parts.append("|---------|------------------|")
                
                # Get feature contributions for PC1
                contributions = {}
                for feature, comps in pca_result["feature_contributions"].items():
                    if "PC1" in comps:
                        contributions[feature] = comps["PC1"]
                
                # Sort by contribution and display top 5
                for feature, contrib in sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    md_parts.append(f"| {feature} | {contrib * 100:.2f}% |")
                md_parts.append("")
            
            # Loadings visualization
            if "loadings_visualization" in pca_result and pca_result["loadings_visualization"]:
                loadings_viz = pca_result["loadings_visualization"]
                # If it's base64 without the data prefix, add it
                if not loadings_viz.startswith("data:"):
                    loadings_viz = f"data:image/png;base64,{loadings_viz}"
                
                md_parts.append("#### Feature Loadings Heatmap")
                md_parts.append("")
                md_parts.append(f"![PCA Loadings Heatmap]({loadings_viz})")
                md_parts.append("")
        
        # t-SNE analysis
        if "tsne" in multivariate_results:
            tsne_result = multivariate_results["tsne"]
            
            md_parts.append("### t-SNE Analysis")
            md_parts.append("")
            
            # t-SNE visualization
            if "visualization" in tsne_result:
                tsne_viz = tsne_result["visualization"]
                # If it's base64 without the data prefix, add it
                if not tsne_viz.startswith("data:"):
                    tsne_viz = f"data:image/png;base64,{tsne_viz}"
                
                md_parts.append("#### t-SNE Visualization")
                md_parts.append("")
                md_parts.append(f"![t-SNE Visualization]({tsne_viz})")
                md_parts.append("")
            
            # t-SNE parameters
            md_parts.append("#### t-SNE Parameters")
            md_parts.append("")
            md_parts.append("| Parameter | Value |")
            md_parts.append("|-----------|-------|")
            md_parts.append(f"| Number of Components | {tsne_result.get('n_components', 'N/A')} |")
            md_parts.append(f"| Perplexity | {tsne_result.get('perplexity', 'N/A')} |")
            md_parts.append(f"| Learning Rate | {tsne_result.get('learning_rate', 'N/A')} |")
            md_parts.append(f"| Iterations | {tsne_result.get('n_iter', 'N/A')} |")
            md_parts.append("")
            
            md_parts.append("> **About t-SNE**: t-SNE is a nonlinear dimensionality reduction technique well-suited "
                          "for visualizing high-dimensional data. Unlike PCA, t-SNE focuses on preserving local structure "
                          "and revealing clusters. Note that t-SNE should be used primarily for visualization, not for "
                          "general dimensionality reduction or as input features for other algorithms.")
            md_parts.append("")
    
    # Time Series Analysis section
    if "time_series" in analysis_results:
        md_parts.append("## Time Series Analysis")
        md_parts.append("")
        md_parts.append("This section analyzes temporal patterns, trends, and seasonality in the data.")
        md_parts.append("")
        
        time_series = analysis_results["time_series"]
        
        # Add content for each time series
        for feature, result in time_series.items():
            if "error" in result:
                continue
            
            md_parts.append(f"### {feature}")
            md_parts.append("")
            
            md_parts.append("#### Time Series Statistics")
            md_parts.append("")
            md_parts.append("| Metric | Value |")
            md_parts.append("|--------|-------|")
            md_parts.append(f"| Start Date | {result.get('start_date', 'N/A')} |")
            md_parts.append(f"| End Date | {result.get('end_date', 'N/A')} |")
            md_parts.append(f"| Duration | {result.get('duration_days', 'N/A')} days |")
            md_parts.append(f"| Mean | {result.get('mean', 'N/A'):.4f} |")
            md_parts.append(f"| Std. Dev. | {result.get('std', 'N/A'):.4f} |")
            md_parts.append(f"| Min | {result.get('min', 'N/A'):.4f} |")
            md_parts.append(f"| Max | {result.get('max', 'N/A'):.4f} |")
            
            if "trend" in result:
                md_parts.append(f"| Trend | {result['trend']} |")
                md_parts.append(f"| Absolute Change | {result.get('absolute_change', 'N/A'):.4f} |")
                md_parts.append(f"| Percent Change | {result.get('percent_change') if result.get('percent_change') is not None else 'N/A'} |")
            md_parts.append("")
            
            # Add plot if available
            if "plot" in result:
                md_parts.append("#### Time Series Plot")
                md_parts.append("")
                md_parts.append(f"![Time Series Plot of {feature}]({result['plot']})")
                md_parts.append("")
            
            # Add seasonality section if available
            if "seasonality" in result and not isinstance(result["seasonality"], str):
                seasonality = result["seasonality"]
                if "error" not in seasonality:
                    md_parts.append("#### Seasonality Analysis")
                    md_parts.append("")
                    
                    if "plot" in seasonality:
                        md_parts.append("![Seasonality Decomposition]({seasonality['plot']})")
                        md_parts.append("")
                    
                    md_parts.append("| Metric | Value |")
                    md_parts.append("|--------|-------|")
                    md_parts.append(f"| Decomposition Model | {seasonality.get('model', 'N/A')} |")
                    md_parts.append(f"| Period | {seasonality.get('period', 'N/A')} |")
                    md_parts.append(f"| Seasonal Strength | {seasonality.get('seasonal', {}).get('strength', 'N/A'):.4f} |")
                    md_parts.append("")
    
    # Join all Markdown parts
    markdown = "\n".join(md_parts)
    
    # Write the Markdown to a file
    md_output_path = output_path
    if not md_output_path.endswith(".md"):
        if md_output_path.endswith(".html"):
            md_output_path = md_output_path.replace(".html", ".md")
        else:
            md_output_path += ".md"
    
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"Markdown report saved to {md_output_path}")
    
    # Convert to HTML if requested
    if convert_to_html:
        html_output_path = output_path
        if not html_output_path.endswith(".html"):
            html_output_path += ".html"
        
        convert_markdown_to_html(markdown, html_output_path, title, include_export_button, df)
        print(f"HTML report saved to {html_output_path}")
    
    # Return the Markdown string
    return markdown


def _dataframe_to_markdown(df, include_header=True):
    """Convert a DataFrame to markdown table format."""
    md_lines = []
    
    # Add header row if requested
    if include_header:
        header = "| |"
        for col in df.columns:
            header += f" {col} |"
        md_lines.append(header)
        
        # Add separator row
        separator = "|---|"
        for _ in range(len(df.columns)):
            separator += "---|"
        md_lines.append(separator)
    
    # Add data rows
    for idx, row in df.iterrows():
        row_str = f"| {idx} |"
        for val in row:
            # Format different data types appropriately
            if pd.isna(val):
                cell = " NaN |"
            elif isinstance(val, (int, np.integer)):
                cell = f" {val} |"
            elif isinstance(val, (float, np.floating)):
                cell = f" {val:.4f} |"
            else:
                cell = f" {val} |"
            row_str += cell
        md_lines.append(row_str)
    
    md_lines.append("")  # Add empty line after table
    return md_lines


def convert_markdown_to_html(markdown_text, output_path, title="Exploratory Data Analysis Report", include_export_button=True, df=None):
    """
    Convert a Markdown report to HTML.
    
    Parameters
    ----------
    markdown_text : str
        The markdown report as a string.
    output_path : str
        The path to save the HTML report.
    title : str, default="Exploratory Data Analysis Report"
        The title of the report.
    include_export_button : bool, default=True
        Whether to include a button to export the report as a Jupyter notebook.
    df : pd.DataFrame, optional
        The dataframe that was analyzed, used for export button functionality.
    """
    try:
        import markdown as markdown_module
        from markdown.extensions.tables import TableExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
    except ImportError:
        print("Markdown Python package not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "markdown"])
        import markdown as markdown_module
        from markdown.extensions.tables import TableExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
    
    # Convert markdown to HTML
    html_body = markdown_module.markdown(markdown_text, extensions=[
        TableExtension(),
        FencedCodeExtension(),
        'nl2br',  # Convert new lines to <br>
        'sane_lists',  # Better handling of lists
    ])
    
    # Create HTML document with proper styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title}</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootswatch@5.2.3/dist/cosmo/bootstrap.min.css">
        <style>
            body {{ padding-top: 20px; padding-bottom: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .card {{ margin-bottom: 20px; }}
            .table-responsive {{ margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; }}
            
            /* Markdown-specific styling */
            h1, h2, h3, h4, h5, h6 {{ margin-top: 1em; margin-bottom: 0.5em; }}
            h1 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3em; }}
            h2 {{ border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
            blockquote {{ padding: 0.5em 1em; border-left: 5px solid #eee; color: #777; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
            table, th, td {{ border: 1px solid #ddd; }}
            th, td {{ padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            th {{ background-color: #f9f9f9; color: #333; }}
            code {{ padding: 0.2em 0.4em; background: #f6f8fa; border-radius: 3px; }}
            pre {{ padding: 1em; overflow: auto; background: #f6f8fa; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_body}
        </div>
    """
    
    # Add export to Jupyter button if enabled
    if include_export_button and df is not None:
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
                let parentHeading = img.previousElementSibling;
                while(parentHeading && !['H1','H2','H3','H4'].includes(parentHeading.tagName)) {{
                    parentHeading = parentHeading.previousElementSibling;
                }}
                
                if (parentHeading) {{                    
                    sectionTitle = parentHeading.textContent.trim();                                
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
        
        # Insert export button
        html = html.replace('<div class="container">', f'<div class="container">\n{export_button}')
    
    # Finish HTML document
    html += """
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return html