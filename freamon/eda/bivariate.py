"""
Bivariate analysis functions for EDA.
"""
from typing import Any, Dict, List, Optional, Union, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from scipy import stats

# Configure matplotlib to avoid treating $ as LaTeX math delimiters
plt.rcParams['text.usetex'] = False


def analyze_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson',
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Analyze correlations between numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    columns : Optional[List[str]], default=None
        The columns to include in the analysis. If None, all numeric columns are used.
    method : str, default='pearson'
        The correlation method to use. Options: 'pearson', 'spearman', 'kendall'.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with correlation analysis results.
    """
    # Select columns to analyze
    if columns is None:
        # Use all numeric columns
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_cols = df.select_dtypes(include=numeric_dtypes).columns.tolist()
    else:
        # Verify all columns are in the dataframe and numeric
        numeric_cols = []
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Column '{col}' is not numeric, skipping")
            else:
                numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        return {"error": "Insufficient numeric columns for correlation analysis"}
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Get top correlations
    corr_pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # only use upper triangle to avoid duplicates
                corr_value = corr_matrix.loc[col1, col2]
                corr_pairs.append((col1, col2, corr_value))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Create dictionary for top correlations
    top_correlations = []
    for col1, col2, corr_value in corr_pairs:
        top_correlations.append({
            "column1": col1,
            "column2": col2,
            "correlation": float(corr_value),
            "abs_correlation": float(abs(corr_value)),
        })
    
    # Prepare result
    result = {
        "method": method,
        "correlation_matrix": corr_matrix.to_dict(),
        "top_correlations": top_correlations
    }
    
    # Create plot if requested
    if include_plot:
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            annot=True, 
            fmt=".2f", 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5}
        )
        
        plt.title(f'{method.capitalize()} Correlation Matrix')
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        result["plot"] = f"data:image/png;base64,{img_str}"
    
    return result


def analyze_feature_target(
    df: pd.DataFrame,
    feature: str,
    target: str,
    include_plot: bool = True,
    include_importance: bool = True,
) -> Dict[str, Any]:
    """
    Analyze the relationship between a feature and a target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    feature : str
        The name of the feature column.
    target : str
        The name of the target column.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    include_importance: bool, default=True
        Whether to include feature importance metrics like chi-squared.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with feature-target analysis results.
    """
    # Validate inputs
    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in dataframe")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    # Initialize result
    result = {
        "feature": feature,
        "target": target,
    }
    
    # Get data types
    feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
    
    # Remove rows with missing values
    df_clean = df[[feature, target]].dropna()
    
    if len(df_clean) == 0:
        return {"error": "No data available after removing missing values"}
    
    # Different analysis based on variable types
    if feature_is_numeric and target_is_numeric:
        # Numeric vs Numeric: correlation and scatter plot
        result["type"] = "numeric_vs_numeric"
        
        # Calculate correlation
        pearson_corr, pearson_p = stats.pearsonr(df_clean[feature], df_clean[target])
        spearman_corr, spearman_p = stats.spearmanr(df_clean[feature], df_clean[target])
        
        result["pearson_correlation"] = float(pearson_corr)
        result["pearson_p_value"] = float(pearson_p)
        result["spearman_correlation"] = float(spearman_corr)
        result["spearman_p_value"] = float(spearman_p)
        
        # Calculate importance metrics
        if include_importance:
            # For numeric->numeric, we can use R-squared
            from sklearn.linear_model import LinearRegression
            
            # Reshape for sklearn
            X = df_clean[feature].values.reshape(-1, 1)
            y = df_clean[target].values
            
            # Fit linear model and get R-squared
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            
            result["importance_metric"] = "r_squared" 
            result["importance_value"] = float(r2)
            result["importance_description"] = "R-squared (coefficient of determination) measures the proportion of variance in the target variable that can be explained by the feature."
        
        # Create plot if requested
        if include_plot:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot with regression line
            sns.regplot(x=feature, y=target, data=df_clean, scatter_kws={"alpha": 0.5})
            
            plt.title(f'{feature} vs {target}')
            plt.xlabel(feature)
            plt.ylabel(target)
            
            # Add correlation annotation
            plt.annotate(
                f"Pearson Corr: {pearson_corr:.3f} (p={pearson_p:.3f})\n"
                f"Spearman Corr: {spearman_corr:.3f} (p={spearman_p:.3f})",
                xy=(0.05, 0.95), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                verticalalignment='top'
            )
            
            # Save plot to BytesIO
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            result["plot"] = f"data:image/png;base64,{img_str}"
    
    elif feature_is_numeric and not target_is_numeric:
        # Numeric vs Categorical: Group by target and calculate statistics
        result["type"] = "numeric_vs_categorical"
        
        # Calculate statistics by target value
        grouped_stats = df_clean.groupby(target)[feature].agg(['mean', 'std', 'count'])
        grouped_stats = grouped_stats.reset_index()
        
        # Convert to dictionary
        result["grouped_stats"] = {}
        for _, row in grouped_stats.iterrows():
            target_val = str(row[target])
            result["grouped_stats"][target_val] = {
                "mean": float(row['mean']),
                "std": float(row['std']),
                "count": int(row['count']),
            }
        
        # Check if we can run ANOVA (at least 2 groups with data)
        if len(grouped_stats) >= 2:
            # Create groups for ANOVA
            groups = []
            for val in df_clean[target].unique():
                group = df_clean[df_clean[target] == val][feature]
                if len(group) > 0:
                    groups.append(group)
            
            if len(groups) >= 2:
                # Run ANOVA
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    result["anova_f_statistic"] = float(f_stat)
                    result["anova_p_value"] = float(p_value)
                    
                    # Add importance metric
                    if include_importance:
                        result["importance_metric"] = "f_statistic"
                        result["importance_value"] = float(f_stat)
                        result["importance_description"] = "F-statistic measures how much the feature discriminates between different target categories. Higher values indicate stronger discrimination."
                except:
                    # ANOVA might fail for various reasons
                    pass
        
        # Create plot if requested
        if include_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Box plot
            sns.boxplot(x=target, y=feature, data=df_clean, ax=ax1)
            ax1.set_title(f'Box Plot of {feature} by {target}')
            ax1.set_xlabel(target)
            ax1.set_ylabel(feature)
            
            # Strip plot for additional detail
            sns.stripplot(x=target, y=feature, data=df_clean, alpha=0.3, jitter=True, ax=ax2)
            ax2.set_title(f'Strip Plot of {feature} by {target}')
            ax2.set_xlabel(target)
            ax2.set_ylabel(feature)
            
            # Add ANOVA result if available
            if "anova_p_value" in result:
                ax1.annotate(
                    f"ANOVA p-value: {result['anova_p_value']:.3f}\nF-statistic: {result['anova_f_statistic']:.3f}",
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    verticalalignment='top'
                )
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            result["plot"] = f"data:image/png;base64,{img_str}"
    
    elif not feature_is_numeric and target_is_numeric:
        # Categorical vs Numeric: Group by feature and calculate statistics
        result["type"] = "categorical_vs_numeric"
        
        # Calculate statistics by feature value
        # Set observed=True to address FutureWarning about observed parameter
        grouped_stats = df_clean.groupby(feature, observed=True)[target].agg(['mean', 'std', 'count'])
        grouped_stats = grouped_stats.reset_index()
        
        # Convert to dictionary
        result["grouped_stats"] = {}
        for _, row in grouped_stats.iterrows():
            feature_val = str(row[feature])
            result["grouped_stats"][feature_val] = {
                "mean": float(row['mean']),
                "std": float(row['std']),
                "count": int(row['count']),
            }
        
        # Check if we can run ANOVA (at least 2 groups with data)
        if len(grouped_stats) >= 2:
            # Create groups for ANOVA
            groups = []
            for val in df_clean[feature].unique():
                group = df_clean[df_clean[feature] == val][target]
                if len(group) > 0:
                    groups.append(group)
            
            if len(groups) >= 2:
                # Run ANOVA
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    result["anova_f_statistic"] = float(f_stat)
                    result["anova_p_value"] = float(p_value)
                    
                    # Add importance metric
                    if include_importance:
                        # Calculate eta squared (effect size)
                        sum_squares_between = sum(len(g) * ((g.mean() - df_clean[target].mean())**2) for g in groups)
                        sum_squares_total = sum([(val - df_clean[target].mean())**2 for val in df_clean[target]])
                        eta_squared = sum_squares_between / sum_squares_total if sum_squares_total > 0 else 0
                        
                        result["importance_metric"] = "eta_squared"
                        result["importance_value"] = float(eta_squared)
                        result["importance_description"] = "Eta-squared measures the proportion of variance in the target variable that can be attributed to the feature (effect size)."
                except:
                    # ANOVA might fail for various reasons
                    pass
        
        # Create plot if requested
        if include_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Box plot
            sns.boxplot(x=feature, y=target, data=df_clean, ax=ax1)
            ax1.set_title(f'Box Plot of {target} by {feature}')
            ax1.set_xlabel(feature)
            ax1.set_ylabel(target)
            
            # Bar plot with error bars
            sns.barplot(x=feature, y=target, data=df_clean, ax=ax2)
            ax2.set_title(f'Mean {target} by {feature}')
            ax2.set_xlabel(feature)
            ax2.set_ylabel(f'Mean {target}')
            
            # Add ANOVA result if available
            if "anova_p_value" in result:
                ax1.annotate(
                    f"ANOVA p-value: {result['anova_p_value']:.3f}\nF-statistic: {result['anova_f_statistic']:.3f}",
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    verticalalignment='top'
                )
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            result["plot"] = f"data:image/png;base64,{img_str}"
    
    else:
        # Categorical vs Categorical: Contingency table and chi-square test
        result["type"] = "categorical_vs_categorical"
        
        # Create contingency table
        contingency = pd.crosstab(df_clean[feature], df_clean[target])
        result["contingency_table"] = contingency.to_dict()
        
        # Calculate percentages
        row_percentages = contingency.div(contingency.sum(axis=1), axis=0).round(3)
        col_percentages = contingency.div(contingency.sum(axis=0), axis=1).round(3)
        
        result["row_percentages"] = row_percentages.to_dict()
        result["col_percentages"] = col_percentages.to_dict()
        
        # Run chi-square test if we have enough data
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            result["chi2_statistic"] = float(chi2)
            result["chi2_p_value"] = float(p)
            result["chi2_dof"] = int(dof)
            
            # Add importance metric
            if include_importance:
                # Calculate Cramer's V (normalized chi-square)
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                cramers_v = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0
                
                result["importance_metric"] = "cramers_v"
                result["importance_value"] = float(cramers_v)
                result["importance_description"] = "Cramer's V is a measure of association between two categorical variables, ranging from 0 (no association) to 1 (perfect association)."
        
        # Create plot if requested
        if include_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Heatmap of counts
            sns.heatmap(contingency, annot=True, fmt="d", cmap="YlGnBu", ax=ax1)
            ax1.set_title(f'Contingency Table: {feature} vs {target}')
            
            # Heatmap of percentages
            sns.heatmap(row_percentages, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax2)
            ax2.set_title(f'Row Percentages: {feature} vs {target}')
            
            # Add chi-square result if available
            if "chi2_p_value" in result:
                ax1.annotate(
                    f"Chi-square p-value: {result['chi2_p_value']:.3f}\nChi-square: {result['chi2_statistic']:.3f}",
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    verticalalignment='top'
                )
                
                if "importance_value" in result:
                    ax2.annotate(
                        f"Cramer's V: {result['importance_value']:.3f}",
                        xy=(0.05, 0.95), 
                        xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        verticalalignment='top'
                    )
            
            # Save plot to BytesIO
            buf = BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            result["plot"] = f"data:image/png;base64,{img_str}"
    
    return result


def calculate_feature_importance(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    method: str = 'random_forest',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    include_plot: bool = True,
) -> Dict[str, Any]:
    """
    Calculate feature importance for a target variable using machine learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    features : List[str]
        List of feature columns to use.
    target : str
        The name of the target column.
    method : str, default='random_forest'
        Method to use for importance calculation. Options: 'random_forest', 'permutation', 'shap'.
    n_estimators : int, default=100
        Number of estimators for tree-based methods.
    max_depth : Optional[int], default=None
        Maximum depth for tree-based methods.
    include_plot : bool, default=True
        Whether to include plot images in the results.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with feature importance results.
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    # Validate inputs
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    # Check which features are actually in the dataframe and numeric
    valid_features = []
    for feature in features:
        if feature not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feature]):
            continue
        valid_features.append(feature)
    
    if not valid_features:
        return {"error": "No valid numeric features found"}
    
    # Remove rows with missing values
    df_clean = df[valid_features + [target]].dropna()
    
    if len(df_clean) == 0:
        return {"error": "No data available after removing missing values"}
    
    # Prepare X and y
    X = df_clean[valid_features]
    y = df_clean[target]
    
    # Check target type (classification or regression)
    is_classification = False
    if not pd.api.types.is_numeric_dtype(y):
        is_classification = True
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    elif y.nunique() < 10:  # Small number of unique values might be a classification problem
        if set(y.unique()).issubset({0, 1}) or len(y.unique()) / len(y) < 0.05:
            is_classification = True
    
    # Initialize model based on problem type
    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    # Train model
    model.fit(X, y)
    
    # Calculate importance based on method
    importance_values = {}
    if method == 'random_forest':
        importance = model.feature_importances_
        for i, feature in enumerate(valid_features):
            importance_values[feature] = float(importance[i])
    elif method == 'permutation':
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        for i, feature in enumerate(valid_features):
            importance_values[feature] = float(perm_importance.importances_mean[i])
    elif method == 'shap':
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For classification with multiple classes, take the mean absolute SHAP value across classes
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            # Calculate mean absolute SHAP value for each feature
            feature_importance = np.abs(shap_values).mean(axis=0)
            for i, feature in enumerate(valid_features):
                importance_values[feature] = float(feature_importance[i])
        except ImportError:
            # Fall back to random forest if SHAP is not available
            importance = model.feature_importances_
            for i, feature in enumerate(valid_features):
                importance_values[feature] = float(importance[i])
    
    # Sort features by importance
    sorted_importance = sorted(importance_values.items(), key=lambda x: x[1], reverse=True)
    
    # Create result dictionary
    result = {
        "method": method,
        "target": target,
        "is_classification": is_classification,
        "importances": importance_values,
        "sorted_importances": {k: v for k, v in sorted_importance}
    }
    
    # Create plot if requested
    if include_plot:
        plt.figure(figsize=(10, 6))
        features = [x[0] for x in sorted_importance]
        importances = [x[1] for x in sorted_importance]
        
        plt.barh(features, importances)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for {target} ({method.replace("_", " ").title()})')
        plt.tight_layout()
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        result["plot"] = f"data:image/png;base64,{img_str}"
    
    return result