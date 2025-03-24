"""
Utility functions and configurations to fix common matplotlib issues.
"""
import logging
import functools
import warnings
import re

def configure_matplotlib_for_currency():
    """
    Configure matplotlib to properly handle dollar signs and other currency symbols.
    
    This function sets matplotlib parameters to prevent LaTeX parsing of dollar signs
    and other special characters that might be interpreted as math mode delimiters.
    
    It should be called before any plotting code that might include currency values.
    
    Returns
    -------
    bool
        True if configuration was successful, False otherwise
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        # Disable LaTeX interpretation
        plt.rcParams['text.usetex'] = False
        
        # Use regular font instead of math font for dollar signs
        plt.rcParams['mathtext.default'] = 'regular'
        
        # Apply comprehensive patches for common issues
        return apply_comprehensive_matplotlib_patches()
    except ImportError:
        logging.warning("Failed to configure matplotlib: matplotlib not installed")
        return False
    except Exception as e:
        logging.warning(f"Failed to configure matplotlib: {str(e)}")
        return False

def apply_comprehensive_matplotlib_patches():
    """
    Apply comprehensive patches to matplotlib to handle problematic text rendering.
    
    This function patches several matplotlib functions to safely handle special characters
    that might cause rendering errors, including dollar signs, underscores, and other
    characters that might be interpreted as math delimiters.
    
    Returns
    -------
    bool
        True if patching was successful, False otherwise
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        # 1. Patch text rendering function in Agg backend
        if hasattr(mpl.backends, 'backend_agg') and hasattr(mpl.backends.backend_agg, 'RendererAgg'):
            original_text_render = mpl.backends.backend_agg.RendererAgg.get_text_width_height_descent
            
            @functools.wraps(original_text_render)
            def patched_text_render(self, s, prop, ismath):
                try:
                    return original_text_render(self, s, prop, ismath)
                except ValueError as e:
                    # On error, replace problematic characters and try again
                    if "Bad character" in str(e) or "$" in s or "_" in s:
                        safe_s = preprocess_text_for_matplotlib(s)
                        try:
                            return original_text_render(self, safe_s, prop, ismath)
                        except Exception:
                            # Last resort: use a completely safe string
                            return original_text_render(self, "placeholder", prop, ismath)
                    else:
                        raise
            
            # Apply the patch
            mpl.backends.backend_agg.RendererAgg.get_text_width_height_descent = patched_text_render
        
        # 2. Patch mathtext parser
        if hasattr(mpl, 'mathtext') and hasattr(mpl.mathtext, 'MathTextParser'):
            if hasattr(mpl.mathtext.MathTextParser, '_parse_cached'):
                original_mathtext_parse = mpl.mathtext.MathTextParser._parse_cached
                
                @functools.wraps(original_mathtext_parse)
                def patched_mathtext_parse(self, s, dpi, prop, antialiased, load_glyph_flags):
                    try:
                        return original_mathtext_parse(self, s, dpi, prop, antialiased, load_glyph_flags)
                    except ValueError as e:
                        if "ParseException" in str(e) or "Bad character" in str(e):
                            # If we get a parsing error, return a dummy result with minimal dimensions
                            warnings.warn(f"MathText parsing failed: {str(e)}. Using placeholder text.")
                            safe_s = "placeholder"
                            return original_mathtext_parse(self, safe_s, dpi, prop, antialiased, load_glyph_flags)
                        else:
                            raise
                
                # Apply the patch
                mpl.mathtext.MathTextParser._parse_cached = patched_mathtext_parse
                
        # 3. Override default text and axes title methods
        # Monkey patch the base text class to preprocess strings
        original_set_text = mpl.text.Text.set_text
        
        @functools.wraps(original_set_text)
        def patched_set_text(self, s):
            if s is not None:
                s = preprocess_text_for_matplotlib(s)
            return original_set_text(self, s)
        
        # Apply the patch
        mpl.text.Text.set_text = patched_set_text
        
        # 4. Override seaborn title and label methods if available
        try:
            import seaborn as sns
            
            # Patch seaborn plot labels
            original_despine = sns.despine
            
            @functools.wraps(original_despine)
            def patched_despine(*args, **kwargs):
                result = original_despine(*args, **kwargs)
                
                # After despine is called, fix any remaining axis labels that might have been set
                if 'ax' in kwargs and kwargs['ax'] is not None:
                    ax = kwargs['ax']
                    if hasattr(ax, 'get_title') and ax.get_title():
                        ax.set_title(preprocess_text_for_matplotlib(ax.get_title()))
                    if hasattr(ax, 'get_xlabel') and ax.get_xlabel():
                        ax.set_xlabel(preprocess_text_for_matplotlib(ax.get_xlabel()))
                    if hasattr(ax, 'get_ylabel') and ax.get_ylabel():
                        ax.set_ylabel(preprocess_text_for_matplotlib(ax.get_ylabel()))
                
                return result
            
            # Apply the patch
            sns.despine = patched_despine
        except ImportError:
            # Seaborn not available, skip this patch
            pass
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to apply comprehensive matplotlib patches: {str(e)}")
        return False

def preprocess_text_for_matplotlib(text):
    """
    Preprocess text to avoid matplotlib rendering issues.
    
    Parameters
    ----------
    text : str or any
        The text to preprocess
        
    Returns
    -------
    str
        Preprocessed text safe for matplotlib rendering
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace dollar signs with [DOLLAR]
    text = text.replace('$', '[DOLLAR]')
    
    # Replace underscores with [UNDERSCORE] to prevent math subscript interpretation
    text = text.replace('_', '[UNDERSCORE]')
    
    # Replace other troublesome characters
    text = text.replace('^', '[CARET]')  # Superscript
    text = text.replace('\\', '[BACKSLASH]')  # Escape character
    text = text.replace('{', '[LBRACE]')  # LaTeX grouping
    text = text.replace('}', '[RBRACE]')  # LaTeX grouping
    
    # Replace percentage signs with [PERCENT] to prevent formatting issues
    text = text.replace('%', '[PERCENT]')
    
    return text

def replace_dollar_signs(text):
    """
    Replace dollar signs with [DOLLAR] to prevent matplotlib LaTeX parsing issues.
    
    Parameters
    ----------
    text : str
        The text containing dollar signs to process
        
    Returns
    -------
    str
        Text with dollar signs replaced with [DOLLAR]
    """
    if not isinstance(text, str):
        return text
    return text.replace('$', '[DOLLAR]')

def safe_process_dataframe(df):
    """
    Process a DataFrame to safely handle dollar signs in string columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain currency values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dollar signs replaced for safe display
    """
    import pandas as pd
    processed_df = df.copy()
    
    for col in processed_df.columns:
        if pd.api.types.is_string_dtype(processed_df[col]):
            processed_df[col] = processed_df[col].apply(preprocess_text_for_matplotlib)
        elif pd.api.types.is_object_dtype(processed_df[col]):
            processed_df[col] = processed_df[col].astype(str).apply(
                lambda x: preprocess_text_for_matplotlib(x) if isinstance(x, str) else x
            )
    
    return processed_df

def patch_freamon_eda():
    """
    Apply comprehensive patches to the freamon EDA module to prevent crashes
    and improve error handling.
    
    Returns
    -------
    bool
        True if patching was successful, False otherwise
    """
    try:
        import matplotlib as mpl
        
        # Apply matplotlib fixes first
        configure_matplotlib_for_currency()
        
        # 5. Patch the EDAAnalyzer.run_full_analysis method to catch and continue on errors
        from freamon.eda.analyzer import EDAAnalyzer
        original_run_full_analysis = EDAAnalyzer.run_full_analysis
        
        @functools.wraps(original_run_full_analysis)
        def patched_run_full_analysis(self, output_path="eda_output", title=None, include_multivariate=True, 
                                     include_feature_importance=True, sample_size=None, 
                                     use_sampling=False, cache_results=True, show_progress=False):
            """Patched version to catch and handle specific errors."""
            try:
                # Run basic stats - should be safe
                self.analyze_basic_stats()
                
                # Run univariate analysis with error handling
                try:
                    self.analyze_univariate(
                        sample_size=sample_size,
                        use_sampling=use_sampling
                    )
                except Exception as e:
                    print(f"Warning: Univariate analysis encountered errors: {str(e)}")
                
                # Run bivariate analysis with error handling
                try:
                    self.analyze_bivariate(
                        sample_size=sample_size,
                        use_sampling=use_sampling
                    )
                except Exception as e:
                    print(f"Warning: Bivariate analysis encountered errors: {str(e)}")
                
                # Run time series analysis if datetime columns are present
                if self.date_column is not None or len(self.datetime_columns) > 0:
                    try:
                        self.analyze_time_series()
                    except Exception as e:
                        print(f"Warning: Time series analysis encountered errors: {str(e)}")
                
                # Run multivariate analysis if requested
                if include_multivariate and len(self.numeric_columns) >= 2:
                    try:
                        self.analyze_multivariate(
                            sample_size=sample_size,
                            use_sampling=use_sampling,
                            cache_results=cache_results
                        )
                    except Exception as e:
                        print(f"Warning: Multivariate analysis encountered errors: {str(e)}")
                
                # Run feature importance analysis if requested and target column is set
                if include_feature_importance and self.target_column is not None:
                    try:
                        self.analyze_feature_importance(
                            target=self.target_column,
                            sample_size=sample_size,
                            use_sampling=use_sampling
                        )
                    except Exception as e:
                        print(f"Warning: Feature importance analysis encountered errors: {str(e)}")
                
                # Generate report
                if output_path is not None:
                    try:
                        self.generate_report(output_path=output_path, title=title)
                    except Exception as e:
                        print(f"Warning: Report generation encountered errors: {str(e)}")
                        
                return True
            except Exception as e:
                print(f"Error in EDA analysis: {str(e)}")
                return False
        
        # Apply the patch
        EDAAnalyzer.run_full_analysis = patched_run_full_analysis
        
        # 6. Patch the HTML report generation to catch and handle errors
        from freamon.eda.report import generate_html_report
        original_generate_html_report = generate_html_report
        
        @functools.wraps(original_generate_html_report)
        def patched_generate_html_report(df, analysis_results, output_path, title="Exploratory Data Analysis Report", theme="cosmo"):
            """Patched version of HTML report generation to handle errors and ensure accordion functionality."""
            try:
                # Safe process any dataframe column values
                import pandas as pd
                df_safe = safe_process_dataframe(df)
                
                # Process analysis results to make sure all text is safe
                safe_results = _make_analysis_results_safe(analysis_results)
                
                # Call the original function with safe data
                return original_generate_html_report(df_safe, safe_results, output_path, title, theme)
            except Exception as e:
                print(f"Error generating HTML report: {str(e)}")
                
                # Try to generate a minimal report with just the available data
                try:
                    minimal_html = _generate_minimal_report(df, analysis_results, title)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(minimal_html)
                    print(f"Minimal report saved to {output_path} due to errors in full report generation")
                    return True
                except Exception as minimal_e:
                    print(f"Error generating minimal report: {str(minimal_e)}")
                    return False
        
        # Apply the patch
        from freamon.eda import report
        report.generate_html_report = patched_generate_html_report
        
        print("Successfully applied enhanced patches to freamon library")
        return True
        
    except ImportError as e:
        print(f"Error patching freamon: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error patching freamon: {e}")
        return False

def _make_analysis_results_safe(results):
    """
    Recursively process analysis results to make all text safe for matplotlib and HTML rendering.
    
    Parameters
    ----------
    results : dict or any
        The analysis results to process
        
    Returns
    -------
    dict or any
        Processed results safe for matplotlib and HTML rendering
    """
    if isinstance(results, dict):
        safe_results = {}
        for k, v in results.items():
            safe_results[k] = _make_analysis_results_safe(v)
        return safe_results
    elif isinstance(results, list):
        return [_make_analysis_results_safe(item) for item in results]
    elif isinstance(results, str):
        return preprocess_text_for_matplotlib(results)
    else:
        return results

def _generate_minimal_report(df, analysis_results, title):
    """
    Generate a minimal HTML report with just the basic statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that was analyzed
    analysis_results : dict
        The analysis results dictionary
    title : str
        The report title
        
    Returns
    -------
    str
        Minimal HTML report
    """
    from datetime import datetime
    
    # Create a basic HTML structure
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{title} (Minimal)</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <style>
            body {{ padding-top: 20px; padding-bottom: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .card {{ margin-bottom: 20px; }}
            .table-responsive {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4">{title} (Minimal)</h1>
            <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="alert alert-warning" role="alert">
                <strong>Note:</strong> This is a minimal report generated due to errors in full report generation.
                Some analyses and visualizations may be missing.
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
    """
    
    # Add basic stats if available
    if "basic_stats" in analysis_results:
        stats = analysis_results["basic_stats"]
        html += f"""
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
                                            <td>{stats.get("n_rows", "N/A")}</td>
                                        </tr>
                                        <tr>
                                            <th>Columns</th>
                                            <td>{stats.get("n_cols", "N/A")}</td>
                                        </tr>
                                        <tr>
                                            <th>Numeric Columns</th>
                                            <td>{stats.get("n_numeric", "N/A")}</td>
                                        </tr>
                                        <tr>
                                            <th>Categorical Columns</th>
                                            <td>{stats.get("n_categorical", "N/A")}</td>
                                        </tr>
                                        <tr>
                                            <th>Datetime Columns</th>
                                            <td>{stats.get("n_datetime", "N/A")}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
        """
        
        # Add missing values section if available
        html += """
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Missing Values</h5>
                            </div>
                            <div class="card-body">
        """
        
        if stats.get("has_missing", False):
            html += f"""
                                <p>This dataset contains <strong>{stats.get("missing_count", "N/A")}</strong> missing values
                                ({stats.get("missing_percent", 0):.2f}% of all values).</p>
            """
        else:
            html += """
                                <p>This dataset does not contain any missing values.</p>
            """
        
        html += """
                            </div>
                        </div>
                    </div>
                </div>
        """
    
    # Add sample data
    html += """
                <div class="card mt-4">
                    <div class="card-header">
                        <h5 class="card-title">Sample Data</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
    """
    
    # Convert the first 5 rows to HTML
    try:
        import pandas as pd
        sample_html = df.head().to_html(classes=["table", "table-striped", "table-hover"], index=True)
        html += sample_html
    except Exception:
        html += "<p>Error displaying sample data.</p>"
    
    html += """
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return html

# Configure matplotlib when this module is imported
configure_matplotlib_for_currency()