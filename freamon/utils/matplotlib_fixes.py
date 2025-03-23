"""
Utility functions and configurations to fix common matplotlib issues.
"""
import logging

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
        import matplotlib.pyplot as plt
        # Disable LaTeX interpretation
        plt.rcParams['text.usetex'] = False
        # Use regular font instead of math font for dollar signs
        plt.rcParams['mathtext.default'] = 'regular'
        return True
    except ImportError:
        logging.warning("Failed to configure matplotlib: matplotlib not installed")
        return False
    except Exception as e:
        logging.warning(f"Failed to configure matplotlib: {str(e)}")
        return False

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
            processed_df[col] = processed_df[col].apply(replace_dollar_signs)
        elif pd.api.types.is_object_dtype(processed_df[col]):
            processed_df[col] = processed_df[col].astype(str).apply(
                lambda x: replace_dollar_signs(x) if isinstance(x, str) else x
            )
    
    return processed_df

# Configure matplotlib when this module is imported
configure_matplotlib_for_currency()