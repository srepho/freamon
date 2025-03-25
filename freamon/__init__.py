"""
Freamon: A package to make data science projects on tabular data easier.
"""

__version__ = "0.3.26"

# Import key components for convenient access
from freamon.modeling.lightgbm import LightGBMModel
from freamon.modeling.tuning import LightGBMTuner
from freamon.modeling.importance import (
    calculate_permutation_importance,
    get_permutation_importance_df,
    plot_permutation_importance,
    select_features_by_importance,
    auto_select_features
)
from freamon.modeling.early_stopping import (
    get_early_stopping_callback,
    get_lr_scheduler
)

# Integration components are not auto-imported to avoid dependencies
# Import them explicitly with: from freamon.integration.allyanonimiser_bridge import AnonymizationStep