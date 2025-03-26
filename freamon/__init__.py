"""
Freamon: A package to make data science projects on tabular data easier.
"""

__version__ = "0.3.48"

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

# Import automated modeling flow
from freamon.modeling.autoflow import AutoModelFlow, auto_model

# Import Polars-optimized deduplication functions
from freamon.deduplication import (
    polars_lsh_deduplication,
    streaming_lsh_deduplication,
    PolarsSupervisedDeduplicationModel
)

# Import useful dataframe utilities
from freamon.utils.dataframe_utils import (
    check_dataframe_type,
    convert_dataframe,
    optimize_dtypes,
    process_in_chunks,
    iterate_chunks
)

# Integration components are not auto-imported to avoid dependencies
# Import them explicitly with: from freamon.integration.allyanonimiser_bridge import AnonymizationStep

# Polars text utilities are not auto-imported to avoid dependencies
# Import them explicitly with: from freamon.utils.polars_text_utils import batch_vectorize_texts, process_text_column, deduplicate_text_column