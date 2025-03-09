"""
Utility functions for the Freamon package.
"""

from freamon.utils.dataframe_utils import (
    check_dataframe_type,
    convert_dataframe,
    optimize_dtypes,
    estimate_memory_usage,
)

from freamon.utils.encoders import (
    EncoderWrapper,
    OneHotEncoderWrapper,
    OrdinalEncoderWrapper,
    TargetEncoderWrapper,
)

from freamon.utils.text_utils import (
    TextProcessor,
)

__all__ = [
    # Dataframe utils
    "check_dataframe_type",
    "convert_dataframe",
    "optimize_dtypes",
    "estimate_memory_usage",
    
    # Encoders
    "EncoderWrapper",
    "OneHotEncoderWrapper",
    "OrdinalEncoderWrapper",
    "TargetEncoderWrapper",
    
    # Text utils
    "TextProcessor",
]