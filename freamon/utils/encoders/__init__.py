"""
Encoders module for categorical variables.
"""
from freamon.utils.encoders.base import (
    EncoderWrapper,
    OneHotEncoderWrapper,
    OrdinalEncoderWrapper,
    TargetEncoderWrapper,
)

from freamon.utils.encoders.category_encoders import (
    BinaryEncoderWrapper,
    HashingEncoderWrapper,
    WOEEncoderWrapper,
)

__all__ = [
    'EncoderWrapper',
    'OneHotEncoderWrapper',
    'OrdinalEncoderWrapper',
    'TargetEncoderWrapper',
    'BinaryEncoderWrapper',
    'HashingEncoderWrapper',
    'WOEEncoderWrapper',
]