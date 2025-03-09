"""Pipeline system integrating feature engineering with model training.

This module provides a unified interface for creating end-to-end machine learning
pipelines that integrate feature engineering, feature selection, model training,
evaluation, and visualization.
"""

from freamon.pipeline.pipeline import Pipeline, PipelineStep
from freamon.pipeline.steps import (
    FeatureEngineeringStep,
    FeatureSelectionStep,
    ModelTrainingStep,
    EvaluationStep,
    ShapIQFeatureEngineeringStep,
)
from freamon.pipeline.visualization import (
    visualize_pipeline,
    generate_interactive_html,
)

__all__ = [
    "Pipeline",
    "PipelineStep",
    "FeatureEngineeringStep",
    "FeatureSelectionStep",
    "ModelTrainingStep",
    "EvaluationStep",
    "ShapIQFeatureEngineeringStep",
    "visualize_pipeline",
    "generate_interactive_html",
]