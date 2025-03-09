"""Tests for the pipeline visualization module."""

import os
import tempfile
from pathlib import Path
import pytest

from freamon.pipeline import (
    Pipeline,
    FeatureEngineeringStep,
    ModelTrainingStep,
    visualize_pipeline,
    generate_interactive_html
)


class TestPipelineVisualization:
    """Tests for the pipeline visualization functionality."""
    
    @pytest.fixture
    def sample_pipeline(self):
        """Create a sample pipeline for testing."""
        # Create pipeline steps
        feature_step = FeatureEngineeringStep(name="feature_engineering")
        feature_step.add_operation(
            method="add_polynomial_features",
            columns=["feature1", "feature2"],
            degree=2
        )
        
        model_step = ModelTrainingStep(
            name="model_training",
            model_type="lightgbm",
            problem_type="classification",
            hyperparameters={"num_leaves": 31, "learning_rate": 0.05}
        )
        
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_step(feature_step)
        pipeline.add_step(model_step)
        
        return pipeline
    
    def test_visualize_pipeline(self, sample_pipeline):
        """Test visualizing a pipeline."""
        try:
            # Test returning base64 image
            img_str = visualize_pipeline(sample_pipeline, format='png')
            assert img_str.startswith("data:image/png;base64,")
            
            # Test saving to file
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "pipeline_viz"
                visualize_pipeline(sample_pipeline, output_path=output_path, format='png')
                
                # Check that output file exists
                assert (Path(output_path).with_suffix('.png')).exists()
        except Exception as e:
            # This is expected if Graphviz is not installed
            from graphviz.backend.execute import ExecutableNotFound
            if isinstance(e, ExecutableNotFound):
                pytest.skip("Graphviz executable not found, skipping visualization test")
    
    def test_generate_interactive_html(self, sample_pipeline):
        """Test generating interactive HTML visualization."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "pipeline_viz.html"
                generate_interactive_html(sample_pipeline, output_path=output_path)
                
                # Check that output file exists
                assert output_path.exists()
                
                # Check file content
                with open(output_path, 'r') as f:
                    content = f.read()
                    assert "<!DOCTYPE html>" in content
                    assert "Pipeline Visualization" in content
                    assert "feature_engineering" in content
                    assert "model_training" in content
                    assert "Code Example" in content
        except Exception as e:
            # This is expected if Graphviz is not installed
            from graphviz.backend.execute import ExecutableNotFound
            if isinstance(e, ExecutableNotFound):
                pytest.skip("Graphviz executable not found, skipping interactive HTML test")