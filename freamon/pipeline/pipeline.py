"""Core pipeline implementation for ML workflows.

This module provides the central Pipeline class and PipelineStep abstract base class
for building modular, reproducible machine learning workflows.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class PipelineStep(ABC):
    """Abstract base class for all pipeline steps.
    
    All pipeline steps must implement the fit, transform, and fit_transform methods.
    Steps can optionally implement save and load methods for persistence.
    """
    name: str
    
    def __init__(self, name: str):
        """Initialize the pipeline step.
        
        Args:
            name: Unique name for this pipeline step
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> PipelineStep:
        """Fit the pipeline step to the data.
        
        Args:
            X: Feature dataframe
            y: Target series (if needed)
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted step
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the input data.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        """Fit and transform the data in one step.
        
        Args:
            X: Feature dataframe
            y: Target series (if needed)
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe
        """
        self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the pipeline step to disk.
        
        Args:
            path: Path to save the step
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path: Union[str, Path]) -> PipelineStep:
        """Load a pipeline step from disk.
        
        Args:
            path: Path to load the step from
            
        Returns:
            Loaded pipeline step
        """
        with open(path, 'rb') as f:
            loaded_step = pickle.load(f)
        
        # Transfer state to current instance
        self.__dict__.update(loaded_step.__dict__)
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if step has been fitted.
        
        Returns:
            Boolean indicating if step has been fitted
        """
        return self._is_fitted


class Pipeline:
    """ML pipeline that connects feature engineering with model training.
    
    The Pipeline class provides a unified interface for creating end-to-end
    machine learning workflows that integrate multiple steps including
    feature engineering, feature selection, model training, and evaluation.
    """
    
    def __init__(self, steps: Optional[List[PipelineStep]] = None):
        """Initialize the pipeline with optional steps.
        
        Args:
            steps: List of pipeline steps to include
        """
        self.steps = steps or []
        self.metadata = {}
        self._step_outputs = {}
    
    def add_step(self, step: PipelineStep) -> Pipeline:
        """Add a step to the pipeline.
        
        Args:
            step: Pipeline step to add
            
        Returns:
            self: For method chaining
        """
        # Validate step name uniqueness
        step_names = [s.name for s in self.steps]
        if step.name in step_names:
            raise ValueError(f"Step name '{step.name}' already exists in pipeline")
        
        self.steps.append(step)
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> Pipeline:
        """Fit all steps in the pipeline.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            self: The fitted pipeline
        """
        data = X.copy()
        self._step_outputs = {}
        
        for step in self.steps:
            # Store input data
            self._step_outputs[f"{step.name}_input"] = data.copy()
            
            # Execute step
            step.fit(data, y, **kwargs)
            data = step.transform(data, **kwargs)
            
            # Store output data
            self._step_outputs[f"{step.name}_output"] = data.copy()
            
        # Store final output
        self._step_outputs["final_output"] = data
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data through all pipeline steps.
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe
        """
        data = X.copy()
        
        for step in self.steps:
            data = step.transform(data, **kwargs)
            
        return data
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            X: Feature dataframe
            y: Target series
            **kwargs: Additional parameters
            
        Returns:
            Transformed dataframe
        """
        self.fit(X, y, **kwargs)
        return self._step_outputs["final_output"]
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Make predictions with the fitted pipeline.
        
        This method transforms the input data through all steps and then
        uses the final step for prediction (assuming it's a model step).
        
        Args:
            X: Feature dataframe
            **kwargs: Additional parameters
            
        Returns:
            Predictions
        """
        # Ensure last step has predict method
        if not hasattr(self.steps[-1], "predict"):
            raise AttributeError("Final pipeline step does not have a predict method")
        
        # Transform through all steps except the last one
        data = X.copy()
        for step in self.steps[:-1]:
            data = step.transform(data, **kwargs)
            
        # Use the last step's predict method
        return self.steps[-1].predict(data, **kwargs)
    
    def get_step_output(self, step_name: str, output_type: str = "output") -> pd.DataFrame:
        """Retrieve the output from a specific step.
        
        Args:
            step_name: Name of the step
            output_type: Type of output ("input" or "output")
            
        Returns:
            Step output dataframe
        """
        key = f"{step_name}_{output_type}"
        if key not in self._step_outputs:
            raise KeyError(f"Output '{key}' not found. Pipeline may not be fitted yet.")
        return self._step_outputs[key]
        
    def get_step(self, step_name: str) -> PipelineStep:
        """Retrieve a step from the pipeline by name.
        
        Args:
            step_name: Name of the step to retrieve
            
        Returns:
            The pipeline step
            
        Raises:
            ValueError: If the step name is not found
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        raise ValueError(f"Step '{step_name}' not found in pipeline")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the entire pipeline to disk.
        
        Creates a directory with individual step files and metadata.
        
        Args:
            path: Directory path to save the pipeline
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "step_names": [step.name for step in self.steps],
            "pipeline_metadata": self.metadata
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save each step
        for step in self.steps:
            step.save(path / f"{step.name}.pkl")
    
    def load(self, path: Union[str, Path]) -> Pipeline:
        """Load a pipeline from disk.
        
        Args:
            path: Directory path to load the pipeline from
            
        Returns:
            Loaded pipeline
        """
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct pipeline
        self.steps = []
        for step_name in metadata["step_names"]:
            step_path = path / f"{step_name}.pkl"
            with open(step_path, 'rb') as f:
                step = pickle.load(f)
            self.steps.append(step)
        
        self.metadata = metadata.get("pipeline_metadata", {})
        
        return self
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the model step.
        
        Returns:
            DataFrame with feature importances
        """
        # Find the model step (usually the last one)
        model_steps = [step for step in self.steps 
                       if hasattr(step, "get_feature_importances")]
        
        if not model_steps:
            raise AttributeError("No steps with feature importance functionality found")
        
        # Get importances from the last model step
        model_step = model_steps[-1]
        return model_step.get_feature_importances()
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline and its steps.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "steps": [
                {
                    "name": step.name,
                    "type": step.__class__.__name__,
                    "is_fitted": step.is_fitted
                }
                for step in self.steps
            ],
            "metadata": self.metadata
        }