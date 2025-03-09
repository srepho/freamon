"""
Custom objective functions for LightGBM models.

This module provides implementations of advanced objective functions
for LightGBM models, including focal loss, Tweedie regression, Huber loss,
and other specialized loss functions not available in the standard library.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.utils import check_consistent_length
import logging
import warnings

# Set up logger
logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("lightgbm package is not installed. Custom objectives will not be available.")


class CustomObjective:
    """
    Base class for custom LightGBM objective functions.
    
    This class provides a common interface for custom objective functions
    and includes validation and utility methods.
    
    Parameters
    ----------
    name : str
        Name of the objective function.
    is_higher_better : bool, default=False
        Whether higher values of the metric indicate better performance.
    """
    
    def __init__(self, name: str, is_higher_better: bool = False):
        """Initialize the custom objective."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm package is required for custom objectives.")
        
        self.name = name
        self.is_higher_better = is_higher_better
        self._fitted = False
        self._params = {}
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for the custom objective.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values (raw scores before any transformation).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            First-order and second-order gradients.
        """
        # This method should be implemented in subclasses
        raise NotImplementedError("Custom objectives must implement __call__ method.")
    
    def eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        Evaluation metric corresponding to the objective.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
        
        Returns
        -------
        Tuple[str, float, bool]
            Metric name, metric value, and whether higher is better.
        """
        # This method should be implemented in subclasses
        raise NotImplementedError("Custom objectives must implement eval_metric method.")
    
    def transform_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw prediction scores into the target metric space.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Raw prediction scores.
            
        Returns
        -------
        np.ndarray
            Transformed predictions.
        """
        # Default implementation is identity transformation
        return y_pred
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters of the objective function.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters.
        """
        return self._params.copy()
    
    def set_params(self, **params) -> 'CustomObjective':
        """
        Set parameters of the objective function.
        
        Parameters
        ----------
        **params : Dict[str, Any]
            Parameters to set.
            
        Returns
        -------
        CustomObjective
            The objective with updated parameters.
        """
        self._params.update(params)
        return self
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate inputs for the objective function.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Validated arrays.
        """
        # Ensure arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check consistent length
        check_consistent_length(y_true, y_pred)
        
        # Reshape if needed
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        return y_true, y_pred


class FocalLoss(CustomObjective):
    """
    Focal Loss for binary classification problems.
    
    Focal loss addresses class imbalance by down-weighting well-classified examples
    and focusing training on hard, misclassified examples:
    
    FL(p) = -alpha * (1 - p)^gamma * log(p) for positive class
    FL(p) = -alpha * p^gamma * log(1 - p) for negative class
    
    Parameters
    ----------
    alpha : float, default=0.25
        Class balancing parameter. Set closer to 0 to address class imbalance
        when positive examples are rare.
    gamma : float, default=2.0
        Focusing parameter. Higher values further reduce the importance of
        well-classified examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss objective."""
        super().__init__(name="focal_loss", is_higher_better=False)
        self.alpha = alpha
        self.gamma = gamma
        self._params = {"alpha": alpha, "gamma": gamma}
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for focal loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            Binary labels (0 or 1).
        y_pred : np.ndarray
            Raw prediction scores.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Apply sigmoid to get predictions in [0, 1]
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Calculate gradients
        pt = p * y_true + (1.0 - p) * (1.0 - y_true)  # Probability of true class
        alpha_t = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)  # Alpha for true class
        
        # Gradient: dL/dp * dp/dy_pred
        grad = alpha_t * (y_true - p) * (pt ** self.gamma) * (1 + self.gamma * np.log(pt + 1e-7))
        
        # Hessian: d²L/dy_pred²
        hess = alpha_t * p * (1.0 - p) * (pt ** self.gamma) * (1 + self.gamma * np.log(pt + 1e-7)) + \
               alpha_t * self.gamma * (y_true - p) * (pt ** (self.gamma - 1.0)) * p * (1.0 - p)
        
        # Stabilize hessians to be positive
        hess = np.maximum(hess, 1e-7)
        
        return grad.flatten(), hess.flatten()
    
    def eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        Evaluation metric for focal loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            Binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted probabilities.
        
        Returns
        -------
        Tuple[str, float, bool]
            Metric name, metric value, and whether higher is better.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Ensure predictions are probabilities
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Calculate focal loss
        pt = p * y_true + (1.0 - p) * (1.0 - y_true)  # Probability of true class
        alpha_t = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)  # Alpha for true class
        
        # Focal loss
        loss = -alpha_t * ((1.0 - pt) ** self.gamma) * np.log(pt + 1e-7)
        
        return "focal_loss", float(np.mean(loss)), False
    
    def transform_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw prediction scores to probabilities.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Raw prediction scores.
            
        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return 1.0 / (1.0 + np.exp(-y_pred))


class HuberLoss(CustomObjective):
    """
    Huber loss for regression problems.
    
    Huber loss is less sensitive to outliers than squared error loss:
    
    L(y, f(x)) = 0.5 * (y - f(x))^2                    if |y - f(x)| <= delta
                 delta * (|y - f(x)| - 0.5 * delta)    otherwise
    
    Parameters
    ----------
    delta : float, default=1.0
        Threshold parameter that determines the transition point from
        quadratic to linear loss.
    """
    
    def __init__(self, delta: float = 1.0):
        """Initialize Huber loss objective."""
        super().__init__(name="huber_loss", is_higher_better=False)
        self.delta = delta
        self._params = {"delta": delta}
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for Huber loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Calculate residuals
        residual = y_pred - y_true
        
        # Calculate gradients
        grad = np.zeros_like(residual)
        
        # Quadratic region (|residual| <= delta)
        quad_idx = np.abs(residual) <= self.delta
        grad[quad_idx] = residual[quad_idx]
        
        # Linear region (|residual| > delta)
        lin_idx = ~quad_idx
        grad[lin_idx] = self.delta * np.sign(residual[lin_idx])
        
        # Calculate hessians
        hess = np.zeros_like(residual)
        hess[quad_idx] = 1.0
        hess[lin_idx] = 0.0
        
        # Ensure positive hessians for stability
        hess = np.maximum(hess, 1e-7)
        
        return grad.flatten(), hess.flatten()
    
    def eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        Evaluation metric for Huber loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted values.
        
        Returns
        -------
        Tuple[str, float, bool]
            Metric name, metric value, and whether higher is better.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Calculate residuals
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        
        # Calculate Huber loss
        loss = np.zeros_like(residual)
        
        # Quadratic region
        quad_idx = abs_residual <= self.delta
        loss[quad_idx] = 0.5 * (residual[quad_idx] ** 2)
        
        # Linear region
        lin_idx = ~quad_idx
        loss[lin_idx] = self.delta * (abs_residual[lin_idx] - 0.5 * self.delta)
        
        return "huber_loss", float(np.mean(loss)), False


class TweedieLoss(CustomObjective):
    """
    Tweedie loss for regression problems with non-negative targets.
    
    Tweedie distribution generalizes Poisson, Gamma, and Normal distributions:
    
    p=1: Poisson
    p=2: Gamma
    p=3: Inverse Gaussian
    
    Parameters
    ----------
    rho : float, default=1.5
        Power parameter of the Tweedie distribution.
        Must be in range [1, 2].
    """
    
    def __init__(self, rho: float = 1.5):
        """Initialize Tweedie loss objective."""
        super().__init__(name="tweedie_loss", is_higher_better=False)
        
        if rho < 1.0 or rho > 2.0:
            raise ValueError("rho must be in range [1, 2]")
        
        self.rho = rho
        self._params = {"rho": rho}
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for Tweedie loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values (non-negative).
        y_pred : np.ndarray
            Predicted values (raw scores).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Apply exponential transformation to ensure positive predictions
        p = np.exp(y_pred)
        
        # Calculate gradients and hessians
        grad = -y_true * (p ** (1 - self.rho)) + p
        hess = -y_true * (1 - self.rho) * (p ** (1 - self.rho)) + p
        
        # Ensure positive hessians for stability
        hess = np.maximum(hess, 1e-7)
        
        return grad.flatten(), hess.flatten()
    
    def eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        Evaluation metric for Tweedie loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values (non-negative).
        y_pred : np.ndarray
            Predicted values.
        
        Returns
        -------
        Tuple[str, float, bool]
            Metric name, metric value, and whether higher is better.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Ensure positive predictions
        p = np.exp(y_pred)
        
        # Calculate Tweedie deviance
        if self.rho == 1.0:  # Poisson
            loss = 2 * (y_true * np.log((y_true + 1e-7) / (p + 1e-7)) - (y_true - p))
        elif self.rho == 2.0:  # Gamma
            loss = 2 * (np.log(p / (y_true + 1e-7)) + (y_true / (p + 1e-7)) - 1)
        else:  # In between
            loss = 2 * (y_true * (np.power(y_true + 1e-7, 1 - self.rho) - np.power(p + 1e-7, 1 - self.rho)) / (1 - self.rho) - \
                        (np.power(y_true + 1e-7, 2 - self.rho) - np.power(p + 1e-7, 2 - self.rho)) / (2 - self.rho))
        
        return "tweedie_loss", float(np.mean(loss)), False
    
    def transform_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw prediction scores to the target space.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Raw prediction scores.
            
        Returns
        -------
        np.ndarray
            Transformed predictions (positive values).
        """
        return np.exp(y_pred)


class MultiClassFocalLoss(CustomObjective):
    """
    Focal Loss for multi-class classification problems.
    
    Focal loss addresses class imbalance and hard examples in multi-class settings.
    
    Parameters
    ----------
    alpha : Optional[Union[float, List[float]]], default=None
        Class weighting parameter. Can be a single value or a list of weights for each class.
        If None, all classes are weighted equally.
    gamma : float, default=2.0
        Focusing parameter. Higher values further reduce the importance of
        well-classified examples.
    num_classes : int, default=None
        Number of classes. If None, it's inferred from the data.
    """
    
    def __init__(
        self, 
        alpha: Optional[Union[float, List[float]]] = None, 
        gamma: float = 2.0,
        num_classes: Optional[int] = None
    ):
        """Initialize multi-class focal loss objective."""
        super().__init__(name="multiclass_focal_loss", is_higher_better=False)
        self.gamma = gamma
        self.num_classes = num_classes
        self._params = {"gamma": gamma, "num_classes": num_classes}
        
        # Handle alpha
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                if num_classes is not None and len(alpha) != num_classes:
                    raise ValueError(f"Length of alpha ({len(alpha)}) must match num_classes ({num_classes})")
                self.alpha = np.array(alpha)
            else:
                self.alpha = float(alpha)
        
        self._params["alpha"] = alpha
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for multi-class focal loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            Class labels as integers (0 to num_classes-1) or one-hot encoded.
        y_pred : np.ndarray
            Raw prediction scores. Shape (n_samples, n_classes).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients and hessians.
        """
        # Handle one-hot vs integer labels
        if y_true.shape == y_pred.shape:
            # One-hot encoded, convert to class indices
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            # Already class indices
            y_true_idx = y_true.flatten()
        
        # Determine number of classes
        if self.num_classes is None:
            if y_pred.shape[1] > 1:
                self.num_classes = y_pred.shape[1]
            else:
                raise ValueError("num_classes must be specified for multi-class focal loss")
        
        # Apply softmax to get probabilities
        max_y_pred = np.max(y_pred, axis=1, keepdims=True)
        exp_y_pred = np.exp(y_pred - max_y_pred)
        softmax_pred = exp_y_pred / np.sum(exp_y_pred, axis=1, keepdims=True)
        
        # Get probability of true class
        batch_size = y_true_idx.shape[0]
        true_class_probs = np.zeros(batch_size)
        for i in range(batch_size):
            true_class_probs[i] = softmax_pred[i, y_true_idx[i]]
        
        # Apply alpha weighting if specified
        alpha_weights = np.ones(batch_size)
        if self.alpha is not None:
            if isinstance(self.alpha, np.ndarray):
                for i in range(batch_size):
                    alpha_weights[i] = self.alpha[y_true_idx[i]]
            else:
                # Single alpha value
                alpha_weights = np.ones(batch_size) * self.alpha
        
        # Calculate focusing factor
        focusing_factor = (1 - true_class_probs) ** self.gamma
        
        # Calculate gradients
        grad = softmax_pred.copy()
        for i in range(batch_size):
            grad[i, y_true_idx[i]] -= 1  # Standard softmax gradient
        
        # Apply focal loss weighting
        for i in range(batch_size):
            grad[i, :] *= alpha_weights[i] * focusing_factor[i]
        
        # Calculate hessians (diagonal approximation)
        hess = softmax_pred * (1 - softmax_pred)
        for i in range(batch_size):
            hess[i, :] *= alpha_weights[i] * focusing_factor[i]
        
        # Ensure positive hessians for stability
        hess = np.maximum(hess, 1e-7)
        
        return grad.flatten(), hess.flatten()
    
    def eval_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        """
        Evaluation metric for multi-class focal loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            Class labels as integers (0 to num_classes-1) or one-hot encoded.
        y_pred : np.ndarray
            Predicted probabilities. Shape (n_samples, n_classes).
        
        Returns
        -------
        Tuple[str, float, bool]
            Metric name, metric value, and whether higher is better.
        """
        # Handle one-hot vs integer labels
        if y_true.shape == y_pred.shape:
            # One-hot encoded, convert to class indices
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            # Already class indices
            y_true_idx = y_true.flatten()
        
        # Apply softmax if needed
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            max_y_pred = np.max(y_pred, axis=1, keepdims=True)
            exp_y_pred = np.exp(y_pred - max_y_pred)
            softmax_pred = exp_y_pred / np.sum(exp_y_pred, axis=1, keepdims=True)
        else:
            softmax_pred = y_pred
        
        # Get probability of true class
        batch_size = y_true_idx.shape[0]
        true_class_probs = np.zeros(batch_size)
        for i in range(batch_size):
            true_class_probs[i] = softmax_pred[i, y_true_idx[i]]
        
        # Apply alpha weighting if specified
        alpha_weights = np.ones(batch_size)
        if self.alpha is not None:
            if isinstance(self.alpha, np.ndarray):
                for i in range(batch_size):
                    alpha_weights[i] = self.alpha[y_true_idx[i]]
            else:
                # Single alpha value
                alpha_weights = np.ones(batch_size) * self.alpha
        
        # Calculate focal loss
        focusing_factor = (1 - true_class_probs) ** self.gamma
        loss = -alpha_weights * focusing_factor * np.log(true_class_probs + 1e-7)
        
        return "multiclass_focal_loss", float(np.mean(loss)), False
    
    def transform_prediction(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform raw prediction scores to class probabilities.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Raw prediction scores. Shape (n_samples, n_classes).
            
        Returns
        -------
        np.ndarray
            Class probabilities after softmax.
        """
        # Apply softmax
        max_y_pred = np.max(y_pred, axis=1, keepdims=True)
        exp_y_pred = np.exp(y_pred - max_y_pred)
        return exp_y_pred / np.sum(exp_y_pred, axis=1, keepdims=True)


def create_objective(
    objective_type: str, 
    **kwargs
) -> CustomObjective:
    """
    Create a custom objective function by name.
    
    Parameters
    ----------
    objective_type : str
        Type of objective function:
        - 'focal_loss': Binary focal loss
        - 'multiclass_focal_loss': Multi-class focal loss
        - 'huber_loss': Huber loss for regression
        - 'tweedie_loss': Tweedie loss for non-negative regression
    **kwargs : Dict[str, Any]
        Parameters for the objective function.
    
    Returns
    -------
    CustomObjective
        The requested objective function.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm package is required for custom objectives.")
    
    objective_map = {
        'focal_loss': FocalLoss,
        'multiclass_focal_loss': MultiClassFocalLoss,
        'huber_loss': HuberLoss,
        'tweedie_loss': TweedieLoss
    }
    
    if objective_type not in objective_map:
        raise ValueError(f"Unknown objective_type: {objective_type}. Available types: {list(objective_map.keys())}")
    
    return objective_map[objective_type](**kwargs)


def register_with_lightgbm(
    objective: CustomObjective, 
    objective_name: Optional[str] = None,
    register_metric: bool = True
) -> Tuple[Callable, Optional[Callable]]:
    """
    Register a custom objective and optionally its metric with LightGBM.
    
    Parameters
    ----------
    objective : CustomObjective
        The custom objective function to register.
    objective_name : Optional[str], default=None
        A custom name for the objective. If None, uses objective.name.
    register_metric : bool, default=True
        Whether to also register a corresponding evaluation metric.
    
    Returns
    -------
    Tuple[Callable, Optional[Callable]]
        The objective function and evaluation metric (if registered).
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("lightgbm package is required for registering custom objectives.")
    
    # Get name for the objective
    obj_name = objective_name or objective.name
    
    # Create wrapper functions that LightGBM can use
    def objective_wrapper(y_true, y_pred):
        return objective(y_true, y_pred)
    
    metric_wrapper = None
    if register_metric:
        def metric_wrapper(y_true, y_pred):
            result = objective.eval_metric(y_true, y_pred)
            return result[0], result[1], result[2]
    
    # Return the functions to be used with LightGBM
    return objective_wrapper, metric_wrapper