"""
Probability calibration for classification models.

This module provides functions for calibrating the probability outputs
of classification models using methods like Platt scaling and isotonic regression.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split


class ProbabilityCalibrator:
    """
    Calibrate the probability outputs of classification models.
    
    Probability calibration is the task of adjusting predicted probabilities
    to be more representative of the true likelihood of an event. This class
    provides methods for calibrating classification models using either
    Platt scaling (sigmoid calibration) or isotonic regression.
    
    Parameters
    ----------
    method : Literal['sigmoid', 'isotonic'], default='sigmoid'
        The calibration method to use:
        - 'sigmoid': Platt scaling, works well with small datasets
        - 'isotonic': Isotonic regression, more flexible but requires more data
    cv : Union[int, str], default=5
        Number of cross-validation folds or 'prefit' if the model is already fitted.
    use_sample_weights : bool, default=False
        Whether to use sample weights when fitting calibrators.
    random_state : Optional[int], default=None
        Controls the randomness of the calibration.
    """
    
    def __init__(
        self,
        method: Literal['sigmoid', 'isotonic'] = 'sigmoid',
        cv: Union[int, str] = 5,
        use_sample_weights: bool = False,
        random_state: Optional[int] = None,
    ):
        """Initialize the ProbabilityCalibrator."""
        self.method = method
        self.cv = cv
        self.use_sample_weights = use_sample_weights
        self.random_state = random_state
        self.calibrator = None
        self.is_fitted = False
        self.n_classes = None
        self.label_encoder = None
    
    def fit(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> 'ProbabilityCalibrator':
        """
        Fit the probability calibrator.
        
        Parameters
        ----------
        model : Any
            The classification model to calibrate.
        X : Union[pd.DataFrame, np.ndarray]
            Training data features.
        y : Union[pd.Series, np.ndarray]
            Target values.
        sample_weight : Optional[np.ndarray], default=None
            Sample weights for fitting the calibrator.
        X_val : Optional[Union[pd.DataFrame, np.ndarray]], default=None
            Validation data features. If provided, cv='prefit' is used.
        y_val : Optional[Union[pd.Series, np.ndarray]], default=None
            Validation target values. Required if X_val is provided.
            
        Returns
        -------
        ProbabilityCalibrator
            Fitted calibrator.
            
        Raises
        ------
        ValueError
            If validation data is partially provided or sample weights
            are provided when use_sample_weights is False.
        """
        # Convert target to numpy array
        if isinstance(y, pd.Series):
            y_np = y.values
        else:
            y_np = y
        
        # Handle multiclass classification with label encoding
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_np)
        self.n_classes = len(self.label_encoder.classes_)
        
        # Validate sample_weight
        if sample_weight is not None and not self.use_sample_weights:
            raise ValueError(
                "Sample weights provided but use_sample_weights is False. "
                "Set use_sample_weights=True to use sample weights."
            )
        
        # Determine whether to use prefit model or CV
        if X_val is not None:
            if y_val is None:
                raise ValueError("y_val must be provided if X_val is provided")
            
            # Use validation data for calibration
            y_val_encoded = self.label_encoder.transform(y_val)
            
            # Create the calibrator with prefit=True
            self.calibrator = CalibratedClassifierCV(
                estimator=model,
                method=self.method,
                cv='prefit',
                n_jobs=None,
                ensemble=True
            )
            
            # Fit on validation data
            fit_params = {}
            if sample_weight is not None and self.use_sample_weights:
                fit_params['sample_weight'] = sample_weight
            
            self.calibrator.fit(X_val, y_val_encoded, **fit_params)
        else:
            # Use cross-validation
            cv_obj = self.cv
            if isinstance(self.cv, int):
                cv_obj = StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=self.random_state
                )
            
            # Create the calibrator
            self.calibrator = CalibratedClassifierCV(
                estimator=model,
                method=self.method,
                cv=cv_obj,
                n_jobs=None,
                ensemble=True
            )
            
            # Fit on training data
            fit_params = {}
            if sample_weight is not None and self.use_sample_weights:
                fit_params['sample_weight'] = sample_weight
            
            self.calibrator.fit(X, y_encoded, **fit_params)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using the calibrated model.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input data.
            
        Returns
        -------
        np.ndarray
            Calibrated probability estimates.
            
        Raises
        ------
        ValueError
            If the calibrator is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        return self.calibrator.predict_proba(X)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict classes using the calibrated model.
        
        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input data.
            
        Returns
        -------
        np.ndarray
            Predicted classes.
            
        Raises
        ------
        ValueError
            If the calibrator is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        probs = self.predict_proba(X)
        pred_encoded = np.argmax(probs, axis=1)
        
        # Convert back to original labels
        return self.label_encoder.inverse_transform(pred_encoded)


def evaluate_calibration(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    n_bins: int = 10,
    class_index: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    return_fig: bool = False,
) -> Union[Dict[str, Any], Any]:
    """
    Evaluate and plot the calibration of a classifier.
    
    Parameters
    ----------
    y_true : Union[pd.Series, np.ndarray]
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int, default=10
        Number of bins for the reliability diagram.
    class_index : Optional[int], default=None
        Index of the class to evaluate for multiclass problems.
        If None and y_prob has shape (n_samples, 2), uses the second column.
        For multiclass with > 2 classes, class_index must be provided.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size.
    show : bool, default=True
        Whether to show the plot.
    return_fig : bool, default=False
        Whether to return the figure.
        
    Returns
    -------
    Union[Dict[str, Any], Any]
        Dictionary with calibration metrics, or the figure if return_fig is True.
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> X, y = make_classification(random_state=0)
    >>> model = RandomForestClassifier(random_state=0).fit(X, y)
    >>> y_prob = model.predict_proba(X)
    >>> result = evaluate_calibration(y, y_prob)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with 'pip install matplotlib'."
        )
    
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    # For binary classification
    if len(y_prob.shape) == 2:
        if y_prob.shape[1] == 2:
            if class_index is None:
                class_index = 1
            y_prob_binary = y_prob[:, class_index]
        elif y_prob.shape[1] > 2:
            if class_index is None:
                raise ValueError(
                    "class_index must be provided for multiclass problems with > 2 classes"
                )
            y_prob_binary = y_prob[:, class_index]
            # Convert to binary labels for evaluation
            y_true_binary = (y_true == class_index).astype(int)
            y_true = y_true_binary
        else:
            y_prob_binary = y_prob[:, 0]
    else:
        y_prob_binary = y_prob
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob_binary, n_bins=n_bins)
    
    # Calculate calibration metrics
    # Expected Calibration Error (ECE)
    ece = 0
    for i in range(len(prob_true)):
        ece += abs(prob_true[i] - prob_pred[i]) * (1 / n_bins)
    
    # Brier score
    brier_score = np.mean((y_prob_binary - y_true) ** 2)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration curve')
    
    # Plot the perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    
    # Plot the histogram of predicted probabilities
    ax2 = ax.twinx()
    ax2.hist(y_prob_binary, range=(0, 1), bins=n_bins, 
             histtype='step', density=True, alpha=0.5, color='gray')
    ax2.set_ylim(0, 4)
    ax2.set_yticks([])
    
    # Add metrics to the plot
    ax.text(
        0.05, 0.95, 
        f'Expected Calibration Error (ECE): {ece:.4f}\n'
        f'Brier Score: {brier_score:.4f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Customize the plot
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    # Prepare results
    results = {
        'ece': ece,
        'brier_score': brier_score,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
    }
    
    if return_fig:
        return fig
    return results


def compare_calibration_methods(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model: Any,
    class_index: Optional[int] = None,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (15, 10),
    show: bool = True,
    return_fig: bool = False,
) -> Union[Dict[str, Any], Any]:
    """
    Compare different calibration methods for a classification model.
    
    This function compares the uncalibrated model against models
    calibrated with Platt scaling (sigmoid) and isotonic regression.
    
    Parameters
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Training data features.
    y_train : Union[pd.Series, np.ndarray]
        Training target values.
    X_test : Union[pd.DataFrame, np.ndarray]
        Test data features.
    y_test : Union[pd.Series, np.ndarray]
        Test target values.
    model : Any
        The classification model to calibrate.
    class_index : Optional[int], default=None
        Index of the class to evaluate for multiclass problems.
    n_bins : int, default=10
        Number of bins for the reliability diagram.
    figsize : Tuple[int, int], default=(15, 10)
        Figure size.
    show : bool, default=True
        Whether to show the plot.
    return_fig : bool, default=False
        Whether to return the figure.
        
    Returns
    -------
    Union[Dict[str, Any], Any]
        Dictionary with calibration metrics for each method,
        or the figure if return_fig is True.
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> model = RandomForestClassifier(random_state=0)
    >>> result = compare_calibration_methods(X_train, y_train, X_test, y_test, model)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with 'pip install matplotlib'."
        )
    
    # Fit the base model
    model.fit(X_train, y_train)
    
    # Get uncalibrated predictions
    y_prob_uncalibrated = model.predict_proba(X_test)
    
    # Fit Platt scaling calibrator
    sigmoid_calibrator = ProbabilityCalibrator(method='sigmoid')
    sigmoid_calibrator.fit(model, X_train, y_train)
    y_prob_sigmoid = sigmoid_calibrator.predict_proba(X_test)
    
    # Fit isotonic regression calibrator
    isotonic_calibrator = ProbabilityCalibrator(method='isotonic')
    isotonic_calibrator.fit(model, X_train, y_train)
    y_prob_isotonic = isotonic_calibrator.predict_proba(X_test)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Define methods and corresponding probabilities
    methods = ['Uncalibrated', 'Platt Scaling', 'Isotonic Regression']
    probabilities = [y_prob_uncalibrated, y_prob_sigmoid, y_prob_isotonic]
    colors = ['darkorange', 'green', 'blue']
    results = {}
    
    # For binary classification
    if len(y_prob_uncalibrated.shape) == 2:
        if y_prob_uncalibrated.shape[1] == 2:
            if class_index is None:
                class_index = 1
            
        elif y_prob_uncalibrated.shape[1] > 2:
            if class_index is None:
                raise ValueError(
                    "class_index must be provided for multiclass problems with > 2 classes"
                )
            
            # Convert to binary labels for evaluation
            y_test_binary = (y_test == class_index).astype(int)
            y_test = y_test_binary
    
    # Plot each method
    for i, (method, y_prob, color) in enumerate(zip(methods, probabilities, colors)):
        if len(y_prob.shape) == 2 and y_prob.shape[1] > 1:
            y_prob_binary = y_prob[:, class_index]
        else:
            y_prob_binary = y_prob
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_prob_binary, n_bins=n_bins)
        
        # Calculate calibration metrics
        # Expected Calibration Error (ECE)
        ece = 0
        for j in range(len(prob_true)):
            ece += abs(prob_true[j] - prob_pred[j]) * (1 / n_bins)
        
        # Brier score
        brier_score = np.mean((y_prob_binary - y_test) ** 2)
        
        # Store results
        results[method] = {
            'ece': ece,
            'brier_score': brier_score,
            'prob_true': prob_true,
            'prob_pred': prob_pred,
        }
        
        # Plot the calibration curve
        axes[i].plot(prob_pred, prob_true, marker='o', linewidth=2, color=color, 
                     label='Calibration curve')
        
        # Plot the perfectly calibrated line
        axes[i].plot([0, 1], [0, 1], linestyle='--', color='gray', 
                     label='Perfectly calibrated')
        
        # Plot the histogram of predicted probabilities
        ax2 = axes[i].twinx()
        ax2.hist(y_prob_binary, range=(0, 1), bins=n_bins, 
                 histtype='step', density=True, alpha=0.5, color='gray')
        ax2.set_ylim(0, 4)
        ax2.set_yticks([])
        
        # Add metrics to the plot
        axes[i].text(
            0.05, 0.95, 
            f'ECE: {ece:.4f}\nBrier: {brier_score:.4f}',
            transform=axes[i].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Customize the plot
        axes[i].set_xlabel('Mean predicted probability')
        if i == 0:
            axes[i].set_ylabel('Fraction of positives')
        axes[i].set_title(method)
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    if return_fig:
        return fig
    return results