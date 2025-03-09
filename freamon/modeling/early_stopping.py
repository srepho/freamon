"""
Enhanced early stopping functionality for model training.

This module provides advanced early stopping implementations including
patience-based stopping, adaptive learning rate scheduling, and more.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from copy import deepcopy

# Set up logger
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Enhanced early stopping with multiple modes and callbacks.
    
    Parameters
    ----------
    monitor : str
        Metric to monitor for improvement.
    mode : str, default='min'
        Direction of improvement ('min' or 'max').
    patience : int, default=10
        Number of epochs/iterations with no improvement before stopping.
    min_delta : float, default=0.0
        Minimum change to qualify as an improvement.
    restore_best_weights : bool, default=True
        Whether to restore model to the best weights when early stopping occurs.
    baseline : Optional[float], default=None
        Baseline value for the monitored metric. Training will stop if this is not
        surpassed by patience steps.
    stopping_threshold : Optional[float], default=None
        Threshold for the monitored metric. Training will stop immediately if this
        value is reached.
    divergence_threshold : Optional[float], default=None
        Threshold for divergence from the best value. Training will stop immediately
        if the monitored metric is worse than the best by this value.
    """
    
    def __init__(
        self,
        monitor: str,
        mode: str = 'min',
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        baseline: Optional[float] = None,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None
    ):
        """Initialize early stopping."""
        self.monitor = monitor
        
        # Validate mode
        if mode not in ['min', 'max']:
            raise ValueError(f"mode '{mode}' is not supported. Expected one of ['min', 'max']")
        self.mode = mode
        
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        
        # Set direction multiplier
        self.multiplier = 1 if self.mode == 'min' else -1
        
        # Initialize state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_weights = None
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        
        # Flags
        self.stop_training = False
        self.converged = False
        self.baseline_surpassed = False if self.baseline is not None else None
        self.stopping_threshold_reached = False
        self.diverged = False
    
    def check_improvement(self, current: float) -> bool:
        """
        Check if the current value shows improvement over the best so far.
        
        Parameters
        ----------
        current : float
            Current metric value.
        
        Returns
        -------
        bool
            True if there is improvement, False otherwise.
        """
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        else:  # max
            return current > (self.best_score + self.min_delta)
    
    def __call__(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Optional[Any] = None
    ) -> bool:
        """
        Check if early stopping condition is met.
        
        Parameters
        ----------
        epoch : int
            Current epoch/iteration.
        logs : Dict[str, float]
            Dictionary of metrics.
        model : Optional[Any], default=None
            Model object to save weights from.
        
        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        # Get current score from logs
        if self.monitor not in logs:
            logger.warning(f"Early stopping metric '{self.monitor}' not found in logs")
            return False
        
        current = logs[self.monitor]
        
        # Check baseline if it exists and hasn't been surpassed yet
        if self.baseline is not None and not self.baseline_surpassed:
            # Check if the baseline is surpassed
            if self.mode == 'min':
                self.baseline_surpassed = current <= self.baseline
            else:
                self.baseline_surpassed = current >= self.baseline
        
        # Check stopping threshold if it exists
        if self.stopping_threshold is not None:
            if self.mode == 'min':
                self.stopping_threshold_reached = current <= self.stopping_threshold
            else:
                self.stopping_threshold_reached = current >= self.stopping_threshold
            
            if self.stopping_threshold_reached:
                self.stopped_epoch = epoch
                self.stop_training = True
                logger.info(f"Early stopping: Stopping threshold reached at epoch {epoch}")
                return True
        
        # Check divergence from best score if threshold exists
        if self.divergence_threshold is not None:
            if self.mode == 'min':
                divergence = current - self.best_score
                self.diverged = divergence > self.divergence_threshold
            else:
                divergence = self.best_score - current
                self.diverged = divergence > self.divergence_threshold
            
            if self.diverged:
                self.stopped_epoch = epoch
                self.stop_training = True
                logger.info(f"Early stopping: Training diverged at epoch {epoch}")
                return True
        
        # Check for improvement
        if self.check_improvement(current):
            # Update best score and reset counter
            self.best_score = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights if requested
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_weights(model)
        else:
            # No improvement, increment counter
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                self.converged = True
                
                # Restore best weights if requested
                if self.restore_best_weights and self.best_weights is not None and model is not None:
                    self._set_weights(model, self.best_weights)
                    logger.info(f"Early stopping: Restoring best weights from epoch {self.best_epoch}")
                
                logger.info(f"Early stopping: Training stopped at epoch {epoch} after {self.patience} epochs with no improvement")
                return True
        
        return False
    
    def _get_weights(self, model: Any) -> Any:
        """Get model weights with appropriate framework handling."""
        # Scikit-learn models don't have the concept of weights to save during training
        
        # Try LightGBM model
        if hasattr(model, 'model_'):
            return deepcopy(model.model_)
        
        # For PyTorch models
        if hasattr(model, 'state_dict'):
            return deepcopy(model.state_dict())
        
        # For Keras/TensorFlow models
        if hasattr(model, 'get_weights'):
            return deepcopy(model.get_weights())
        
        # For other models, store the entire model
        return deepcopy(model)
    
    def _set_weights(self, model: Any, weights: Any) -> None:
        """Set model weights with appropriate framework handling."""
        # Try LightGBM model
        if hasattr(model, 'model_'):
            model.model_ = weights
            return
        
        # For PyTorch models
        if hasattr(model, 'load_state_dict') and hasattr(weights, 'keys'):
            model.load_state_dict(weights)
            return
        
        # For Keras/TensorFlow models
        if hasattr(model, 'set_weights') and isinstance(weights, list):
            model.set_weights(weights)
            return
        
        # For other models, try to replace the model
        try:
            model = weights
        except:
            logger.warning("Could not restore best weights. Weight format not recognized.")
    
    def get_monitor_value(self, logs: Dict[str, float]) -> Optional[float]:
        """
        Get the current value of the monitored metric.
        
        Parameters
        ----------
        logs : Dict[str, float]
            Dictionary of metrics.
        
        Returns
        -------
        Optional[float]
            Current value of the monitored metric, or None if not found.
        """
        return logs.get(self.monitor, None)
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the early stopping object.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the current state.
        """
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'stop_training': self.stop_training,
            'converged': self.converged,
            'baseline_surpassed': self.baseline_surpassed,
            'stopping_threshold_reached': self.stopping_threshold_reached,
            'diverged': self.diverged
        }


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler with multiple strategies.
    
    Parameters
    ----------
    monitor : str
        Metric to monitor for improvement.
    mode : str, default='min'
        Direction of improvement ('min' or 'max').
    factor : float, default=0.5
        Factor by which to reduce learning rate.
    patience : int, default=5
        Number of epochs/iterations with no improvement before reducing lr.
    min_delta : float, default=1e-4
        Minimum change to qualify as an improvement.
    min_lr : float, default=1e-6
        Minimum learning rate.
    cooldown : int, default=0
        Number of epochs to wait after a lr reduction before resuming normal operation.
    strategy : str, default='reduce_on_plateau'
        Strategy for adjusting learning rate:
        - 'reduce_on_plateau': Reduce by factor if no improvement for patience epochs
        - 'cosine_annealing': Use cosine annealing schedule
        - 'exponential_decay': Exponential decay by factor every patience epochs
        - 'one_cycle': One-cycle policy with warmup and cooldown
    warmup_epochs : int, default=0
        Number of epochs for warmup (only for certain strategies).
    cycle_epochs : int, default=0
        Number of epochs for a full cycle (only for certain strategies).
    initial_lr : float, default=0.1
        Initial learning rate.
    max_lr : float, default=0.1
        Maximum learning rate (for one_cycle strategy).
    """
    
    def __init__(
        self,
        monitor: str,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        min_delta: float = 1e-4,
        min_lr: float = 1e-6,
        cooldown: int = 0,
        strategy: str = 'reduce_on_plateau',
        warmup_epochs: int = 0,
        cycle_epochs: int = 0,
        initial_lr: float = 0.1,
        max_lr: float = 0.1
    ):
        """Initialize the scheduler."""
        self.monitor = monitor
        
        # Validate mode
        if mode not in ['min', 'max']:
            raise ValueError(f"mode '{mode}' is not supported. Expected one of ['min', 'max']")
        self.mode = mode
        
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.cooldown = cooldown
        
        # Validate strategy
        valid_strategies = [
            'reduce_on_plateau',
            'cosine_annealing',
            'exponential_decay',
            'one_cycle'
        ]
        if strategy not in valid_strategies:
            raise ValueError(f"strategy '{strategy}' is not supported. Expected one of {valid_strategies}")
        self.strategy = strategy
        
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        
        # Set direction multiplier
        self.multiplier = 1 if self.mode == 'min' else -1
        
        # Initialize state
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.wait = 0
        self.cooldown_counter = 0
        self.current_lr = initial_lr
        self.current_epoch = 0
    
    def check_improvement(self, current: float) -> bool:
        """
        Check if the current value shows improvement over the best so far.
        
        Parameters
        ----------
        current : float
            Current metric value.
        
        Returns
        -------
        bool
            True if there is improvement, False otherwise.
        """
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        else:  # max
            return current > (self.best_score + self.min_delta)
    
    def __call__(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> float:
        """
        Compute new learning rate based on the selected strategy.
        
        Parameters
        ----------
        epoch : int
            Current epoch/iteration.
        logs : Dict[str, float]
            Dictionary of metrics.
        model : Optional[Any], default=None
            Model object.
        optimizer : Optional[Any], default=None
            Optimizer object.
        
        Returns
        -------
        float
            New learning rate.
        """
        self.current_epoch = epoch
        
        if self.strategy == 'reduce_on_plateau':
            return self._reduce_on_plateau(epoch, logs, model, optimizer)
        elif self.strategy == 'cosine_annealing':
            return self._cosine_annealing(epoch, model, optimizer)
        elif self.strategy == 'exponential_decay':
            return self._exponential_decay(epoch, model, optimizer)
        elif self.strategy == 'one_cycle':
            return self._one_cycle(epoch, model, optimizer)
        else:
            return self.current_lr
    
    def _reduce_on_plateau(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> float:
        """Reduce learning rate on plateau strategy."""
        # Get current score from logs
        if self.monitor not in logs:
            logger.warning(f"Learning rate scheduler metric '{self.monitor}' not found in logs")
            return self.current_lr
        
        current = logs[self.monitor]
        
        # Check if we're in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr
        
        # Check for improvement
        if self.check_improvement(current):
            # Update best score and reset counter
            self.best_score = current
            self.wait = 0
        else:
            # No improvement, increment counter
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait = 0
                self.cooldown_counter = self.cooldown
                
                # Apply the new learning rate
                self._set_learning_rate(model, optimizer, self.current_lr)
                
                logger.info(f"Reducing learning rate: {old_lr:.6f} -> {self.current_lr:.6f}")
        
        return self.current_lr
    
    def _cosine_annealing(
        self,
        epoch: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> float:
        """Cosine annealing learning rate schedule."""
        # Determine effective cycle length
        cycle_length = self.cycle_epochs if self.cycle_epochs > 0 else 1
        
        # Handle warmup if needed
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = epoch / max(1, self.warmup_epochs)
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * warmup_progress
        else:
            # Adjust epoch for warmup
            effective_epoch = epoch - self.warmup_epochs
            cycle_position = effective_epoch % cycle_length
            
            # Cosine schedule from initial_lr to min_lr
            # When cycle_position is 0, cosine_factor is 1.0
            # When cycle_position is cycle_length, cosine_factor is 0.0
            cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_position / cycle_length))
            
            # Ensure we actually reach min_lr at the end of the cycle
            if cycle_position == cycle_length - 1:
                self.current_lr = self.min_lr
            else:
                self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        
        # Apply the new learning rate
        self._set_learning_rate(model, optimizer, self.current_lr)
        
        return self.current_lr
    
    def _exponential_decay(
        self,
        epoch: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> float:
        """Exponential decay learning rate schedule."""
        # Determine schedule step boundary
        if self.patience > 0 and epoch > 0 and epoch % self.patience == 0:
            # Calculate new learning rate with exponential decay
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            
            # Apply the new learning rate
            self._set_learning_rate(model, optimizer, self.current_lr)
            
            logger.info(f"Reducing learning rate: {old_lr:.6f} -> {self.current_lr:.6f}")
        
        return self.current_lr
    
    def _one_cycle(
        self,
        epoch: int,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None
    ) -> float:
        """One-cycle learning rate policy."""
        # Determine total cycle length
        total_length = self.cycle_epochs if self.cycle_epochs > 0 else 1
        
        # Calculate mid_point (e.g., 45% of cycle is warmup)
        mid_point = int(total_length * 0.45)
        
        if epoch < mid_point:
            # Linear warmup from initial_lr to max_lr
            progress = epoch / mid_point
            self.current_lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Cosine annealing from max_lr to min_lr
            remaining = total_length - mid_point
            progress = (epoch - mid_point) / remaining
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        
        # Apply the new learning rate
        self._set_learning_rate(model, optimizer, self.current_lr)
        
        return self.current_lr
    
    def _set_learning_rate(
        self,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        lr: float = 0.1
    ) -> None:
        """
        Set the learning rate in the model or optimizer.
        
        Handles different frameworks and model types.
        """
        # Handle different model types and frameworks
        
        # LightGBM
        if model is not None and hasattr(model, 'model') and hasattr(model.model, 'set_params'):
            if hasattr(model.model, 'learning_rate'):
                model.model.set_params(learning_rate=lr)
                return
            elif hasattr(model.model, 'learning_rate_'):
                model.model.set_params(learning_rate=lr)
                return
        
        # Generic scikit-learn
        if model is not None and hasattr(model, 'set_params'):
            try:
                model.set_params(learning_rate=lr)
                return
            except:
                pass
        
        # PyTorch
        if optimizer is not None and hasattr(optimizer, 'param_groups'):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return
        
        # Keras/TensorFlow
        if optimizer is not None and hasattr(optimizer, 'learning_rate'):
            import tensorflow as tf
            if hasattr(optimizer.learning_rate, 'assign'):
                optimizer.learning_rate.assign(lr)
            elif isinstance(optimizer.learning_rate, tf.Variable):
                optimizer.learning_rate.assign(lr)
            else:
                setattr(optimizer, 'learning_rate', lr)
            return
        
        # If we get here, we couldn't set the learning rate
        logger.warning("Could not set learning rate. Unsupported model/optimizer type.")


def get_early_stopping_callback(
    monitor: str = 'val_loss',
    mode: str = 'min',
    patience: int = 10,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
    baseline: Optional[float] = None,
    stopping_threshold: Optional[float] = None,
    divergence_threshold: Optional[float] = None
) -> EarlyStopping:
    """
    Create an early stopping callback with the specified parameters.
    
    Parameters
    ----------
    monitor : str, default='val_loss'
        Metric to monitor for improvement.
    mode : str, default='min'
        Direction of improvement ('min' or 'max').
    patience : int, default=10
        Number of epochs with no improvement before stopping.
    min_delta : float, default=0.0
        Minimum change to qualify as an improvement.
    restore_best_weights : bool, default=True
        Whether to restore model to the best weights when stopping.
    baseline : Optional[float], default=None
        Baseline value for the monitored metric.
    stopping_threshold : Optional[float], default=None
        Threshold for the monitored metric to stop immediately.
    divergence_threshold : Optional[float], default=None
        Threshold for divergence from the best value to stop immediately.
    
    Returns
    -------
    EarlyStopping
        Early stopping callback object.
    """
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=restore_best_weights,
        baseline=baseline,
        stopping_threshold=stopping_threshold,
        divergence_threshold=divergence_threshold
    )


def get_lr_scheduler(
    monitor: str = 'val_loss',
    mode: str = 'min',
    factor: float = 0.5,
    patience: int = 5,
    min_delta: float = 1e-4,
    min_lr: float = 1e-6,
    cooldown: int = 0,
    strategy: str = 'reduce_on_plateau',
    initial_lr: float = 0.1,
    max_lr: float = 0.1,
    warmup_epochs: int = 0,
    cycle_epochs: int = 0
) -> AdaptiveLearningRateScheduler:
    """
    Create a learning rate scheduler with the specified parameters.
    
    Parameters
    ----------
    monitor : str, default='val_loss'
        Metric to monitor for improvement.
    mode : str, default='min'
        Direction of improvement ('min' or 'max').
    factor : float, default=0.5
        Factor by which to reduce learning rate.
    patience : int, default=5
        Number of epochs with no improvement before reducing lr.
    min_delta : float, default=1e-4
        Minimum change to qualify as an improvement.
    min_lr : float, default=1e-6
        Minimum learning rate.
    cooldown : int, default=0
        Number of epochs to wait after lr reduction.
    strategy : str, default='reduce_on_plateau'
        Strategy for adjusting learning rate.
    initial_lr : float, default=0.1
        Initial learning rate.
    max_lr : float, default=0.1
        Maximum learning rate (for one_cycle strategy).
    warmup_epochs : int, default=0
        Number of epochs for warmup.
    cycle_epochs : int, default=0
        Number of epochs for a full cycle.
    
    Returns
    -------
    AdaptiveLearningRateScheduler
        Learning rate scheduler object.
    """
    return AdaptiveLearningRateScheduler(
        monitor=monitor,
        mode=mode,
        factor=factor,
        patience=patience,
        min_delta=min_delta,
        min_lr=min_lr,
        cooldown=cooldown,
        strategy=strategy,
        warmup_epochs=warmup_epochs,
        cycle_epochs=cycle_epochs,
        initial_lr=initial_lr,
        max_lr=max_lr
    )