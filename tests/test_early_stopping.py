"""
Tests for early stopping functionality.
"""
import pytest
import numpy as np
import pandas as pd

from freamon.modeling.early_stopping import (
    EarlyStopping, 
    AdaptiveLearningRateScheduler,
    get_early_stopping_callback,
    get_lr_scheduler
)


class TestEarlyStopping:
    """Test cases for the EarlyStopping class."""
    
    def test_init(self):
        """Test initialization of EarlyStopping."""
        es = EarlyStopping(monitor='val_loss')
        assert es.monitor == 'val_loss'
        assert es.mode == 'min'
        assert es.patience == 10
        assert es.min_delta == 0.0
        assert es.restore_best_weights is True
        assert es.baseline is None
        assert es.stopping_threshold is None
        assert es.divergence_threshold is None
        assert es.best_score == float('inf')
        assert es.wait == 0
        assert es.stop_training is False
    
    def test_check_improvement_min_mode(self):
        """Test improvement checking in min mode."""
        es = EarlyStopping(monitor='val_loss', mode='min')
        es.best_score = 0.5
        es.min_delta = 0.01
        
        # Better value: 0.45 < (0.5 - 0.01)
        assert es.check_improvement(0.45) is True
        
        # Equal value: 0.5 == 0.5 (not better by min_delta)
        assert es.check_improvement(0.5) is False
        
        # Slightly better but not by min_delta: 0.49 < 0.5 but 0.49 !< (0.5 - 0.01)
        assert es.check_improvement(0.49) is False
        
        # Worse value: 0.6 > 0.5
        assert es.check_improvement(0.6) is False
    
    def test_check_improvement_max_mode(self):
        """Test improvement checking in max mode."""
        es = EarlyStopping(monitor='val_accuracy', mode='max')
        es.best_score = 0.8
        es.min_delta = 0.01
        
        # Better value: 0.85 > (0.8 + 0.01)
        assert es.check_improvement(0.85) is True
        
        # Equal value: 0.8 == 0.8 (not better by min_delta)
        assert es.check_improvement(0.8) is False
        
        # Slightly better but not by min_delta: 0.805 > 0.8 but 0.805 !> (0.8 + 0.01)
        assert es.check_improvement(0.805) is False
        
        # Worse value: 0.7 < 0.8
        assert es.check_improvement(0.7) is False
    
    def test_call_with_improvement(self):
        """Test __call__ with improvement."""
        es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
        
        # Initial call with improvement
        result = es(epoch=0, logs={'val_loss': 0.5})
        assert result is False  # Don't stop
        assert es.best_score == 0.5
        assert es.wait == 0
        
        # Second call with improvement
        result = es(epoch=1, logs={'val_loss': 0.4})
        assert result is False  # Don't stop
        assert es.best_score == 0.4
        assert es.wait == 0
    
    def test_call_without_improvement(self):
        """Test __call__ without improvement."""
        es = EarlyStopping(monitor='val_loss', mode='min', patience=2)
        
        # Initial call
        result = es(epoch=0, logs={'val_loss': 0.5})
        assert result is False
        assert es.best_score == 0.5
        assert es.wait == 0
        
        # Second call with no improvement
        result = es(epoch=1, logs={'val_loss': 0.6})
        assert result is False
        assert es.best_score == 0.5
        assert es.wait == 1
        
        # Third call with no improvement - should trigger early stopping
        result = es(epoch=2, logs={'val_loss': 0.55})
        assert result is True
        assert es.best_score == 0.5
        assert es.wait == 2
        assert es.stopped_epoch == 2
        assert es.stop_training is True
    
    def test_stopping_threshold(self):
        """Test stopping threshold."""
        es = EarlyStopping(
            monitor='val_accuracy', 
            mode='max', 
            stopping_threshold=0.9
        )
        
        # First call, below threshold
        result = es(epoch=0, logs={'val_accuracy': 0.85})
        assert result is False
        
        # Second call, reaches threshold, should stop
        result = es(epoch=1, logs={'val_accuracy': 0.92})
        assert result is True
        assert es.stopping_threshold_reached is True
    
    def test_divergence_threshold(self):
        """Test divergence threshold."""
        es = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            divergence_threshold=0.2
        )
        
        # First call, establish baseline
        result = es(epoch=0, logs={'val_loss': 0.5})
        assert result is False
        
        # Second call, minor increase, shouldn't stop
        result = es(epoch=1, logs={'val_loss': 0.6})
        assert result is False
        
        # Third call, major increase (diverged), should stop
        result = es(epoch=2, logs={'val_loss': 0.75})  # 0.75 - 0.5 > 0.2
        assert result is True
        assert es.diverged is True
    
    def test_restore_best_weights(self):
        """Test restore_best_weights functionality."""
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.weights = 'initial'
        
        model = MockModel()
        
        # Create early stopping with restore_best_weights=True
        es = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        # Override weight handling methods for testing
        es._get_weights = lambda m: m.weights
        es._set_weights = lambda m, w: setattr(m, 'weights', w)
        
        # Initial call
        result = es(epoch=0, logs={'val_loss': 0.5}, model=model)
        model.weights = 'best'  # Simulate model improvement
        
        # Second call with improvement
        result = es(epoch=1, logs={'val_loss': 0.4}, model=model)
        assert es.best_weights == 'best'
        model.weights = 'worse'  # Simulate model degradation
        
        # Calls without improvement to trigger early stopping
        result = es(epoch=2, logs={'val_loss': 0.45}, model=model)
        result = es(epoch=3, logs={'val_loss': 0.5}, model=model)
        
        # Check that weights were restored
        assert model.weights == 'best'


class TestAdaptiveLearningRateScheduler:
    """Test cases for the AdaptiveLearningRateScheduler class."""
    
    def test_init(self):
        """Test initialization of AdaptiveLearningRateScheduler."""
        scheduler = AdaptiveLearningRateScheduler(monitor='val_loss')
        assert scheduler.monitor == 'val_loss'
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5
        assert scheduler.min_delta == 1e-4
        assert scheduler.min_lr == 1e-6
        assert scheduler.cooldown == 0
        assert scheduler.strategy == 'reduce_on_plateau'
        assert scheduler.current_lr == 0.1
    
    def test_reduce_on_plateau(self):
        """Test reduce_on_plateau strategy."""
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.lr = 0.1
        
        model = MockModel()
        
        # Create scheduler
        scheduler = AdaptiveLearningRateScheduler(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=2,
            initial_lr=0.1
        )
        
        # Override _set_learning_rate for testing
        scheduler._set_learning_rate = lambda m, o, lr: setattr(m, 'lr', lr)
        
        # Initial call
        lr = scheduler(epoch=0, logs={'val_loss': 0.5}, model=model)
        assert lr == 0.1
        
        # Second call with improvement
        lr = scheduler(epoch=1, logs={'val_loss': 0.4}, model=model)
        assert lr == 0.1
        
        # Third call without improvement
        lr = scheduler(epoch=2, logs={'val_loss': 0.45}, model=model)
        assert lr == 0.1
        assert scheduler.wait == 1
        
        # Fourth call without improvement - should reduce lr
        lr = scheduler(epoch=3, logs={'val_loss': 0.42}, model=model)
        assert lr == 0.05
        assert model.lr == 0.05
        assert scheduler.wait == 0
    
    def test_cosine_annealing(self):
        """Test cosine_annealing strategy."""
        scheduler = AdaptiveLearningRateScheduler(
            monitor='val_loss',
            strategy='cosine_annealing',
            initial_lr=0.1,
            min_lr=0.001,
            warmup_epochs=2,
            cycle_epochs=10
        )
        
        # Override _set_learning_rate for testing
        scheduler._set_learning_rate = lambda m, o, lr: None
        
        # Warmup phase
        lr1 = scheduler(epoch=0, logs={}, model=None)
        lr2 = scheduler(epoch=1, logs={}, model=None)
        
        # Cosine phase
        lrs = [scheduler(epoch=i, logs={}, model=None) for i in range(2, 12)]
        
        # Check warmup
        assert lr1 < lr2  # Learning rate should increase during warmup
        
        # Check cosine pattern
        assert lrs[0] > lrs[5]  # First half of cycle: LR decreases
        assert lrs[-1] < lrs[0]  # End of cycle: LR is lower than start
        assert abs(lrs[-1] - 0.001) < 0.0001  # End of cycle approaches min_lr
    
    def test_exponential_decay(self):
        """Test exponential_decay strategy."""
        scheduler = AdaptiveLearningRateScheduler(
            monitor='val_loss',
            strategy='exponential_decay',
            initial_lr=0.1,
            factor=0.5,
            patience=3  # Decay every 3 epochs
        )
        
        # Override _set_learning_rate for testing
        scheduler._set_learning_rate = lambda m, o, lr: None
        
        # Initial epochs (no decay)
        lr1 = scheduler(epoch=0, logs={}, model=None)
        lr2 = scheduler(epoch=2, logs={}, model=None)
        
        # Decay point
        lr3 = scheduler(epoch=3, logs={}, model=None)
        
        # Another decay point
        lr4 = scheduler(epoch=6, logs={}, model=None)
        
        assert lr1 == 0.1
        assert lr2 == 0.1  # No change before patience epochs
        assert lr3 == 0.05  # Decayed at epoch=3
        assert lr4 == 0.025  # Decayed again at epoch=6
    
    def test_one_cycle(self):
        """Test one_cycle strategy."""
        scheduler = AdaptiveLearningRateScheduler(
            monitor='val_loss',
            strategy='one_cycle',
            initial_lr=0.01,
            max_lr=0.1,
            min_lr=0.001,
            cycle_epochs=10
        )
        
        # Override _set_learning_rate for testing
        scheduler._set_learning_rate = lambda m, o, lr: None
        
        # Sample across the cycle
        lrs = [scheduler(epoch=i, logs={}, model=None) for i in range(10)]
        
        # Check one-cycle pattern
        assert lrs[0] < lrs[2]  # Learning rate increases in first phase
        assert lrs[2] < lrs[4]  # Continues to increase
        assert lrs[4] > lrs[7]  # Decreases in second phase
        assert lrs[7] > lrs[9]  # Continues to decrease
        assert abs(lrs[4] - 0.1) < 0.01  # Should approach max_lr
        assert lrs[9] > 0.001  # Should approach min_lr


def test_get_early_stopping_callback():
    """Test the helper function for creating early stopping callbacks."""
    es = get_early_stopping_callback(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        min_delta=0.01
    )
    
    assert isinstance(es, EarlyStopping)
    assert es.monitor == 'val_accuracy'
    assert es.mode == 'max'
    assert es.patience == 5
    assert es.min_delta == 0.01


def test_get_lr_scheduler():
    """Test the helper function for creating learning rate schedulers."""
    scheduler = get_lr_scheduler(
        monitor='val_loss',
        strategy='cosine_annealing',
        initial_lr=0.2,
        min_lr=0.001
    )
    
    assert isinstance(scheduler, AdaptiveLearningRateScheduler)
    assert scheduler.monitor == 'val_loss'
    assert scheduler.strategy == 'cosine_annealing'
    assert scheduler.initial_lr == 0.2
    assert scheduler.min_lr == 0.001