"""Utility functions for data processing, metrics and training.

This package contains helper functions used throughout the project
for computing evaluation metrics and training PyTorch models.

Modules
-------
metrics
    Functions to compute accuracy, F1 score, expected calibration
    error (ECE), negative log‑likelihood (NLL) and Brier score.

train_utils
    Generic training routines for PyTorch models and wrappers
    around scikit‑learn estimators.

"""

from .metrics import accuracy, f1_macro, expected_calibration_error, negative_log_likelihood, brier_score
from .train_utils import train_sklearn_model, train_torch_model, VibrationDataset, evaluate_torch_model

__all__ = [
    'accuracy',
    'f1_macro',
    'expected_calibration_error',
    'negative_log_likelihood',
    'brier_score',
    'train_sklearn_model',
    'train_torch_model',
    'VibrationDataset',
    'evaluate_torch_model',
]