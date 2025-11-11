"""Evaluation metrics for classification models.

This module implements a number of commonly used metrics for
multi‑class classification tasks in fault diagnosis.  The
implementations here rely on NumPy and scikit‑learn where
appropriate.  Each function accepts NumPy arrays as input and
returns a scalar value.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        The proportion of correct predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    correct = (y_true == y_pred).sum()
    return correct / len(y_true)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the macro‑averaged F1 score.

    The F1 score is the harmonic mean of precision and recall.  For
    multi‑class problems, the macro average calculates the metric
    independently for each class and then takes the unweighted mean.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        The macro‑averaged F1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return f1_score(y_true, y_pred, average='macro')


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE) for probabilistic predictions.

    The ECE measures the discrepancy between confidence estimates and
    empirical accuracies.  Predictions are grouped into a fixed
    number of bins based on their maximum predicted probability.  For
    each bin, the average confidence and accuracy are computed.  The
    ECE is the weighted sum of the absolute differences between
    confidence and accuracy across all bins.

    Parameters
    ----------
    probs : np.ndarray
        Array of predicted class probabilities of shape (n_samples, n_classes).
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    n_bins : int, optional
        Number of bins to use when computing the ECE.  Default is 15.

    Returns
    -------
    float
        The expected calibration error.  Smaller values indicate better calibration.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    assert probs.ndim == 2, "probs must be a 2D array of shape (n_samples, n_classes)"
    assert probs.shape[0] == y_true.shape[0], "probs and y_true must have the same number of samples"
    # Predicted class and its confidence
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    # Define bin edges from 0 to 1
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Indices of samples where confidence falls into bin
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            # Average confidence and accuracy within bin
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(predictions[in_bin] == y_true[in_bin])
            ece += prop_in_bin * np.abs(avg_acc - avg_conf)
    return ece


def negative_log_likelihood(
    probs: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute the negative log‑likelihood (cross entropy) for predictions.

    For each sample, the log probability of the true class is
    accumulated.  A small epsilon is added to probabilities to avoid
    taking the log of zero.

    Parameters
    ----------
    probs : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    eps : float, optional
        Small value added to probabilities for numerical stability.

    Returns
    -------
    float
        The average negative log‑likelihood across all samples.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, eps, 1.0)
    # Extract probabilities of the true class for each sample
    true_probs = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(true_probs)))


def brier_score(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute the Brier score for probabilistic multi‑class predictions.

    The Brier score is the mean squared difference between the
    predicted probabilities and the one‑hot encoded true labels.  It
    captures both calibration and discrimination: lower scores are
    better.

    Parameters
    ----------
    probs : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).

    Returns
    -------
    float
        The average Brier score across all samples.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    n_samples, n_classes = probs.shape
    # One‑hot encode true labels
    true_one_hot = np.zeros_like(probs)
    true_one_hot[np.arange(n_samples), y_true] = 1.0
    # Compute squared error and average
    sq_error = (probs - true_one_hot) ** 2
    return float(np.mean(sq_error))