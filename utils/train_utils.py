"""Utility functions for training and evaluating models.

This module provides helper routines to train PyTorch models and
scikit‑learn classifiers on vibration signal datasets.  It defines
datasets, training loops and evaluation functions that compute
predictions and probabilistic outputs.
"""

from typing import Tuple, Optional, Callable, Iterable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

from .metrics import (
    accuracy,
    f1_macro,
    expected_calibration_error,
    negative_log_likelihood,
    brier_score,
)


class VibrationDataset(Dataset):
    """Torch dataset for vibration signals.

    Each sample consists of a one‑dimensional signal and a label.
    Signals are expected to be 1D NumPy arrays of length ``seq_len``.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, seq_len) containing the signals.
    y : np.ndarray
        Array of shape (n_samples,) containing integer labels.
    dtype : torch.dtype, optional
        Data type for converting signals to tensors.  Default is
        ``torch.float32``.
    device : torch.device, optional
        Device on which to place tensors.  Default is CPU.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None):
        super().__init__()
        assert len(X) == len(y), "Features and labels must have the same length"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.dtype = dtype
        self.device = device

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        signal = torch.as_tensor(self.X[idx], dtype=self.dtype)
        label = torch.as_tensor(self.y[idx], dtype=torch.long)
        return signal, label


def train_sklearn_model(model: BaseEstimator,
                        X_train: np.ndarray,
                        y_train: np.ndarray) -> BaseEstimator:
    """Train a scikit‑learn classifier on flattened vibration signals.

    For scikit‑learn models like SVM or RandomForest, signals must
    first be reshaped from (n_samples, seq_len) to (n_samples, seq_len)
    (flattened).  After fitting, the model can be used to predict
    class labels.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The scikit‑learn model to fit.
    X_train : np.ndarray
        Training signals of shape (n_samples, seq_len).
    y_train : np.ndarray
        Training labels of shape (n_samples,).

    Returns
    -------
    sklearn.base.BaseEstimator
        The fitted model.
    """
    # Flatten signals if they are not already 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    model.fit(X_train_flat, y_train)
    return model


def train_torch_model(model: torch.nn.Module,
                      train_loader: DataLoader,
                      num_epochs: int = 10,
                      learning_rate: float = 1e-3,
                      device: Optional[torch.device] = None,
                      weight_decay: float = 0.0) -> torch.nn.Module:
    """Train a PyTorch model on vibration signals.

    This function performs a standard training loop using cross‑
    entropy loss and the Adam optimiser.  It does not implement
    early stopping or learning rate scheduling.  After training,
    the model is returned.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    train_loader : torch.utils.data.DataLoader
        Dataloader providing batches of (signal, label) tuples.
    num_epochs : int, optional
        Number of training epochs.  Default is 10.
    learning_rate : float, optional
        Initial learning rate for Adam optimiser.  Default is 1e‑3.
    device : torch.device, optional
        Device to train on.  If None, uses CUDA if available.
    weight_decay : float, optional
        Weight decay (L2 regularisation) for optimiser.

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for signals, labels in train_loader:
            # signals: (batch, seq_len)
            signals = signals.to(device)
            labels = labels.to(device)
            # Reshape signals for different model types
            if isinstance(model, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)):
                # Should not be needed; we handle in dataset or wrapper
                inputs = signals.unsqueeze(-1)
            else:
                # For CNN/ResNet/MSSRBNN/GCN expect shape (batch, 1, seq_len)
                if signals.ndim == 2:
                    inputs = signals.unsqueeze(1)
                else:
                    inputs = signals
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # Optionally print training progress
        # print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    return model


def evaluate_torch_model(model: torch.nn.Module,
                         data_loader: DataLoader,
                         device: Optional[torch.device] = None,
                         mc_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a PyTorch model and return predictions and probabilities.

    The function switches the model into evaluation mode unless it
    defines a ``predict`` method for Monte Carlo sampling.  For
    Bayesian models with MC dropout, multiple stochastic forward
    passes are averaged.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model to evaluate.
    data_loader : torch.utils.data.DataLoader
        Loader providing batches of (signal, label).
    device : torch.device, optional
        Device to use for evaluation.  Default uses CUDA if
        available.
    mc_samples : int, optional
        Number of Monte Carlo samples for Bayesian models.  Ignored
        for deterministic models.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple (preds, probs) where preds are the predicted labels
        and probs are the predicted class probabilities for each sample.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            # Reshape input accordingly
            # Determine if model expects 3D input (batch, 1, seq_len) or 2D (batch, seq_len, 1)
            if hasattr(model, 'lstm') or hasattr(model, 'embedding'):
                # LSTM/Transformer expects (batch, seq_len, input_size)
                inputs = signals.unsqueeze(-1)
            else:
                # CNN/ResNet/GCN/MSSRBNN expects (batch, 1, seq_len)
                inputs = signals.unsqueeze(1)
            # Compute probabilities
            if hasattr(model, 'predict'):
                # Bayesian model with MC dropout
                probs = model.predict(inputs, n_samples=mc_samples)
            else:
                logits = model(inputs)
                probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    probs_array = np.concatenate(all_probs, axis=0)
    preds_array = np.concatenate(all_preds, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)
    return preds_array, probs_array, targets_array