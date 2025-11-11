"""Main script to train and evaluate models for gearbox fault diagnosis.

This script provides a command‑line interface to train various models
defined in the project on the synthetic vibration dataset.  It
supports both scikit‑learn classifiers (SVM, RandomForest) and
PyTorch deep learning models (CNN, LSTM, Transformer, GCN, ResNet,
MSSR‑BNN).  After training, the script reports a suite of metrics
including accuracy, macro F1 score, expected calibration error (ECE),
negative log‑likelihood (NLL) and Brier score.

Usage examples
--------------

Train a CNN for 10 epochs with batch size 64 on the default dataset:

.. code-block:: bash

    python train.py --model cnn --epochs 10 --batch-size 64

Train a support vector machine on the same dataset:

.. code-block:: bash

    python train.py --model svm

If the dataset ``dataset.npz`` does not exist in the ``data`` folder,
the script will attempt to generate it by invoking the data
generation script.
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from models import (
    CNNClassifier,
    LSTMClassifier,
    TransformerClassifier,
    GCNClassifier,
    ResNet1D,
    MSSRBNNClassifier,
)

from utils import (
    VibrationDataset,
    train_sklearn_model,
    train_torch_model,
    evaluate_torch_model,
    accuracy,
    f1_macro,
    expected_calibration_error,
    negative_log_likelihood,
    brier_score,
)


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the vibration dataset from a .npz file.

    Parameters
    ----------
    path : str
        Path to the .npz file containing ``X_train``, ``y_train``,
        ``X_test`` and ``y_test`` arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The training signals, training labels, test signals and test labels.
    """
    data = np.load(path)
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def generate_dataset_if_missing() -> str:
    """Ensure the synthetic dataset exists; generate it if necessary.

    The dataset is expected to reside at ``gearbox_fault_diagnosis/data/dataset.npz``.
    If the file is missing, this function calls the data generation
    script to create it.

    Returns
    -------
    str
        Path to the generated dataset file.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    dataset_path = os.path.join(data_dir, 'dataset.npz')
    if not os.path.isfile(dataset_path):
        print("Dataset not found. Generating synthetic data...")
        import subprocess
        script_path = os.path.join(data_dir, 'generate_data.py')
        subprocess.run(['python', script_path], check=True)
    return dataset_path


def run_single_model(model_name: str,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     seq_len: int,
                     batch_size: int,
                     epochs: int,
                     mc_samples: int) -> None:
    """Train and evaluate a single model and print metrics.

    Parameters
    ----------
    model_name : str
        Name of the model to train (svm, rf, cnn, lstm, transformer,
        gcn, resnet or mssr_bnn).
    X_train, y_train, X_test, y_test : np.ndarray
        Training and testing data.
    seq_len : int
        Length of each input signal.
    batch_size : int
        Batch size for neural network training.
    epochs : int
        Number of epochs to train neural networks.
    mc_samples : int
        Number of Monte Carlo samples for Bayesian evaluation.
    """
    unique_classes = len(np.unique(y_train))
    if model_name in ['svm', 'rf']:
        # Flatten signals
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        if model_name == 'svm':
            model = SVC(kernel='rbf', C=1.5, gamma=0.01, probability=True)
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=0)
        start_time = time.time()
        model = train_sklearn_model(model, X_train, y_train)
        train_time = time.time() - start_time
        probs = model.predict_proba(X_test_flat)
        preds = probs.argmax(axis=1)
        targets = y_test
    else:
        # Create PyTorch datasets and loaders
        train_dataset = VibrationDataset(X_train, y_train)
        test_dataset = VibrationDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Instantiate model
        if model_name == 'cnn':
            model = CNNClassifier(input_length=seq_len, num_classes=unique_classes)
        elif model_name == 'lstm':
            model = LSTMClassifier(input_size=1, hidden_size=128, num_layers=1,
                                   num_classes=unique_classes)
        elif model_name == 'transformer':
            model = TransformerClassifier(input_size=1, d_model=64, nhead=4,
                                         num_layers=2, dim_feedforward=128,
                                         num_classes=unique_classes,
                                         dropout=0.1, max_len=seq_len)
        elif model_name == 'gcn':
            num_nodes = 50
            model = GCNClassifier(num_nodes=num_nodes, k=2, hidden_dim=32,
                                  num_classes=unique_classes)
        elif model_name == 'resnet':
            model = ResNet1D(num_classes=unique_classes)
        elif model_name == 'mssr_bnn':
            model = MSSRBNNClassifier(num_experts=3, input_length=seq_len,
                                      num_classes=unique_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        # Train
        start_time = time.time()
        model = train_torch_model(model, train_loader, num_epochs=epochs)
        train_time = time.time() - start_time
        # Evaluate
        preds, probs, targets = evaluate_torch_model(model, test_loader,
                                                     mc_samples=mc_samples)
    # Compute metrics
    acc = accuracy(targets, preds)
    f1 = f1_macro(targets, preds)
    ece = expected_calibration_error(probs, targets)
    nll = negative_log_likelihood(probs, targets)
    bs = brier_score(probs, targets)
    # Print results
    print(f"=== Model: {model_name} ===")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print(f"Negative Log-Likelihood: {nll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate gearbox fault diagnosis models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['svm', 'rf', 'cnn', 'lstm', 'transformer', 'gcn', 'resnet', 'mssr_bnn', 'all'],
                        help='Which model to train or "all" to evaluate all models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs for neural networks (ignored for sklearn models)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for neural network training')
    parser.add_argument('--mc-samples', type=int, default=10,
                        help='Number of Monte Carlo samples for Bayesian models during evaluation')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to .npz dataset file; if not provided, use the generated synthetic dataset')
    args = parser.parse_args()
    # Ensure dataset exists
    dataset_path = args.dataset or generate_dataset_if_missing()
    X_train, y_train, X_test, y_test = load_dataset(dataset_path)
    seq_len = X_train.shape[1]
    if args.model == 'all':
        model_names = ['svm', 'rf', 'cnn', 'lstm', 'transformer', 'gcn', 'resnet', 'mssr_bnn']
        for m in model_names:
            run_single_model(m, X_train, y_train, X_test, y_test, seq_len,
                             batch_size=args.batch_size, epochs=args.epochs,
                             mc_samples=args.mc_samples)
    else:
        run_single_model(args.model, X_train, y_train, X_test, y_test, seq_len,
                         batch_size=args.batch_size, epochs=args.epochs,
                         mc_samples=args.mc_samples)


if __name__ == '__main__':
    main()