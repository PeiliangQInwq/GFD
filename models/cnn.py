"""Simple convolutional neural network for 1D vibration signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """A basic 1D CNN with two convolutional layers followed by a classifier.

    Args:
        input_length: Length of the input signal (default 1000).
        num_classes: Number of output classes (default 4).
        dropout_prob: Dropout probability for regularisation.
    """

    def __init__(self, input_length: int = 1000, num_classes: int = 4, dropout_prob: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=dropout_prob)
        # Compute the feature dimension after pooling
        # We'll use two max pooling layers with kernel size 2
        pooled_length = input_length // 4  # after two poolings of stride 2
        self.fc = nn.Linear(64 * pooled_length, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits