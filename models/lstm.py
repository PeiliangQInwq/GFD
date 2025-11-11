"""Long Short‑Term Memory classifier for 1D vibration signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for sequential data.

    Args:
        input_size: Dimensionality of input features (default 1 for raw signal).
        hidden_size: Number of hidden units in the LSTM.
        num_layers: Number of LSTM layers.
        num_classes: Number of output classes.
        dropout_prob: Dropout probability on the fully connected layer.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 128,
                 num_layers: int = 1, num_classes: int = 4,
                 dropout_prob: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        # LSTM returns outputs and (hidden_state, cell_state)
        out, (hn, cn) = self.lstm(x)
        # Use the last hidden state
        last_hidden = hn[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits