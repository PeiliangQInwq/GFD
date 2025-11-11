"""Transformer encoder classifier for 1D vibration signals."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer.

    Adds sinusoidal positional information to the inputs to allow the
    model to capture order information in sequences.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class TransformerClassifier(nn.Module):
    """A minimal Transformer encoder classifier for 1D signals.

    Args:
        input_size: Dimensionality of each input element (default 1).
        d_model: Dimension of embedding space.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Dimension of feedforward network in Transformer.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(self, input_size: int = 1, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, num_classes: int = 4,
                 dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits