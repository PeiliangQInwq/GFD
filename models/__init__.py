"""Model registry.

This module exposes classes for all implemented models.  Each model
inherits from ``torch.nn.Module`` and can be imported directly from
``models``.
"""

from .cnn import CNNClassifier
from .lstm import LSTMClassifier
from .transformer import TransformerClassifier
from .resnet import ResNet1D
from .gcn import GCNClassifier
from .mssr_bnn import MSSRBNNClassifier

__all__ = [
    'CNNClassifier',
    'LSTMClassifier',
    'TransformerClassifier',
    'ResNet1D',
    'GCNClassifier',
    'MSSRBNNClassifier',
]