"""Simple Graph Convolutional Network for down‑sampled vibration signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_knn_graph(num_nodes: int, k: int = 2) -> torch.Tensor:
    """Construct a k‑nearest neighbour adjacency matrix for a 1D chain of nodes.

    Each node is connected to its k nearest neighbours on both sides.  Self
    connections are included.  The resulting adjacency matrix is
    symmetric.

    Args:
        num_nodes: Number of nodes.
        k: Number of neighbours to connect on each side.

    Returns:
        A binary adjacency matrix of shape (num_nodes, num_nodes).
    """
    A = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        A[i, i] = 1.0  # self loop
        for offset in range(1, k + 1):
            if i - offset >= 0:
                A[i, i - offset] = 1.0
                A[i - offset, i] = 1.0
            if i + offset < num_nodes:
                A[i, i + offset] = 1.0
                A[i + offset, i] = 1.0
    return A


class GraphConv(nn.Module):
    """Graph convolution layer for undirected graphs.

    Implements X' = D^{-1/2} A D^{-1/2} X W
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        # X: (batch, num_nodes, in_channels)
        # A: (num_nodes, num_nodes)
        # Compute degree matrix
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        # Perform graph convolution
        X = self.linear(X)
        # Multiply adjacency
        X = torch.matmul(A_norm, X)
        return X


class GCNClassifier(nn.Module):
    """A basic Graph Convolution Network classifier.

    Each input signal is down‑sampled into a set of nodes, each node
    corresponding to the mean amplitude of a segment of the original
    signal.  A fixed k‑nearest neighbour graph is constructed and two
    graph convolution layers are applied, followed by a global average
    pooling and a fully connected classifier.

    Args:
        num_nodes: Number of nodes to down‑sample each signal into.
        k: Number of neighbours for the adjacency matrix.
        hidden_dim: Dimensionality of the graph convolution output.
        num_classes: Number of output classes.
    """

    def __init__(self, num_nodes: int = 50, k: int = 2,
                 hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.k = k
        self.gc1 = GraphConv(in_channels=1, out_channels=hidden_dim)
        self.gc2 = GraphConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Build adjacency matrix once and register as buffer
        A = build_knn_graph(num_nodes, k)
        self.register_buffer('A', A)

    def forward(self, x):
        # x: (batch_size, 1, seq_len) -> downsample to (batch, num_nodes, 1)
        batch_size, _, seq_len = x.size()
        segment_length = seq_len // self.num_nodes
        # Reshape and take mean across segments
        x_reshaped = x.view(batch_size, self.num_nodes, segment_length)
        node_features = x_reshaped.mean(dim=2, keepdim=True)  # shape (batch, num_nodes, 1)
        # Graph conv layers
        X = node_features  # (batch, num_nodes, 1)
        X = F.relu(self.gc1(X, self.A))
        X = F.relu(self.gc2(X, self.A))
        # Global average pooling over nodes
        X = X.mean(dim=1)
        logits = self.fc(X)
        return logits