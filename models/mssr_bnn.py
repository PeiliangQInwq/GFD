"""Multistable Stochastic Resonance Bayesian Neural Network (MSSR‑BNN).

This module implements a simplified version of the proposed MSSR‑BNN
model described in the paper.  To keep the implementation
computationally tractable, the following approximations are used:

1. **Experts** – Three identical convolutional neural networks (CNNs)
   with dropout serve as experts.  During training and inference, the
   dropout layers remain active to approximate sampling from a
   Bayesian posterior.
2. **Gating network** – A lightweight feed‑forward network takes a
   two‑dimensional summary of the input signal (mean and standard
   deviation) and outputs a probability distribution over the experts.
   These probabilities are used as weights to fuse the experts’
   predictions.
3. **Simplified MSSR** – The true multi‑well stochastic resonance
   mechanism is not simulated.  Instead, the experts implicitly
   account for noise augmentation through dropout and random noise
   injection in the data generation phase.

This implementation still captures the essence of combining multiple
stochastic models with Bayesian inference and a gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn import CNNClassifier


class MSSRBNNClassifier(nn.Module):
    """Uncertainty‑aware MSSR‑BNN classifier.

    Args:
        num_experts: Number of expert CNNs to ensemble.
        input_length: Length of the input signal.
        num_classes: Number of fault classes.
    """

    def __init__(self, num_experts: int = 3, input_length: int = 1000,
                 num_classes: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.input_length = input_length
        self.num_classes = num_classes
        # Create expert models
        self.experts = nn.ModuleList([
            CNNClassifier(input_length=input_length, num_classes=num_classes, dropout_prob=0.3)
            for _ in range(num_experts)
        ])
        # Gating network: maps summary statistics (mean & std) to expert weights
        self.gating_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward pass through MSSR‑BNN.

        Args:
            x: Input tensor of shape (batch_size, 1, input_length)

        Returns:
            Weighted sum of expert output probabilities (batch_size, num_classes)
        """
        batch_size = x.size(0)
        # Compute summary statistics for gating: mean and standard deviation across the sequence
        mean = x.mean(dim=2)  # (batch, 1)
        std = x.std(dim=2)    # (batch, 1)
        summary = torch.cat([mean, std], dim=1)  # (batch, 2)
        gating_weights = self.gating_net(summary)  # (batch, num_experts)
        gating_weights = gating_weights.unsqueeze(-1)  # (batch, num_experts, 1)

        # Collect expert predictions
        expert_probs = []
        for expert in self.experts:
            logits = expert(x)  # (batch, num_classes)
            probs = F.softmax(logits, dim=-1)  # (batch, num_classes)
            expert_probs.append(probs.unsqueeze(1))  # (batch, 1, num_classes)
        expert_probs = torch.cat(expert_probs, dim=1)  # (batch, num_experts, num_classes)

        # Weighted average of expert predictions
        # gating_weights: (batch, num_experts, 1)
        # expert_probs:  (batch, num_experts, num_classes)
        weighted_probs = (gating_weights * expert_probs).sum(dim=1)  # (batch, num_classes)
        return weighted_probs

    def predict(self, x, n_samples: int = 10):
        """Generate predictions with Monte Carlo sampling.

        During evaluation we perform multiple stochastic forward passes
        (with dropout enabled) and average the outputs.  This approximates
        Bayesian inference and produces a more calibrated prediction.

        Args:
            x: Input tensor of shape (batch_size, 1, input_length)
            n_samples: Number of Monte Carlo samples to draw.

        Returns:
            Averaged probability predictions of shape (batch_size, num_classes).
        """
        self.train()  # keep dropout active
        probs_list = []
        for _ in range(n_samples):
            probs = self.forward(x)
            probs_list.append(probs.unsqueeze(0))  # (1, batch, num_classes)
        probs_stack = torch.cat(probs_list, dim=0)  # (n_samples, batch, num_classes)
        return probs_stack.mean(dim=0)