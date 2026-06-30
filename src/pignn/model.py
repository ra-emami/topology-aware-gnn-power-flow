"""Topology-aware GNN surrogate for AC power flow."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv


class ProxySolverGNN(torch.nn.Module):
    """Three TAGConv layers + linear head, predicting standardized ``[V, theta]``.

    Each TAGConv layer aggregates a ``K``-hop electrical neighborhood; the per-unit
    admittance edge weights modulate every message when ``use_edge_weights`` is set.
    """

    def __init__(self, in_channels: int = 4, hidden=(128, 64, 32),
                 ks=(3, 3, 2), use_edge_weights: bool = True):
        super().__init__()
        self.use_edge_weights = use_edge_weights
        h1, h2, h3 = hidden
        k1, k2, k3 = ks
        self.conv1 = TAGConv(in_channels, h1, K=k1)
        self.conv2 = TAGConv(h1, h2, K=k2)
        self.conv3 = TAGConv(h2, h3, K=k3)
        self.lin = torch.nn.Linear(h3, 2)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        ew = data.edge_weight if self.use_edge_weights else None
        x = F.leaky_relu(self.conv1(x, ei, ew))
        x = F.leaky_relu(self.conv2(x, ei, ew))
        x = F.leaky_relu(self.conv3(x, ei, ew))
        return self.lin(x)
