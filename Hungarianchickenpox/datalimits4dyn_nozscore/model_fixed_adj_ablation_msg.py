"""
Fixed Adjacency Model with Configurable Message Features
==========================================================
Self features: always FULL (all 6).
Message features: configurable (same configs as model_ablation_msg.py).
Adjacency: FIXED (fully_connected or geographic), not learnable.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch_geometric.nn import MessagePassing

from model_ablation_msg import (
    MSG_CONFIGS, FULL_SELF_INDICES, FEATURE_NAMES_MAP,
    get_msg_in_dim, describe_msg_config,
    build_fully_connected_edge_index, count_parameters,
)


class _MLP(Seq):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        layers = [Lin(in_dim, hidden), ReLU()]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.extend([Lin(hidden, hidden), ReLU()])
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.extend([Lin(hidden, hidden), ReLU()])
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(Lin(hidden, out_dim))
        super().__init__(*layers)


class GraphDerivNNFixedAdjMsgAblation(MessagePassing):
    """Fixed adjacency + configurable message features.
    Self features are always FULL (all 6).
    """

    def __init__(
        self,
        n_f: int,
        msg_dim: int,
        ndim: int,
        delt_t: float,
        num_nodes: int,
        fixed_adj: torch.Tensor,
        msg_config: str,
        hidden: int = 64,
        aggr: str = "add",
        flow: str = "source_to_target",
        dropout: float = 0.10,
    ):
        super().__init__(aggr=aggr, flow=flow)
        self.num_nodes = int(num_nodes)
        self.delt_t = float(delt_t)
        self.msg_config = msg_config
        self._msg_cfg = MSG_CONFIGS[msg_config]

        msg_in_dim = get_msg_in_dim(msg_config)
        self_in_dim = len(FULL_SELF_INDICES)

        self.msg_fnc = _MLP(in_dim=msg_in_dim, hidden=hidden, out_dim=msg_dim, dropout=dropout)
        self.node_fnc_self = _MLP(in_dim=self_in_dim, hidden=hidden, out_dim=1, dropout=dropout)

        full_edge_index = build_fully_connected_edge_index(self.num_nodes)
        self.register_buffer("full_edge_index", full_edge_index)

        src = full_edge_index[0]
        tgt = full_edge_index[1]
        edge_weights = fixed_adj[tgt, src].float()
        self.register_buffer("fixed_edge_weights", edge_weights)

    def get_edge_weights(self):
        return self.fixed_edge_weights

    def get_adjacency_matrix(self):
        adj = self.fixed_edge_weights.new_zeros((self.num_nodes, self.num_nodes))
        src = self.full_edge_index[0]
        tgt = self.full_edge_index[1]
        adj[tgt, src] = self.fixed_edge_weights
        return adj

    def expand_edge_weights_for_batch(self, x, edge_weight):
        total_nodes = int(x.shape[0])
        repeat_factor = total_nodes // self.num_nodes
        if repeat_factor == 1:
            return edge_weight
        return edge_weight.repeat(repeat_factor)

    def forward(self, x, edge_index=None, **kwargs):
        active_edge_index = self.full_edge_index if edge_index is None else edge_index
        edge_weight = self.expand_edge_weights_for_batch(x, self.fixed_edge_weights)
        return self.propagate(active_edge_index, x=x, edge_weight=edge_weight)

    def build_phi(self, x_i, x_j):
        parts = []
        for idx in self._msg_cfg["j_indices"]:
            parts.append(x_j[:, idx:idx + 1])
        for idx in self._msg_cfg["diff_indices"]:
            parts.append(x_j[:, idx:idx + 1] - x_i[:, idx:idx + 1])
        return torch.cat(parts, dim=1)

    def message(self, x_i, x_j, edge_weight=None):
        msg = self.msg_fnc(self.build_phi(x_i, x_j))
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1).to(msg.dtype)
        return msg

    def update(self, aggr_out, x=None):
        self_in = torch.cat(
            [x[:, idx:idx + 1] for idx in FULL_SELF_INDICES],
            dim=1,
        )
        self_term = self.node_fnc_self(self_in)
        return self_term + aggr_out

    def prediction(self, g):
        return self.forward(g.x, edge_index=getattr(g, "edge_index", None))

    def data_loss(self, g, square=True):
        pred = self.prediction(g)
        return torch.sum((g.y - pred) ** 2) if square else torch.sum(torch.abs(g.y - pred))

    def loss(self, g, square=True, **kwargs):
        return self.data_loss(g, square=square)
