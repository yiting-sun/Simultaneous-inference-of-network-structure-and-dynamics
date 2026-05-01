"""
Fixed Adjacency + Configurable Message Features
================================================
Same architecture as ``datalimits4dyn_nozscore/model_fixed_adj_ablation_msg.py``
but self-contained (no cross-directory imports), so this file can be used as a
reference implementation in the Step-1 of the struc-only experiment.

Self features: always FULL (6 indices).
Message features: fixed to ``rate_x``.
Adjacency: FIXED (passed in at construction time, never updated).
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch_geometric.nn import MessagePassing


# ----------------------------- helpers --------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    if num_nodes <= 1:
        raise ValueError(f"num_nodes must be > 1, got {num_nodes}")
    src, tgt = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            src.append(j)
            tgt.append(i)
    return torch.tensor([src, tgt], dtype=torch.long)


# ----------------------------- feature indices ------------------------
IDX_XT = 0
IDX_XTM1 = 1
IDX_MA3 = 2
IDX_DIFF3 = 3
IDX_SEASONAL_T = 4
IDX_SEASONAL_RATE12TO1_T = 5
IDX_MA5 = 6
IDX_MA10 = 7
IDX_RATE5_T = 8
IDX_RATE10_T = 9

FULL_SELF_INDICES = [IDX_XT, IDX_XTM1, IDX_MA5, IDX_MA10, IDX_SEASONAL_T, IDX_SEASONAL_RATE12TO1_T]

FEATURE_NAMES_MAP = {
    IDX_XT: "x_t", IDX_XTM1: "x_tm1", IDX_MA3: "ma3", IDX_DIFF3: "diff3",
    IDX_SEASONAL_T: "seasonal", IDX_SEASONAL_RATE12TO1_T: "seasonal_rate",
    IDX_MA5: "ma5", IDX_MA10: "ma10", IDX_RATE5_T: "rate5", IDX_RATE10_T: "rate10",
}

MSG_CONFIGS = {
    "rate_x": {
        "j_indices": [IDX_RATE5_T, IDX_RATE10_T, IDX_XT],
        "diff_indices": [],
        "description": "rates + neighbor incidence x_t_j",
    },
}


def get_msg_in_dim(msg_config_name: str) -> int:
    cfg = MSG_CONFIGS[msg_config_name]
    return len(cfg["j_indices"]) + len(cfg["diff_indices"])


def describe_msg_config(msg_config_name: str) -> str:
    cfg = MSG_CONFIGS[msg_config_name]
    j_names = [FEATURE_NAMES_MAP[i] for i in cfg["j_indices"]]
    diff_names = [f"({FEATURE_NAMES_MAP[i]}_j - {FEATURE_NAMES_MAP[i]}_i)" for i in cfg["diff_indices"]]
    return f"{msg_config_name} ({get_msg_in_dim(msg_config_name)}d): {', '.join(j_names + diff_names)}"


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
    """Fixed adjacency + configurable message features. Self features always FULL."""

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
        if msg_config not in MSG_CONFIGS:
            raise ValueError(f"Unknown msg_config: {msg_config}. Options: {list(MSG_CONFIGS.keys())}")
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
