"""
Model / utilities for Lorenz subgraph inference.

Classes:
    Graph_deriv_NN          — known-structure: fixed adjacency, learn dynamics.
    tau_soft_regu_deriv_MP  — simultaneous: learnable soft adjacency + dynamics.

Edge-index convention (PyG flow='source_to_target'):
    edge_index[0] = source ,  edge_index[1] = target
    message(x_i, x_j):  x_i = x[edge_index[1]] = target
                        x_j = x[edge_index[0]] = source
    msg_fnc input (per reference): concat( x_i_dim0 , x_j_dim0 )  i.e. [x_target, x_source]
"""

import numpy as np
import torch
from torch import nn
from torch.functional import F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_edge_index(Adj):
    """Build edge_index = np.where(Adj). Pass in Adj that is indexed as
    Adj[src, tgt] (col = target) so the resulting edge_index has
    edge_index[0] = source, edge_index[1] = target.

    Our saved binary adjacency has row = destination, col = source
    (A[i,j]=1 means j->i), so callers should pass `objectAij.T`.
    """
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index


def calculate_auc(objectAij, weights):
    """AUC / AUPRC for simul inference: compare learned soft adjacency to true A.
    Same as HR: soft_weight[k] approximates objectAij[target, source]
    (i.e. entry objectAij.T after dropping the diagonal).
    """
    weights = F.softmax(weights, dim=1)[:, 0].view(-1, 1)
    weights = weights.cpu()
    objectAij = objectAij.T

    mask = np.ones_like(objectAij, dtype=bool)
    np.fill_diagonal(mask, 0)
    objectAij_no_dig = objectAij[mask].reshape(-1, 1)

    fpr, tpr, _ = roc_curve(objectAij_no_dig, weights.detach().numpy())
    auc_value = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(objectAij_no_dig, weights.detach().numpy())
    auprc_value = auc(recall, precision)

    return auc_value, auprc_value


def uniform_subsample_indices(length, num_samples):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples >= length:
        return np.arange(length, dtype=np.int64)
    if num_samples == 1:
        return np.array([length // 2], dtype=np.int64)
    idx = np.linspace(0, length - 1, num=num_samples)
    idx = np.unique(np.round(idx).astype(np.int64))
    if idx.size < num_samples:
        all_idx = set(range(length))
        used = set(idx.tolist())
        remaining = sorted(all_idx - used)
        need = num_samples - idx.size
        idx = np.sort(np.concatenate([idx, np.array(remaining[:need], dtype=np.int64)]))
    return idx


def safe_corrcoef(x, y):
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    if x.size == 0 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if not np.allclose(ss_tot, 0) else np.nan


def evaluate_on_set(model, X_set, y_set, edge_index, device):
    model.eval()
    preds, trues = [], []
    edge_index_dev = edge_index.to(device)
    with torch.no_grad():
        for t in range(X_set.shape[0]):
            x = X_set[t].to(device)
            pred = model(x, edge_index=edge_index_dev)
            preds.append(pred.cpu().numpy())
            trues.append(y_set[t].numpy())
    preds = np.array(preds)
    trues = np.array(trues)

    mse = float(np.mean((trues - preds) ** 2))
    mae = float(np.mean(np.abs(trues - preds)))
    r2 = compute_r2(trues.reshape(-1), preds.reshape(-1))
    corr = safe_corrcoef(trues.reshape(-1), preds.reshape(-1))

    n_nodes = preds.shape[1]
    node_r2 = []
    for i in range(n_nodes):
        yi, yhi = trues[:, i, :].reshape(-1), preds[:, i, :].reshape(-1)
        node_r2.append(compute_r2(yi, yhi))

    return {
        "mse": mse, "mae": mae, "r2": r2, "corr": corr,
        "node_r2": node_r2,
        "node_r2_mean": float(np.nanmean(node_r2)),
        "node_r2_median": float(np.nanmedian(node_r2)),
        "node_r2_min": float(np.nanmin(node_r2)),
        "node_r2_max": float(np.nanmax(node_r2)),
    }


class GraphDataset(Dataset):
    def __init__(self, X_train, y_train, edge_index):
        if not isinstance(X_train, torch.Tensor):
            self.X_train = torch.FloatTensor(X_train)
        else:
            self.X_train = X_train
        if not isinstance(y_train, torch.Tensor):
            self.y_train = torch.FloatTensor(y_train)
        else:
            self.y_train = y_train
        self.edge_index = edge_index
        self.X_train = self.X_train.share_memory_()
        self.y_train = self.y_train.share_memory_()
        self.edge_index = self.edge_index.share_memory_()

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return Data(x=self.X_train[idx], edge_index=self.edge_index, y=self.y_train[idx])


# =============================================================================
# Simultaneous inference: learnable soft adjacency + L1 regularization
# =============================================================================
class tau_soft_regu_deriv_MP(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam,
                 hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, msg_dim)
        )
        self.node_fnc_x = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        self.node_fnc_y = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        self.node_fnc_z = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        self._init_weights()
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam
        self.weights = Parameter(torch.Tensor(edge_num, 2), requires_grad=True)
        torch.nn.init.normal_(self.weights, 0, 0.1)

    def _init_weights(self):
        for module in [self.msg_fnc, self.node_fnc_x, self.node_fnc_y, self.node_fnc_z]:
            for layer in module:
                if isinstance(layer, Lin):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i = x[edge_index[1]] = target ; x_j = x[edge_index[0]] = source
        tmp = torch.cat([x_i[:, 0], x_j[:, 0]])
        tmp = tmp.reshape(2, -1).t()
        self.soft_weights = F.softmax(self.weights / self.tau, dim=1)[:, 0].view(-1, 1)
        Len = int(x_i[:, 0].shape[0]) / int(self.soft_weights.shape[0])
        w = self.soft_weights.repeat(int(Len), 1)
        msg = self.msg_fnc(tmp) * w
        return msg

    def update(self, aggr_out, x=None):
        if self.ndim == 1:
            return torch.cat([self.node_fnc_x(x) + aggr_out], dim=1)
        elif self.ndim == 2:
            return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)
        elif self.ndim == 3:
            return torch.cat([self.node_fnc_x(x) + aggr_out,
                              self.node_fnc_y(x),
                              self.node_fnc_z(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g))**2) + self.lam * torch.sum(self.soft_weights)
        else:
            return torch.sum(torch.abs(g.y - self.prediction(g))) + self.lam * torch.sum(self.soft_weights)

    def update_tau(self, newtau):
        self.tau = newtau


# =============================================================================
# Known-structure: fixed adjacency, learn dynamics only
# =============================================================================
class Graph_deriv_NN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, msg_dim)
        )
        for layer in self.msg_fnc:
            if isinstance(layer, Lin):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.node_fnc_x = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        for layer in self.node_fnc_x:
            if isinstance(layer, Lin):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.node_fnc_y = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        for layer in self.node_fnc_y:
            if isinstance(layer, Lin):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.node_fnc_z = Seq(
            Lin(n_f, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, 1)
        )
        for layer in self.node_fnc_z:
            if isinstance(layer, Lin):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        self.ndim = ndim
        self.delt_t = delt_t

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        if self.ndim == 1:
            tmp = torch.cat([x_i, x_j], dim=1)
        else:
            tmp = torch.cat([x_i[:, 0], x_j[:, 0]])
            tmp = tmp.reshape(2, -1).t()
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        if self.ndim == 1:
            return torch.cat([self.node_fnc_x(x) + aggr_out], dim=1)
        elif self.ndim == 2:
            return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)
        elif self.ndim == 3:
            return torch.cat([self.node_fnc_x(x) + aggr_out,
                              self.node_fnc_y(x),
                              self.node_fnc_z(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g))**2)
        else:
            return torch.sum(torch.abs(g.y - self.prediction(g)))
