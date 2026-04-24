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
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_edge_index(Adj):
    """Build edge_index from adjacency matrix. Each row = target, each col = source."""
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index


def build_fully_connected_edge_index(num_nodes):
    """Fully connected (no self-loops) edge index."""
    src, tgt = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(j)
                tgt.append(i)
    return torch.tensor([src, tgt], dtype=torch.long)


def calculate_auc(objectAij, weights):
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
    """Return uniformly spaced integer indices in [0, length-1] of size num_samples."""
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


def select_kept_node_indices(total_nodes, num_keep, seed):
    """Reproducibly choose `num_keep` node indices to RETAIN out of `total_nodes`.
    Uses a dedicated np.random.default_rng(seed) so the choice is independent
    of the global numpy/torch RNG state (model init reproducibility unaffected).
    """
    if num_keep > total_nodes:
        raise ValueError(f"num_keep={num_keep} > total_nodes={total_nodes}")
    if num_keep <= 0:
        raise ValueError(f"num_keep must be positive, got {num_keep}")
    rng = np.random.default_rng(int(seed))
    kept = rng.choice(total_nodes, size=num_keep, replace=False)
    return np.sort(kept).astype(np.int64)


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
    """Evaluate model on a dataset and return metrics."""
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
        return Data(
            x=self.X_train[idx],
            edge_index=self.edge_index,
            y=self.y_train[idx]
        )


# =============================================================================
# Simultaneous inference model: learnable soft adjacency + L1 regularization
# =============================================================================
class tau_soft_regu_deriv_MP(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam,
                 hidden=50, aggr='add', flow='source_to_target'):
        super(tau_soft_regu_deriv_MP, self).__init__(aggr=aggr, flow=flow)
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
        tmp = torch.cat([x_i[:, 0], x_j[:, 0]])
        tmp = tmp.reshape(2, -1).t()
        self.soft_weights = F.softmax(self.weights / self.tau, dim=1)
        self.soft_weights = self.soft_weights[:, 0].view(-1, 1)
        Len = int(x_i[:, 0].shape[0]) / int(self.soft_weights.shape[0])
        w = self.soft_weights.repeat(int(Len), 1)
        msg = self.msg_fnc(tmp) * w
        del tmp, w
        return msg

    def update(self, aggr_out, x=None):
        if self.ndim == 1:
            fx = self.node_fnc_x(x)
            dxdt = fx + aggr_out
            return torch.cat([dxdt], dim=1)
        elif self.ndim == 2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx + aggr_out
            dydt = fy
            return torch.cat([dxdt, dydt], dim=1)
        elif self.ndim == 3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx + aggr_out
            dydt = fy
            dzdt = fz
            return torch.cat([dxdt, dydt, dzdt], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim) * augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        edge_index = g.edge_index
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g))**2) + self.lam * torch.sum(self.soft_weights)
        else:
            return torch.sum(torch.abs(g.y - self.prediction(g))) + self.lam * torch.sum(self.soft_weights)

    def update_tau(self, newtau):
        self.tau = newtau


# =============================================================================
# Known structure model: fixed adjacency, learn dynamics only
# =============================================================================
class Graph_deriv_NN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):
        super(Graph_deriv_NN, self).__init__(aggr=aggr, flow=flow)
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
            fx = self.node_fnc_x(x)
            dxdt = fx + aggr_out
            return torch.cat([dxdt], dim=1)
        elif self.ndim == 2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx + aggr_out
            dydt = fy
            return torch.cat([dxdt, dydt], dim=1)
        elif self.ndim == 3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx + aggr_out
            dydt = fy
            dzdt = fz
            return torch.cat([dxdt, dydt, dzdt], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim) * augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        edge_index = g.edge_index
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g))**2)
        else:
            return torch.sum(torch.abs(g.y - self.prediction(g)))
