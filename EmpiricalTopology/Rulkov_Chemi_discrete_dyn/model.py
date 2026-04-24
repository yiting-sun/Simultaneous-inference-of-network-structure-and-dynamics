"""
Model / utilities for Rulkov chemical-synapse inference.

We learn three components that match the code-defined dynamics:

    dx_i/dt = self_x(x_i, y_i) + sum_j A[i,j] * message(x_i, x_j)
    dy_i/dt = self_y(x_i, y_i)

where, in the current formal dynamics,

    self_x_true = (alpha / (1 + x_i^2) + y_i - x_i) / dt
    self_y_true = -(mu / dt) * (x_i - sigma)
    message_true(target=i, source=j)
                 = -(gc / dt) * (x_i - v_s) * Gamma(x_j)

and Gamma(x_j) = 1 / (1 + exp(-lambda * (x_j - theta))).

Edge-index convention (PyG flow='source_to_target'):
    edge_index[0] = source , edge_index[1] = target
    message(x_i, x_j): x_i = target state, x_j = source state
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
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index


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
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
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


class tau_soft_regu_deriv_MP(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam,
                 hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
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

        self._init_weights()
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam
        self.weights = Parameter(torch.Tensor(edge_num, 2), requires_grad=True)
        torch.nn.init.normal_(self.weights, 0, 0.1)

    def _init_weights(self):
        for module in [self.msg_fnc, self.node_fnc_x, self.node_fnc_y]:
            for layer in module:
                if isinstance(layer, Lin):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.stack([x_i[:, 0], x_j[:, 0]], dim=1)
        self.soft_weights = F.softmax(self.weights / self.tau, dim=1)[:, 0].view(-1, 1)
        repeat_factor = int(x_i[:, 0].shape[0] / self.soft_weights.shape[0])
        w = self.soft_weights.repeat(repeat_factor, 1)
        msg = self.msg_fnc(tmp) * w
        return msg

    def update(self, aggr_out, x=None):
        return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g)) ** 2) + self.lam * torch.sum(self.soft_weights)
        return torch.sum(torch.abs(g.y - self.prediction(g))) + self.lam * torch.sum(self.soft_weights)

    def update_tau(self, newtau):
        self.tau = newtau


class Graph_deriv_NN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
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

        self.ndim = ndim
        self.delt_t = delt_t

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.stack([x_i[:, 0], x_j[:, 0]], dim=1)
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g)) ** 2)
        return torch.sum(torch.abs(g.y - self.prediction(g)))


# -----------------------------------------------------------------------------
# Factorized-message variants
# -----------------------------------------------------------------------------
# message(x_i, x_j) = msg_recv(x_i) * msg_send(x_j)  (elementwise)
#
# Matches the multiplicative receiver/sender structure of the true Rulkov
# chemical synapse message: -(gc/dt)*(x_i - v_s) * Gamma(x_j). The receiver
# factor only depends on x_i and the sender factor only on x_j, so the
# joint-MLP ambiguity (absorbing x_i-dependent parts of the message into
# node_fnc_x) is structurally removed.
# -----------------------------------------------------------------------------


class Graph_deriv_NN_factored(MessagePassing):
    """Factored message-passing variant, PRIOR2-style inductive biases applied:

      - msg_send: bound output to (0,1) via nn.Sigmoid  --  matches the true
        Gamma(x_j) = sigmoid(lambda*(x_j - theta)) chemical-synapse kernel.
      - msg_recv: simplified to a single hidden layer  --  the true receiver
        kernel -(gc/dt)*(x_i - v_s) is affine; reducing depth cuts gauge freedom.
      - xavier_uniform_ init on the pre-sigmoid output of msg_send keeps the
        initial logit near 0 where sigmoid gradient is maximal; matching
        init on the msg_recv output keeps the initial factor product tame.
    """
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        # msg_recv target is affine -(gc/dt)*(x_i - v_s); one hidden layer suffices
        self.msg_recv = Seq(
            Lin(1, hidden), ReLU(),
            Lin(hidden, msg_dim)
        )
        # msg_send target is Gamma(x_j) = sigmoid(lambda*(x_j - theta)); bound output
        self.msg_send = Seq(
            Lin(1, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, msg_dim), nn.Sigmoid()
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

        self._init_weights()
        self.ndim = ndim
        self.delt_t = delt_t

    def _init_weights(self):
        # ReLU-regime inits for every Lin layer (msg_recv hidden, msg_send hiddens, node_fnc_x/y)
        for module in [self.msg_recv, self.msg_send, self.node_fnc_x, self.node_fnc_y]:
            for layer in module:
                if isinstance(layer, Lin):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        # Override the output Lin of msg_send (feeds nn.Sigmoid) so pre-activation starts near 0.
        lins_send = [m for m in self.msg_send if isinstance(m, Lin)]
        torch.nn.init.xavier_uniform_(lins_send[-1].weight, gain=1.0)
        if lins_send[-1].bias is not None:
            torch.nn.init.zeros_(lins_send[-1].bias)
        # msg_recv output is unbounded; small-gain xavier keeps initial product tame.
        lins_recv = [m for m in self.msg_recv if isinstance(m, Lin)]
        torch.nn.init.xavier_uniform_(lins_recv[-1].weight, gain=1.0)
        if lins_recv[-1].bias is not None:
            torch.nn.init.zeros_(lins_recv[-1].bias)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.msg_recv(x_i[:, 0:1]) * self.msg_send(x_j[:, 0:1])

    def update(self, aggr_out, x=None):
        return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g)) ** 2)
        return torch.sum(torch.abs(g.y - self.prediction(g)))


class tau_soft_regu_deriv_MP_factored(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam,
                 hidden=50, aggr='add', flow='source_to_target'):
        super().__init__(aggr=aggr, flow=flow)
        # msg_recv target is affine -(gc/dt)*(x_i - v_s); one hidden layer suffices
        self.msg_recv = Seq(
            Lin(1, hidden), ReLU(),
            Lin(hidden, msg_dim)
        )
        # msg_send target is Gamma(x_j) = sigmoid(lambda*(x_j - theta)); bound output
        self.msg_send = Seq(
            Lin(1, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, hidden), ReLU(),
            Lin(hidden, msg_dim), nn.Sigmoid()
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

        self._init_weights()
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam
        self.weights = Parameter(torch.Tensor(edge_num, 2), requires_grad=True)
        torch.nn.init.normal_(self.weights, 0, 0.1)

    def _init_weights(self):
        # ReLU-regime inits for hidden layers of every submodule
        for module in [self.msg_recv, self.msg_send, self.node_fnc_x, self.node_fnc_y]:
            for layer in module:
                if isinstance(layer, Lin):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
        # Override the output Lin of msg_send (feeds Sigmoid) with xavier so
        # initial pre-activations sit near 0 and sigmoid stays in its high-gradient region.
        lins_send = [m for m in self.msg_send if isinstance(m, Lin)]
        torch.nn.init.xavier_uniform_(lins_send[-1].weight, gain=1.0)
        if lins_send[-1].bias is not None:
            torch.nn.init.zeros_(lins_send[-1].bias)
        # msg_recv output is an (unbounded) scalar factor; use a small-gain xavier
        # so initial product with msg_send isn't dominated by random large values.
        lins_recv = [m for m in self.msg_recv if isinstance(m, Lin)]
        torch.nn.init.xavier_uniform_(lins_recv[-1].weight, gain=1.0)
        if lins_recv[-1].bias is not None:
            torch.nn.init.zeros_(lins_recv[-1].bias)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        with torch.amp.autocast('cuda', enabled=False):
            raw_msg = self.msg_recv(x_i[:, 0:1].float()) * self.msg_send(x_j[:, 0:1].float())
        self.soft_weights = F.softmax(self.weights / self.tau, dim=1)[:, 0].view(-1, 1)
        repeat_factor = int(x_i[:, 0].shape[0] / self.soft_weights.shape[0])
        w = self.soft_weights.repeat(repeat_factor, 1)
        return (raw_msg * w).to(x_i.dtype)

    def update(self, aggr_out, x=None):
        return torch.cat([self.node_fnc_x(x) + aggr_out, self.node_fnc_y(x)], dim=1)

    def prediction(self, g, augment=False, augmentation=3):
        x = g.x
        if augment:
            aug = torch.randn(1, self.ndim) * augmentation
            aug = aug.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(self.ndim).to(x.device), aug)
        return self.propagate(g.edge_index, size=(x.size(0), x.size(0)), x=x)

    def loss(self, g, square=False, legacy=False, **kwargs):
        pred = self.prediction(g)
        if legacy:
            if square:
                return torch.sum((g.y - pred) ** 2) + self.lam * torch.sum(self.soft_weights)
            return torch.sum(torch.abs(g.y - pred)) + self.lam * torch.sum(self.soft_weights)
        n_data = g.y.numel()
        reg = (self.lam / n_data) * torch.sum(self.soft_weights)
        if square:
            return torch.mean((g.y - pred) ** 2) + reg
        return torch.mean(torch.abs(g.y - pred)) + reg

    def update_tau(self, newtau):
        self.tau = newtau
