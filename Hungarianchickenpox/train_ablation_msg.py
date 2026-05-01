import os
import sys
import argparse
import time
import math
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_ablation_msg import (
    GraphDerivNNMsgAblation,
    build_fully_connected_edge_index,
    count_parameters,
    MSG_CONFIGS,
    FULL_SELF_INDICES,
    FEATURE_NAMES_MAP,
    describe_msg_config,
)


class WeightedGraphDataset(Dataset):
    def __init__(self, X, Y, edge_index):
        self.X = X
        self.Y = Y
        self.edge_index = edge_index

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return Data(x=self.X[idx], y=self.Y[idx], edge_index=self.edge_index)


def build_id_to_name_from_edges(edges_df: pd.DataFrame) -> dict:
    id_to_name = {}
    for _, r in edges_df.iterrows():
        id_to_name[int(r["id_1"])] = str(r["name_1"])
        id_to_name[int(r["id_2"])] = str(r["name_2"])
    return id_to_name


def reorder_timeseries_to_edge_id_order(ts_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in ts_df.columns:
        raise ValueError("Time series CSV must have a 'Date' column.")
    id_to_name = build_id_to_name_from_edges(edges_df)
    ids = sorted(id_to_name.keys())
    node_order = [id_to_name[i] for i in ids]
    missing = [n for n in node_order if n not in ts_df.columns]
    if missing:
        raise ValueError(f"These counties from edge file are missing in time series columns: {missing}")
    return ts_df[["Date"] + node_order]


def maybe_smooth(X: np.ndarray, method: str) -> np.ndarray:
    if method == "gaussian":
        return gaussian_filter1d(X, sigma=1.5, axis=0, mode="nearest")
    if method == "none":
        return X
    raise ValueError(f"Unknown smoothing method: {method}")


def rolling_mean_prefix(X: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(X)
    csum = np.cumsum(X, axis=0)
    for t in range(X.shape[0]):
        start = max(0, t - window + 1)
        total = csum[t] - (csum[start - 1] if start > 0 else 0.0)
        out[t] = total / float(t - start + 1)
    return out


def lagged_difference(X: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(X)
    for t in range(X.shape[0]):
        prev_idx = max(0, t - lag)
        out[t] = X[prev_idx] - X[t]
    return out


def compute_seasonal_components(X: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        raise ValueError(f"seasonal period must be > 1, got {period}")
    if X.shape[0] < 2 * period:
        raise ValueError(
            f"Need at least two full periods for seasonal_decompose, got T={X.shape[0]}, period={period}"
        )
    seasonal = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        result = seasonal_decompose(X[:, i], model="additive", period=period, extrapolate_trend="freq")
        seasonal[:, i] = result.seasonal.astype(np.float32)
    return seasonal


def build_longwindow_features_t(X: np.ndarray, dt: float, seasonal_period: int):
    if X.shape[0] < 13:
        raise ValueError("Need at least 13 weeks to build features from t=12 onward and predict t+1.")

    ma3 = rolling_mean_prefix(X, 3)
    ma5 = rolling_mean_prefix(X, 5)
    ma10 = rolling_mean_prefix(X, 10)
    diff3 = lagged_difference(X, 3)
    seasonal = compute_seasonal_components(X, period=seasonal_period)
    valid_t = np.arange(11, X.shape[0] - 1, dtype=np.int64)

    x_t = X[valid_t, :, None]
    x_tm1 = X[valid_t - 1, :, None]
    ma3_t = ma3[valid_t, :, None]
    diff3_t = diff3[valid_t, :, None]
    seasonal_t = seasonal[valid_t, :, None]
    seasonal_rate12to1_t = ((seasonal[valid_t, :] - seasonal[valid_t - 11, :]) / 11.0)[:, :, None]
    ma5_t = ma5[valid_t, :, None]
    ma10_t = ma10[valid_t, :, None]
    rate5_t = ((X[valid_t - 5, :] - X[valid_t, :]) / 5.0)[:, :, None]
    rate10_t = ((X[valid_t - 10, :] - X[valid_t, :]) / 10.0)[:, :, None]

    features = np.concatenate(
        [x_t, x_tm1, ma3_t, diff3_t, seasonal_t, seasonal_rate12to1_t, ma5_t, ma10_t, rate5_t, rate10_t],
        axis=2,
    ).astype(np.float32)
    dxdt_next = ((X[valid_t + 1, :] - X[valid_t, :]) / dt)[:, :, None].astype(np.float32)
    feature_names = [
        "x_t", "x_t_minus_1", "i_ma3_hist", "i_prev_diff3",
        "i_seasonal_t", "i_seasonal_t_rate12to1_from_t_minus_12_to_t_minus_1",
        "j_ma5_hist", "j_ma10_hist",
        "j_rate5_from_t_minus_5_to_t", "j_rate10_from_t_minus_10_to_t",
    ]
    return features, dxdt_next, feature_names


def normalize_features_targets(X_train, X_val, X_test, y_train, y_val, y_test):
    x_mu = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, keepdim=True) + 1e-6
    y_mu = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True) + 1e-6
    X_train = (X_train - x_mu) / x_std
    y_train = (y_train - y_mu) / y_std
    if X_val is not None:
        X_val = (X_val - x_mu) / x_std
    if y_val is not None:
        y_val = (y_val - y_mu) / y_std
    if X_test is not None:
        X_test = (X_test - x_mu) / x_std
    if y_test is not None:
        y_test = (y_test - y_mu) / y_std
    return X_train, X_val, X_test, y_train, y_val, y_test, x_mu, x_std, y_mu, y_std


def identity_norm_template(X_train, y_train):
    x_mu = torch.zeros_like(X_train[:1])
    x_std = torch.ones_like(X_train[:1])
    y_mu = torch.zeros_like(y_train[:1])
    y_std = torch.ones_like(y_train[:1])
    return x_mu, x_std, y_mu, y_std


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.allclose(ss_tot, 0.0):
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def evaluate_sequence_metrics(model, X_seq, y_seq, edge_index, device):
    model.eval()
    pred_list, true_list = [], []
    edge_index = edge_index.to(device)
    with torch.no_grad():
        for t in range(X_seq.shape[0]):
            x = X_seq[t].to(device)
            y = y_seq[t].to(device)
            pred = model(x, edge_index=edge_index)
            pred_list.append(pred.detach().cpu().numpy())
            true_list.append(y.detach().cpu().numpy())
    pred_dxdt = np.asarray(pred_list)
    true_dxdt = np.asarray(true_list)
    overall_corr = safe_corrcoef(true_dxdt.reshape(-1), pred_dxdt.reshape(-1))
    overall_r2 = compute_r2(true_dxdt.reshape(-1), pred_dxdt.reshape(-1))
    _, n_nodes, _ = pred_dxdt.shape
    node_corr, node_r2 = [], []
    for i in range(n_nodes):
        yi = true_dxdt[:, i, 0]
        yhi = pred_dxdt[:, i, 0]
        node_corr.append(safe_corrcoef(yi, yhi))
        node_r2.append(compute_r2(yi, yhi))
    return {
        "overall_corr": float(overall_corr) if not np.isnan(overall_corr) else np.nan,
        "overall_r2": float(overall_r2) if not np.isnan(overall_r2) else np.nan,
        "node_corr_mean": float(np.nanmean(node_corr)),
        "node_corr_median": float(np.nanmedian(node_corr)),
        "node_corr_min": float(np.nanmin(node_corr)),
        "node_corr_max": float(np.nanmax(node_corr)),
        "node_r2_mean": float(np.nanmean(node_r2)),
        "node_r2_median": float(np.nanmedian(node_r2)),
        "node_r2_min": float(np.nanmin(node_r2)),
        "node_r2_max": float(np.nanmax(node_r2)),
    }


def compute_structure_metrics_soft(true_adj: np.ndarray, pred_adj: np.ndarray) -> dict:
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    y_true = true_adj[mask].astype(np.float64).reshape(-1)
    y_score = pred_adj[mask].astype(np.float64).reshape(-1)
    return {
        "auc": float(roc_auc_score(y_true.astype(np.int64), y_score)),
        "auprc": float(average_precision_score(y_true.astype(np.int64), y_score)),
        "corr": safe_corrcoef(y_true, y_score),
        "mse": float(np.mean((y_true - y_score) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_score))),
        "mean_weight": float(np.mean(y_score)),
        "std_weight": float(np.std(y_score)),
    }


def save_inferred_adjacency(path: str, matrix: np.ndarray):
    np.save(path, matrix.astype(np.float32))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y", "t"}:
        return True
    if v.lower() in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def default_data_path(filename: str) -> str:
    """Resolve bundled dataset files relative to this script directory."""
    return os.path.join(CURRENT_DIR, filename)


def main():
    parser = argparse.ArgumentParser(description="Message-feature ablation for simultaneous inference")
    parser.add_argument("--ts_csv", type=str, default=default_data_path("hungary_chickenpox_reordered_trimmed.csv"))
    parser.add_argument("--ts_reordered_csv", type=str, default="")
    parser.add_argument("--edge_csv", type=str, default=default_data_path("hungary_county_edges.csv"))
    parser.add_argument("--adj_npy", type=str, default=default_data_path("hungary_chickenpox_adj.npy"))
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--msg_config", type=str, default="rate_x",
                        choices=list(MSG_CONFIGS.keys()),
                        help="Which message-feature config to use")
    parser.add_argument("--log", type=str2bool, default=False)
    parser.add_argument("--zscore", type=str2bool, default=False)
    parser.add_argument("--smooth", type=str, default="gaussian", choices=["gaussian", "none"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_seed", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--metrics_every", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--tau_decay", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--seasonal_period", type=int, default=52)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    checkpoints_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    if args.use_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- Load data ---
    if args.ts_reordered_csv and os.path.exists(args.ts_reordered_csv):
        ts = pd.read_csv(args.ts_reordered_csv)
    else:
        ts = pd.read_csv(args.ts_csv)
        if "Date" not in ts.columns:
            edges = pd.read_csv(args.edge_csv)
            ts = reorder_timeseries_to_edge_id_order(ts, edges)

    X_raw = ts.iloc[:, 1:].to_numpy(dtype=np.float32)
    X = np.log1p(X_raw) if args.log else X_raw
    X = maybe_smooth(X, args.smooth)

    X_feat, dXdt, feature_names = build_longwindow_features_t(X=X, dt=args.dt, seasonal_period=args.seasonal_period)
    X_t = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(dXdt, dtype=torch.float32)

    A_true = np.load(args.adj_npy)
    if A_true.ndim != 2 or A_true.shape[0] != A_true.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A_true.shape}")
    if A_true.shape[0] != X_feat.shape[1]:
        raise ValueError(f"Adjacency size {A_true.shape[0]} != node count {X_feat.shape[1]}")
    A_true = (A_true != 0).astype(np.float32)
    np.fill_diagonal(A_true, 0.0)

    num_nodes = X_feat.shape[1]
    edge_index = build_fully_connected_edge_index(num_nodes)

    # --- 60/20/20 temporal split (no shuffle for time series) ---
    T = X_t.shape[0]
    train_end = int(T * args.train_ratio)
    val_end = int(T * (args.train_ratio + args.val_ratio))

    X_train = X_t[:train_end]
    y_train = Y_t[:train_end]
    X_val = X_t[train_end:val_end]
    y_val = Y_t[train_end:val_end]
    X_test = X_t[val_end:]
    y_test = Y_t[val_end:]

    print(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} (total={T})")

    # --- Normalization ---
    if args.zscore:
        X_train, X_val, X_test, y_train, y_val, y_test, x_mu, x_std, y_mu, y_std = normalize_features_targets(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
    else:
        x_mu, x_std, y_mu, y_std = identity_norm_template(X_train, y_train)

    # --- DataLoaders ---
    train_ds = WeightedGraphDataset(X_train, y_train, edge_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    val_ds = WeightedGraphDataset(X_val, y_val, edge_index)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch // 2), shuffle=False, num_workers=0)
    test_ds = WeightedGraphDataset(X_test, y_test, edge_index)
    test_loader = DataLoader(test_ds, batch_size=max(1, args.batch // 2), shuffle=False, num_workers=0)

    # --- Message config ---
    msg_config = args.msg_config
    msg_description = describe_msg_config(msg_config)

    target_definition = "dxdt_next = (x[t+1] - x[t]) / dt"
    feature_definition = (
        f"self features: FULL (all 6); msg features: {msg_description}; "
        f"seasonal_decompose(period={args.seasonal_period})"
    )

    torch.save({
        "x_mu": x_mu.cpu(),
        "x_std": x_std.cpu(),
        "y_mu": y_mu.cpu(),
        "y_std": y_std.cpu(),
        "feature_names": feature_names,
        "target_definition": target_definition,
        "feature_definition": feature_definition,
        "seasonal_period": args.seasonal_period,
        "msg_config": msg_config,
        "msg_description": msg_description,
    }, os.path.join(args.outdir, "norm_stats.pt"))
    np.save(os.path.join(args.outdir, "ground_truth_adjacency.npy"), A_true)
    np.save(os.path.join(args.outdir, "fullconnect_edge_index.npy"), edge_index.cpu().numpy())

    # --- Build model ---
    Dyn = GraphDerivNNMsgAblation(
        n_f=X_train.shape[-1],
        msg_dim=1,
        ndim=1,
        delt_t=args.dt,
        num_nodes=num_nodes,
        tau=args.tau,
        lam=args.lam,
        msg_config=msg_config,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(Dyn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    sched = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs, final_div_factor=1e4)
    tau_decay_step = args.tau_decay ** (1.0 / steps_per_epoch)

    print(f"Msg config: {msg_description}")
    print(f"Self features: FULL (all 6)")
    print(f"Feature names: {feature_names}")
    print(f"Target: {target_definition}")
    print(f"Seasonal period: {args.seasonal_period}")
    print(f"Tau: {args.tau}, Tau decay per epoch: {args.tau_decay}")
    print(f"Lambda: {args.lam}, Zscore: {args.zscore}, Smooth: {args.smooth}")
    print(f"Node count: {num_nodes}, edge count: {edge_index.shape[1]}")
    print(f"Model params: {count_parameters(Dyn):,}")

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_hist, val_hist, test_hist, sparse_metrics_rows = [], [], [], []
    start = time.time()
    tau = float(args.tau)

    for epoch in tqdm(range(args.epochs), desc="training"):
        # --- Train ---
        Dyn.train()
        tr_loss = 0.0
        tr_items = 0
        for g in train_loader:
            g = g.to(device)
            opt.zero_grad()
            loss = Dyn.loss(g, square=False)
            bsz = int(getattr(g, "num_graphs", 1))
            (loss / max(1, bsz)).backward()
            torch.nn.utils.clip_grad_norm_(Dyn.parameters(), max_norm=5.0)
            opt.step()
            sched.step()
            tau *= tau_decay_step
            Dyn.update_tau(tau)
            tr_loss += loss.item()
            tr_items += bsz
        train_mse = tr_loss / max(1, tr_items)
        train_hist.append(train_mse)

        # --- Validation (for early stopping) ---
        Dyn.eval()
        va_loss = 0.0
        va_items = 0
        with torch.no_grad():
            for g in val_loader:
                g = g.to(device)
                loss = Dyn.eval_loss(g, square=True)
                bsz = int(getattr(g, "num_graphs", 1))
                va_loss += loss.item()
                va_items += bsz
        val_mse = va_loss / max(1, va_items)
        val_hist.append(val_mse)

        # --- Test loss ---
        te_loss = 0.0
        te_items = 0
        with torch.no_grad():
            for g in test_loader:
                g = g.to(device)
                loss = Dyn.eval_loss(g, square=True)
                bsz = int(getattr(g, "num_graphs", 1))
                te_loss += loss.item()
                te_items += bsz
        test_mse = te_loss / max(1, te_items)
        test_hist.append(test_mse)

        # --- Early stopping on validation ---
        if val_mse < (best_val - args.early_stop_min_delta):
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in Dyn.state_dict().items()}
            epochs_without_improvement = 0
            inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
            structure_metrics = compute_structure_metrics_soft(A_true, inferred_adj)
            torch.save({
                "epoch": epoch + 1,
                "state_dict": best_state,
                "config": vars(args),
                "nodes": list(ts.columns[1:]),
                "feature_names": feature_names,
                "target_definition": target_definition,
                "feature_definition": feature_definition,
                "msg_config": msg_config,
                "msg_config": msg_config,
                "msg_description": msg_description,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "structure_metrics": structure_metrics,
                "inferred_adjacency": inferred_adj,
            }, os.path.join(args.outdir, "best_model.pt"))
            save_inferred_adjacency(os.path.join(args.outdir, "best_inferred_adjacency.npy"), inferred_adj)
        else:
            epochs_without_improvement += 1

        # --- Periodic metrics ---
        should_eval_metrics = args.metrics_every > 0 and (((epoch + 1) % args.metrics_every == 0) or (epoch + 1 == args.epochs))
        metrics_row = None
        if should_eval_metrics:
            inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
            structure_metrics = compute_structure_metrics_soft(A_true, inferred_adj)
            train_metrics = evaluate_sequence_metrics(Dyn, X_train, y_train, edge_index, device)
            val_metrics = evaluate_sequence_metrics(Dyn, X_val, y_val, edge_index, device)
            test_metrics = evaluate_sequence_metrics(Dyn, X_test, y_test, edge_index, device)
            metrics_row = {
                "epoch": epoch + 1,
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "test_mse": float(test_mse),
                "train_overall_corr": train_metrics["overall_corr"],
                "train_overall_r2": train_metrics["overall_r2"],
                "train_node_corr_mean": train_metrics["node_corr_mean"],
                "train_node_r2_mean": train_metrics["node_r2_mean"],
                "train_node_corr_median": train_metrics["node_corr_median"],
                "train_node_r2_median": train_metrics["node_r2_median"],
                "train_node_corr_min": train_metrics["node_corr_min"],
                "train_node_corr_max": train_metrics["node_corr_max"],
                "train_node_r2_min": train_metrics["node_r2_min"],
                "train_node_r2_max": train_metrics["node_r2_max"],
                "val_overall_corr": val_metrics["overall_corr"],
                "val_overall_r2": val_metrics["overall_r2"],
                "val_node_corr_mean": val_metrics["node_corr_mean"],
                "val_node_r2_mean": val_metrics["node_r2_mean"],
                "test_overall_corr": test_metrics["overall_corr"],
                "test_overall_r2": test_metrics["overall_r2"],
                "test_node_corr_mean": test_metrics["node_corr_mean"],
                "test_node_r2_mean": test_metrics["node_r2_mean"],
                "test_node_corr_median": test_metrics["node_corr_median"],
                "test_node_r2_median": test_metrics["node_r2_median"],
                "test_node_corr_min": test_metrics["node_corr_min"],
                "test_node_corr_max": test_metrics["node_corr_max"],
                "test_node_r2_min": test_metrics["node_r2_min"],
                "test_node_r2_max": test_metrics["node_r2_max"],
                "struct_auc": structure_metrics["auc"],
                "struct_auprc": structure_metrics["auprc"],
                "struct_corr": structure_metrics["corr"],
                "struct_mse": structure_metrics["mse"],
                "struct_mae": structure_metrics["mae"],
                "struct_mean_weight": structure_metrics["mean_weight"],
                "struct_std_weight": structure_metrics["std_weight"],
            }
            sparse_metrics_rows.append(metrics_row)
            save_inferred_adjacency(
                os.path.join(args.outdir, f"inferred_adjacency_epoch_{epoch + 1:04d}.npy"),
                inferred_adj,
            )

        if (epoch + 1) % args.save_every == 0 or (epoch + 1 == args.epochs):
            inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
            payload = {
                "epoch": epoch + 1,
                "state_dict": {k: v.detach().cpu().clone() for k, v in Dyn.state_dict().items()},
                "config": vars(args),
                "nodes": list(ts.columns[1:]),
                "feature_names": feature_names,
                "msg_config": msg_config,
                "msg_config": msg_config,
                "msg_description": msg_description,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "inferred_adjacency": inferred_adj,
            }
            if metrics_row is not None:
                payload.update(metrics_row)
            torch.save(payload, os.path.join(checkpoints_dir, f"model_epoch_{epoch + 1:04d}.pt"))

        if (epoch + 1) % 50 == 0 or (epoch + 1 == args.epochs):
            if metrics_row is None:
                inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
                structure_metrics = compute_structure_metrics_soft(A_true, inferred_adj)
                train_metrics = evaluate_sequence_metrics(Dyn, X_train, y_train, edge_index, device)
                val_metrics = evaluate_sequence_metrics(Dyn, X_val, y_val, edge_index, device)
                test_metrics = evaluate_sequence_metrics(Dyn, X_test, y_test, edge_index, device)
                metrics_row = {
                    "train_overall_r2": train_metrics["overall_r2"],
                    "val_overall_r2": val_metrics["overall_r2"],
                    "test_overall_r2": test_metrics["overall_r2"],
                    "struct_auc": structure_metrics["auc"],
                    "struct_auprc": structure_metrics["auprc"],
                    "struct_corr": structure_metrics["corr"],
                    "struct_mse": structure_metrics["mse"],
                    "struct_mae": structure_metrics["mae"],
                }
            print(
                f"tau={Dyn.tau:.6g} "
                f"epoch={epoch+1} train_mse={train_mse:.6g} "
                f"val_mse={val_mse:.6g} best_val={best_val:.6g} "
                f"test_mse={test_mse:.6g} "
                f"train_R2={metrics_row['train_overall_r2']:.6g} "
                f"val_R2={metrics_row['val_overall_r2']:.6g} "
                f"test_R2={metrics_row['test_overall_r2']:.6g} "
                f"Adj_AUC={metrics_row['struct_auc']:.6g} "
                f"Adj_AUPRC={metrics_row['struct_auprc']:.6g} "
                f"Adj_corr={metrics_row['struct_corr']:.6g} "
                f"Adj_mse={metrics_row['struct_mse']:.6g} "
                f"Adj_mae={metrics_row['struct_mae']:.6g}"
            )

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch={epoch+1}")
            break

    if best_state is not None:
        Dyn.load_state_dict(best_state)

    elapsed = time.time() - start
    print(f"Done. elapsed={elapsed:.1f}s")

    # --- Save metrics ---
    metrics_df = pd.DataFrame(sparse_metrics_rows)
    metrics_csv_path = os.path.join(args.outdir, f"training_metrics_every{args.metrics_every}.csv")
    metrics_pkl_path = os.path.join(args.outdir, f"training_metrics_every{args.metrics_every}.pkl")
    metrics_df.to_csv(metrics_csv_path, index=False)
    with open(metrics_pkl_path, "wb") as f:
        pickle.dump(sparse_metrics_rows, f)
    with open(os.path.join(args.outdir, "loss_curves.pkl"), "wb") as f:
        pickle.dump({
            "train_mse": train_hist,
            "val_mse": val_hist,
            "test_mse": test_hist,
            "metrics_every": args.metrics_every,
            "metrics_epochs": metrics_df["epoch"].tolist() if len(metrics_df) else [],
            "train_overall_r2": metrics_df["train_overall_r2"].tolist() if len(metrics_df) else [],
            "val_overall_r2": metrics_df["val_overall_r2"].tolist() if (len(metrics_df) and "val_overall_r2" in metrics_df.columns) else [],
            "test_overall_r2": metrics_df["test_overall_r2"].tolist() if (len(metrics_df) and "test_overall_r2" in metrics_df.columns) else [],
            "struct_auc": metrics_df["struct_auc"].tolist() if (len(metrics_df) and "struct_auc" in metrics_df.columns) else [],
            "struct_auprc": metrics_df["struct_auprc"].tolist() if (len(metrics_df) and "struct_auprc" in metrics_df.columns) else [],
            "struct_corr": metrics_df["struct_corr"].tolist() if (len(metrics_df) and "struct_corr" in metrics_df.columns) else [],
            "struct_mse": metrics_df["struct_mse"].tolist() if (len(metrics_df) and "struct_mse" in metrics_df.columns) else [],
            "struct_mae": metrics_df["struct_mae"].tolist() if (len(metrics_df) and "struct_mae" in metrics_df.columns) else [],
        }, f)

    # --- Final model ---
    final_inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
    final_structure_metrics = compute_structure_metrics_soft(A_true, final_inferred_adj)
    final_train_metrics = evaluate_sequence_metrics(Dyn, X_train, y_train, edge_index, device)
    final_val_metrics = evaluate_sequence_metrics(Dyn, X_val, y_val, edge_index, device)
    final_test_metrics = evaluate_sequence_metrics(Dyn, X_test, y_test, edge_index, device)
    final_payload = {
        "epoch": len(train_hist),
        "state_dict": {k: v.detach().cpu().clone() for k, v in Dyn.state_dict().items()},
        "config": vars(args),
        "nodes": list(ts.columns[1:]),
        "feature_names": feature_names,
        "target_definition": target_definition,
        "feature_definition": feature_definition,
        "msg_config": msg_config,
        "msg_config": msg_config,
        "msg_description": msg_description,
        "train_mse": train_hist[-1] if len(train_hist) else np.nan,
        "val_mse": val_hist[-1] if len(val_hist) else np.nan,
        "test_mse": test_hist[-1] if len(test_hist) else np.nan,
        "inferred_adjacency": final_inferred_adj,
        "structure_metrics": final_structure_metrics,
        "train_metrics": final_train_metrics,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
    }
    if len(metrics_df) > 0:
        final_payload.update(metrics_df.iloc[-1].to_dict())
    torch.save(final_payload, os.path.join(args.outdir, "final_model.pt"))
    save_inferred_adjacency(os.path.join(args.outdir, "final_inferred_adjacency.npy"), final_inferred_adj)
    np.save(os.path.join(args.outdir, "X_features_seasonal_t.npy"), X_feat)
    np.save(os.path.join(args.outdir, "dXdt_next.npy"), dXdt)

    # --- Summary ---
    print(f"\n=== Final Results (msg_config={msg_config}) ===")
    print(f"Msg: {msg_description}")
    print(f"Best val MSE: {best_val:.6g} | Test MSE: {test_hist[-1] if test_hist else 'N/A':.6g}")
    print(f"Train R2: {final_train_metrics['overall_r2']:.6g}")
    print(f"Val R2:   {final_val_metrics['overall_r2']:.6g}")
    print(f"Test R2:  {final_test_metrics['overall_r2']:.6g}")
    print(f"Adj AUC:  {final_structure_metrics['auc']:.6g}")
    print(f"Adj AUPRC: {final_structure_metrics['auprc']:.6g}")
    print(f"Adj Corr: {final_structure_metrics['corr']:.6g}")


if __name__ == "__main__":
    main()
