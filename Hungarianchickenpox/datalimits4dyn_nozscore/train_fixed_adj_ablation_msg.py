"""
Training Script: Fixed Adjacency + Message-Feature Ablation
==========================================================
Data split: 60% train / 20% val (early stopping) / 20% test
Adjacency types: shared-edge masks loaded from saved nozscore npz files
Message-feature subset configurable via --msg_config flag. Self always FULL.
Smoothing: gaussian. This script only supports nozscore.
"""

import os, sys, argparse, time, math, pickle, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_fixed_adj_ablation_msg import (
    GraphDerivNNFixedAdjMsgAblation,
    build_fully_connected_edge_index,
    count_parameters,
    MSG_CONFIGS,
    FEATURE_NAMES_MAP,
    describe_msg_config,
)


SUPPORTED_MSG_CONFIGS = ("rate_x",)
FIXED_ADJ_NPZ_MAP = {
    "runs3thr0p6": os.path.join(PROJECT_DIR, "surrogateAdj_shared_edge_masks_runs3_thr0p6.npz"),
}


class WeightedGraphDataset(Dataset):
    def __init__(self, X, Y, edge_index):
        self.X, self.Y, self.edge_index = X, Y, edge_index
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return Data(x=self.X[idx], y=self.Y[idx], edge_index=self.edge_index)


def rolling_mean_prefix(X, window):
    out = np.zeros_like(X)
    csum = np.cumsum(X, axis=0)
    for t in range(X.shape[0]):
        start = max(0, t - window + 1)
        total = csum[t] - (csum[start - 1] if start > 0 else 0.0)
        out[t] = total / float(t - start + 1)
    return out

def lagged_difference(X, lag):
    out = np.zeros_like(X)
    for t in range(X.shape[0]):
        prev_idx = max(0, t - lag)
        out[t] = X[prev_idx] - X[t]
    return out

def compute_seasonal_components(X, period):
    seasonal = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[1]):
        result = seasonal_decompose(X[:, i], model="additive", period=period, extrapolate_trend="freq")
        seasonal[:, i] = result.seasonal.astype(np.float32)
    return seasonal

def build_longwindow_features_t(X, dt=1.0, seasonal_period=52):
    ma3  = rolling_mean_prefix(X, 3)
    ma5  = rolling_mean_prefix(X, 5)
    ma10 = rolling_mean_prefix(X, 10)
    diff3 = lagged_difference(X, 3)
    seasonal = compute_seasonal_components(X, period=seasonal_period)
    valid_t = np.arange(11, X.shape[0] - 1, dtype=np.int64)

    features = np.concatenate([
        X[valid_t, :, None], X[valid_t - 1, :, None],
        ma3[valid_t, :, None], diff3[valid_t, :, None],
        seasonal[valid_t, :, None],
        ((seasonal[valid_t, :] - seasonal[valid_t - 11, :]) / 11.0)[:, :, None],
        ma5[valid_t, :, None], ma10[valid_t, :, None],
        ((X[valid_t - 5, :] - X[valid_t, :]) / 5.0)[:, :, None],
        ((X[valid_t - 10, :] - X[valid_t, :]) / 10.0)[:, :, None],
    ], axis=2).astype(np.float32)

    dxdt = ((X[valid_t + 1, :] - X[valid_t, :]) / dt)[:, :, None].astype(np.float32)
    feature_names = [
        "x_t", "x_t_minus_1", "i_ma3_hist", "i_prev_diff3",
        "i_seasonal_t", "i_seasonal_t_rate12to1",
        "j_ma5_hist", "j_ma10_hist",
        "j_rate5", "j_rate10",
    ]
    return features, dxdt, feature_names


def normalize_features_targets(X_train, X_val, X_test, y_train, y_val, y_test):
    x_mu = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, keepdim=True) + 1e-6
    y_mu = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True) + 1e-6
    X_train = (X_train - x_mu) / x_std
    y_train = (y_train - y_mu) / y_std
    X_val = (X_val - x_mu) / x_std
    y_val = (y_val - y_mu) / y_std
    X_test = (X_test - x_mu) / x_std
    y_test = (y_test - y_mu) / y_std
    return X_train, X_val, X_test, y_train, y_val, y_test, x_mu, x_std, y_mu, y_std


def identity_norm_template(X_train, y_train):
    x_mu = torch.zeros_like(X_train[:1])
    x_std = torch.ones_like(X_train[:1])
    y_mu = torch.zeros_like(y_train[:1])
    y_std = torch.ones_like(y_train[:1])
    return x_mu, x_std, y_mu, y_std


def get_saved_msg_configs(npz_data):
    if "msg_configs" in npz_data:
        values = npz_data["msg_configs"]
    elif "msg_config" in npz_data:
        values = npz_data["msg_config"]
    else:
        raise KeyError("Neither 'msg_configs' nor 'msg_config' found in adjacency npz.")
    return [str(x) for x in np.asarray(values).reshape(-1).tolist()]


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
    node_mse, node_r2, node_corr = [], [], []
    for i in range(n_nodes):
        yi, yhi = trues[:, i, 0], preds[:, i, 0]
        node_mse.append(float(np.mean((yi - yhi) ** 2)))
        node_r2.append(compute_r2(yi, yhi))
        node_corr.append(safe_corrcoef(yi, yhi))

    return {
        "mse": mse, "mae": mae, "r2": r2, "corr": corr,
        "node_mse": node_mse, "node_r2": node_r2, "node_corr": node_corr,
        "node_r2_mean": float(np.nanmean(node_r2)),
        "node_r2_median": float(np.nanmedian(node_r2)),
        "node_r2_min": float(np.nanmin(node_r2)),
        "node_r2_max": float(np.nanmax(node_r2)),
        "node_corr_mean": float(np.nanmean(node_corr)),
    }


def load_shared_edge_mask_adj(adj_type, msg_config, num_nodes):
    npz_path = FIXED_ADJ_NPZ_MAP[adj_type]
    data = np.load(npz_path, allow_pickle=True)
    key = f"mask_{msg_config}"
    if key not in data:
        raise KeyError(f"{key} not found in {npz_path}")

    saved_msg_configs = get_saved_msg_configs(data)
    if msg_config not in saved_msg_configs:
        raise ValueError(f"msg_config={msg_config} not listed in {npz_path}: {saved_msg_configs}")

    A = data[key].astype(np.float32)
    if A.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Adjacency shape mismatch for {adj_type}/{msg_config}: "
            f"expected {(num_nodes, num_nodes)}, got {A.shape}"
        )
    np.fill_diagonal(A, 0)
    return A


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    if v.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def main():
    parser = argparse.ArgumentParser(description="Fixed adj + message-feature ablation training")
    parser.add_argument("--adj_type", type=str, required=True,
                        choices=list(FIXED_ADJ_NPZ_MAP.keys()))
    parser.add_argument("--msg_config", type=str, default="rate_x",
                        choices=list(SUPPORTED_MSG_CONFIGS))
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--zscore", type=str2bool, default=False)
    parser.add_argument("--smooth", type=str, default="gaussian", choices=["gaussian", "none"])
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--metrics_every", type=int, default=50)
    parser.add_argument("--seasonal_period", type=int, default=52)
    parser.add_argument("--use_seed", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.use_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device(args.device)

    # Paths
    ts_csv = os.path.join(PROJECT_DIR, "hungary_chickenpox_reordered_trimmed.csv")

    msg_config = args.msg_config
    if args.zscore:
        raise ValueError("This script now only supports --zscore false (nozscore).")

    print(f"{'=' * 60}")
    print(f"Fixed Adj Ablation: adj_type={args.adj_type}, msg={msg_config}")
    print(f"zscore={args.zscore}, smooth={args.smooth}, seed={args.seed}")
    print(f"Output: {args.outdir}")
    print(f"{'=' * 60}")

    # Load data
    ts = pd.read_csv(ts_csv)
    county_names = list(ts.columns[1:])
    X_raw = ts.iloc[:, 1:].to_numpy(dtype=np.float32)
    X = gaussian_filter1d(X_raw, sigma=1.5, axis=0, mode="nearest") if args.smooth == "gaussian" else X_raw
    X_feat, dXdt, feature_names = build_longwindow_features_t(X, dt=1.0, seasonal_period=args.seasonal_period)

    X_t = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(dXdt, dtype=torch.float32)
    num_nodes = X_feat.shape[1]
    T = X_t.shape[0]

    # 60/20/20 temporal split
    train_end = int(T * args.train_ratio)
    val_end = int(T * (args.train_ratio + args.val_ratio))

    X_train, Y_train = X_t[:train_end], Y_t[:train_end]
    X_val, Y_val = X_t[train_end:val_end], Y_t[train_end:val_end]
    X_test, Y_test = X_t[val_end:], Y_t[val_end:]

    print(f"Data split: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]} (total={T})")

    # Normalization
    x_mu, x_std, y_mu, y_std = identity_norm_template(X_train, Y_train)

    # Build fixed adjacency
    fixed_adj = load_shared_edge_mask_adj(args.adj_type, msg_config, num_nodes)

    np.save(os.path.join(args.outdir, "fixed_adjacency.npy"), fixed_adj)
    n_edges_active = int((fixed_adj > 0).sum())
    print(f"Adjacency: {args.adj_type}, active edges={n_edges_active}, "
          f"density={n_edges_active / (num_nodes * (num_nodes - 1)):.3f}")

    # Ablation config
    msg_description = describe_msg_config(msg_config)
    print(f"Msg config: {msg_description}")

    # Build model
    edge_index = build_fully_connected_edge_index(num_nodes)
    fixed_adj_tensor = torch.tensor(fixed_adj, dtype=torch.float32)

    model = GraphDerivNNFixedAdjMsgAblation(
        n_f=X_train.shape[-1],
        msg_dim=1,
        ndim=1,
        delt_t=1.0,
        num_nodes=num_nodes,
        fixed_adj=fixed_adj_tensor,
        msg_config=msg_config,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    print(f"Model params: {count_parameters(model):,} (adj is fixed)")

    # Save norm stats
    torch.save({
        "x_mu": x_mu.cpu(), "x_std": x_std.cpu(),
        "y_mu": y_mu.cpu(), "y_std": y_std.cpu(),
        "feature_names": feature_names,
        "msg_config": msg_config,
        "adj_type": args.adj_type,
        "msg_description": msg_description,
    }, os.path.join(args.outdir, "norm_stats.pt"))

    # DataLoaders
    train_ds = WeightedGraphDataset(X_train, Y_train, edge_index)
    val_ds = WeightedGraphDataset(X_val, Y_val, edge_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    sched = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
                       epochs=args.epochs, final_div_factor=1e4)

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_hist, val_hist = [], []
    metrics_rows = []
    start = time.time()

    for epoch in tqdm(range(args.epochs), desc=f"{args.adj_type}_{msg_config}"):
        model.train()
        tr_loss, tr_items = 0.0, 0
        for g in train_loader:
            g = g.to(device)
            opt.zero_grad()
            loss = model.loss(g, square=False)
            bsz = int(getattr(g, "num_graphs", 1))
            (loss / max(1, bsz)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            sched.step()
            tr_loss += loss.item()
            tr_items += bsz
        train_mse = tr_loss / max(1, tr_items)
        train_hist.append(train_mse)

        # Validate
        model.eval()
        va_loss, va_items = 0.0, 0
        with torch.no_grad():
            for g in val_loader:
                g = g.to(device)
                loss = model.data_loss(g, square=False)
                bsz = int(getattr(g, "num_graphs", 1))
                va_loss += loss.item()
                va_items += bsz
        val_mse = va_loss / max(1, va_items)
        val_hist.append(val_mse)

        if val_mse < (best_val_loss - args.early_stop_min_delta):
            best_val_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "state_dict": best_state,
                "adj_type": args.adj_type,
                "msg_config": msg_config,
                "msg_description": msg_description,
                "config": vars(args),
                "train_mse": train_mse,
                "val_mse": val_mse,
            }, os.path.join(args.outdir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        # Periodic metrics
        if args.metrics_every > 0 and ((epoch + 1) % args.metrics_every == 0 or epoch + 1 == args.epochs):
            train_metrics = evaluate_on_set(model, X_train, Y_train, edge_index, device)
            val_metrics = evaluate_on_set(model, X_val, Y_val, edge_index, device)
            test_metrics = evaluate_on_set(model, X_test, Y_test, edge_index, device)

            row = {
                "epoch": epoch + 1,
                "train_loss": train_mse, "val_loss": val_mse,
                "train_mse": train_metrics["mse"], "train_r2": train_metrics["r2"],
                "val_mse": val_metrics["mse"], "val_r2": val_metrics["r2"],
                "test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"], "test_corr": test_metrics["corr"],
                "test_node_r2_mean": test_metrics["node_r2_mean"],
                "test_node_r2_median": test_metrics["node_r2_median"],
                "test_node_r2_min": test_metrics["node_r2_min"],
                "test_node_r2_max": test_metrics["node_r2_max"],
            }
            metrics_rows.append(row)

            if (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
                print(f"  epoch={epoch+1} train_loss={train_mse:.4f} val_loss={val_mse:.4f} "
                      f"test_R2={test_metrics['r2']:.4f} node_R2_mean={test_metrics['node_r2_mean']:.4f}")

        if epochs_no_improve >= args.early_stop_patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best and final eval
    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed:.1f}s")

    final_train = evaluate_on_set(model, X_train, Y_train, edge_index, device)
    final_val = evaluate_on_set(model, X_val, Y_val, edge_index, device)
    final_test = evaluate_on_set(model, X_test, Y_test, edge_index, device)

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS ({args.adj_type}, {msg_config}):")
    print(f"  Train: MSE={final_train['mse']:.6f}, R2={final_train['r2']:.4f}")
    print(f"  Val:   MSE={final_val['mse']:.6f}, R2={final_val['r2']:.4f}")
    print(f"  Test:  MSE={final_test['mse']:.6f}, R2={final_test['r2']:.4f}, MAE={final_test['mae']:.6f}")
    print(f"  Test per-node R2: mean={final_test['node_r2_mean']:.4f}, "
          f"median={final_test['node_r2_median']:.4f}")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "adj_type": args.adj_type,
        "msg_config": msg_config,
        "msg_description": msg_description,
        "zscore": args.zscore,
        "smooth": args.smooth,
        "seed": args.seed,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_hist),
        "elapsed_seconds": elapsed,
        "train": {k: float(v) if isinstance(v, (float, np.floating)) else v
                  for k, v in final_train.items() if k not in ("node_mse", "node_r2", "node_corr")},
        "val": {k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in final_val.items() if k not in ("node_mse", "node_r2", "node_corr")},
        "test": {k: float(v) if isinstance(v, (float, np.floating)) else v
                 for k, v in final_test.items() if k not in ("node_mse", "node_r2", "node_corr")},
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.outdir, "training_metrics.csv"), index=False)

    with open(os.path.join(args.outdir, "loss_curves.pkl"), "wb") as f:
        pickle.dump({"train": train_hist, "val": val_hist}, f)

    node_r2_df = pd.DataFrame({
        "county": county_names,
        "test_r2": final_test["node_r2"],
        "test_corr": final_test["node_corr"],
        "test_mse": final_test["node_mse"],
    })
    node_r2_df.to_csv(os.path.join(args.outdir, "test_per_node_metrics.csv"), index=False)

    # Save features and targets
    np.save(os.path.join(args.outdir, "X_features.npy"), X_feat)
    np.save(os.path.join(args.outdir, "dXdt.npy"), dXdt)

    print(f"Results saved to {args.outdir}")


if __name__ == "__main__":
    main()
