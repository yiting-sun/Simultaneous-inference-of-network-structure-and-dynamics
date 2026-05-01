"""
Step 2 Training Script: Structure-Only Inference with Frozen Self/Msg MLPs
==========================================================================
For each (seed, num_nodes_keep) combination:
  1. Build the same sub-node selection / features / temporal split as
     Step 1 (so the kept-node identities match exactly).
  2. Optionally uniform-subsample the training rows
     (``--num_train_samples``), in the same way as
     ``datalimits4simul_nozscore/train_ablation_msg_subnodes.py``.
  3. Load the matching Step-1 ``best_model.pt`` and copy its
     ``msg_fnc`` / ``node_fnc_self`` weights into the Step-2 model, then
     freeze them.
  4. Train ``edge_logits`` only, with simul-style softmax(./tau) +
     L1-on-edge-weights regulariser, and a per-step ``tau`` decay so the
     learned structure converges to a sparse 0/1 mask.

The Step-1 checkpoint to load is resolved as
``<pretrained_dir>/run<seed>_<msg_config>_<adj_type>_nozscore_gaussian/best_model.pt``
(matching the naming in ``run_struc_subnodes.sh``). Override with
``--pretrained_pt`` if needed.
"""

import os, sys, argparse, time, math, pickle, json
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
SEASONAL_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = SEASONAL_DIR
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_struc_only import (
    GraphDerivNNStrucOnly,
    build_fully_connected_edge_index,
    count_parameters,
    MSG_CONFIGS,
    FULL_SELF_INDICES,
    FEATURE_NAMES_MAP,
    describe_msg_config,
)


ORIGINAL_NUM_NODES = 20

# For comparing the learned adj against a reference shared-edge mask.
# Same npz files as Step 1.
REF_NPZ_MAP = {
    "runs3thr0p6": os.path.join(PROJECT_DIR, "surrogateAdj_shared_edge_masks_runs3_thr0p6.npz"),
}


class WeightedGraphDataset(Dataset):
    def __init__(self, X, Y, edge_index):
        self.X, self.Y, self.edge_index = X, Y, edge_index
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return Data(x=self.X[idx], y=self.Y[idx], edge_index=self.edge_index)


# ----------------------------- features --------------------------------

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


def uniform_subsample_indices(length, num_samples):
    """Uniformly spaced integer indices in [0, length-1] of size num_samples.
    Same logic as datalimits4simul_nozscore/train_ablation_msg_subnodes.py."""
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
    """Reproducible — must match the Step-1 selection exactly."""
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
    pred = np.asarray(pred_list)
    true = np.asarray(true_list)
    overall_r2 = compute_r2(true.reshape(-1), pred.reshape(-1))
    overall_corr = safe_corrcoef(true.reshape(-1), pred.reshape(-1))
    n_nodes = pred.shape[1]
    node_r2 = [compute_r2(true[:, i, 0], pred[:, i, 0]) for i in range(n_nodes)]
    node_corr = [safe_corrcoef(true[:, i, 0], pred[:, i, 0]) for i in range(n_nodes)]
    return {
        "overall_r2": float(overall_r2) if not np.isnan(overall_r2) else np.nan,
        "overall_corr": float(overall_corr) if not np.isnan(overall_corr) else np.nan,
        "node_r2_mean": float(np.nanmean(node_r2)),
        "node_r2_median": float(np.nanmedian(node_r2)),
        "node_r2_min": float(np.nanmin(node_r2)),
        "node_r2_max": float(np.nanmax(node_r2)),
        "node_corr_mean": float(np.nanmean(node_corr)),
    }


def compute_structure_metrics_soft(true_adj: np.ndarray, pred_adj: np.ndarray) -> dict:
    mask = ~np.eye(true_adj.shape[0], dtype=bool)
    y_true = true_adj[mask].astype(np.float64).reshape(-1)
    y_score = pred_adj[mask].astype(np.float64).reshape(-1)
    unique_labels = np.unique(y_true.astype(np.int64))
    if unique_labels.size < 2:
        auc = np.nan
        auprc = np.nan
    else:
        auc = float(roc_auc_score(y_true.astype(np.int64), y_score))
        auprc = float(average_precision_score(y_true.astype(np.int64), y_score))
    return {
        "auc": auc,
        "auprc": auprc,
        "corr": safe_corrcoef(y_true, y_score),
        "mse": float(np.mean((y_true - y_score) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_score))),
        "mean_weight": float(np.mean(y_score)),
        "std_weight": float(np.std(y_score)),
    }


def load_reference_mask_full(ref_label, msg_config, expected_full_nodes):
    """Used as the "ground truth" for structure metrics. Same role as
    A_true in the simul training script — here it comes from the reference
    npz, sliced to kept nodes by the caller."""
    npz_path = REF_NPZ_MAP[ref_label]
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"reference npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    key = f"mask_{msg_config}"
    if key not in data:
        raise KeyError(f"{key} not found in {npz_path}")
    if "msg_configs" in data:
        saved_msg_configs = [str(x) for x in data["msg_configs"].tolist()]
    elif "msg_config" in data:
        raw = data["msg_config"].tolist()
        if isinstance(raw, (list, tuple)):
            saved_msg_configs = [str(x) for x in raw]
        else:
            saved_msg_configs = [str(raw)]
    else:
        saved_msg_configs = [msg_config]
    if msg_config not in saved_msg_configs:
        raise ValueError(f"msg_config={msg_config} not listed in {npz_path}: {saved_msg_configs}")
    A = data[key].astype(np.float32)
    if A.shape != (expected_full_nodes, expected_full_nodes):
        raise ValueError(
            f"Reference adj shape mismatch: expected "
            f"{(expected_full_nodes, expected_full_nodes)}, got {A.shape}"
        )
    np.fill_diagonal(A, 0)
    return A


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


def resolve_pretrained_pt(pretrained_dir, seed, msg_config, adj_type, num_nodes_keep,
                          override_pt=None):
    """Resolve which Step-1 best_model.pt to load.

    Default scheme matches ``run_struc_subnodes.sh``:
        <pretrained_dir>/runs_struc_msg_nodes<N>/run<seed>_<msg>_<adj>_nozscore_gaussian/best_model.pt
    For N=20 the seed in the path is hard-coded to 1, since Step 1 only runs
    one seed at the full graph size.
    """
    if override_pt:
        return override_pt
    n_for_path = num_nodes_keep
    seed_for_path = seed if num_nodes_keep < ORIGINAL_NUM_NODES else 1
    run_name = f"run{seed_for_path}_{msg_config}_{adj_type}_nozscore_gaussian"
    return os.path.join(
        pretrained_dir,
        f"runs_struc_msg_nodes{n_for_path}",
        run_name,
        "best_model.pt",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: train edge_logits only with frozen self/msg MLPs "
                    "loaded from a Step-1 fixed-adj checkpoint.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--msg_config", type=str, default="rate_x",
                        choices=list(MSG_CONFIGS.keys()))
    parser.add_argument("--adj_type", type=str, default="runs3thr0p6",
                        choices=list(REF_NPZ_MAP.keys()),
                        help="Reference adj for structure metrics, AND used to "
                             "resolve which Step-1 checkpoint to load.")
    parser.add_argument("--pretrained_dir", type=str, default=CURRENT_DIR,
                        help="Root dir containing the Step-1 runs_struc_msg_nodes<N>/ folders.")
    parser.add_argument("--pretrained_pt", type=str, default="",
                        help="Direct path to a Step-1 best_model.pt; overrides --pretrained_dir resolution.")
    parser.add_argument("--zscore", type=str2bool, default=False)
    parser.add_argument("--smooth", type=str, default="gaussian", choices=["gaussian", "none"])
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=100,
                        help="MUST match the Step-1 checkpoint's hidden width.")
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="MUST match the Step-1 checkpoint's dropout.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--batch_ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--metrics_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seasonal_period", type=int, default=52)
    parser.add_argument("--use_seed", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2,
                        help="Sub-graph / sub-node selection seed. Also used to "
                             "resolve the matching Step-1 checkpoint.")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="Seed for torch / numpy random initialization. "
                             "Decoupled from --seed so that, for each sub-graph, "
                             "we can run multiple NN-init replicates. Defaults "
                             "to --seed when not provided.")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--tau_decay", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--num_train_samples", type=int, default=-1,
                        help="Uniformly subsample the training set to this many samples "
                             "AFTER features are built. -1 = use all.")
    parser.add_argument("--num_nodes_keep", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    checkpoints_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    if args.init_seed is None:
        args.init_seed = args.seed
    if args.use_seed:
        torch.manual_seed(args.init_seed)
        np.random.seed(args.init_seed)
    device = torch.device(args.device)

    if args.zscore:
        raise ValueError("This script only supports --zscore false (nozscore).")

    msg_config = args.msg_config
    msg_description = describe_msg_config(msg_config)

    # ---- Resolve & load the Step-1 checkpoint ---------------------------
    pretrained_pt = resolve_pretrained_pt(
        pretrained_dir=args.pretrained_dir,
        seed=args.seed,
        msg_config=msg_config,
        adj_type=args.adj_type,
        num_nodes_keep=args.num_nodes_keep,
        override_pt=args.pretrained_pt or None,
    )
    if not os.path.isfile(pretrained_pt):
        raise FileNotFoundError(f"Step-1 checkpoint not found: {pretrained_pt}")
    print(f"Loading pretrained MLPs from: {pretrained_pt}")
    payload = torch.load(pretrained_pt, map_location="cpu", weights_only=False)
    pretrained_state = payload["state_dict"]

    print(f"{'=' * 60}")
    print(f"Step2 (struc-only): adj_type={args.adj_type}, msg={msg_config}")
    print(f"seed={args.seed}, init_seed={args.init_seed}, "
          f"num_nodes_keep={args.num_nodes_keep}, "
          f"num_train_samples={args.num_train_samples}")
    print(f"Output: {args.outdir}")
    print(f"{'=' * 60}")

    # --- Load + slice data (must match Step 1) ---------------------------
    ts_csv = os.path.join(PROJECT_DIR, "hungary_chickenpox_reordered_trimmed.csv")
    edge_csv = os.path.join(PROJECT_DIR, "hungary_county_edges.csv")
    adj_npy = os.path.join(PROJECT_DIR, "hungary_chickenpox_adj.npy")
    print(f"Data sources: ts_csv={ts_csv}, edge_csv={edge_csv}, adj_npy={adj_npy}")
    ts = pd.read_csv(ts_csv)
    full_county_names = list(ts.columns[1:])
    X_raw_full = ts.iloc[:, 1:].to_numpy(dtype=np.float32)
    full_n = X_raw_full.shape[1]
    if full_n != ORIGINAL_NUM_NODES:
        print(f"WARNING: expected {ORIGINAL_NUM_NODES} columns, got {full_n}.")

    kept_node_indices = select_kept_node_indices(
        total_nodes=full_n, num_keep=args.num_nodes_keep, seed=args.seed
    )
    dropped_node_indices = np.sort(np.array(
        [i for i in range(full_n) if i not in set(kept_node_indices.tolist())],
        dtype=np.int64,
    ))
    county_names = [full_county_names[i] for i in kept_node_indices]
    X_raw = X_raw_full[:, kept_node_indices]
    print(f"Node dropping: kept {len(kept_node_indices)}/{full_n} nodes")
    print(f"  kept_indices    = {kept_node_indices.tolist()}")
    np.save(os.path.join(args.outdir, "kept_node_indices.npy"), kept_node_indices)
    np.save(os.path.join(args.outdir, "dropped_node_indices.npy"), dropped_node_indices)

    # Sanity-check: kept indices should match the Step-1 checkpoint.
    if "kept_node_indices" in payload:
        ck_kept = list(payload["kept_node_indices"])
        if ck_kept != kept_node_indices.tolist():
            print(
                "[warn] kept_node_indices in this Step-2 run "
                f"({kept_node_indices.tolist()}) does not match the Step-1 "
                f"checkpoint ({ck_kept}). Did you change --seed or --num_nodes_keep?"
            )

    X = gaussian_filter1d(X_raw, sigma=1.5, axis=0, mode="nearest") if args.smooth == "gaussian" else X_raw

    X_feat, dXdt, feature_names = build_longwindow_features_t(X, dt=1.0, seasonal_period=args.seasonal_period)
    X_t = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(dXdt, dtype=torch.float32)

    num_nodes = X_feat.shape[1]
    assert num_nodes == args.num_nodes_keep, (num_nodes, args.num_nodes_keep)
    edge_index = build_fully_connected_edge_index(num_nodes)

    # --- 60/20/20 split --------------------------------------------------
    T = X_t.shape[0]
    train_end = int(T * args.train_ratio)
    val_end = int(T * (args.train_ratio + args.val_ratio))

    X_train_full = X_t[:train_end]
    Y_train_full = Y_t[:train_end]
    X_val, Y_val = X_t[train_end:val_end], Y_t[train_end:val_end]
    X_test, Y_test = X_t[val_end:], Y_t[val_end:]

    # --- Uniform subsampling (same as simul script) ----------------------
    train_full_size = X_train_full.shape[0]
    if args.num_train_samples is None or args.num_train_samples <= 0 or args.num_train_samples >= train_full_size:
        train_indices = np.arange(train_full_size, dtype=np.int64)
        effective_num_train = train_full_size
    else:
        train_indices = uniform_subsample_indices(train_full_size, args.num_train_samples)
        effective_num_train = int(train_indices.size)

    X_train = X_train_full[train_indices]
    Y_train = Y_train_full[train_indices]
    print(
        f"Data split: train={X_train.shape[0]} (subsampled from {train_full_size}), "
        f"val={X_val.shape[0]}, test={X_test.shape[0]} (total feat rows={T})"
    )
    np.save(os.path.join(args.outdir, "train_indices.npy"), train_indices)

    # --- Reference adjacency (sliced to kept nodes) for structure metrics --
    A_ref_full = load_reference_mask_full(args.adj_type, msg_config, full_n)
    A_ref = A_ref_full[np.ix_(kept_node_indices, kept_node_indices)].astype(np.float32)
    np.fill_diagonal(A_ref, 0)
    np.save(os.path.join(args.outdir, "reference_adjacency.npy"), A_ref)
    np.save(os.path.join(args.outdir, "reference_adjacency_full.npy"), A_ref_full)

    # --- DataLoaders -----------------------------------------------------
    train_ds = WeightedGraphDataset(X_train, Y_train, edge_index)
    if args.batch_ratio is not None and args.batch_ratio > 0:
        effective_batch = max(1, int(math.ceil(args.batch_ratio * len(train_ds))))
    else:
        effective_batch = max(1, args.batch)
    print(f"Batch size: {effective_batch} (batch_ratio={args.batch_ratio}, "
          f"train_ds_size={len(train_ds)})")
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=False, num_workers=0)
    val_loader = DataLoader(WeightedGraphDataset(X_val, Y_val, edge_index),
                            batch_size=max(1, effective_batch // 2), shuffle=False)
    test_loader = DataLoader(WeightedGraphDataset(X_test, Y_test, edge_index),
                             batch_size=max(1, effective_batch // 2), shuffle=False)

    # --- Build model & load frozen MLPs ----------------------------------
    Dyn = GraphDerivNNStrucOnly(
        n_f=X_train.shape[-1],
        msg_dim=1,
        ndim=1,
        delt_t=1.0,
        num_nodes=num_nodes,
        tau=args.tau,
        lam=args.lam,
        msg_config=msg_config,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    loaded_keys = Dyn.load_pretrained_mlps(pretrained_state, strict=True)
    print(f"Loaded {len(loaded_keys)} frozen MLP params from Step-1 checkpoint.")
    n_trainable = sum(p.numel() for p in Dyn.trainable_parameters())
    print(f"Trainable params (edge_logits only): {n_trainable:,}")

    # Save norm stats / metadata.
    torch.save({
        "feature_names": feature_names,
        "msg_config": msg_config,
        "msg_description": msg_description,
        "adj_type": args.adj_type,
        "num_train_samples": effective_num_train,
        "train_full_size": train_full_size,
        "num_nodes": int(num_nodes),
        "kept_node_indices": kept_node_indices.tolist(),
        "dropped_node_indices": dropped_node_indices.tolist(),
        "kept_county_names": county_names,
        "pretrained_pt": pretrained_pt,
        "hidden": int(args.hidden),
        "dropout": float(args.dropout),
    }, os.path.join(args.outdir, "norm_stats.pt"))

    # --- Optimiser only over edge_logits --------------------------------
    opt = torch.optim.Adam(Dyn.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    sched = OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
                       epochs=args.epochs, final_div_factor=1e4)
    tau_decay_step = args.tau_decay ** (1.0 / steps_per_epoch)

    print(f"Tau: {args.tau}, Tau decay per epoch: {args.tau_decay}, Lambda: {args.lam}")

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_hist, val_hist, test_hist = [], [], []
    sparse_metrics_rows = []
    start = time.time()
    tau = float(args.tau)

    desc = f"struc_{msg_config}_N{num_nodes}_n{effective_num_train}"
    for epoch in tqdm(range(args.epochs), desc=desc):
        Dyn.train()
        # Make sure frozen MLPs stay in eval mode (no dropout) so the
        # signal coming through them is deterministic for the structure
        # search.
        Dyn.msg_fnc.eval()
        Dyn.node_fnc_self.eval()

        tr_loss = 0.0
        tr_items = 0
        for g in train_loader:
            g = g.to(device)
            opt.zero_grad()
            loss = Dyn.loss(g, square=False)
            bsz = int(getattr(g, "num_graphs", 1))
            (loss / max(1, bsz)).backward()
            torch.nn.utils.clip_grad_norm_(Dyn.trainable_parameters(), max_norm=5.0)
            opt.step()
            sched.step()
            tau *= tau_decay_step
            Dyn.update_tau(tau)
            tr_loss += loss.item()
            tr_items += bsz
        train_mse = tr_loss / max(1, tr_items)
        train_hist.append(train_mse)

        Dyn.eval()
        va_loss = 0.0
        va_items = 0
        with torch.no_grad():
            for g in val_loader:
                g = g.to(device)
                loss = Dyn.eval_loss(g, square=False)
                bsz = int(getattr(g, "num_graphs", 1))
                va_loss += loss.item()
                va_items += bsz
        val_mse = va_loss / max(1, va_items)
        val_hist.append(val_mse)

        te_loss = 0.0
        te_items = 0
        with torch.no_grad():
            for g in test_loader:
                g = g.to(device)
                loss = Dyn.eval_loss(g, square=False)
                bsz = int(getattr(g, "num_graphs", 1))
                te_loss += loss.item()
                te_items += bsz
        test_mse = te_loss / max(1, te_items)
        test_hist.append(test_mse)

        if val_mse < (best_val - args.early_stop_min_delta):
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in Dyn.state_dict().items()}
            epochs_without_improvement = 0
            inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
            structure_metrics = compute_structure_metrics_soft(A_ref, inferred_adj)
            torch.save({
                "epoch": epoch + 1,
                "state_dict": best_state,
                "config": vars(args),
                "kept_node_indices": kept_node_indices.tolist(),
                "msg_config": msg_config,
                "msg_description": msg_description,
                "adj_type": args.adj_type,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "structure_metrics": structure_metrics,
                "inferred_adjacency": inferred_adj,
                "num_train_samples": effective_num_train,
                "num_nodes": int(num_nodes),
                "pretrained_pt": pretrained_pt,
            }, os.path.join(args.outdir, "best_model.pt"))
            save_inferred_adjacency(os.path.join(args.outdir, "best_inferred_adjacency.npy"), inferred_adj)
        else:
            epochs_without_improvement += 1

        # Periodic full metrics
        if args.metrics_every > 0 and ((epoch + 1) % args.metrics_every == 0 or epoch + 1 == args.epochs):
            inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
            structure_metrics = compute_structure_metrics_soft(A_ref, inferred_adj)
            train_metrics = evaluate_sequence_metrics(Dyn, X_train, Y_train, edge_index, device)
            val_metrics = evaluate_sequence_metrics(Dyn, X_val, Y_val, edge_index, device)
            test_metrics = evaluate_sequence_metrics(Dyn, X_test, Y_test, edge_index, device)
            row = {
                "epoch": epoch + 1,
                "tau": float(Dyn.tau),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "test_mse": float(test_mse),
                "train_overall_r2": train_metrics["overall_r2"],
                "val_overall_r2": val_metrics["overall_r2"],
                "test_overall_r2": test_metrics["overall_r2"],
                "test_node_r2_mean": test_metrics["node_r2_mean"],
                "struct_auc": structure_metrics["auc"],
                "struct_auprc": structure_metrics["auprc"],
                "struct_corr": structure_metrics["corr"],
                "struct_mse": structure_metrics["mse"],
                "struct_mae": structure_metrics["mae"],
                "struct_mean_weight": structure_metrics["mean_weight"],
            }
            sparse_metrics_rows.append(row)
            save_inferred_adjacency(
                os.path.join(args.outdir, f"inferred_adjacency_epoch_{epoch + 1:04d}.npy"),
                inferred_adj,
            )
            if (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
                print(f"  epoch={epoch+1} train={train_mse:.4f} val={val_mse:.4f} test={test_mse:.4f} "
                      f"test_R2={test_metrics['overall_r2']:.4f} "
                      f"AUC={structure_metrics['auc']} AUPRC={structure_metrics['auprc']}")

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        Dyn.load_state_dict(best_state)

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed:.1f}s")

    # --- Final evaluation on best model ---------------------------------
    final_inferred_adj = Dyn.get_adjacency_matrix().detach().cpu().numpy()
    final_structure_metrics = compute_structure_metrics_soft(A_ref, final_inferred_adj)
    final_train = evaluate_sequence_metrics(Dyn, X_train, Y_train, edge_index, device)
    final_val = evaluate_sequence_metrics(Dyn, X_val, Y_val, edge_index, device)
    final_test = evaluate_sequence_metrics(Dyn, X_test, Y_test, edge_index, device)

    print(f"\n{'=' * 60}")
    print(f"FINAL ({args.adj_type}, {msg_config}, N={num_nodes}, n_train={effective_num_train})")
    print(f"  test R2 = {final_test['overall_r2']:.4f}")
    print(f"  AUC     = {final_structure_metrics['auc']}")
    print(f"  AUPRC   = {final_structure_metrics['auprc']}")
    print(f"{'=' * 60}")

    # --- Save final adj + summary ---------------------------------------
    save_inferred_adjacency(os.path.join(args.outdir, "final_inferred_adjacency.npy"), final_inferred_adj)

    def _f(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    results = {
        "run_name": os.path.basename(os.path.normpath(args.outdir)),
        "msg_config": msg_config,
        "adj_type": args.adj_type,
        "num_nodes_keep": int(num_nodes),
        "num_train_samples": int(effective_num_train),
        "total_samples": int(num_nodes) * int(effective_num_train),
        "epochs_run": len(train_hist),
        "best_val_mse": _f(best_val),
        "elapsed_seconds": float(elapsed),
        "pretrained_pt": pretrained_pt,
        "train": {"mse": _f(train_hist[-1] if train_hist else np.nan),
                  "r2": _f(final_train["overall_r2"]),
                  "node_r2_mean": _f(final_train["node_r2_mean"])},
        "val": {"mse": _f(val_hist[-1] if val_hist else np.nan),
                "r2": _f(final_val["overall_r2"]),
                "node_r2_mean": _f(final_val["node_r2_mean"])},
        "test": {"mse": _f(test_hist[-1] if test_hist else np.nan),
                 "r2": _f(final_test["overall_r2"]),
                 "node_r2_mean": _f(final_test["node_r2_mean"])},
        "structure": {k: _f(v) for k, v in final_structure_metrics.items()},
        "config": {
            "lam": _f(args.lam), "tau": _f(args.tau), "tau_decay": _f(args.tau_decay),
            "seed": int(args.seed), "init_seed": int(args.init_seed),
            "hidden": int(args.hidden), "dropout": float(args.dropout),
        },
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    metrics_df = pd.DataFrame(sparse_metrics_rows)
    metrics_df.to_csv(os.path.join(args.outdir, f"training_metrics_every{args.metrics_every}.csv"), index=False)
    with open(os.path.join(args.outdir, "loss_curves.pkl"), "wb") as f:
        pickle.dump({
            "train_mse": train_hist,
            "val_mse": val_hist,
            "test_mse": test_hist,
            "metrics_every": args.metrics_every,
            "metrics_epochs": metrics_df["epoch"].tolist() if len(metrics_df) else [],
            "train_overall_r2": metrics_df["train_overall_r2"].tolist() if len(metrics_df) else [],
            "val_overall_r2": metrics_df["val_overall_r2"].tolist() if len(metrics_df) else [],
            "test_overall_r2": metrics_df["test_overall_r2"].tolist() if len(metrics_df) else [],
            "struct_auc": metrics_df["struct_auc"].tolist() if len(metrics_df) else [],
            "struct_auprc": metrics_df["struct_auprc"].tolist() if len(metrics_df) else [],
            "struct_corr": metrics_df["struct_corr"].tolist() if len(metrics_df) else [],
            "struct_mse": metrics_df["struct_mse"].tolist() if len(metrics_df) else [],
            "struct_mae": metrics_df["struct_mae"].tolist() if len(metrics_df) else [],
        }, f)

    print(f"Results saved to {args.outdir}")


if __name__ == "__main__":
    main()
