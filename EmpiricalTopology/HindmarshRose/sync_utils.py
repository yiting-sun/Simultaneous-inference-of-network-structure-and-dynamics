import json
import os
import pickle
import re

import numpy as np
from scipy.signal import hilbert


SERIES_RE = re.compile(
    r"Series_N(?P<N>\d+)_Celegans_T(?P<T>\d+)_dt(?P<dt>[^_]+)_gc(?P<gc>[^_]+)_d(?P<d>[^_]+)_seed(?P<seed>\d+)\.pickle$"
)


def tag_to_float(tag):
    return float(tag.replace("p", "."))


def parse_series_filename(path):
    name = os.path.basename(path)
    match = SERIES_RE.match(name)
    if match is None:
        raise ValueError(f"Unrecognized series filename: {path}")
    meta = match.groupdict()
    return {
        "N": int(meta["N"]),
        "T": int(meta["T"]),
        "dt_tag": meta["dt"],
        "gc_tag": meta["gc"],
        "d_tag": meta["d"],
        "seed": int(meta["seed"]),
        "gc": tag_to_float(meta["gc"]),
        "degree": tag_to_float(meta["d"]),
    }


def sync_stats_path_from_series_path(series_path):
    return series_path.replace("/Series_", "/SyncStats_").replace("Series_", "SyncStats_").replace(".pickle", ".json")


def compute_sync_metrics(series, num_nodes, dims=3, discard_frac=0.5):
    series = np.asarray(series)
    if series.ndim != 2 or series.shape[1] != num_nodes * dims:
        raise ValueError(f"Unexpected series shape {series.shape} for num_nodes={num_nodes}, dims={dims}")

    series_3d = series.reshape(series.shape[0], num_nodes, dims)
    V = series_3d[:, :, 0]
    start = min(max(int(discard_frac * V.shape[0]), 0), max(V.shape[0] - 1, 0))
    V_tail = V[start:]

    inst_std = V_tail.std(axis=1)
    node_means = V_tail.mean(axis=0)
    node_stds = V_tail.std(axis=0)

    corr = np.corrcoef(V_tail.T)
    if corr.ndim == 0:
        pairwise_corr = np.array([1.0])
    else:
        iu = np.triu_indices_from(corr, k=1)
        pairwise_corr = corr[iu]
        pairwise_corr = pairwise_corr[np.isfinite(pairwise_corr)]
        if pairwise_corr.size == 0:
            pairwise_corr = np.array([np.nan])

    analytic = hilbert(V_tail, axis=0)
    phase = np.angle(analytic)
    kuramoto_t = np.abs(np.exp(1j * phase).mean(axis=1))

    mean_pairwise_corr = float(np.nanmean(pairwise_corr))
    mean_kuramoto = float(np.nanmean(kuramoto_t))
    std_mean = float(np.mean(inst_std))

    sync_flag = bool((mean_pairwise_corr >= 0.95) and (mean_kuramoto >= 0.95))

    return {
        "num_steps": int(series.shape[0]),
        "num_nodes": int(num_nodes),
        "discard_start": int(start),
        "mean_inst_std_v": std_mean,
        "max_inst_std_v": float(np.max(inst_std)),
        "min_inst_std_v": float(np.min(inst_std)),
        "mean_node_std_v": float(np.mean(node_stds)),
        "mean_node_mean_v": float(np.mean(node_means)),
        "mean_pairwise_corr_v": mean_pairwise_corr,
        "median_pairwise_corr_v": float(np.nanmedian(pairwise_corr)),
        "min_pairwise_corr_v": float(np.nanmin(pairwise_corr)),
        "mean_kuramoto_v": mean_kuramoto,
        "median_kuramoto_v": float(np.median(kuramoto_t)),
        "min_kuramoto_v": float(np.min(kuramoto_t)),
        "sync_flag": sync_flag,
    }


def compute_sync_metrics_from_series_path(series_path, dims=3, discard_frac=0.5):
    meta = parse_series_filename(series_path)
    with open(series_path, "rb") as f:
        _, series = pickle.load(f)
    metrics = compute_sync_metrics(series, meta["N"], dims=dims, discard_frac=discard_frac)
    metrics.update(meta)
    metrics["series_path"] = series_path
    metrics["sync_stats_path"] = sync_stats_path_from_series_path(series_path)
    return metrics


def save_sync_metrics(metrics, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
