import argparse
import os
import pickle
import time

import numpy as np
from scipy.integrate import odeint

from sync_utils import compute_sync_metrics, save_sync_metrics, sync_stats_path_from_series_path


def float_to_tag(value):
    return format(value, "g").replace(".", "")


def degree_to_tag(value):
    return format(value, ".3f").rstrip("0").rstrip(".").replace(".", "p")


def gc_to_tag(value):
    return format(value, ".6f").rstrip("0").rstrip(".").replace(".", "p")


def is_weakly_connected(adj):
    n = adj.shape[0]
    if n <= 1:
        return True
    undirected = np.logical_or(adj != 0, adj.T != 0)
    if not np.any(undirected):
        return False
    start = int(np.flatnonzero(undirected.sum(axis=1) > 0)[0])
    seen = np.zeros(n, dtype=bool)
    stack = [start]
    seen[start] = True
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(undirected[node])
        for nxt in neighbors:
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(int(nxt))
    active = undirected.sum(axis=1) > 0
    return bool(np.all(seen[active]))


def select_degree_matched_subgraph(full_A, num_keep, seed, target_degree,
                                    num_trials=20000, topk_refine=64):
    total_nodes = full_A.shape[0]
    if num_keep >= total_nodes:
        kept = np.arange(total_nodes, dtype=np.int64)
        sub_A = full_A.copy()
        return kept, sub_A, int(np.count_nonzero(sub_A)), is_weakly_connected(sub_A)

    rng = np.random.default_rng(int(seed))
    target_edges = target_degree * num_keep
    best = None
    candidates = []

    for _ in range(num_trials):
        kept = np.sort(rng.choice(total_nodes, size=num_keep, replace=False)).astype(np.int64)
        sub_A = full_A[np.ix_(kept, kept)]
        edge_count = int(np.count_nonzero(sub_A))
        gap = abs(edge_count - target_edges)
        candidates.append((gap, kept, edge_count))
        if best is None or gap < best[0]:
            best = (gap, kept, edge_count)
        if gap == 0:
            break

    candidates.sort(key=lambda item: item[0])
    refined = candidates[:max(1, min(topk_refine, len(candidates)))]

    best_choice = None
    for gap, kept, edge_count in refined:
        sub_A = full_A[np.ix_(kept, kept)]
        connected = is_weakly_connected(sub_A)
        zero_edge_penalty = 1 if edge_count == 0 else 0
        score = (zero_edge_penalty, 0 if connected else 1, gap, -edge_count)
        if best_choice is None or score < best_choice[0]:
            best_choice = (score, kept, sub_A, edge_count, connected)

    if best_choice is None:
        kept = best[1]
        sub_A = full_A[np.ix_(kept, kept)]
        return kept, sub_A, best[2], is_weakly_connected(sub_A)

    return best_choice[1], best_choice[2], best_choice[3], best_choice[4]


def HR_generate_vectorized(x, t, A, gc):
    dxdt = np.zeros_like(x)
    a = 1
    b = 3
    c = 1
    d = 5
    s = 4
    r = 0.004
    p0 = -1.6
    Iext = 3.24
    Vsyn1 = 2
    k = -10
    nodes = A.shape[0]

    x_reshaped = x.reshape((nodes, 3))
    V = x_reshaped[:, 0]
    tmp = gc * (Vsyn1 - V[:, None]) * A / (1 + np.exp(k * (V[None, :] - 1)))
    tmp_sum = tmp.sum(axis=1)

    dxdt[::3] = x_reshaped[:, 1] - a * V ** 3 + b * V ** 2 - x_reshaped[:, 2] + Iext + tmp_sum
    dxdt[1::3] = c - d * V ** 2 - x_reshaped[:, 1]
    dxdt[2::3] = r * (s * (V - p0) - x_reshaped[:, 2])
    return dxdt


def main():
    parser = argparse.ArgumentParser(
        description="Generate HR time series on degree-matched Celegans subgraphs")
    parser.add_argument("--num_nodes_keep", type=int, default=279,
                        help="Number of nodes to keep. 279 = full network.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for subgraph node selection and initial conditions.")
    parser.add_argument("--T", type=int, default=500, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--gc", type=float, default=0.15, help="Coupling strength")
    parser.add_argument("--degree_tolerance", type=float, default=1.0,
                        help="Acceptable absolute deviation from full-graph average degree "
                             "before rescaling coupling.")
    parser.add_argument("--num_trials", type=int, default=20000,
                        help="Number of random subgraph candidates to evaluate.")
    args = parser.parse_args()

    T = args.T
    dt = args.dt
    gc = args.gc
    original_num_nodes = 279
    net_name = "Celegans"
    dims = 3

    src_path = f"HR_dyn/data/Aij_N{original_num_nodes}_{net_name}_ind1.npy"
    full_A = np.asarray(np.load(src_path))

    full_edge_count = int(np.count_nonzero(full_A))
    full_avg_degree = float(full_edge_count / full_A.shape[0])
    full_target_degree = full_avg_degree
    target_edges = full_target_degree * args.num_nodes_keep

    print(f"Loaded full adjacency: shape={full_A.shape}, "
          f"edges={int(np.count_nonzero(full_A))}, avg_in_degree={full_avg_degree:.3f}")
    print(f"Sampling N={args.num_nodes_keep}, target_degree={full_target_degree:.3f}, "
          f"target_edges={target_edges:.2f}, seed={args.seed}, trials={args.num_trials}")

    kept, A, edge_count, connected = select_degree_matched_subgraph(
        full_A, args.num_nodes_keep, args.seed, full_target_degree, num_trials=args.num_trials)
    actual_degree = edge_count / args.num_nodes_keep
    degree_tag = degree_to_tag(actual_degree)
    if actual_degree > 0 and abs(actual_degree - full_target_degree) > args.degree_tolerance:
        effective_gc = gc * full_target_degree / actual_degree
    else:
        effective_gc = gc
    gc_tag = gc_to_tag(effective_gc)

    print(f"Selected subgraph: edges={edge_count}, avg_in_degree={actual_degree:.3f}, "
          f"weakly_connected={connected}, effective_gc={effective_gc:.6f}")
    print(f"First kept indices: {kept.tolist()[:10]}")

    num_steps = int(T / dt)
    print(f"Total steps: {num_steps}, T={T}, dt={dt}, base_gc={gc}, effective_gc={effective_gc}")

    ic_seed = 42
    np.random.seed(ic_seed)
    full_init_V = np.random.uniform(-1.5, 1.5, original_num_nodes)
    full_init_y = np.random.uniform(-10, 0, original_num_nodes)
    full_init_z = np.random.uniform(1.0, 3.5, original_num_nodes)

    init = np.zeros(args.num_nodes_keep * dims)
    init[0::3] = full_init_V[kept]
    init[1::3] = full_init_y[kept]
    init[2::3] = full_init_z[kept]

    tspan = np.arange(0, T, dt)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start time: {start_time}")
    print("Simulating HR dynamics on degree-matched Celegans subgraph ...")
    series = odeint(HR_generate_vectorized, init, tspan, args=(A, effective_gc))
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print(f"Simulation finished! series shape: {series.shape}")
    print(f"End time: {end_time}")

    data_dir = "HR_dyn/data_new"
    os.makedirs(data_dir, exist_ok=True)

    dt_str = float_to_tag(dt)
    save_path = (
        f"{data_dir}/Series_N{args.num_nodes_keep}_{net_name}_T{T}_dt{dt_str}"
        f"_gc{gc_tag}_d{degree_tag}_seed{args.seed}.pickle"
    )
    kept_path = f"{data_dir}/kept_nodes_N{args.num_nodes_keep}_d{degree_tag}_seed{args.seed}.npy"

    with open(save_path, "wb") as f:
        pickle.dump([A, series], f)
    np.save(kept_path, kept)
    sync_metrics = compute_sync_metrics(series, args.num_nodes_keep, dims=dims, discard_frac=0.5)
    sync_metrics.update({
        "series_path": save_path,
        "kept_nodes_path": kept_path,
        "effective_gc": effective_gc,
        "base_gc": gc,
        "degree_tag": degree_tag,
        "gc_tag": gc_tag,
        "seed": int(args.seed),
    })
    sync_path = sync_stats_path_from_series_path(save_path)
    save_sync_metrics(sync_metrics, sync_path)

    print(f"Saved series to: {save_path}")
    print(f"Saved kept nodes to: {kept_path}")
    print(f"Saved sync stats to: {sync_path}")
    print(f"Sync summary: sync_flag={sync_metrics['sync_flag']}, "
          f"mean_pairwise_corr_v={sync_metrics['mean_pairwise_corr_v']:.4f}, "
          f"mean_kuramoto_v={sync_metrics['mean_kuramoto_v']:.4f}")
    print(f"series shape: {series.shape} (expected ({num_steps}, {args.num_nodes_keep * dims}))")


if __name__ == "__main__":
    main()
