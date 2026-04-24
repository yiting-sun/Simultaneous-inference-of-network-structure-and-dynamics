"""
Unified data generator for the discrete Rulkov chemical-synapse network.

Sub-graph selection strategy (controlled automatically by --num_nodes_keep):
  *  N <  80  ->  degree-matched subgraph (many random trials, pick the one
                  with edge-count closest to full-graph degree, tie-broken by
                  weakly-connectedness). This is the 'small network, match
                  full-graph density' branch; saved to --save_dir_small.
  *  N >= 80  ->  random subgraph (first sample whose restricted adjacency is
                  non-empty). This is the 'large network, let topology fall
                  out naturally' branch; saved to --save_dir_large.

Coupling strength rescaling:
  We rescale the per-edge gc so that (sub_avg_degree * effective_gc) matches
  (full_avg_degree * base_gc). This keeps the expected aggregate message
  strength per node roughly comparable to the reference network.

Divergence guard:
  Uses the stricter max |state| threshold of 50 (well above healthy Rulkov
  trajectories which stay within |x| < 10; catches numerical bursts that
  silently corrupt downstream MSE training).

Adjacency self-loop check:
  Refuses to run if the loaded adjacency still has any diagonal entries;
  build_ciona_cns_topology.py is expected to have stripped them already.

Filename convention (same in both output dirs):
  Series_N{N}_RulkovChemicalODE_T{T}_dt{dt_str}_gc{eff_gc}_d{deg}_seed{s}.pickle
  TrueDxdt_N{N}_...
"""

import argparse
import os
import pickle
import time

import numpy as np

from rulkov_chemical_ode import (
    NET_NAME,
    euler_rollout_checked,
    make_initial_state,
    make_node_parameters,
    reversal_potential,
    select_kept_node_indices,
)
from sync_utils import compute_sync_metrics, save_sync_metrics, sync_stats_path_from_series_path


SMALL_LARGE_THRESHOLD = 80  # N < this uses degree-matched, else random


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
        for nxt in np.flatnonzero(undirected[node]):
            if not seen[nxt]:
                seen[nxt] = True
                stack.append(int(nxt))
    active = undirected.sum(axis=1) > 0
    return bool(np.all(seen[active]))


def select_degree_matched_subgraph(full_A, num_keep, seed, target_degree,
                                   num_trials=20000, topk_refine=64):
    """Random-sample many subgraphs of size `num_keep`, return the one whose
    induced edge count is closest to `target_degree * num_keep`, with
    preference for weakly-connected results (ties broken by smallest gap)."""
    total_nodes = full_A.shape[0]
    if num_keep >= total_nodes:
        kept = np.arange(total_nodes, dtype=np.int64)
        sub_A = full_A.copy()
        return kept, sub_A, int(np.count_nonzero(sub_A)), is_weakly_connected(sub_A)

    rng = np.random.default_rng(int(seed))
    target_edges = target_degree * num_keep
    candidates = []
    best = None
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


def select_random_subgraph(full_A, num_keep, seed):
    """Random rejection-sampled subgraph (reject all-zero restrictions).
    Matches the legacy large-N pipeline."""
    kept = select_kept_node_indices(full_A.shape[0], num_keep, seed, full_A)
    sub_A = full_A[np.ix_(kept, kept)]
    edge_count = int(np.count_nonzero(sub_A))
    connected = is_weakly_connected(sub_A)
    return kept, sub_A, edge_count, connected


def load_and_verify_adjacency(adj_path):
    ext = os.path.splitext(adj_path)[1].lower()
    if ext == ".npy":
        A = np.load(adj_path).astype(np.float64)
    elif ext == ".npz":
        with np.load(adj_path, allow_pickle=True) as data:
            key = "A" if "A" in data else list(data.keys())[0]
            A = data[key].astype(np.float64)
    else:
        raise ValueError(f"Unsupported adjacency format: {adj_path}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A.shape}")
    self_loops = int(np.trace(A))
    if self_loops != 0:
        raise ValueError(
            f"Adjacency at {adj_path} has {self_loops} self-loop(s). Run "
            "build_ciona_cns_topology.py first to produce a self-loop-free adj."
        )
    return A


def main():
    parser = argparse.ArgumentParser(
        description="Generate Rulkov discrete-map time series for sub-graphs of the ciona CNS network."
    )
    default_adj = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "ciona_cns_177_binary_adj.npy",
    )
    parser.add_argument("--num_nodes_keep", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--gc", type=float, default=0.1,
                        help="Base per-edge coupling strength (before degree rescaling).")
    parser.add_argument("--synapse_type", type=str, default="excitatory",
                        choices=["excitatory", "inhibitory"])
    parser.add_argument("--v_s", type=float, default=None)
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--theta", type=float, default=-1.0)
    parser.add_argument("--adj_path", type=str, default=default_adj)
    parser.add_argument("--save_dir_small", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_small"))
    parser.add_argument("--save_dir_large", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_large"))
    parser.add_argument("--num_trials", type=int, default=20000,
                        help="Used only for degree-matched subgraph search (small-N branch).")
    parser.add_argument("--degree_tolerance", type=float, default=1.0,
                        help="Do not rescale gc if |d_sub - d_full| <= tolerance.")
    parser.add_argument("--divergence_threshold", type=float, default=50.0,
                        help="Reject rollout if any |state| exceeds this. Healthy Rulkov keeps "
                             "|x|<10; 50 is a 5x safety margin that still catches numerical bursts.")
    parser.add_argument("--gc_backoff", type=float, default=0.85,
                        help="Multiply gc by this and retry if a rollout diverges.")
    parser.add_argument("--max_gc_retries", type=int, default=20)
    args = parser.parse_args()

    if not (0.0 < args.gc_backoff < 1.0):
        raise ValueError(f"--gc_backoff must be in (0,1), got {args.gc_backoff}")
    if args.divergence_threshold <= 0:
        raise ValueError(f"--divergence_threshold must be positive, got {args.divergence_threshold}")
    if args.max_gc_retries < 0:
        raise ValueError(f"--max_gc_retries must be >= 0, got {args.max_gc_retries}")

    T, dt, N = args.T, args.dt, args.num_nodes_keep
    num_steps = int(T / dt)

    full_A = load_and_verify_adjacency(args.adj_path)
    full_edge_count = int(np.count_nonzero(full_A))
    full_avg_degree = full_edge_count / full_A.shape[0]

    # Branch on N to choose sub-graph selection strategy
    if N < SMALL_LARGE_THRESHOLD:
        branch = "small"
        save_dir = args.save_dir_small
        kept, sub_A, edge_count, connected = select_degree_matched_subgraph(
            full_A, N, args.seed, full_avg_degree, num_trials=args.num_trials,
        )
    else:
        branch = "large"
        save_dir = args.save_dir_large
        kept, sub_A, edge_count, connected = select_random_subgraph(full_A, N, args.seed)

    # Self-loop check on the induced subgraph too (should be automatic if full_A is clean).
    sub_self_loops = int(np.trace(sub_A))
    if sub_self_loops != 0:
        raise ValueError(f"Sub-graph has {sub_self_loops} self-loop(s); aborting.")

    actual_degree = edge_count / N
    if actual_degree > 0 and abs(actual_degree - full_avg_degree) > args.degree_tolerance:
        effective_gc = args.gc * full_avg_degree / actual_degree
    else:
        effective_gc = args.gc

    alpha, sigma, mu = make_node_parameters(N, args.seed)
    init = make_initial_state(N, args.seed)
    v_s = reversal_potential(args.synapse_type, args.v_s)

    print(f"Network: {NET_NAME}")
    print(f"Adjacency: {args.adj_path}, shape={full_A.shape}, edges={full_edge_count}, "
          f"avg_deg={full_avg_degree:.3f} (self-loops=0 verified)")
    print(f"Branch: {branch} (threshold N<{SMALL_LARGE_THRESHOLD} -> degree-matched; else random)")
    print(f"Sub-graph: N={N}, edges={edge_count}, avg_deg={actual_degree:.3f}, "
          f"weakly_connected={connected}")
    print(f"base_gc={args.gc}, effective_gc={effective_gc:.6f}, gc_backoff={args.gc_backoff}, "
          f"divergence_threshold={args.divergence_threshold}")
    print(f"num_steps={num_steps}, T={T}, dt={dt}")
    print(f"Parameter ranges: alpha=[{alpha.min():.3f},{alpha.max():.3f}], "
          f"sigma=[{sigma.min():.3f},{sigma.max():.3f}], mu={mu[0]:.4g}, "
          f"v_s={v_s}, lam={args.lam}, theta={args.theta}")

    start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start: {start}  — simulating ...")

    gc_attempt = effective_gc
    series = None
    for attempt in range(args.max_gc_retries + 1):
        series_try, stable, div_step, max_abs_state = euler_rollout_checked(
            init_state=init, num_steps=num_steps, dt=dt,
            A=sub_A, alpha=alpha, sigma=sigma, mu=mu,
            gc=gc_attempt, v_s=v_s, lam=args.lam, theta=args.theta,
            divergence_threshold=args.divergence_threshold,
        )
        if stable:
            series = series_try
            print(f"Stable at gc={gc_attempt:.6f} after {attempt} retries; "
                  f"max|state|={max_abs_state:.3f}")
            break
        if attempt == args.max_gc_retries:
            raise RuntimeError(
                f"Failed to get stable rollout after {attempt + 1} attempts; "
                f"last gc={gc_attempt:.6g}, divergence_step={div_step}, "
                f"max|state|={max_abs_state:.6g}"
            )
        next_gc = gc_attempt * args.gc_backoff
        print(f"  unstable at gc={gc_attempt:.6f} (step {div_step}, max|state|={max_abs_state:.3f}); "
              f"retrying with gc={next_gc:.6f}")
        gc_attempt = next_gc

    final_gc = gc_attempt
    retries_used = attempt
    end = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"End: {end}; series shape={series.shape}; "
          f"x range=[{series[:, 0::2].min():.3f}, {series[:, 0::2].max():.3f}], "
          f"y range=[{series[:, 1::2].min():.3f}, {series[:, 1::2].max():.3f}]")

    os.makedirs(save_dir, exist_ok=True)
    dt_str = float_to_tag(dt)
    gc_tag = gc_to_tag(final_gc)
    deg_tag = degree_to_tag(actual_degree)
    series_path = os.path.join(
        save_dir,
        f"Series_N{N}_{NET_NAME}_T{int(T)}_dt{dt_str}_gc{gc_tag}_d{deg_tag}_seed{args.seed}.pickle",
    )
    with open(series_path, "wb") as f:
        pickle.dump([sub_A, series], f)

    kept_path = os.path.join(
        save_dir, f"kept_nodes_N{N}_{NET_NAME}_d{deg_tag}_seed{args.seed}.npy"
    )
    np.save(kept_path, kept)

    sync_metrics = compute_sync_metrics(series, N, dims=2, discard_frac=0.5)
    sync_metrics.update({
        "series_path": series_path,
        "kept_nodes_path": kept_path,
        "branch": branch,
        "base_gc": float(args.gc),
        "effective_gc": float(final_gc),
        "full_avg_degree": float(full_avg_degree),
        "sub_avg_degree": float(actual_degree),
        "weakly_connected": bool(connected),
        "self_loops_full_adj": 0,
        "self_loops_sub_adj": 0,
        "divergence_threshold": float(args.divergence_threshold),
        "gc_backoff": float(args.gc_backoff),
        "retries_used": int(retries_used),
        "seed": int(args.seed),
        "degree_tag": deg_tag,
        "gc_tag": gc_tag,
        "stability_max_abs_state": float(max_abs_state),
    })
    sync_path = sync_stats_path_from_series_path(series_path)
    save_sync_metrics(sync_metrics, sync_path)

    print(f"Saved series : {series_path}")
    print(f"Saved kept   : {kept_path}")
    print(f"Saved sync   : {sync_path}")
    print(f"Sync summary: pairwise_corr={sync_metrics['mean_pairwise_corr_x']:.3f}, "
          f"kuramoto={sync_metrics['mean_kuramoto_x']:.3f}, sync_flag={sync_metrics['sync_flag']}")


if __name__ == "__main__":
    main()
