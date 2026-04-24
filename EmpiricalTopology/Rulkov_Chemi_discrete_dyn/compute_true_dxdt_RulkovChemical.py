"""
Compute true dx/dt for a pre-generated Rulkov discrete-map time series.

Auto-detects whether the series lives in `data_small/` (N<80) or `data_large/`
(N>=80) based on --num_nodes_keep, and writes TrueDxdt_... next to the Series
file it opened.

Refuses to run if the stored adjacency has any self-loops.
"""

import argparse
import glob
import os
import pickle
import re
import time

import numpy as np

from rulkov_chemical_ode import (
    NET_NAME,
    make_node_parameters,
    reversal_potential,
    rulkov_chemical_rhs_batch,
)


SMALL_LARGE_THRESHOLD = 80


def float_to_tag(value):
    return format(value, "g").replace(".", "")


def main():
    parser = argparse.ArgumentParser(
        description="Compute true dx/dt for the discrete Rulkov chemical-synapse dataset."
    )
    parser.add_argument("--num_nodes_keep", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--synapse_type", type=str, default="excitatory",
                        choices=["excitatory", "inhibitory"])
    parser.add_argument("--v_s", type=float, default=None)
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--theta", type=float, default=-1.0)
    parser.add_argument("--data_dir", type=str, default="",
                        help="Override auto-routing; otherwise use data_small/ for "
                             "N<80 else data_large/.")
    args = parser.parse_args()

    N = args.num_nodes_keep
    dt_str = float_to_tag(args.dt)
    v_s = reversal_potential(args.synapse_type, args.v_s)

    base = os.path.dirname(os.path.abspath(__file__))
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(base, "data_small" if N < SMALL_LARGE_THRESHOLD else "data_large")

    pattern = os.path.join(
        data_dir,
        f"Series_N{N}_{NET_NAME}_T{int(args.T)}_dt{dt_str}_gc*_d*_seed{args.seed}.pickle",
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one series file for pattern {pattern}, got {len(matches)}: {matches}"
        )
    series_path = matches[0]

    m = re.search(r"_gc([^_]+)_d([^_]+)_seed", os.path.basename(series_path))
    if m is None:
        raise ValueError(f"Could not parse gc/degree tags from {series_path}")
    gc_tag, degree_tag = m.group(1), m.group(2)
    gc = float(gc_tag.replace("p", "."))

    with open(series_path, "rb") as f:
        A, series = pickle.load(f)

    A = np.asarray(A, dtype=np.float64)
    if int(np.trace(A)) != 0:
        raise ValueError(
            f"Stored adjacency has {int(np.trace(A))} self-loop(s). Regenerate "
            "the topology (build_ciona_cns_topology.py) and series."
        )

    alpha, sigma, mu = make_node_parameters(N, args.seed)
    print(f"Loaded series shape={series.shape}; A shape={A.shape}, edges={int(np.count_nonzero(A))}, "
          f"self-loops=0 verified; gc={gc}")

    start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start: {start} — computing true dx/dt ...")

    dxdt = rulkov_chemical_rhs_batch(
        series=series, A=A, alpha=alpha, sigma=sigma, mu=mu,
        gc=gc, v_s=v_s, lam=args.lam, theta=args.theta, ode_dt=args.dt,
    )

    end = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"End: {end} — dxdt shape={dxdt.shape}")

    out_path = os.path.join(
        data_dir,
        f"TrueDxdt_N{N}_{NET_NAME}_T{int(args.T)}_dt{dt_str}_gc{gc_tag}_d{degree_tag}_seed{args.seed}.pickle",
    )
    with open(out_path, "wb") as f:
        pickle.dump([A, dxdt], f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
