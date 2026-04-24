import argparse
import glob
import os
import pickle
import re
import time

import numpy as np


def float_to_tag(value):
    return format(value, "g").replace(".", "")


def main():
    parser = argparse.ArgumentParser(description="Compute true dx/dt for HR subgraph time series")
    parser.add_argument("--num_nodes_keep", type=int, default=279,
                        help="Number of nodes (must match generation_HR_new.py).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (must match generation_HR_new.py).")
    parser.add_argument("--T", type=int, default=500, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    args = parser.parse_args()

    T = args.T
    dt = args.dt
    nodes_num = args.num_nodes_keep
    net_name = "Celegans"
    dims = 3

    dt_str = float_to_tag(dt)

    data_dir = "HR_dyn/data_new"
    pattern = (
        f"{data_dir}/Series_N{nodes_num}_{net_name}_T{T}_dt{dt_str}"
        f"_gc*_d*_seed{args.seed}.pickle"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one series file for pattern {pattern}, found {len(matches)}: {matches}")
    data_path = matches[0]
    match = re.search(r"_gc([^_]+)_d([^_]+)_seed", os.path.basename(data_path))
    if match is None:
        raise ValueError(f"Could not parse degree tag from {data_path}")
    gc_tag = match.group(1)
    degree_tag = match.group(2)
    gc = float(gc_tag.replace("p", "."))
    with open(data_path, "rb") as f:
        objectAij, series = pickle.load(f)

    A = np.asarray(objectAij)
    print(f"Loaded series: shape={series.shape}, A: shape={A.shape}, "
          f"edges={int(np.count_nonzero(A))}, gc={gc}, degree_tag={degree_tag}")

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

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start time: {start_time}")
    print("Computing true dx/dt ...")

    num_steps = series.shape[0]
    dxdt = np.zeros_like(series)

    series_3d = series.reshape(num_steps, nodes_num, dims)
    V_all = series_3d[:, :, 0]
    y_all = series_3d[:, :, 1]
    z_all = series_3d[:, :, 2]

    sigmoid = 1.0 / (1.0 + np.exp(k * (V_all - 1.0)))

    chunk_size = 10000
    for start in range(0, num_steps, chunk_size):
        end = min(start + chunk_size, num_steps)
        V_chunk = V_all[start:end]
        sig_chunk = sigmoid[start:end]

        coupling = gc * (Vsyn1 - V_chunk) * (sig_chunk @ A.T)

        dxdt[start:end, 0::3] = (
            y_all[start:end] - a * V_chunk ** 3 + b * V_chunk ** 2 - z_all[start:end] + Iext + coupling
        )
        dxdt[start:end, 1::3] = c - d * V_chunk ** 2 - y_all[start:end]
        dxdt[start:end, 2::3] = r * (s * (V_chunk - p0) - z_all[start:end])

        if (start // chunk_size) % 10 == 0:
            print(f"  processed {end}/{num_steps} steps")

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Done! dxdt shape: {dxdt.shape}")
    print(f"End time: {end_time}")

    save_path = (
        f"{data_dir}/TrueDxdt_N{nodes_num}_{net_name}_T{T}_dt{dt_str}"
        f"_gc{gc_tag}_d{degree_tag}_seed{args.seed}.pickle"
    )
    with open(save_path, "wb") as f:
        pickle.dump([objectAij, dxdt], f)

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
