#%%
import numpy as np
import os
import pickle
import time
import argparse

parser = argparse.ArgumentParser(description='Compute true dx/dt for Lorenz subgraph time series')
parser.add_argument('--num_nodes_keep', type=int, default=58,
                    help='Number of nodes (must match generation_Lorenz.py).')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed (must match generation_Lorenz.py).')
parser.add_argument('--T', type=float, default=100.0, help='Total simulation time')
parser.add_argument('--dt', type=float, default=0.001, help='Time step')
parser.add_argument('--epsilon', type=float, default=None,
                    help='Coupling strength. Default: 0.2 if num_nodes_keep == 5, else 0.1. Must match generation_Lorenz.py.')
args = parser.parse_args()

if args.epsilon is None:
    args.epsilon = 0.2 if args.num_nodes_keep == 5 else 0.1

# ===== Parameters =====
T = args.T
dt = args.dt
epsilon = args.epsilon
nodes_num = args.num_nodes_keep
net_name = 'Lorenz'
dims = 3

dt_str = str(dt).replace('.', '')

# ===== Load generated series =====
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
data_path = os.path.join(data_dir,
    f'Series_N{nodes_num}_{net_name}_T{int(T)}_dt{dt_str}_seed{args.seed}.pickle')
with open(data_path, 'rb') as f:
    objectAij, series = pickle.load(f)

A = objectAij.astype(np.float64)
print(f'Loaded series: shape={series.shape}, A: shape={A.shape}')
A_rowsum = A.sum(axis=1)

# ===== Lorenz dynamics parameters =====
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

# ===== Compute true dx/dt for all time steps (vectorized, chunked) =====
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f'Start time: {start_time}')
print('Computing true dx/dt ...')

num_steps = series.shape[0]
dxdt = np.empty_like(series)

series_3d = series.reshape(num_steps, nodes_num, dims)
x_all = series_3d[:, :, 0]   # (T, N)
y_all = series_3d[:, :, 1]
z_all = series_3d[:, :, 2]

chunk = 20000
for start in range(0, num_steps, chunk):
    end = min(start + chunk, num_steps)
    x = x_all[start:end]    # (chunk, N)
    y = y_all[start:end]
    z = z_all[start:end]

    # coupling_i = epsilon * sum_j A[i,j] * (x_j - x_i)
    #            = epsilon * ((x @ A.T) - A_rowsum * x)
    coupling = epsilon * (x @ A.T - A_rowsum[None, :] * x)

    dxdt[start:end, 0::3] = sigma * (y - x) + coupling
    dxdt[start:end, 1::3] = rho * x - y - x * z
    dxdt[start:end, 2::3] = x * y - beta * z

    if (start // chunk) % 10 == 0:
        print(f'  processed {end}/{num_steps} steps')

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f'End time: {end_time}')
print(f'Done! dxdt shape: {dxdt.shape}')

# ===== Save =====
save_path = os.path.join(data_dir,
    f'TrueDxdt_N{nodes_num}_{net_name}_T{int(T)}_dt{dt_str}_seed{args.seed}.pickle')
res = [objectAij, dxdt]
with open(save_path, 'wb') as f:
    pickle.dump(res, f)

print(f'Saved to: {save_path}')
