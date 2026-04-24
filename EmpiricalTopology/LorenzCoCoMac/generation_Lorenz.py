#%%
import numpy as np
from scipy.integrate import odeint
import os
import pickle
import time
import argparse

parser = argparse.ArgumentParser(description='Generate Lorenz time series on CoCoMac FV91 58-node cortical network (or subgraph)')
parser.add_argument('--num_nodes_keep', type=int, default=58,
                    help='Number of nodes to keep. 58 = full CoCoMac FV91 network.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed for subgraph node selection.')
parser.add_argument('--T', type=float, default=100.0, help='Total simulation time')
parser.add_argument('--dt', type=float, default=0.001, help='Time step')
parser.add_argument('--epsilon', type=float, default=None,
                    help='Coupling strength. Default: 0.2 if num_nodes_keep == 5, else 0.1.')
args = parser.parse_args()

if args.epsilon is None:
    args.epsilon = 0.2 if args.num_nodes_keep == 5 else 0.1

# ===== Parameters =====
T = args.T
dt = args.dt
epsilon = args.epsilon
ORIGINAL_NUM_NODES = 58     # CoCoMac FV91 macaque cortex
net_name = 'Lorenz'
dims = 3

# ===== Load full binary adjacency matrix =====
# CoCoMac FV91 macaque cortical connectivity, 58 areas — same adjacency used
# by Wilsoncowan_dyn/generation_WilsonCowan.py. Only the dynamics differ.
# Convention: A[i, j] = 1 means edge j -> i (col = source, row = destination).
# Prefer the file sitting next to this script; fall back to the Wilsoncowan_dyn
# copy so we don't have to duplicate the asset.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ADJ_FILENAME = 'cocomac_FV91_A_binary_N58.npz'
_local_adj = os.path.join(_THIS_DIR, 'data', _ADJ_FILENAME)
_wc_adj = os.path.join(os.path.dirname(_THIS_DIR),
                       'Wilsoncowan_dyn', 'data', _ADJ_FILENAME)
if os.path.isfile(_local_adj):
    src_path = _local_adj
elif os.path.isfile(_wc_adj):
    src_path = _wc_adj
else:
    raise FileNotFoundError(
        f'Could not find {_ADJ_FILENAME} in either:\n  {_local_adj}\n  {_wc_adj}')
with np.load(src_path, allow_pickle=True) as _d:
    full_A = _d['A'].astype(np.float64)
print(f'Loaded full adjacency (CoCoMac FV91) from {src_path}: '
      f'shape={full_A.shape}, nonzero={int(full_A.sum())}')
assert full_A.shape == (ORIGINAL_NUM_NODES, ORIGINAL_NUM_NODES)

# ===== Subgraph node selection =====
# Reject subgraphs with zero edges: resample until the induced adjacency has
# at least one edge. Deterministic given (seed, num_keep) since the rng state
# advances on each rejection.
def select_kept_node_indices(total_nodes, num_keep, seed, full_adj, max_retries=10000):
    if num_keep >= total_nodes:
        return np.arange(total_nodes, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for attempt in range(max_retries):
        kept = np.sort(rng.choice(total_nodes, size=num_keep, replace=False)).astype(np.int64)
        if full_adj[np.ix_(kept, kept)].sum() > 0:
            if attempt > 0:
                print(f'  (resampled {attempt} time(s) to get a subgraph with edges)')
            return kept
    raise RuntimeError(f'Could not find a non-empty subgraph after {max_retries} retries '
                       f'(num_keep={num_keep}, seed={seed}).')

nodes_num = args.num_nodes_keep
kept = select_kept_node_indices(ORIGINAL_NUM_NODES, nodes_num, args.seed, full_A)
A = full_A[np.ix_(kept, kept)]
print(f'Subgraph: {nodes_num} nodes (seed={args.seed}), kept indices={kept.tolist()[:20]}...')
print(f'Sub-adjacency: shape={A.shape}, nonzero={np.count_nonzero(A)}')

# Precompute per-node incoming degree (row sum, since A[i,j]=1 => j->i).
# Used to avoid recomputing A.sum(axis=1) at every RHS call.
A_rowsum = A.sum(axis=1)

# ===== Vectorized Lorenz dynamics =====
# Per-node ODE (standard Lorenz x,y,z):
#   dx_i/dt = sigma*(y_i - x_i)   +  epsilon * sum_j A[i,j] * (x_j - x_i)
#   dy_i/dt = rho*x_i - y_i - x_i*z_i
#   dz_i/dt = -beta*z_i + x_i*y_i
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

def Lorenz_generate_vectorized(state, t, A, A_rowsum):
    # state layout: [x_1, y_1, z_1, x_2, y_2, z_2, ...]
    x = state[0::3]   # (N,)
    y = state[1::3]
    z = state[2::3]

    # Diffusive coupling on x:  sum_j A[i,j] * (x_j - x_i)
    #                         = (A @ x) - A_rowsum * x
    coupling = epsilon * (A @ x - A_rowsum * x)

    dxdt = np.empty_like(state)
    dxdt[0::3] = sigma * (y - x) + coupling
    dxdt[1::3] = rho * x - y - x * z
    dxdt[2::3] = x * y - beta * z
    return dxdt

# ===== Generate time series =====
num_steps = int(T / dt)
print(f'\nTotal steps: {num_steps}, T={T}, dt={dt}')

# Initial conditions: fixed seed on the full 58 nodes, then slice to kept nodes,
# so subgraphs share per-node initial conditions with the full network.
IC_SEED = 42
rng_ic = np.random.default_rng(IC_SEED)
# same U(1, 2) as reference Lorenz generator (init = 1 + uniform(0,1))
full_init_x = 1.0 + rng_ic.uniform(0.0, 1.0, ORIGINAL_NUM_NODES)
full_init_y = 1.0 + rng_ic.uniform(0.0, 1.0, ORIGINAL_NUM_NODES)
full_init_z = 1.0 + rng_ic.uniform(0.0, 1.0, ORIGINAL_NUM_NODES)

init = np.zeros(nodes_num * dims)
init[0::3] = full_init_x[kept]
init[1::3] = full_init_y[kept]
init[2::3] = full_init_z[kept]

tspan = np.arange(0, T, dt)

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f'Start time: {start_time}')
print(f'Simulating Lorenz dynamics on subgraph (N={nodes_num}, seed={args.seed})...')

# Tight tolerances keep the dynamics accurate; LSODA (odeint) adapts its
# internal step so vectorized RHS evaluates only as often as needed.
series = odeint(Lorenz_generate_vectorized, init, tspan,
                args=(A, A_rowsum),
                rtol=1e-8, atol=1e-10, mxstep=5000)

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f'End time: {end_time}')
print(f'Simulation finished! series shape: {series.shape}')

# ===== Save data =====
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(save_dir, exist_ok=True)

dt_str = str(dt).replace('.', '')
save_path = os.path.join(save_dir,
    f'Series_N{nodes_num}_{net_name}_T{int(T)}_dt{dt_str}_seed{args.seed}.pickle')

res = [A, series]
with open(save_path, 'wb') as f:
    pickle.dump(res, f)

np.save(os.path.join(save_dir, f'kept_nodes_N{nodes_num}_Lorenz_seed{args.seed}.npy'), kept)

print(f'Saved to: {save_path}')
print(f'series shape: {series.shape}  (should be ({num_steps}, {nodes_num * dims}))')
