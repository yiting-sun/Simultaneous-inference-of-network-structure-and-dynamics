import numpy as np
import pandas as pd
import pickle 

N = 50
print(f'We are simulating {N} nodes network!')
Degree = 2
Time = 50
M = int(N*Degree)
with open (f'Series_N{N}_M{M}_T{Time}_ind1.pickle','rb') as f:
    A1,data = pickle.load(f)

sigma = 10
rho  = 28
beta = 8/3
g1 = 0.2

#%% Pre-allocate result array with correct size to avoid hstack operations
dxdt_all = np.zeros((len(data), 3*N))

# Pre-compute node indices for vectorized operations
x_indices = np.arange(0, 3*N, 3)
y_indices = np.arange(1, 3*N, 3)
z_indices = np.arange(2, 3*N, 3)

for l, x in enumerate(data):
    # Reshape x to facilitate vectorized operations
    x_reshaped = x.reshape(N, 3)
    
    x_nodes = x[x_indices]  # Extract x-component of each node
    coupling = g1 * (A1 @ x_nodes - np.sum(A1, axis=1) * x_nodes)
    
    dxdt_all[l, x_indices] = -sigma * (x[x_indices] - x[y_indices]) + coupling
    dxdt_all[l, y_indices] = rho * x[x_indices] - x[y_indices] - x[x_indices] * x[z_indices]
    dxdt_all[l, z_indices] = -beta * x[z_indices] + x[x_indices] * x[y_indices]
np.save(f'Derivs_N{N}_M{M}_T{Time}_ind1.npy', dxdt_all)
