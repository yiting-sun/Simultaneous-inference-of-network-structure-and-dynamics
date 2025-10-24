import networkx as nx
import numpy as np
from scipy.integrate import odeint
import argparse
import os
import pickle
import concurrent.futures
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('--node_num', type=int, default=20, help='Number of nodes, default=100')
parser.add_argument('--dims', type=int, default=3, help='dims of nodes, default=1')
parser.add_argument('--times', type=int, default=500, help='Length of time, default=100')
parser.add_argument('--Model', type=str, default='HR', help='Attractor')
parser.add_argument('--Net', type=str, default='ER', help='BA or ER, all directed')
parser.add_argument('--dt', type=float, default=0.01, help='time interval, default=0.01')
args, unknown = parser.parse_known_args()

gc = 0.15
def HR_generate(x, t, A, gc):
    # gc is coupling strength
    dxdt = np.zeros((x.shape[0],))
    a, b, c, d = 1, 3, 1, 5
    s, r, p0 = 4, 0.005, -1.6
    Iext = 3.24
    Vsyn1 = 2
    Vsyn2 = -1.5
    k = -10
    nodes = A.shape[0]
    for i in range(nodes):
        tmp = 0
        for j in range(nodes):
            if A[i,j] >= 0:
                tmp += gc * (Vsyn1 - x[3*i]) * A[i,j] / (1 + np.exp(k * (x[3*j] - 1)))
            else:
                tmp += gc * (Vsyn2 - x[3*i]) * abs(A[i,j]) / (1 + np.exp(k * (x[3*j] - 1)))
        dxdt[3*i] = x[3*i+1] - a * x[3*i]**3 + b * x[3*i]**2 - x[3*i+2] + Iext + tmp
        dxdt[3*i+1] = c - d * x[3*i]**2 - x[3*i+1]
        dxdt[3*i+2] = r * (s * (x[3*i] - p0) - x[3*i+2])
    return dxdt

def generate_data(A):
    num_steps = int(args.times/args.dt)
    init = 1+np.random.uniform(0.,1.,size=(args.node_num * args.dims,))
    tspan=np.arange(0,args.times,args.dt)
    y = odeint(HR_generate, init, tspan,args=(A, gc))
    return y

def save_data(data,A):
    num_steps = int(args.times/args.dt)
    file_path = f'./data/'
    idata = data[0:num_steps,:,:]
    res = [A,idata]
    with open(file_path+'timeseries.pickle', 'wb') as f:
        pickle.dump(res, f)
    pass

A = np.load(f'./data/TriER_nodes{args.node_num}_edges100.npy')

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
print('Simulating time series...')
new_data = generate_data(A)
data = new_data[:,:,np.newaxis]

save_data(data, A)

print('Simulation finished!')
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)
