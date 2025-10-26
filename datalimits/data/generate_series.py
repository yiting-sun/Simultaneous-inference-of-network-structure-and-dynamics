import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import networkx as nx
from scipy.integrate import odeint

Nodes = 50
node_num = Nodes
dims = 3
degree = 2
edges = int(Nodes*degree)
Time = 50
dt = 0.001

G = nx.gnm_random_graph(node_num, edges, directed=True)
A = nx.to_numpy_array(G)
print('number of edges is:',np.sum(A))

def Lorenz_generate(x,t,A):
    dxdt = np.zeros((x.shape[0],))
    epsilon = 0.2
    sigma = 10
    rho  = 28
    beta = 8/3
    nodes = A.shape[0]
    for i in range(nodes):
        tmp = 0
        for j in range(nodes):
            tmp = tmp + epsilon*A[i][j]*(x[3*j]-x[3*i])
        dxdt[3*i] = -sigma*(x[3*i]-x[3*i+1])+tmp
        dxdt[3*i+1] = rho*x[3*i]-x[3*i+1]-x[3*i]*x[3*i+2]
        dxdt[3*i+2] = -beta*x[3*i+2] + x[3*i]*x[3*i+1]

    return(dxdt)

def generate_data(A):
    np.random.seed(os.getpid())# use PID as random seed
    init = 1+np.random.uniform(0.,1.,size=(node_num * dims))
    tspan=np.arange(0,Time,dt)
    y = odeint(Lorenz_generate, init, tspan,args=(A,))
    return y

def generate_and_save_data(i):
    data = generate_data(A)
    res = [A, data]
    savepath = './'
    with open(savepath+f'Series_N{node_num}_M{edges}_T{Time}_ind{i}.pickle', 'wb') as f:
        pickle.dump(res, f)

i = 1
print(f'We are simulating {Nodes} nodes network!')
generate_and_save_data(i)


