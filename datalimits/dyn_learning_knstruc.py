
import torch
import copy
import math
import pandas as pd
import numpy as np
import pickle
import time
import torch.nn.utils as U
import torch.optim as optim
from utils.model import *
from utils.tools import *
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import seaborn as snb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Nodes_num', type=int, default=5,help='5,10,15,20,50')
parser.add_argument('--m', type=int, default=50, help='timepoints undersample, M[i]')
parser.add_argument('--exp', type=int, default=1, help='1,2,3')
parser.add_argument('--device_id', type=int, default=0, help='0,1,2')
parser.add_argument('--Model', type=str, default='Lorenz', help='Attractor')
parser.add_argument('--r_batch', type=int, default=10, help='ratio of batch size')
parser.add_argument('--hidden', type=int, default=50, help='hidden size')
args, unknown = parser.parse_known_args()

torch.cuda.set_device(args.device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter setting
Epochs = 3000
degree = 2
Time = 50
hidden = args.hidden
edges= int(args.Nodes_num*degree)
nodes_num = args.Nodes_num
deltat = 0.001
tponits = int(Time/deltat)
r_batch = args.r_batch

# print(args.Nodes_num)
# if args.Nodes_num == 50:
#     Mall = [400,350,300,250,200,150,100,50]
#     M = [int(Mall[i]/nodes_num) for i in range(len(Mall))]
# elif args.Nodes_num == 100:
#     M = [1,2,3,4,5,6,7,8,9,10]
# elif args.Nodes_num == 300:
#     M = [1,2,3,4,5,6,7,8,9,10]
# elif args.Nodes_num == 20:
#     Mall = [400,380,360,340,320,300,280,260,240,220,200,180,160,140,120,100,80,60,40]
#     M = [int(Mall[i]/nodes_num) for i in range(len(Mall))]
# else:
#     Mall = [400,390,380,370,360,350,340,330,320,310,300,290,280,270,260,250,240,230,220,210,200,190,180,170,160,150,140,130,120,110,100,90,80,70,60,50]
#     M = [int(Mall[i]/nodes_num) for i in range(len(Mall))]

Times = 40 # length of time series we will use
total_points = int(Times/deltat)
if args.m == 1:
    dt = 100 
    indices = np.random.randint(5, total_points, args.m)
else:
    dt,indices = get_m_index(args.m, Times, deltat)

print(args.m)

path = f'./data/Series_N{nodes_num}_M{edges}_T{Time}_ind1.pickle'
with open(path, 'rb') as f:
    objectAij, series= pickle.load(f)

print(series.shape)

Dimension = int(series.shape[1]/nodes_num)
series = series.reshape(tponits, nodes_num, Dimension)
series = series[:total_points,:,:]
sample_series = series[indices - 1]
derivs = np.load(f'./data/Derivs_N{nodes_num}_M{edges}_T{Time}_ind1.npy')

derivs = derivs.reshape(tponits, nodes_num, Dimension)
dxdt = derivs[:total_points,:,:]
dxdt = dxdt[indices - 1]

timeseries = sample_series[:,:,:]
dXdt = dxdt[:,:,:]
print(dXdt.shape==timeseries.shape)

mapping_data = timeseries
goal_data = dXdt

X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))

edge_index = get_edge_index(objectAij.T) # source to target

# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
# X_train = X_train.float()
# y_train = y_train.float()
# X_test = X_test.float()
# y_test = y_test.float()

X_train = X.float()
y_train = y.float()

len_train = X_train.shape[0]
print('X_train.shape[0] is:',len_train)
batch = math.ceil(len_train*r_batch*0.01)

print(f'batch size is {batch}')

os.makedirs(f'./knstruc/knStru_Nodes{args.Nodes_num}_Times{Times}_m{args.m}_exp{args.exp}/', exist_ok=True)
save_path = f'./knstruc/knStru_Nodes{args.Nodes_num}_Times{Times}_m{args.m}_exp{args.exp}/'

train_dataset = GraphDataset(X_train, y_train, edge_index)
# test_dataset = GraphDataset(X_test, y_test, edge_index)

trainloader = DataLoader(
    train_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,           
    pin_memory=True,        
    persistent_workers=True  
)
# testloader = DataLoader(test_dataset, batch_size=int(batch/2), shuffle=True, num_workers=4, pin_memory=True)

print('trainloader:', len(trainloader))
# print('testloader:', len(testloader))

aggr = 'add'
n_f = Dimension
msg_dim = 1
dim = Dimension
edge_num = edge_index.shape[1]

Dyn = Graph_deriv_NN(n_f, msg_dim, ndim=Dimension, delt_t=dt, hidden=hidden, aggr='add').to(device)
Dyn = torch.compile(Dyn)
print(f'The model has {count_parameters(Dyn):,} trainable parameters')

init_lr = 1e-3
opt = torch.optim.Adam(Dyn.parameters(), lr=init_lr, weight_decay=1e-8)
total_epochs = Epochs
batch_per_epoch = math.ceil(len(trainloader))

sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,#len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

recorded_models = []
messages_over_time = []
selfDyn_over_time = []
train_loss = []
val_loss = []

import copy
for epoch in tqdm(range(0, total_epochs)):
    Dyn.train()
    total_loss = 0.0
    i = 0
    j = 0
    num_items = 0
    valid_loss = 0
    valid_num_items = 0
    while i < batch_per_epoch:
        for ginput in trainloader:
            ginput = ginput.to(device)
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ginput.x = ginput.x
            ginput.y = ginput.y
            ginput.edge_index = ginput.edge_index
            ginput.batch = ginput.batch
            loss = Dyn.loss(ginput)
            (loss/int(ginput.batch[-1]+1)).backward()
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1]+1)
    
    # Dyn.eval()
    # with torch.no_grad():
    #     while j < batch_per_epoch:
    #         for ginput in testloader:
    #             ginput = ginput.to(device)
    #             if j>= batch_per_epoch:
    #                 break
    #             ginput.x = ginput.x
    #             ginput.y = ginput.y
    #             ginput.edge_index = ginput.edge_index
    #             ginput.batch = ginput.batch
    #             loss = Dyn.loss(ginput)#/int(ginput.batch[-1]+1)
    #             valid_loss += loss.item()
    #             valid_num_items += int(ginput.batch[-1]+1)
    #             j += 1


    cur_loss = total_loss/num_items
    # cur_valid_loss = valid_loss/valid_num_items
    print(cur_loss)
    # print(cur_valid_loss)
    train_loss.append(cur_loss)
    # val_loss.append(cur_valid_loss)
    
    Dyn.to(device)

    if (epoch+1)>2000:
        recorded_models.append(copy.deepcopy(Dyn.state_dict()))
        
        if (epoch+1)%100 == 0:
            torch.save(recorded_models, save_path+f'recorded_models_e{epoch}.pt')
            recorded_models = []

import matplotlib.pyplot as plt
plt.plot(train_loss,label='train_loss')
plt.legend()
plt.savefig(save_path + 'train_loss.png')
# plt.plot(val_loss,label='valid_loss')
# plt.legend()
# plt.savefig(save_path + 'val_loss.png')

train_loss = np.array(train_loss)
np.save(save_path + 'train_loss.npy',train_loss)
# val_loss = np.array(val_loss)
# np.save(save_path + 'val_loss.npy',val_loss)
