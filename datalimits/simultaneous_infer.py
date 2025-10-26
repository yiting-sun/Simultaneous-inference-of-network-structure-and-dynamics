import torch
import copy
import math
import pandas as pd
import numpy as np
import pickle
import torch.nn.utils as U
import torch.optim as optim
from utils.model import *
from utils.tools import *
import argparse
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
import seaborn as snb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Nodes_num', type=int, default=10,help='5,10,15,20,50')
parser.add_argument('--m', type=int, default=50, help='timepoints undersample')
parser.add_argument('--exp', type=int, default=1, help='1,2,3,4,5')
parser.add_argument('--device_id', type=int, default=0, help='0,1,2')
parser.add_argument('--Model', type=str, default='Lorenz', help='Attractor')
parser.add_argument('--tau_update', type=float, default=0.999, help='decay rate of tau')
parser.add_argument('--r_batch', type=int, default=5, help='ratio of batch size')
args, unknown = parser.parse_known_args()

torch.cuda.set_device(args.device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter setting
Epochs = 3000
hidden = 50
nodes_num = args.Nodes_num

# Define M, degree, Time based on nodes_num
# if args.Nodes_num < 50:
#     M1 = np.arange(5,99,2)
#     M = M1.tolist()
#     M.append(100) 
#     degree = 2
#     Time = 50
# elif args.Nodes_num == 50:
#     M1 = np.arange(5,99,2)
#     M = M1.tolist()
#     M.append(100) 
#     degree = 10
#     Time = 50
# else:
#     M1 = np.arange(0.1,3.1,0.1)
#     m = M1.tolist()
#     M = [int(m[i]*nodes_num) for i in range(len(m))] # timepoints by subsampling
#     degree = 10
#     Time = 100

degree = 2
Time = 50
edges= int(args.Nodes_num*degree)
deltat = 0.001
tponits = int(Time/deltat)
Times = 40 # length of time series we will use

total_points = int(Times/deltat)
dt,indices = get_m_index(args.m, Times, deltat)

os.makedirs(f'./simul_infer/Nodes{args.Nodes_num}_Times{Times}_m{args.m}_hidden{hidden}_exp{args.exp}/', exist_ok=True)
save_path = f'./simul_infer/Nodes{args.Nodes_num}_Times{Times}_m{args.m}_hidden{hidden}_exp{args.exp}/'

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

timeseries = sample_series
dXdt = dxdt
print(dXdt.shape==timeseries.shape)

mapping_data = timeseries
goal_data = dXdt

X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))

initialA = np.ones((nodes_num,nodes_num)) # fully connected
np.fill_diagonal(initialA,0) # delete diagonal links
edge_index = get_edge_index(initialA)#i to j, to be revised and ipynb

X_train = X.float()
y_train = y.float()

init_lr = 1e-3
len_train = X_train.shape[0]
print('X_train.shape[0] is:',len_train)
batch = math.ceil(len_train*args.r_batch*0.01)
print(f'batch size is {batch}')

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

aggr = 'add'
n_f = Dimension
msg_dim = 1
dim = Dimension
edge_num = edge_index.shape[1]
lam = 0.1

print(f'lam: {lam}')
tau = 1

Dyn = simultaneous_infer_deriv(n_f, msg_dim, dim, dt, edge_num, tau=tau, lam=lam, aggr=aggr, hidden=hidden).to(device)

opt = torch.optim.Adam(Dyn.parameters(), lr=init_lr, weight_decay=1e-8)
total_epochs = Epochs
batch_per_epoch = math.ceil(len(trainloader))

sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,#len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

recorded_models = []
structure_weights = []
messages_over_time = []
selfDyn_over_time = []
loss_over_time = []
val_loss_over_time = []
AUC_over_time = []
AUPRC_over_time = []
Time_over_epc = []


for epoch in tqdm(range(total_epochs)):
    total_loss = 0.0
    i = 0
    j = 0
    num_items = 0
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
            (loss/int(ginput.batch[-1]+1)).backward() #int(ginput.batch[-1]+1) is batchsize
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1]+1)

    # Dyn.eval()
    # valid_num_items = 0
    # valid_loss = 0
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
    #             loss = torch.sum(torch.abs(ginput.y-Dyn.prediction(ginput)))
    #             valid_loss += loss.item()
    #             valid_num_items += int(ginput.batch[-1]+1)
    #             j += 1

    tau *= args.tau_update
    Dyn.update_tau(tau)
    
    cur_loss = total_loss/num_items 
    # cur_valid_loss = valid_loss/valid_num_items
    print(cur_loss)
    # print(cur_valid_loss)
    cur_AUC, cur_AUPRC = calculate_auc(objectAij, Dyn.weights)
    print("AUC: {:.5f}, AUPRC: {:.5f}".format(cur_AUC, cur_AUPRC))

    loss_over_time.append(cur_loss)
    # val_loss_over_time.append(cur_valid_loss)
    if (epoch+1)>1500:
        cur_weights = Dyn.weights.cpu().detach().numpy()
        structure_weights.append(cur_weights)
        AUC_over_time.append(cur_AUC)
        AUPRC_over_time.append(cur_AUPRC)
        recorded_models.append(copy.deepcopy(Dyn.state_dict()))
        
        if (epoch+1)%100 == 0:
            data_dict = {
                f'weights_over_time_lam{lam}_e{epoch}.pkl': structure_weights
            }

            for filename, data in data_dict.items():
                with open(save_path + filename, 'wb') as f:
                    pickle.dump(data, f)

            AUC_over_time = pd.DataFrame(AUC_over_time)
            AUPRC_over_time = pd.DataFrame(AUPRC_over_time)
            Eva = pd.concat((AUC_over_time,AUPRC_over_time),axis = 1)

            Eva.to_csv(save_path+f'Eva_lam{lam}_e{epoch}.csv')
            torch.save(recorded_models, save_path+f'recorded_models_lam{lam}_e{epoch}.pt')

            structure_weights = []
            AUC_over_time = []
            AUPRC_over_time = []
            recorded_models = []

plt.plot(loss_over_time,label='train_loss')
plt.legend()
plt.savefig(save_path + 'train_loss.png')
train_loss = np.array(loss_over_time)
np.save(save_path + 'train_loss.npy',train_loss)
with open(save_path + f'loss_over_time_lam{lam}.pkl','wb') as f:
    pickle.dump(loss_over_time,f)

# plt.plot(val_loss_over_time,label='valid_loss')
# plt.legend()
# plt.savefig(save_path + 'val_loss.png')
# val_loss = np.array(val_loss_over_time)
# np.save(save_path + 'val_loss.npy',val_loss)
# with open(save_path + f'val_loss_over_time_lam{lam}.pkl','wb') as f:
#     pickle.dump(val_loss_over_time,f)

