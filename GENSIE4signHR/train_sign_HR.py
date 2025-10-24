import torch
import pickle
import numpy as np
import torch.nn.utils as U
import torch.optim as optim
import math
import argparse
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
import seaborn as snb
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm
from copy import deepcopy as copy
from utils.model import *
from utils.tools import *

# Parameters configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# Parameter setting
Dimension = 3
dt = 0.01
Time = 500
nodes_num = 20
batch = 256
Net = 'ER'
tponits = int(Time/dt)

# Load data
data_path = f'./data/'
path = data_path + 'timeseries.pickle'
with open(path, 'rb') as f:
    objectAij, series= pickle.load(f)

series = series.reshape(-1, nodes_num, Dimension)
print(series.shape)
goal_data = series[1:-1,:,0:Dimension]
mapping_data = series[0:-2,:,0:Dimension]
print(goal_data.shape==mapping_data.shape)
X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))

np.random.seed(65)

initialA = np.ones((nodes_num,nodes_num)) # fully connected
np.fill_diagonal(initialA,0) # delete diagonal links
edge_index_j2i = get_edge_index(initialA.T) # note the transpose here for j->i

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)
X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()

trainloader = DataLoader(
    [Data(
        X_train[i],
        edge_index=edge_index_j2i,
        y=y_train[i]) for i in range(len(y_train))],
    batch_size=batch,
    shuffle=False
)

testloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index_j2i,
        y=y_test[i]) for i in range(len(y_test))],
    batch_size=int(batch/2),
    shuffle=True
)

# Model training
aggr = 'add'
model = 'HR'
hidden = 50
n_f = Dimension
msg_dim = 1
dim = Dimension
edge_num = edge_index_j2i.shape[1]
tau = 1
lam = 1e-4

Dyn = Get_sign_MP_ori(n_f, msg_dim, dim, dt, edge_num, tau=tau, lam=lam, aggr=aggr, hidden=hidden).to(device)

init_lr = 5e-4
opt = torch.optim.Adam(Dyn.parameters(), lr=init_lr, weight_decay=1e-8)
total_epochs = 2000
batch_per_epoch = 200

sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,#len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

recorded_models = []
AUC_over_time = []
AUPRC_over_time = []
structure_weights = []

os.makedirs('Training_process', exist_ok=True)
save_path = f'Training_process/'
save_interval = 100

for epoch in range(total_epochs):
    show_progress = (epoch + 1) % 20 == 0 or epoch == 0

    total_loss = 0.0
    i = 0
    num_items = 0

    with tqdm(trainloader, total=batch_per_epoch, desc=f"Epoch {epoch+1}/{total_epochs}", disable=not show_progress) as pbar:
        for ginput in pbar:
            if i >= batch_per_epoch:
                break
            ginput = ginput.to(device)
            
            opt.zero_grad()
            loss = Dyn.loss(ginput, norm=lam, square=False)
            (loss / int(ginput.batch[-1] + 1)).backward()
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1] + 1)
            
            if show_progress:
                pbar.set_postfix(loss=f"{(total_loss/num_items)}")

    tau *= 0.999
    Dyn.update_tau(tau)

    cur_loss = total_loss / num_items
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch+1} Average Loss: {cur_loss}")
        cur_weights = Dyn.weights.cpu().detach().numpy()
        structure_weights.append(cur_weights)
        error1, error2 = cal_structure_performance(objectAij, Dyn.weights, epoch)
        print(f'Structure Error (Type1): {error1:.4f}, (Type2): {error2:.4f}')
    else:
        cur_weights = Dyn.weights.cpu().detach().numpy()
        structure_weights.append(cur_weights)

    if epoch % save_interval == 0:
        recorded_models.append(copy(Dyn.state_dict()))

if (total_epochs - 1) % save_interval != 0:
    recorded_models.append(copy(Dyn.state_dict()))

# save model and weights
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_save_path = f"{save_path}recorded_models_lam{lam}_{timestamp}.pt"
weights_save_path = f"{save_path}weights_tau0.999_lam{lam}_{timestamp}.pkl"

print(f"Saving model to: {model_save_path}")
torch.save(recorded_models, model_save_path)

print(f"Saving weights to: {weights_save_path}")
with open(weights_save_path, 'wb') as f:
    pickle.dump(structure_weights, f)
