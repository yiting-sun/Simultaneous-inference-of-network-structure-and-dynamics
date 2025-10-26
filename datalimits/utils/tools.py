import torch
import numpy as np
from math import log
import math
import pandas as pd
from sklearn.preprocessing import normalize 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

def get_index(M, Times, i, deltat):
    prec = int(1/deltat)
    rounded_dt = [(Times*prec-1)//(M[i]-1)/prec for i in range(len(M))] 
    dt = rounded_dt[i]
    interval = rounded_dt[i]/deltat
    ind = [1 + ii*interval for ii in range(M[i])]
    ind = np.array(ind)
    ind = ind.astype(int)
    return dt,ind

def get_m_index(m, Times, deltat):
    prec = int(1/deltat)
    rounded_dt = (Times*prec-1)//(m-1)/prec
    dt = rounded_dt
    interval = rounded_dt/deltat
    ind = [1 + ii*interval for ii in range(m)]
    ind = np.array(ind)
    ind = ind.astype(int)
    return dt,ind

def get_edge_index(Adj):
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index

def calculate_auc(objectAij, weights):
    weights = F.softmax(weights, dim = 1)[:, 0].view(-1,1)
    weights = weights.cpu()
    objectAij = objectAij.T
    
    mask = np.ones_like(objectAij, dtype=bool)
    np.fill_diagonal(mask, 0)
    objectAij_no_dig = objectAij[mask].reshape(-1,1)

    fpr,tpr,_ = roc_curve(objectAij_no_dig, weights.detach().numpy())
    auc_value = auc(fpr,tpr)
    precision, recall, _ = precision_recall_curve(objectAij_no_dig, weights.detach().numpy())
    auprc_value = auc(recall, precision)

    return auc_value, auprc_value

def calculate_aic(n,mse,num_params):
    aic = n * log(mse) + 2 * num_params 
    return aic 

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
def terms_sort_fit(X_lib,Y_goal,intercept):
    reg = LassoCV(cv=5, fit_intercept=intercept,  n_jobs=-1, max_iter=1000).fit(X_lib,Y_goal)
    coef = pd.Series(reg.coef_, index=X_lib.columns)
    if intercept == True:
        coef['constant'] = reg.intercept_
        num_params = len(coef)
    else:
        num_params = len(coef)    
    P = X_lib
    Score = reg.score(X_lib,Y_goal)
    yhat = reg.predict(P)
    mse = mean_squared_error(Y_goal, yhat)
    aic = calculate_aic(len(Y_goal), mse, num_params)
    #print('label of function: %.3f' % time)
    sort = coef.sort_values()
    print(coef)
    return Score, mse, aic

'''Example: Lorenze system, dim=3, msg_dim=1, newtestloader should be defined previously'''
def get_messages(tDyn, dim, msg_dim, loader, device):

    def get_message_info(tmp):
        tDyn.cpu()

        s1 = tmp.x[tmp.edge_index[0]] #source
        s1 = s1[:,0]
        s2 = tmp.x[tmp.edge_index[1]] #target
        s2 = s2[:,0]
        Tmp = torch.cat([s2, s1]) # tmp --> xi,xj
        Tmp = Tmp.reshape(2,-1)
        Tmp = Tmp.t()# tmp has shape [E, 2 * in_channels]

        Gweights = F.softmax(tDyn.weights/(0.999**epc),dim=1)
        Gweights = Gweights[:,0].view(-1,1)
        Len = int(s1.shape[0])/int(Gweights.shape[0])
        w = Gweights.repeat(int(Len),1)
        tmpW = torch.cat([Tmp,w],dim=1)
        tmpW = tmpW.to(torch.float32)
        Gxixj = tDyn.msg_fnc(Tmp)
        m12 = Gxixj*w#/source_x[:,Dimension].reshape(source_x.shape[0],1)

        all_messages = torch.cat((
            tmpW, Gxixj,
             m12), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]

        return pd.DataFrame(
              data=all_messages.cpu().detach().numpy(),
             columns=columns
        )
        #print(all_messages.shape)
        return pd.DataFrame(all_messages)

    msg_info = []
    for i, g in enumerate(loader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    
    return msg_info

def get_selfDynamics(tDyn, dim, loader, device):
    def get_selfDynamics_info(tmp):
        tDyn.cpu()
        
        tmp = tmp.x[tmp.edge_index[1]]
        # tmp = tmp.to(device)
        #tmp = tmp[:,0:Dimension].reshape(-1,Dimension)
        if dim==1:
            self_dyn_x = tDyn.node_fnc_x(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x), dim=1)
            columns = ['x','s1']
            
        if dim==2:
            self_dyn_x = tDyn.node_fnc_x(tmp)
            self_dyn_y = tDyn.node_fnc_y(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x,self_dyn_y), dim=1)
            columns = ['x','y','s1','s2']
        if dim==3:
            self_dyn_x = tDyn.node_fnc_x(tmp)
            self_dyn_y = tDyn.node_fnc_y(tmp)
            self_dyn_z = tDyn.node_fnc_z(tmp)
            self_dyn_all = torch.cat((tmp,self_dyn_x,self_dyn_y,self_dyn_z), dim=1)
            columns = ['x','y','z','s1','s2','s3']
            
        return pd.DataFrame(
              data=self_dyn_all.cpu().detach().numpy(),
             columns=columns
        )
        return pd.DataFrame(self_dyn_all)

    selfDyn_info = []
    for i, g in enumerate(loader):
        selfDyn_info.append(get_selfDynamics_info(g))

    selfDyn_info = pd.concat(selfDyn_info)
    return selfDyn_info      

def get_messages_after_binaryAij(tDyn, dim, msg_dim, bi_bestWei, loader, device):

    def get_message_info(tmp):
        tDyn.cpu()

        s1 = tmp.x[tmp.edge_index[0]] #source
        s1 = s1[:,0]
        #print(s1)
        s2 = tmp.x[tmp.edge_index[1]] #target
        s2 = s2[:,0]
        #print(s2)
        Tmp = torch.cat([s2, s1]) # tmp --> xi,xj
        Tmp = Tmp.reshape(2,-1)
        Tmp = Tmp.t()# tmp has shape [E, 2 * in_channels]

        # Gweights = F.softmax(tDyn.weights,dim=1)
        # Gweights = Gweights[:,0].view(-1,1)
        Gweights = torch.tensor(bi_bestWei)
        Len = int(s1.shape[0])/int(Gweights.shape[0])
        w = Gweights.repeat(int(Len),1)
        # Tmp = Tmp.to(device)
        tmpW = torch.cat([Tmp,w],dim=1)
        tmpW = tmpW.to(torch.float32)
        #source_x = tmp.x[tmp.edge_index[1]]
        #tmpW = torch.cat([Tmp,source_x[:,Dimension].reshape(source_x.shape[0],1)],dim=1)
        #print(tmp)
        Gxixj = tDyn.msg_fnc(Tmp)
        m12 = Gxixj*w#/source_x[:,Dimension].reshape(source_x.shape[0],1)

        all_messages = torch.cat((
            tmpW, Gxixj,
             m12), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G_fnc']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G_fnc']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['G_fnc']
            columns += ['e%d'%(k,) for k in range(msg_dim)]

        return pd.DataFrame(
              data=all_messages.cpu().detach().numpy(),
             columns=columns
        )
        #print(all_messages.shape)
        return pd.DataFrame(all_messages)

    msg_info = []
    for i, g in enumerate(loader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    
    return msg_info  

'''get interaction is used for unweightd Aij'''
def get_interaction(tDyn, dim, msg_dim, loader, device):

    def get_message_info(tmp):
        tDyn.cpu()

        s1 = tmp.x[tmp.edge_index[0]] #source
        s1 = s1[:,0]
        s2 = tmp.x[tmp.edge_index[1]] #target
        s2 = s2[:,0]
        Tmp = torch.cat([s2, s1]) # tmp --> xi,xj
        Tmp = Tmp.reshape(2,-1)
        Tmp = Tmp.t()# tmp has shape [E, 2 * in_channels]

        #source_x = tmp.x[tmp.edge_index[1]]
        #tmpW = torch.cat([Tmp,source_x[:,Dimension].reshape(source_x.shape[0],1)],dim=1)
        #print(tmp)
        Gxixj = tDyn.msg_fnc(Tmp)

        all_messages = torch.cat((
            Tmp, 
            Gxixj), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['G']
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['G']
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['G']

        return pd.DataFrame(
              data=all_messages.cpu().detach().numpy(),
             columns=columns
        )
        #print(all_messages.shape)
        return pd.DataFrame(all_messages)

    msg_info = []
    for i, g in enumerate(loader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    
    return msg_info


def get_messages_with_trueAij(tDyn, dim, loader, objectAij, device, msg_dim=1,epc=2800):

    def get_message_info(tmp):
        tDyn.cpu()

        s1 = tmp.x[tmp.edge_index[0]] #source
        s1 = s1[:,0]
        s2 = tmp.x[tmp.edge_index[1]] #target
        s2 = s2[:,0]
        Tmp = torch.cat([s2, s1]) # tmp --> xi,xj
        Tmp = Tmp.reshape(2,-1)
        Tmp = Tmp.t()# tmp has shape [E, 2 * in_channels]

        Gweights = F.softmax(tDyn.weights/(0.999**epc),dim=1)
        Gweights = Gweights[:,0].view(-1,1)
        Len = int(s1.shape[0])/int(Gweights.shape[0])
        w = Gweights.repeat(int(Len),1)
        tmpW = torch.cat([Tmp,w],dim=1)
        tmpW = tmpW.to(torch.float32)
        Gxixj = tDyn.msg_fnc(Tmp)
        m12 = Gxixj*w#/source_x[:,Dimension].reshape(source_x.shape[0],1)

        nodes_num = int(objectAij.shape[0])
        mask = np.ones((nodes_num, nodes_num), dtype=bool)
        np.fill_diagonal(mask, 0)
        selected_elements = objectAij.T[mask]
        object_edges_rem_diag = selected_elements.reshape(-1,1)
        object_edges = np.tile(object_edges_rem_diag, (int(Len), 1))
        object_edges = torch.tensor(object_edges, dtype=torch.float32)
        object_edges = object_edges.view(-1,1)

        all_messages = torch.cat((
            tmpW, object_edges, Gxixj,
             m12), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['Aij']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['Aij']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['w']
            columns += ['Aij']
            columns += ['G']
            columns += ['e%d'%(k,) for k in range(msg_dim)]

        return pd.DataFrame(
              data=all_messages.cpu().detach().numpy(),
             columns=columns
        )
        #print(all_messages.shape)
        return pd.DataFrame(all_messages)

    msg_info = []
    for i, g in enumerate(loader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    
    return msg_info
