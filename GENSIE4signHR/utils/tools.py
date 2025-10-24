import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import log

def get_edge_index(Adj): #target to source
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index

def cal_structure_performance(true_A, weights, epc):
    Gweights = F.softmax(weights/(0.999**epc),dim=1)
    Gweights = (1*Gweights[:,0])+((-1)*Gweights[:,1]) #Gweights[:,0] is excitatory, Gweights[:,1] is inhibitory
    Gweights = Gweights.view(-1,1).cpu()
    bestWei = Gweights.detach().numpy()
    nodes_num = 20
    edge_num = nodes_num*(nodes_num-1)
    bestWei = bestWei[:edge_num]

    Stru = np.zeros((nodes_num, nodes_num))
    mask = np.ones((nodes_num, nodes_num), dtype=bool)
    np.fill_diagonal(mask, 0)
    Stru[mask] = bestWei.squeeze()
    error1 = np.sum(np.abs(Stru - true_A))
    error2 = np.sum(np.abs(-Stru - true_A))
    return error1, error2

def calculate_aic(n,mse,num_params):
    aic = n * log(mse) + 2 * num_params 
    return aic 

def terms_sort_fit(X_lib,Y_goal,intercept):
    reg = LassoCV(cv=5, fit_intercept=intercept, n_jobs=-1, max_iter=1000).fit(X_lib,Y_goal)
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

def get_messages_ori(tDyn, dim, msg_dim, loader):

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
        Gweights = (1*Gweights[:,0])+((-1)*Gweights[:,1]) #Gweights[:,0] is excitatory, Gweights[:,1] is inhibitory
        Gweights = Gweights.view(-1,1)
        Len = int(s1.shape[0])/int(Gweights.shape[0])
        T = Gweights.repeat(int(Len),1)
        print(T.shape)
        T_excit = torch.where(T>0,T,0)
        T_inh = torch.where(T<0,-T,0)

        tmpT = torch.cat([Tmp,T],dim=1)
        tmpT = tmpT.to(torch.float32)

        m12_excit = tDyn.msg_fnc_excit(Tmp)
        m12_inh = tDyn.msg_fnc_inh(Tmp)
        m12 = m12_excit*T_excit+m12_inh*T_inh

        all_messages = torch.cat((
            tmpT, m12_excit, m12_inh,
             m12), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1,3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
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

def get_messages(tDyn, dim, msg_dim, loader):

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
        Gweights = (1*Gweights[:,0])+((-1)*Gweights[:,1]) # Gweights[:,0] is type1, Gweights[:,1] is type2
        Gweights = Gweights.view(-1,1)
        Len = int(s1.shape[0])/int(Gweights.shape[0])
        T = Gweights.repeat(int(Len),1)
        print(T.shape)
        T_type1 = torch.where(T>0,T,0)
        T_type2 = torch.where(T<0,-T,0)

        tmpT = torch.cat([Tmp,T],dim=1)
        tmpT = tmpT.to(torch.float32)

        m12_type1 = tDyn.msg_fnc_type1(Tmp)
        m12_type2 = tDyn.msg_fnc_type2(Tmp)
        m12 = m12_type1*T_type1+m12_type2*T_type2

        all_messages = torch.cat((
            tmpT, m12_type1, m12_type2,
             m12), dim=1)
        if dim == 1:
            columns = [elem%(k) for k in range(1,3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
            columns += ['e%d'%(k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem%(k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['T']
            columns += ['type1']
            columns += ['type2']
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


