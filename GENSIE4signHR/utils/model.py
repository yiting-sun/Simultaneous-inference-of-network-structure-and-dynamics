import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn.parameter import Parameter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid, Softmax
from torch.autograd import Variable, grad
from torch.utils.data import Dataset
from torch_geometric.data import Data
use_cuda = torch.cuda.is_available()

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Get_sign_MP_ori(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(Get_sign_MP_ori, self).__init__(aggr=aggr, flow=flow)
        # we suppose this type is excitatory, the other type is inhibitory.
        # But the sign could be changed during training.
        self.msg_fnc_excit = Seq( 
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        
        self.msg_fnc_inh = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        self.ndim = ndim
        self.delt_t = delt_t
        self.Tweights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True) # (edge_num, 1) or (edge_num, 2)
        torch.nn.init.normal_(self.weights, 0, 0.1) # could be changed

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # assert isinstance(self.weights, torch.Tensor), "weights must be a PyTorch tensor"
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        soft_weights = F.softmax(self.weights/self.tau,dim=1) 
        self.Tweights = soft_weights[:,0]-soft_weights[:,1] #[:,0] is excit, [:,1] is inhibit
        self.Tweights = self.Tweights.view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.Tweights.shape[0])

        w_all = self.Tweights.repeat(int(Len),1)
        w_excit = torch.where(w_all>0, w_all, 0)
        w_inh = torch.where(w_all<0, -w_all, 0)
        return self.msg_fnc_excit(tmp)*w_excit+self.msg_fnc_inh(tmp)*w_inh

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            # return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
            return torch.cat([x+dxdt*self.delt_t], dim=1)
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            # return torch.cat([x_update,y_update,dxdt,dydt], dim=1)
            return torch.cat([x_update,y_update], dim=1)
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_update = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            # return torch.cat([x_update,y_update,z_update,dxdt,dydt,dzdt], dim=1)
            return torch.cat([x_update,y_update,z_update], dim=1)
        
    def prediction(self, g, augment=False, augmentation=3): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
    
    def loss(self, g, square=False, **kwargs):
        pred = self.prediction(g)  
        if self.lam:
            if square:
                return torch.sum((g.y - pred)**2) + self.lam*torch.sum(torch.abs(self.Tweights))
            else:
                return torch.sum(torch.abs(g.y - pred)) + self.lam*torch.sum(torch.abs(self.Tweights))
        else:
            if square:
                return torch.sum((g.y - pred)**2)
            else:
                return torch.sum(torch.abs(g.y - pred))
  
    def update_tau(self, newtau):
        self.tau = newtau

# in this module, we change the message flow, and optimize the function name 'type1' and 'type2'
class Get_sign_MP(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(Get_sign_MP, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_type1 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        
        self.msg_fnc_type2 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        self.ndim = ndim
        self.delt_t = delt_t
        self.Tweights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True) # (edge_num, 1) or (edge_num, 2)
        torch.nn.init.normal_(self.weights, 0, 0.1) # could be changed

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # assert isinstance(self.weights, torch.Tensor), "weights must be a PyTorch tensor"
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        soft_weights = F.softmax(self.weights/self.tau,dim=1) 
        self.Tweights = soft_weights[:,0]-soft_weights[:,1] #[:,0] is excit, [:,1] is inhibit
        self.Tweights = self.Tweights.view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.Tweights.shape[0])

        w_all = self.Tweights.repeat(int(Len),1)
        w_type1 = torch.where(w_all>0, w_all, 0)
        w_type2 = torch.where(w_all<0, -w_all, 0)
        return self.msg_fnc_type1(tmp)*w_type1+self.msg_fnc_type2(tmp)*w_type2

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            # return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
            return torch.cat([x+dxdt*self.delt_t], dim=1)
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            # return torch.cat([x_update,y_update,dxdt,dydt], dim=1)
            return torch.cat([x_update,y_update], dim=1)
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_update = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            # return torch.cat([x_update,y_update,z_update,dxdt,dydt,dzdt], dim=1)
            return torch.cat([x_update,y_update,z_update], dim=1)
        
    def prediction(self, g, augment=False, augmentation=3): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
    
    def loss(self, g, square=True, **kwargs):
        pred = self.prediction(g)  
        if self.lam:
            if square:
                return torch.sum((g.y - pred)**2) + self.lam*torch.sum(torch.abs(self.Tweights))
            else:
                return torch.sum(torch.abs(g.y - pred)) + self.lam*torch.sum(torch.abs(self.Tweights))
        else:
            if square:
                return torch.sum((g.y - pred)**2)
            else:
                return torch.sum(torch.abs(g.y - pred))
  
    def update_tau(self, newtau):
        self.tau = newtau


from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Get_sign_MP_se(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(Get_sign_MP_se, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_type1 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        
        self.msg_fnc_type2 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        self.ndim = ndim
        self.delt_t = delt_t
        self.Tweights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True) # (edge_num, 1) or (edge_num, 2)
        torch.nn.init.normal_(self.weights, 0, 0.1) # could be changed

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # assert isinstance(self.weights, torch.Tensor), "weights must be a PyTorch tensor"
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        soft_weights = F.softmax(self.weights/self.tau,dim=1) 
        self.Tweights = soft_weights[:,0]-soft_weights[:,1] #[:,0] is excit, [:,1] is inhibit
        self.Tweights = self.Tweights.view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.Tweights.shape[0])

        w_all = self.Tweights.repeat(int(Len),1)
        w_type1 = torch.where(w_all>0, w_all, 0)
        w_type2 = torch.where(w_all<0, -w_all, 0)
        return self.msg_fnc_type1(tmp)*w_type1+self.msg_fnc_type2(tmp)*w_type2

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            # return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
            return torch.cat([x+dxdt*self.delt_t], dim=1)
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            # return torch.cat([x_update,y_update,dxdt,dydt], dim=1)
            return torch.cat([x_update,y_update], dim=1)
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_update = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            # return torch.cat([x_update,y_update,z_update,dxdt,dydt,dzdt], dim=1)
            return torch.cat([x_update,y_update,z_update], dim=1)
        
    def prediction(self, g, augment=False, augmentation=3): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
  
    def loss(self, g, square=False, **kwargs):
        pred = self.prediction(g)  
        if self.lam:
            if square:
                return torch.sum((g.y - pred)**2) + self.lam*torch.sum(torch.abs(self.Tweights))
            else:
                return torch.sum(torch.abs(g.y - pred)) + self.lam*torch.sum(torch.abs(self.Tweights))
        else:
            if square:
                return torch.sum((g.y - pred)**2)
            else:
                return torch.sum(torch.abs(g.y - pred))
            
    def update_tau(self, newtau):
        self.tau = newtau


from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Retrain_sign_MP(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edgetype, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(Retrain_sign_MP, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc_type1 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )
        
        self.msg_fnc_type2 = Seq(
            Lin(2,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,msg_dim)
        )

        self.node_fnc_x = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_z = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        self.ndim = ndim
        self.delt_t = delt_t
        self.edgetype = edgetype
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.msg_fnc_type1.apply(weight_init)
        self.msg_fnc_type2.apply(weight_init)
        self.node_fnc_x.apply(weight_init)
        self.node_fnc_y.apply(weight_init)
        self.node_fnc_z.apply(weight_init)
    

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # assert isinstance(self.weights, torch.Tensor), "weights must be a PyTorch tensor"
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        Len = int(x_i[:,0].shape[0])/int(self.edgetype.shape[0])
        T = self.edgetype.repeat(int(Len),1)
        T = T.clone().detach()
        w_type1 = torch.where(T>0, T, 0) #bistru==1, 58
        w_type2 = torch.where(T<0, -T, 0) #bistru==-1, 42
        return self.msg_fnc_type1(tmp)*w_type1+self.msg_fnc_type2(tmp)*w_type2

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            # return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
            return torch.cat([x+dxdt*self.delt_t], dim=1)
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            # return torch.cat([x_update,y_update,dxdt,dydt], dim=1)
            return torch.cat([x_update,y_update], dim=1)
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            z_update = x[:,2].reshape(-1,1)+dzdt*self.delt_t
            # return torch.cat([x_update,y_update,z_update,dxdt,dydt,dzdt], dim=1)
            return torch.cat([x_update,y_update,z_update], dim=1)
        
    def prediction(self, g): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
  
    def loss(self, g, square=True, **kwargs):
        if square:
            return torch.sum((g.y - self.prediction(g))**2)
        else:
            return torch.sum(torch.abs(g.y - self.prediction(g)))

class GraphDataset(Dataset):
    def __init__(self, data_list, labels, edge_index):
        self.data_list = data_list
        self.labels = labels
        self.edge_index = edge_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        y = self.labels[idx]
        return Data(x=x, edge_index=self.edge_index, y=y)