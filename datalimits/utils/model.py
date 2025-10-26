import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn.parameter import Parameter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid, Softmax
from torch.autograd import Variable, grad
use_cuda = torch.cuda.is_available()

'''tau_soft_MessPassing is a model that can sparsely sample by ajustment of tau'''
class simultaneous_infer_deriv(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(simultaneous_infer_deriv, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
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
        self._init_weights()
    
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True) # (edge_num, 1) or (edge_num, 2)
        torch.nn.init.normal_(self.weights, 0, 0.1) # could be changed

    def _init_weights(self):
        """Initialize all Linear layers with Kaiming (He) initialization."""
        for module in [self.msg_fnc, self.node_fnc_x, self.node_fnc_y, self.node_fnc_z]:
            for layer in module:
                if isinstance(layer, Lin): 
                    torch.nn.init.kaiming_normal_(
                        layer.weight, 
                        mode='fan_in',     
                        nonlinearity='relu'
                    )
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)  

    def forward(self, x, edge_index):
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        self.soft_weights = F.softmax(self.weights/self.tau,dim=1) 
        self.soft_weights = self.soft_weights[:,0].view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.soft_weights.shape[0])
        w = self.soft_weights.repeat(int(Len),1)
        msg = self.msg_fnc(tmp)*w
        del tmp, w
        return msg

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            # updated = torch.cat([x+dxdt*self.delt_t], dim=1)
            # return torch.cat([x+dxdt*self.delt_t,dxdt], dim=1)
            return torch.cat([dxdt], dim=1)
            return updated
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            # updated = torch.cat([x_update,y_update], dim=1)
            # del x_update, y_update
            return torch.cat([dxdt,dydt], dim=1)
            return updated
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
            # updated = torch.cat([x_update,y_update,z_update], dim=1)
            # del x_update, y_update, z_update
            return torch.cat([dxdt,dydt,dzdt], dim=1)
            return updated
        
    def prediction(self, g, augment=False, augmentation=3): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            if augment:
                augmentation = torch.randn(1, ndim)*augmentation
                augmentation = augmentation.repeat(len(x), 1).to(x.device)
                x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
  
    def loss(self, g,square=False, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)+self.lam*torch.sum(self.soft_weights)
            else:
                return torch.sum(torch.abs(g.y - self.prediction(g)))+self.lam*torch.sum(self.soft_weights)
            
    def update_tau(self, newtau):
        self.tau = newtau


class simultaneous_infer_morelayer(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, edge_num, tau, lam, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(simultaneous_infer_morelayer, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
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
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        
        self.node_fnc_y = Seq(
            Lin(n_f,hidden),
            ReLU(),
            Lin(hidden,hidden),
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
            Lin(hidden,hidden),
            ReLU(),
            Lin(hidden,1)
        )
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True)
        torch.nn.init.normal_(self.weights, 0, 1) # could be changed

    def forward(self, x, edge_index):
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i[:,0], x_j[:,0]])
        tmp = tmp.reshape(2,-1)
        tmp = tmp.t()
        self.soft_weights = F.softmax(self.weights/self.tau,dim=1)
        self.soft_weights = self.soft_weights[:,0].view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.soft_weights.shape[0])
        w = self.soft_weights.repeat(int(Len),1)
        return self.msg_fnc(tmp)*w

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
            if augment:
                augmentation = torch.randn(1, ndim)*augmentation
                augmentation = augmentation.repeat(len(x), 1).to(x.device)
                x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
  
    def loss(self, g,square=False, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)+self.lam*torch.sum(self.soft_weights)
            else:
                return torch.sum(torch.abs(g.y - self.prediction(g)))+self.lam*torch.sum(self.soft_weights)
            
    def update_tau(self, newtau):
        self.tau = newtau


'''now, when we get binary Aij result, then we may use class of GraphNN
to precisely learn F and G'''

from torch.nn import Sequential as Seq, Linear as Lin, Sigmoid, ReLU
from torch_geometric.nn import MessagePassing

class GraphNN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i, otherwise is (i,j)'"""

        super(GraphNN, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
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

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        if self.ndim==1:
            tmp = torch.cat([x_i,x_j], dim=1)
        else:
            tmp = torch.cat([x_i[:,0], x_j[:,0]]) # tmp has shape [E, 2 * in_channels]
            tmp = tmp.reshape(2,-1)
            tmp = tmp.t()
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            return x+dxdt*self.delt_t
            return x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            return torch.cat([x_update,y_update], dim=1)
            return x_update,y_update
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
            return torch.cat([x_update,y_update,z_update], dim=1)
            return x_update,y_update,z_update
        
    def prediction(self, g, augment=False, augmentation=3):
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            if augment:
                augmentation = torch.randn(1, ndim)*augmentation
                augmentation = augmentation.repeat(len(x), 1).to(x.device)
                x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)

    
    def loss(self, g,square=True, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)
            else:
                return torch.sum(torch.abs(g.y - self.prediction(g)))

    def get_trajectories(self, g, **kwargs):
            if self.ndim == 1:
                xUpdate = self.prediction(g)
                return xUpdate
            if self.ndim == 2:
                xUpdate, yUpdate = self.prediction(g)
                return xUpdate, yUpdate
            if self.ndim == 3:
                xUpdate,yUpdate,zUpdate = self.prediction(g)
                return xUpdate, yUpdate, zUpdate

from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data

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

from torch.nn import Sequential as Seq, Linear as Lin, Sigmoid, ReLU
from torch_geometric.nn import MessagePassing

class Graph_deriv_NN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, delt_t, hidden=50, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i, otherwise is (i,j)'"""

        super(Graph_deriv_NN, self).__init__(aggr=aggr, flow=flow)
        self.msg_fnc = Seq(
            Lin(2,hidden),
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

    def forward(self, x, edge_index):
        # x has shape [N, number_of_features]
        # edge_index has shape [2,E]
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        if self.ndim==1:
            tmp = torch.cat([x_i,x_j], dim=1)
        else:
            tmp = torch.cat([x_i[:,0], x_j[:,0]]) # tmp has shape [E, 2 * in_channels]
            tmp = tmp.reshape(2,-1)
            tmp = tmp.t()
        return self.msg_fnc(tmp)

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            return torch.cat([dxdt], dim=1)
            return x+dxdt*self.delt_t
            return x_update
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            x_update = x[:,0].reshape(-1,1)+dxdt*self.delt_t
            y_update = x[:,1].reshape(-1,1)+dydt*self.delt_t
            return torch.cat([dxdt,dydt], dim=1)
            return torch.cat([x_update,y_update], dim=1)
            return x_update,y_update
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
            return torch.cat([dxdt,dydt,dzdt], dim=1)
            return torch.cat([x_update,y_update,z_update], dim=1)
            return x_update,y_update,z_update
        
    def prediction(self, g, augment=False, augmentation=3):
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            if augment:
                augmentation = torch.randn(1, ndim)*augmentation
                augmentation = augmentation.repeat(len(x), 1).to(x.device)
                x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)

    
    def loss(self, g,square=False, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)
            else:
                return torch.sum(torch.abs(g.y - self.prediction(g)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class edegsInfer(MessagePassing):
    def __init__(self, ndim, delt_t, edge_num, tau, lam, aggr='add', flow='source_to_target'):

        """If flow is 'source_to_target', the relation is (j,i), means information is passed from x_j to x_i'"""
        '''edge_num should be defined previously'''
        
        super(edegsInfer, self).__init__(aggr=aggr, flow=flow)
        
        self.ndim = ndim
        self.delt_t = delt_t
        self.soft_weights = None
        self.tau = tau
        self.lam = lam

        self.weights = Parameter(torch.Tensor(edge_num,2), requires_grad=True) # (edge_num, 1) or (edge_num, 2)
        torch.nn.init.normal_(self.weights, 0, 0.1) # could be changed

    def node_fnc_x(self,x):
        Fx = -10*x[:,0]+10*x[:,1]
        return Fx.reshape(-1,1)
    def node_fnc_y(self,x):
        Fy = 28*x[:,0]-x[:,1]-x[:,0]*x[:,2]
        return Fy.reshape(-1,1)
    def node_fnc_z(self,x):
        Fz = -8/3*x[:,2]+x[:,0]*x[:,1]
        return Fz.reshape(-1,1)
    def msg_fnc(self,tmp):
        Gx = 0.2*(tmp[:,1]-tmp[:,0])
        return Gx.reshape(-1,1)
         
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
        self.soft_weights = F.softmax(self.weights/self.tau,dim=1) # True or False
        self.soft_weights = self.soft_weights[:,0].view(-1,1)
        Len = int(x_i[:,0].shape[0])/int(self.soft_weights.shape[0])
        w = self.soft_weights.repeat(int(Len),1)
        msg = self.msg_fnc(tmp)*w
        return msg

    def update(self, aggr_out, x=None):
        if self.ndim==1:
            fx = self.node_fnc_x(x)
            dxdt = fx+aggr_out
            return torch.cat([dxdt], dim=1)
            return updated
        elif self.ndim==2:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            dxdt = fx+aggr_out
            dydt = fy
            return torch.cat([dxdt,dydt], dim=1)
            return updated
        elif self.ndim==3:
            fx = self.node_fnc_x(x)
            fy = self.node_fnc_y(x)
            fz = self.node_fnc_z(x)
            dxdt = fx+aggr_out
            dydt = fy
            dzdt = fz
            return torch.cat([dxdt,dydt,dzdt], dim=1)
            return updated
        
    def prediction(self, g): # How to use prediction
            #x is [n, n_f]f
            x = g.x
            ndim = self.ndim
            edge_index = g.edge_index
            return self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
  
    def loss(self, g,square=False, **kwargs):
            if square:
                return torch.sum((g.y - self.prediction(g))**2)+self.lam*torch.sum(self.soft_weights)
            else:
                return torch.sum(torch.abs(g.y - self.prediction(g)))+self.lam*torch.sum(self.soft_weights)
            
    def update_tau(self, newtau):
        self.tau = newtau