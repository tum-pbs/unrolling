# -*- coding: utf-8 -*-
"""

KS test network archs

"""
import numpy as np
import scipy as scp
import torch, os
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU

DISABLE_GCN = int(os.environ.get("KS_DISABLE_GCN", "0"))>0
if not DISABLE_GCN:
    from torch_geometric.nn import GCNConv, MessagePassing
    import torch_geometric

class ConvResNet1D(torch.nn.Module):
    def __init__(self, res_net_channels, res_net_depth, device):
        super(ConvResNet1D, self).__init__()
        # note - removed "flow_size" param, this was the resolution of the grid, shouldnt matter
        
        USE_BIAS = True
        self.upsample_conv_layer = torch.nn.Conv1d(2, res_net_channels, kernel_size=3, padding=1,
                                                   padding_mode='circular', bias=USE_BIAS).to(device)   
        self.conv_layers = []
        self.relu_layers = []
        self.res_net_depth = res_net_depth
        for i in range(res_net_depth):
            self.conv_layers.append(torch.nn.Conv1d(res_net_channels, res_net_channels, kernel_size=3, padding=1,
                                                    padding_mode='circular', bias=USE_BIAS).to(device))
            self.relu_layers.append(torch.nn.LeakyReLU(inplace=True).to(device))
            self.conv_layers.append(torch.nn.Conv1d(res_net_channels, res_net_channels, kernel_size=3, padding=1, 
                                                    padding_mode='circular', bias=USE_BIAS).to(device))
            
            self.relu_layers.append(torch.nn.LeakyReLU(inplace=True).to(device))

        if res_net_depth==0:
            self.relu_layers.append(torch.nn.LeakyReLU(inplace=True).to(device))

        self.downsample_conv_layer = torch.nn.Conv1d(res_net_channels, 1, kernel_size=3, padding=1,
                                                   padding_mode='circular', bias=USE_BIAS).to(device) 
        self.net = torch.nn.Sequential(self.upsample_conv_layer, *self.conv_layers,
                                       *self.relu_layers, self.downsample_conv_layer)
                       
    def forward(self,x):
        x = self.upsample_conv_layer(x)
        for i in range(self.res_net_depth):
            x_skip = x
            x = self.conv_layers[i*2](x)
            x = self.relu_layers[i*2](x)
            x = self.conv_layers[i*2+1](x)+x_skip
            x = self.relu_layers[i*2+1](x)
        if self.res_net_depth==0:
            x = self.relu_layers[0](x)
        x = self.downsample_conv_layer(x)
        return x



# GCN networks are optional

def direction_feature(edge_index: Tensor, edge_dim_channels, device) -> Tensor:
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    edge_ft = torch.unsqueeze(edge_index[0,:] - edge_index[1,:],1).to(device)

    #Shape: [2,E] edge_indices[0,:] has all other nodes connecting to edge_indices[1,:] (less often changing)
    for i in range(edge_ft.shape[0]):
        if edge_ft[i,0] < 0:
            edge_ft[i,0] = -1
        elif edge_ft[i,0] > 0:
            edge_ft[i,0] = 1
    
    edge_ft = edge_ft.repeat(1, edge_dim_channels)
    
    return edge_ft

if not DISABLE_GCN:

    class Directional_EdgeConv(MessagePassing):
        def __init__(self, in_channels, out_channels, direction_ft):
            #super().__init__(aggr='sum') #  "Sum" aggregation.
            super().__init__(aggr='mean') # 'Mean' Aggregation
            #self.mlp = Seq(Linear(2 * in_channels + 1, out_channels))
            self.direction_ft = direction_ft
            
            self.mlp = Seq(Linear(2 * in_channels + direction_ft.shape[-1], out_channels))
            

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            # x has shape [N, in_channels]
            # edge_index has shape [2, E]

            #propagate_type: (x: Tensor)
        
            # Since the edge_index doesn't change dynamically direction_ft also doesn't
            self.dir_ft = self.direction_ft.unsqueeze(0).repeat(x.shape[0], 1, 1)
            
            return self.propagate(edge_index=edge_index, x=x, size=None)

        def message(self, x_i, x_j):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]

            tmp = torch.cat([x_i, x_j - x_i, self.dir_ft], dim=-1)
            # tmp = torch.cat([x_i, x_j-x_i], dim=-1)
            return self.mlp(tmp)

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, depth=3, device="", nn_inputs = 2, edgeconv=False, direction_ft=None):
            super().__init__()
            #torch.manual_seed(1234567)
            self.depth = depth
            self.device = device
            if edgeconv:
                self.convIn = Directional_EdgeConv(nn_inputs, hidden_channels, direction_ft).to(device)
            else:
                self.convIn = GCNConv(nn_inputs, hidden_channels).to(device)

            for i in range(1, depth * 2 + 1):
                if edgeconv:
                    self.add_module(f'GCNConv{i}',
                                    Directional_EdgeConv(hidden_channels, hidden_channels, direction_ft).to(device))
                else:
                    self.add_module(f'GCNConv{i}',
                                GCNConv(hidden_channels, hidden_channels).to(device))

            if edgeconv:
                self.convOut = Directional_EdgeConv(hidden_channels, 1, direction_ft).to(device)
            else:
                self.convOut = GCNConv(hidden_channels, 1).to(device)

        def forward(self, x: Tensor, edge_index: Tensor):

            x_2 =  self.convIn(x, edge_index)
            
            for i in range(1, self.depth * 2 + 1, 2):
                # no inplace=True for compile?
                x_1 = torch.nn.LeakyReLU(inplace=True)(getattr(self, f'GCNConv{i}')(x_2, edge_index))
                x_2 = torch.nn.LeakyReLU(inplace=True)(getattr(self, f'GCNConv{i + 1}')(x_1, edge_index) + x_2)
                #x_1 = torch.nn.LeakyReLU()(getattr(self, f'GCNConv{i}')(x_2, edge_index))
                #x_2 = torch.nn.LeakyReLU()(getattr(self, f'GCNConv{i + 1}')(x_1, edge_index) + x_2)
            if self.depth==0:
                x_2 = torch.nn.LeakyReLU(inplace=True)(x_2)
            
            # x_out = self.convOut(x_2, edge_index) + x # variant, residual
            x_out = self.convOut(x_2, edge_index) 
            
            return x_out
else:

    # dummy
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, depth=3, device="", nn_inputs = 2, edgeconv=False, direction_ft=None):
            pass


# TRAINING DATASET helpers - uses torch.utils.data to create iterable 
# functions to arange dataset into training trajectories
def stack_prediction_inputs(data, prediction_horizon, input_slices):
    indices = np.arange(data.shape[0]-prediction_horizon - input_slices)[:,None] + np.arange(0,input_slices)
    return data[indices]

def stack_prediction_targets(data, prediction_horizon, input_slices):
    indices = np.arange(data.shape[0]-prediction_horizon - input_slices)[:,None] + np.arange(input_slices,input_slices+prediction_horizon)
    return data[indices]


