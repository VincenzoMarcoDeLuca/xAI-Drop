from torch_geometric.nn import GINConv, BatchNorm
from torch.nn import Linear, Dropout, ReLU, Softmax
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F
import torch


import torch.nn as nn

class GIN_node_classification(nn.Module):
    def __init__(self, input_features, hidden_channels = [], dropout = [], num_classes=2, dropout_edge_p = 0.0, use_bn = True):
        super(GIN_node_classification, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.batch_norm0 = BatchNorm(input_features)
            self.batch_norm1 = BatchNorm(hidden_channels[0])

        self.conv1 = GINConv(input_features, hidden_channels[0])
        self.relu = ReLU()#torch.nn.LeakyReLU(leaky_relu1)
        self.dropout_p = dropout
        self.conv2 = GINConv(hidden_channels[0], num_classes)#hidden_channels[1])
        #self.linear = Linear(hidden_channels[0], num_classes)
        #self.linear = Linear(hidden_channels[1], num_classes)
        #self.batch_norm2 = BatchNorm(hidden_channels[1])
        self.dropout_edge_p = dropout_edge_p

    def forward(self,x,edge_index):
        if self.use_bn:
           x = self.batch_norm0(x)
        if self.training:
            edge_index, _ = dropout_edge(edge_index, self.dropout_edge_p)
            #noise_std = 0.01  # standard deviation of the noise
            # Inject noise
            #x = x + noise_std * torch.randn_like(x)
        x = F.dropout(x, self.dropout_p[0],self.training)
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.batch_norm1(x)
        x= self.relu(x)
        x = F.dropout(x, self.dropout_p[1], self.training)
        x = self.conv2(x, edge_index)
        #x = self.batch_norm2(x)
        #x = self.relu(x)
        #x = F.dropout(x, self.dropout_p[2], self.training)
        #x = self.linear(x)
        return x



    def full_forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout2(x)
        return x

    def forward_single(self, x, edge_index):
        sm = Softmax(dim = -1)
        x = self.forward(x, edge_index)
        x = sm(x)
        return x