
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn import Dropout
import torch.nn as nn
from torch.nn.functional import relu

class GCN_link_prediction(nn.Module):
    def __init__(self, in_channels, hidden_channels=[256,32], dropout_probability = [0.0,0.0], use_bn = False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.batch_norm1 = BatchNorm(in_channels)
            self.batch_norm2 = BatchNorm(hidden_channels[0])
        self.dropout1 = Dropout(p=dropout_probability[0])
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.dropout2 = Dropout(p=dropout_probability[1])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])

    def encode(self, x, edge_index):
        if self.use_bn:
            x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.batch_norm2(x)
        x = relu(x)
        x = self.dropout2(x)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
