from torch_geometric.nn import GATConv, BatchNorm
from torch.nn import Linear, ReLU, Dropout, Softmax
import torch.nn as nn
import torch.nn.functional as F


class GAT_node_classification(nn.Module):
    def __init__(self, input_features, hidden_channels=[], num_heads=[], dropout=[], num_classes=2):
        super(GAT_node_classification, self).__init__()


        if self.use_bn:
            self.batch_norm0 = BatchNorm(input_features)
            self.batch_norm1 = BatchNorm(hidden_channels[0])
        self.dropout1 = Dropout(p=dropout[0])
        self.conv1 = GATConv(input_features, hidden_channels[0], heads = num_heads[0])
        self.relu = ReLU()
        self.dropout2 = Dropout(p=dropout[1])
        self.linear = Linear(hidden_channels[0]*num_heads[0], num_classes = num_classes)

    def forward(self,x,edge_index):
        if self.use_bn:
            x = self.batch_norm0(x)
        x = F.dropout(x, self.dropout_p[0],self.training)
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.batch_norm1(x)
        x= self.relu(x)
        x = F.dropout(x, self.dropout_p[1],self.training)
        x = self.linear(x)
        return x

    def full_forward(self, x, edge_index):
        if self.use_bn:
            x = self.batch_norm0(x)
        x = F.dropout(x, self.dropout_p[0],self.training)
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.batch_norm1(x)
        x= self.relu(x)
        x = F.dropout(x, self.dropout_p[1],self.training)
        return x

    def forward_single(self, x, edge_index):
        sm = Softmax(dim = -1)
        x = self.forward(x, edge_index)
        x = sm(x)
        return x
