from torch_geometric.nn import GATConv, BatchNorm
from torch.nn import Linear, ReLU, Dropout, Softmax
import torch.nn as nn

class GAT_node_classification(nn.Module):
    def __init__(self, input_features, hidden_channels=[], num_heads=[], dropout=[], num_classes=2):
        super(GAT_node_classification, self).__init__()


        self.dropout1 = Dropout(p=dropout[0])
        self.conv1 = GATConv(input_features, hidden_channels[0], heads = num_heads[0])
        self.relu = ReLU()
        self.dropout2 = Dropout(p=dropout[1])

        self.conv2 = GATConv(hidden_channels[0]*num_heads[0], hidden_channels[1], heads = num_heads[1])
        self.linear = Linear(hidden_channels[1]*num_heads[1], num_classes = num_classes)

    def forward(self,x,edge_index):
        x = self.conv1(x, edge_index)

        #x = self.batchnorm1(x)

        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)

        #x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.leaky_relu2(x)
        x = self.linear(x)
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