from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm
from torch.nn import Linear, ReLU, Sequential, Dropout
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F
import torch.nn as nn
import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch_geometric.utils import dropout_node
from metrics.uncertainty import most_confident
from metrics.explainability import get_explanation, sufficiency
from dropping.drop import confidence_dropping, node_dropping


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

class GIN_node_classification(nn.Module):
    def __init__(self, input_features, hidden_channels = [], dropout = [], num_classes=2, dropout_edge_p = 0.0, use_bn = True):
        super(GIN_node_classification, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.batch_norm0 = BatchNorm(input_features)
            self.batch_norm1 = BatchNorm(hidden_channels[0])
        nn1 = Sequential(Linear(input_features, hidden_channels[0]), ReLU())
        self.conv1 = GINConv(nn1)
        self.relu = ReLU()
        self.dropout_p = dropout
        self.linear = Linear(hidden_channels[0], num_classes)
        self.dropout_edge_p = dropout_edge_p

    def forward(self,x,edge_index):
        if self.use_bn:
           x = self.batch_norm0(x)
        if self.training:
            edge_index, _ = dropout_edge(edge_index, self.dropout_edge_p)
        x = F.dropout(x, self.dropout_p[0],self.training)
        x = self.conv1(x, edge_index)
        if self.use_bn:
           x = self.batch_norm1(x)
        x = F.dropout(x, self.dropout_p[1],self.training)
        x = self.linear(x)
        return x

class GCN_node_classification(nn.Module):
    def __init__(self, input_features, hidden_channels = [], dropout = [], num_classes=2, dropout_edge_p = 0.0, use_bn = True):
        super(GCN_node_classification, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.batch_norm0 = BatchNorm(input_features)
            self.batch_norm1 = BatchNorm(hidden_channels[0])
        self.conv1 = GCNConv(input_features, hidden_channels[0])
        self.relu = ReLU()
        self.dropout_p = dropout
        self.conv2 = GCNConv(hidden_channels[0], num_classes)
        self.dropout_edge_p = dropout_edge_p

    def forward(self,x,edge_index):
        if self.use_bn:
           x = self.batch_norm0(x)
        if self.training:
            edge_index, _ = dropout_edge(edge_index, self.dropout_edge_p)
        x = F.dropout(x, self.dropout_p[0],self.training)
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.batch_norm1(x)
        x= self.relu(x)
        x = F.dropout(x, self.dropout_p[1], self.training)
        x = self.conv2(x, edge_index)
        return x




def train(model, data, mask, criterion, optimizer, num_classes, dropping_probability = 0.1, explainer = None,  sparsity_value = 0.1, threshold_confidence_value = 0.8, device = "cuda"):

  most_confident_nodes, confidence_values = most_confident(model, data.x, data.edge_index, threshold_confidence_value = threshold_confidence_value)
  if dropping_probability > 0.0 and most_confident_nodes is not None and most_confident_nodes.shape[0]>1:

    most_confident_nodes = most_confident_nodes.reshape(-1)
    node_imp, explanation = get_explanation(explainer, most_confident_nodes, data.x, data.edge_index)
    node_imp = torch.abs(node_imp)

    faithfulness_sufficiency, mapping_filtered, explanation_sparsity = sufficiency(model, data.x, node_imp, explanation=explanation,  sparsity_value = sparsity_value)
    node_dropping_probs = confidence_dropping(most_confident_nodes, faithfulness_sufficiency, data.x.shape[0], default_probability=dropping_probability, device = device)
  else:
      node_dropping_probs = None
  model.train()
  optimizer.zero_grad()

  if dropping_probability > 0.0:
        if node_dropping_probs is not None:
                edge_index, _, node_mask = node_dropping(data.edge_index, node_dropping_probs,data.train_mask, default_drop_probability=dropping_probability)#mean_confidence_nodes)

        else:
               edge_index, _, node_mask = dropout_node(data.edge_index, dropping_probability, relabel_nodes=True)

        x = data.x[node_mask]
        y = data.y[node_mask]
        mask = mask[node_mask]
  else:
        edge_index = data.edge_index
        x = data.x
        y = data.y
  y_pred = model(x, edge_index)
  loss = criterion(y_pred[mask], y[mask])
  acc = multiclass_f1_score(y_pred[mask], y[mask], num_classes=num_classes, average="micro")
  loss.backward()
  optimizer.step()
  return loss.item(), acc.item()

def test(model, data, mask, criterion, num_classes):
    with torch.no_grad():
        model.eval()
        y_pred = model(data.x, data.edge_index)
        loss = criterion(y_pred[mask], data.y[mask])
        acc = multiclass_f1_score(y_pred[mask], data.y[mask], num_classes=num_classes, average="micro")

    return loss.item(),acc.item()