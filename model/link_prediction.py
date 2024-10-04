
from torch_geometric.nn import GCNConv, GATConv, GINConv, BatchNorm
from torch.nn import Dropout, ReLU, Linear
import torch.nn as nn
from torch.nn.functional import relu
import torch
from metrics.uncertainty import most_confident
from dropping.drop import confidence_dropping, xai_edge_dropping
from metrics.explainability import get_explanation, normalize, sufficiency_edges
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torch_geometric.utils import dropout_edge

def train_link_predictor(model, train_data, optimizer, criterion, drop_edge_probability = None, explainer = None, threshold_confidence_value = 0.95, sparsity_value = 0.25, device = 'cuda'):
   all_edges = torch.cat((train_data.pos_edge_label_index, train_data.neg_edge_label_index), dim=-1)
   most_confident_edges, most_confident_values = most_confident(model=model,
                                                               x=train_data.x,
                                                               edge_index=all_edges,
                                                               threshold_confidence_value=threshold_confidence_value,
                                                               need_softmax=False,
                                                               task='link_prediction')
   if most_confident_edges is not None:
       most_confident_edges = most_confident_edges.reshape(-1)
       edge_imp, explanation = get_explanation(explainer,
                                           node_idx=None,
                                           x=train_data.x,
                                        edge_index=all_edges,
                                        edge_idx=most_confident_edges,
                                        k_hop_info=None,
                                        task='link_prediction')

       edge_imp = 1 - normalize(edge_imp)
       all_edges = torch.tensor([i for i in range(train_data.edge_index.shape[1])]).to(device)

       faithfulness_sufficiency = sufficiency_edges(model = model,
                                                    x= train_data.x,
                                                    edge_imp = edge_imp,
                                                    explanation=explanation,
                                                    sparsity_value=sparsity_value)

       edge_dropping_probs = confidence_dropping(most_confident_edges,
                                                 faithfulness_sufficiency,
                                                 train_data.edge_index.shape[1],
                                                 default_probability=drop_edge_probability,
                                                 device = device)
   else:
       edge_dropping_probs = None
   if edge_dropping_probs is not None:
        edge_mask = xai_edge_dropping(edge_dropping_probs)
   else:
        _, edge_mask = dropout_edge(all_edges)

   edge_pos_mask = edge_mask[:train_data.pos_edge_label.shape[0]]
   edge_neg_mask = edge_mask[train_data.pos_edge_label.shape[0]:]

   all_pos_remaining_edges = train_data.pos_edge_label_index[:,edge_pos_mask]
   all_pos_remaining_edges_label = train_data.pos_edge_label[edge_pos_mask]

   all_neg_remaining_edges = train_data.neg_edge_label_index[:,edge_neg_mask]
   all_neg_remaining_edges_label = train_data.neg_edge_label[edge_neg_mask]

   all_edges = torch.cat((all_pos_remaining_edges, all_neg_remaining_edges), dim=-1)
   all_edges_label = torch.cat((all_pos_remaining_edges_label, all_neg_remaining_edges_label), dim=-1)

   model.train()
   optimizer.zero_grad()
   z = model.encode(train_data.x, all_pos_remaining_edges)
   out = model.decode(z, all_edges).view(-1)
   loss = criterion(out, all_edges_label)
   loss.backward()
   optimizer.step()
   out = out.sigmoid()
   AUC_metric = BinaryAUROC()
   AUC_metric.update(out, all_edges_label)
   trn_auc = AUC_metric.compute()
   AP_metric = BinaryAUPRC()
   AP_metric.update(out, all_edges_label)
   trn_ap = AP_metric.compute()
   pos_pr = out > 0.5
   trn_acc = torch.sum((pos_pr == all_edges_label).int()) / all_edges_label.shape[0]

   return trn_auc.item(), trn_ap.item(), trn_acc.item(), loss.item()

@torch.no_grad()
def eval_link_predictor(model, data, criterion):
    with torch.no_grad():
       model.eval()
       z = model.encode(data.x, data.pos_edge_label_index)

       all_eval_edges_label_index = torch.cat((data['pos_edge_label_index'], data['neg_edge_label_index']), dim=-1)
       all_eval_edges_label = torch.cat((data['pos_edge_label'], data['neg_edge_label']), dim=-1)

       out = model.decode(z, all_eval_edges_label_index).view(-1)
       eval_loss = criterion(out, all_eval_edges_label)
       out = out.sigmoid()
       AUC_metric = BinaryAUROC()
       AUC_metric.update(out, all_eval_edges_label)
       eval_auc = AUC_metric.compute()
       AP_metric = BinaryAUPRC()
       AP_metric.update(out, all_eval_edges_label)
       eval_ap = AP_metric.compute()

       pos_pr = out > 0.5
       eval_acc = torch.sum((pos_pr == all_eval_edges_label).int())/all_eval_edges_label.shape[0]

       return eval_auc.item(), eval_ap.item(), eval_loss.item(), eval_acc.item()

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

class GAT_link_prediction(nn.Module):
    def __init__(self, in_channels, hidden_channels=[256,32], dropout_probability = [0.0,0.0], num_heads = [4,2], use_bn = False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.batch_norm1 = BatchNorm(in_channels)
            self.batch_norm2 = BatchNorm(hidden_channels[0])
        self.dropout1 = Dropout(p=dropout_probability[0])
        self.conv1 = GATConv(in_channels, hidden_channels[0], num_heads = num_heads[0])
        self.dropout2 = Dropout(p=dropout_probability[1])
        self.conv2 = GATConv(hidden_channels[0], hidden_channels[1], num_heads[1])

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


class GIN_link_prediction(nn.Module):
    def __init__(self, in_channels, hidden_channels=[256, 32], dropout_probability=[0.0, 0.0], use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        if use_bn:
            self.batch_norm1 = BatchNorm(in_channels)
            self.batch_norm2 = BatchNorm(hidden_channels[0])
        self.dropout1 = Dropout(p=dropout_probability[0])
        self.conv1 = GINConv(Linear(in_channels, hidden_channels[0]), ReLU())
        self.dropout2 = Dropout(p=dropout_probability[1])
        self.conv2 = GINConv(Linear(hidden_channels[0], hidden_channels[1]), ReLU())

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