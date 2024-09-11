import torch
from model.node_classification.utils import train, test
from metrics.uncertainty import most_confident
from dropping.drop import confidence_dropping, xai_edge_dropping
from metrics.explainability import get_explanation, normalize, sufficiency_edges
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
#from torch_geometric.loader import RandomNodeLoader

from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, dropout_edge, dropout_node
from torch_geometric.explain import CaptumExplainer, Explainer

def train_link_predictor(model, train_data, optimizer, criterion, drop_edge_strategy = None, drop_edge_probability = None, explainer = None, threshold_confidence_value = 0.95, confidence_strategy= 'threshold', sparsity_value = 0.25):
   auc = 0.0
   loss = 0.0
   trn_acc = 0.0
   mean_edge_dropping_probs = drop_edge_probability
   std_edge_dropping_probs = 0.0
   all_edges = torch.cat((train_data.pos_edge_label_index, train_data.neg_edge_label_index), dim=-1)
   all_edges_label = torch.cat((train_data.pos_edge_label, train_data.neg_edge_label), dim=-1)
   most_confident_edges, most_confident_values, mean_confidence, std_confidence = most_confident(model=model,
                                                                                                 x=train_data.x,
                                                                                                 edge_index=all_edges,
                                                                                                 percentage=None,
                                                                                                 threshold_confidence_value=threshold_confidence_value,
                                                                                                 confidence_strategy=confidence_strategy,
                                                                                                 need_softmax=False,
                                                                                                 task='link_prediction',
                                                                                                 criterion='Most',
                                                                                                 mask=None)
   """
   if most_confident_edges is not None and most_confident_edges.shape[0]>2:
       most_confident_edges = most_confident_edges.reshape(-1)
       most_confident_values = most_confident_values.reshape(-1)
       ###Guarantee balancement of the dataset  in picking most_confident_edges
       negative_label_mask = (all_edges_label[most_confident_edges]==0)
       positive_label_mask = (all_edges_label[most_confident_edges]==1)
       count_negative_label = torch.sum(negative_label_mask.int())
       count_positive_label = torch.sum(positive_label_mask.int())

       most_confident_edges_negative = most_confident_edges[negative_label_mask]
       most_confident_edges_positive = most_confident_edges[positive_label_mask]
       most_confident_values_edges_negative = most_confident_values[negative_label_mask]
       most_confident_values_edges_positive = most_confident_values[positive_label_mask]
       if count_negative_label > count_positive_label:
          values, indexes_sorted = torch.sort(most_confident_values_edges_negative)
          most_confident_values_edges_negative = most_confident_values_edges_negative[-count_positive_label:]
          indexes_sorted = indexes_sorted[-count_positive_label:]
          most_confident_edges_negative = most_confident_edges[indexes_sorted]
       elif count_positive_label > count_negative_label:
          values, indexes_sorted = torch.sort(most_confident_values_edges_positive)
          most_confident_values_edges_positive = most_confident_values_edges_positive[-count_positive_label:]
          indexes_sorted = indexes_sorted[-count_positive_label:]
          most_confident_edges_positive = most_confident_edges[indexes_sorted]
       most_confident_edges = torch.cat((most_confident_edges_positive, most_confident_edges_negative),dim=-1)
       most_confident_values = torch.cat((most_confident_values_edges_positive, most_confident_values_edges_negative))
   """
   #most_confident_edges = None
   #mean_confidence = 0.0
   #std_confidence = 0.0
   num_confident_edges = most_confident_edges.shape[0] if most_confident_edges is not None else 0

   if True:
              if drop_edge_strategy == 'confidence' or drop_edge_strategy == 'xai':
                  if most_confident_edges is not None and drop_edge_strategy == 'confidence':
                      most_confident_edges = most_confident_edges.reshape(-1)
                      #edge_imp, explanation = get_explanation(explainer, node_idx=None, x=trn_batch.x, edge_index=trn_batch.edge_index,
                      #                                        edge_idx=most_confident_edges, k_hop_info=None, task='link_prediction')
                      #def get_explanation(explainer, node_idx, x, edge_index, edge_idx = None, k_hop_info = None, task = 'node_classification'):

                      num_edges = train_data.edge_index.shape[1]
                      most_confident_values = most_confident_values.reshape(-1)
                      edge_probability, mean_edge_probability, std_edge_probability = confidence_dropping(most_confident_edges, mapping_filtered=None, confidence_values = most_confident_values, num_nodes=num_edges,
                                      default_probability=0.5, mapping='gaussian', normalize_score=False)
                      edge_mask = edge_dropping(edge_probability)
                  elif most_confident_edges is not None and drop_edge_strategy == 'xai':
                      most_confident_edges = most_confident_edges.reshape(-1)

                      #all_pos_remaining_edges_label = train_data.pos_edge_label[edge_pos_mask]
                      edge_imp, explanation = get_explanation(explainer,
                                                              node_idx=None,
                                                              x=train_data.x,
                                                              edge_index=all_edges,
                                                              edge_idx=most_confident_edges,
                                                              k_hop_info=None,
                                                              task='link_prediction')

                      edge_imp = 1 - normalize(edge_imp)
                      all_edges = torch.tensor([i for i in range(train_data.edge_index.shape[1])]).to('cuda:0')

                      faithfulness_sufficiency = sufficiency_edges(model = model,
                                                             expl_edge_idx = most_confident_edges,
                                                             x= train_data.x,
                                                             edge_index = train_data.edge_index,
                                                             edge_imp = edge_imp,
                                                             explanation=explanation,
                                                             sparsity_value=sparsity_value)

                      #if most_confident_edges.shape[0] > 1:
                      #    import pdb; pdb.set_trace()
                      edge_dropping_probs, mean_probability, std_probability = confidence_dropping(most_confident_edges,
                                                                                                   most_confident_edges,
                                                                                                   faithfulness_sufficiency,
                                                                                                   train_data.edge_index.shape[1],
                                                                                                   default_probability=drop_edge_probability,
                                                                                                   normalize_score=False,
                                                                                                   mean_confidence_dropping=0.0,
                                                                                                   vary_probability=False,
                                                                                                   distribution='gaussian')
                      edge_mask = xai_edge_dropping(edge_dropping_probs, default_drop_probability=0.5, most_confident_edges = most_confident_edges)
                      edge_pos_mask = edge_mask[:int(edge_mask.shape[0]/2)]
                      edge_neg_mask = edge_mask[int(edge_mask.shape[0]/2):]
                      all_remaining_x = train_data.x
                      mean_edge_dropping_probs = torch.mean(edge_dropping_probs).item()
                      std_edge_dropping_probs = torch.std(edge_dropping_probs).item()
                  else:
                      edge_pos_mask = torch.ones(train_data.pos_edge_label_index.shape[1]).bool()
                      edge_neg_mask = torch.ones(train_data.neg_edge_label_index.shape[1]).bool()
                      all_remaining_x = train_data.x

              elif drop_edge_strategy == 'fair_drop':
                  delta = 0.16
                  protected_attribute = train_data.y
                  Y = protected_attribute
                  Y_aux = (Y[train_data.pos_edge_label_index[0, :]] != Y[train_data.pos_edge_label_index[1, :]]).to('cuda')
                  randomization = (torch.FloatTensor(Y_aux.size(0)).uniform_() < 0.5 + delta).to('cuda')
                  keep = torch.where(randomization, Y_aux, ~Y_aux)
                  edge_pos_mask = torch.zeros(train_data.pos_edge_label_index.shape[1]).bool()
                  edge_pos_mask[keep] = True
                  all_remaining_x = train_data.x

                  edge_neg_mask = torch.ones(train_data.neg_edge_label_index.shape[1]).bool()

                  print("Total edges: ", all_edges.shape, " used edges: ", torch.sum(torch.cat((edge_pos_mask, edge_neg_mask)).int()))

              elif drop_edge_strategy == 'random_edge':
                  _, edge_pos_mask = dropout_edge(edge_index=train_data.pos_edge_label_index, p=drop_edge_probability)

                  _, edge_neg_mask = dropout_edge(edge_index=train_data.neg_edge_label_index, p=drop_edge_probability)

                  all_remaining_x = train_data.x

              elif drop_edge_strategy == 'random_node':
                  ###Even if we ignore pos_node_mask e neg_node_mask, the nodes of interest will be isolated thanks  to pos_edge_mask and neg_edge_mask
                  _, edge_pos_mask, pos_node_mask = dropout_node(edge_index=train_data.pos_edge_label_index, p = 0.5)
                  _, edge_neg_mask, neg_node_mask = dropout_node(edge_index=train_data.neg_edge_label_index, p = 0.5)
                  #all_remaining_x = train_data.x[torch.logical_or(pos_node_mask, neg_node_mask)]
                  all_remaining_x = train_data.x

              elif drop_edge_strategy=='baseline':
                  edge_pos_mask = torch.ones(train_data.pos_edge_label_index.shape[1]).bool()
                  edge_neg_mask = torch.ones(train_data.neg_edge_label_index.shape[1]).bool()
                  all_remaining_x = train_data.x
              else:
                  print("Drop edge strategy: ", drop_edge_strategy, " is not available")
                  exit(1)
              try:
                  all_pos_remaining_edges = train_data.pos_edge_label_index[:,edge_pos_mask]
                  all_pos_remaining_edges_label = train_data.pos_edge_label[edge_pos_mask]

                  all_neg_remaining_edges = train_data.neg_edge_label_index[:,edge_neg_mask]
                  all_neg_remaining_edges_label = train_data.neg_edge_label[edge_neg_mask]

                  all_edges = torch.cat((all_pos_remaining_edges, all_neg_remaining_edges), dim=-1)
                  all_edges_label = torch.cat((all_pos_remaining_edges_label, all_neg_remaining_edges_label), dim=-1)

              except:
                  import pdb; pdb.set_trace()


              model.train()
              optimizer.zero_grad()
              z = model.encode(all_remaining_x,all_pos_remaining_edges)
              out = model.decode(z, all_edges).view(-1)
              loss = criterion(out, all_edges_label)
              loss.backward()
              optimizer.step()


              out = out.sigmoid()
              AUC_metric = BinaryAUROC()
              AUC_metric.update(out, all_edges_label)
              trn_auc = AUC_metric.compute()
              #auc = roc_auc_score(all_edges_label.cpu().numpy(), out.detach().cpu().numpy())

              AP_metric = BinaryAUPRC()
              AP_metric.update(out, all_edges_label)
              trn_ap = AP_metric.compute()

              pos_pr = out > 0.5
              trn_acc = torch.sum((pos_pr == all_edges_label).int()) / all_edges_label.shape[0]


   return trn_auc, trn_ap.item(), trn_acc, loss.item(), num_confident_edges, mean_edge_dropping_probs, std_edge_dropping_probs, mean_confidence, std_confidence

@torch.no_grad()
def eval_link_predictor(model, data, criterion, device='cuda'):
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
       #eval_auc = roc_auc_score(all_eval_edges_label.cpu().numpy(), out.cpu().numpy())
       AP_metric = BinaryAUPRC()
       AP_metric.update(out, all_eval_edges_label)
       eval_ap = AP_metric.compute()

       pos_pr = out > 0.5
       eval_acc = torch.sum((pos_pr == all_eval_edges_label).int())/all_eval_edges_label.shape[0]
       return eval_auc, eval_ap, eval_loss, eval_acc