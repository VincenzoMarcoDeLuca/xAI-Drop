import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch_geometric.utils import dropout_node
from metrics.uncertainty import most_confident
from metrics.explainability import get_explanation, sufficiency
from dropping.drop import confidence_dropping, node_dropping


def train(model, data, mask, criterion, optimizer, num_classes, confidence_sampling = 0.25, dropping_probability = 0.1, explainer = None,  sparsity_value = 0.1, threshold_confidence_value = 0.8, confidence_strategy='ranking', normalize_score = False, sparsity_strategy = 'percentage', sparsity_threshold = 0.5, vary_probability = False, dropping_distribution='cumulative'):

  most_confident_nodes, confidence_values, mean_confidence_nodes, std_confidence_nodes = most_confident(model, data.x, data.edge_index, percentage=confidence_sampling, threshold_confidence_value = threshold_confidence_value, confidence_strategy=confidence_strategy)
  if dropping_probability > 0.0 and most_confident_nodes is not None and most_confident_nodes.shape[0]>1:

    most_confident_nodes = most_confident_nodes.reshape(-1)
    node_imp, explanation = get_explanation(explainer, most_confident_nodes, data.x, data.edge_index)
    node_imp = torch.abs(node_imp)

    faithfulness_sufficiency, mapping_filtered, explanation_sparsity = sufficiency(model, most_confident_nodes, data.x, data.edge_index, node_imp, explanation=explanation,  sparsity_value = sparsity_value, sparsity_strategy = sparsity_strategy, sparsity_threshold = sparsity_threshold)
    node_dropping_probs, mean_probability, std_probability = confidence_dropping(most_confident_nodes, mapping_filtered, faithfulness_sufficiency, data.x.shape[0], default_probability=dropping_probability, normalize_score=normalize_score, mean_confidence_dropping = mean_confidence_nodes, vary_probability=vary_probability, distribution=dropping_distribution)
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
