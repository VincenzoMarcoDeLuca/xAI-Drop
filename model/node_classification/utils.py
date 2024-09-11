import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch_geometric.utils import dropout_node
from metrics.uncertainty import most_confident
from metrics.explainability import get_explanation, sufficiency, process_node, normalize
from dropping.drop import confidence_dropping, node_dropping, gaussian_mapping


def train(model, data, mask, criterion, optimizer, num_classes, confidence_sampling = 0.25, dropping_probability = 0.1, explainer = None, k_hop_info_list = None, epoch = 50, starting_drop_epoch = 100, drop_strategy='xai', sparsity_value = 0.1, threshold_confidence_value = 0.8, confidence_strategy='ranking', normalize_score = False, confidence_criterion='Most', retain_sufficiency = True,  sparsity_strategy = 'percentage', sparsity_threshold = 0.5, vary_probability = False, dropping_distribution='cumulative'):
  all_sufficiency_mean =0.0
  all_sufficiency_std = 0.0
  explanation_sparsity = 0.0
  if retain_sufficiency is not True:
      all_nodes = torch.tensor([i for i in range(data.x.shape[0])]).to('cuda:0')
      node_imp, explanation = get_explanation(explainer, all_nodes, data.x, data.edge_index)
      node_imp = 1 - normalize(node_imp)
      all_sufficiency, _= sufficiency(model, all_nodes, data.x, data.edge_index,
                                                               node_imp, explanation=explanation,
                                                               sparsity_value=sparsity_value)
      all_sufficiency_mean = torch.mean(all_sufficiency).item()
      all_sufficiency_std = torch.std(all_sufficiency).item()
  if epoch >= starting_drop_epoch:
   most_confident_nodes, confidence_values, mean_confidence_nodes, std_confidence_nodes = most_confident(model, data.x, data.edge_index, percentage=confidence_sampling, threshold_confidence_value = threshold_confidence_value, confidence_strategy=confidence_strategy, criterion=confidence_criterion)
   if dropping_probability > 0.0 and drop_strategy=='xai' and most_confident_nodes is not None and most_confident_nodes.shape[0]>1:

    most_confident_nodes = most_confident_nodes.reshape(-1)
    most_confident_nodes_num = most_confident_nodes.shape[0]
    faithfulness_sufficiency = []
    node_imp, explanation = get_explanation(explainer, most_confident_nodes, data.x, data.edge_index)
    #normalized_node_imp, pt = gaussian_mapping(node_imp)
    #import pdb; pdb.set_trace()

    #node_imp = -node_imp
    #node_imp = 1-normalize(node_imp)
    node_imp = torch.abs(node_imp)
    top_12,_ = torch.topk(node_imp, int(node_imp.shape[0]*0.12))
    top_25,_ = torch.topk(node_imp, int(node_imp.shape[0]*0.25))
    top_50,_ = torch.topk(node_imp, int(node_imp.shape[0]*0.5))
    print(f"Questa è la node_imp: {torch.mean(node_imp).item():.3f} {torch.std(node_imp).item():.3f}"
          f"Node_imp50: {torch.mean(top_50).item():.3f} {torch.std(top_50):.3f}"
          f"Node_imp25: {torch.mean(top_25).item():.3f} {torch.std(top_25):.3f}"
          f"Node_imp12: {torch.mean(top_12).item():.3f} {torch.std(top_12):.3f}")

    faithfulness_sufficiency, mapping_filtered, explanation_sparsity = sufficiency(model, most_confident_nodes, data.x, data.edge_index, node_imp, explanation=explanation,  sparsity_value = sparsity_value, sparsity_strategy = sparsity_strategy, sparsity_threshold = sparsity_threshold)
    node_dropping_probs, mean_probability, std_probability = confidence_dropping(most_confident_nodes, mapping_filtered, faithfulness_sufficiency, data.x.shape[0], default_probability=dropping_probability, normalize_score=normalize_score, mean_confidence_dropping = mean_confidence_nodes, vary_probability=vary_probability, distribution=dropping_distribution)
    ###Extract training nodes from confident set:
    trn_indexes = torch.nonzero(data.train_mask).reshape(-1).to('cuda')
    mapping_filtered = mapping_filtered.to('cuda')
    useful_map_index = torch.where(torch.isin(mapping_filtered, trn_indexes))[0].to('cuda')
    if useful_map_index.shape[0] > 0:
        #trn_confident_index = mapping_filtered[useful_map_index].to('cuda')
        trn_faithfulness_sufficiency = faithfulness_sufficiency[useful_map_index]
        mean_trn_faithfulness_sufficiency = torch.mean(trn_faithfulness_sufficiency).item()
        std_trn_faithfulness_sufficiency = torch.std(trn_faithfulness_sufficiency).item()
    else:
        mean_trn_faithfulness_sufficiency = 1.0
        std_trn_faithfulness_sufficiency = 0.0
    mean_faithfulness_sufficiency = torch.mean(faithfulness_sufficiency).item()
    std_faithfulness_sufficiency = torch.std(faithfulness_sufficiency).item()
    """
    for idx, node_idx in enumerate(most_confident_nodes):
        if k_hop_info_list is not None:
           node_imp, explanation = get_explanation(explainer, [node_idx], data.x, data.edge_index, k_hop_info_list[node_idx])
        faithfulness_sufficiency.append(sufficiency(model, node_idx, data.x, data.edge_index, node_imp, explanation=explanation))
    """
    #if True:
    #    with ThreadPoolExecutor() as executor:
    #        futures = [executor.submit(process_node, explainer, [node_idx], data, model, k_hop_info_list) for node_idx in most_confident_nodes]#explainer, node_idx, data, model))
    #        faithfulness_sufficiency = [future.result() for future in futures]
    #faithfulness_sufficiency = torch.tensor(faithfulness_sufficiency)
    #node_dropping_probs = confidence_dropping(most_confident_nodes, faithfulness_sufficiency, num_nodes = data.x.shape[0], default_probability = dropping_probability)
   else:
    mean_faithfulness_sufficiency = 1.0
    std_faithfulness_sufficiency = 0.0
    mean_trn_faithfulness_sufficiency = 1.0
    std_trn_faithfulness_sufficiency = 0.0
    most_confident_nodes_num = 0
    node_dropping_probs = None
    mean_probability = 0.5
    std_probability = 0.0
  model.train()
  optimizer.zero_grad()

  if dropping_probability > 0.0 and epoch >= starting_drop_epoch:
        #edge_index, _, node_mask = node_dropping(data.edge_index, node_dropping_probs)#dropout_node(data.edge_index, drop_prob, relabel_nodes= True)
        if drop_strategy == 'random':
            print("Random strategy")
            edge_index, _, node_mask = dropout_node(data.edge_index, dropping_probability, relabel_nodes=True)
            mean_dropping = dropping_probability

        elif drop_strategy == 'xai':
            if node_dropping_probs is not None:
                #model_certainty = (most_confident_nodes_num / data.x.shape[0])
                #node_dropping_probs = model_certainty * (node_dropping_probs + 0.5)
                mean_dropping = torch.mean(node_dropping_probs).item()
                print("Node_dropping_probs: ", node_dropping_probs, " ", torch.sum(node_dropping_probs).item())
                print("DROP Mean: ", mean_dropping, " Max: ", torch.max(node_dropping_probs).item(), " Min: ", torch.min(node_dropping_probs).item(), " ", mean_confidence_nodes)
                edge_index, _, node_mask = node_dropping(data.edge_index, node_dropping_probs,data.train_mask, default_drop_probability=dropping_probability)#mean_confidence_nodes)

            else:
               mean_dropping = dropping_probability
               edge_index, _, node_mask = dropout_node(data.edge_index, dropping_probability, relabel_nodes=True)

        else:
            print("Not known strategy")
            exit(1)
        #print("Ei: ", edge_index.shape, " Ni: ", node_mask.int().sum())
        x = data.x[node_mask]
        y = data.y[node_mask]
        mask = mask[node_mask]
  else:
        edge_index = data.edge_index
        node_mask = torch.ones(data.x.shape[0]).bool()
        mean_dropping = dropping_probability
        x = data.x
        y = data.y
        #mask = mask
  y_pred = model(x, edge_index)
  loss = criterion(y_pred[mask], y[mask])
  acc = multiclass_f1_score(y_pred[mask], y[mask], num_classes=num_classes, average="micro")
  node_backprop = y_pred[mask].shape[0]
  loss.backward()
  optimizer.step()
  loss_eval, acc_eval = test(model, data, data.train_mask, criterion, num_classes)
  return [loss_eval, acc_eval], x.shape[0], edge_index.shape[1], mean_probability, std_probability, most_confident_nodes_num, mean_dropping, mean_confidence_nodes, std_confidence_nodes, all_sufficiency_mean, all_sufficiency_std, explanation_sparsity, [loss.item(), acc.item()], node_backprop, mean_faithfulness_sufficiency, std_faithfulness_sufficiency, mean_trn_faithfulness_sufficiency, std_trn_faithfulness_sufficiency

def test(model, data, mask, criterion, num_classes):
    with torch.no_grad():
        model.eval()
        y_pred = model(data.x, data.edge_index)
        loss = criterion(y_pred[mask], data.y[mask])
        acc = multiclass_f1_score(y_pred[mask], data.y[mask], num_classes=num_classes, average="micro")

    return loss.item(),acc.item()