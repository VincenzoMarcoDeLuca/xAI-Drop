from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import subgraph
import torch
from torch.nn import Softmax, KLDivLoss
from torch.nn.functional import log_softmax

import torch.nn.functional as F
from math import ceil
import numpy as np

# Go through all objects tracked by the garbage collector
"""
import gc
for obj in gc.get_objects():
try:
    if torch.is_tensor(obj):
        if obj.is_cuda:
            tensors_on_gpu.append(obj)
            #print(f"Tensor on GPU: {obj.shape}, dtype: {obj.dtype}, device: {obj.device}")
except Exception as e:
    pass
"""
def acc_sufficiency(file_name):
    data_list_xai = []
    data_list_or = []
    file_path = f'{file_name}_EXPL.txt'
    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and add the line to the list
            data_list_xai.append(int(line))
    file_path = f'{file_name}_OR.txt'
    with open(file_path, 'r') as f:
        for line in f:
            data_list_or.append(int(line))
    data_list_xai = np.array(data_list_xai)
    data_list_or = np.array(data_list_or)
    correct = np.sum((data_list_xai==data_list_or).astype(int))
    tot = correct/data_list_xai.shape[0]
    print("ACCURACY SUF: ", tot)


def whole_explanation(x, edge_index, model, explainer, num_classes, num_hops = 2, sparsity_expl = 0.66, batch_size=250, file_name = None, all = None):
    if all is None:
        all = x.shape[0]
        mask = torch.tensor([i for i in range(all)])
    else:
        mask = torch.zeros(x.shape[0]).bool()
        for i in range(all):
            mask[i] = True
    batch = 5000
    expl_pred_class = torch.tensor([]).to('cpu')
    for node_idx in range(all):
            if node_idx%300==0:
                print("Node idx: ", node_idx)#, " ", len(gc.get_objects()))
            all_explanations = produce_explanation(x, edge_index, node_idx, explainer)

            _, edge_index_sub, _, _ = k_hop_subgraph(node_idx, num_hops=num_hops, edge_index=edge_index, relabel_nodes=True)
            _, topk_index = torch.topk(all_explanations.edge_mask, int(round(sparsity_expl * edge_index_sub.shape[1])))

            filtered_edges = edge_index[:, topk_index]
            output_expl = model(x, filtered_edges)[node_idx].reshape(1, -1)
            tmp_expl_pred_class = torch.argmax(output_expl, dim=-1)

            expl_pred_class = torch.cat((expl_pred_class, tmp_expl_pred_class))
            del all_explanations
    output_original = model(x, edge_index)[mask]
    mod_pred_class = torch.argmax(output_original, dim=-1)
    if file_name is not None:
        file_path = f'{file_name}_EXPL.txt'
        with open(file_path, 'w') as f:
            for item in expl_pred_class.tolist():
                f.write(f"{int(item)}\n")

        file_path = f'{file_name}_OR.txt'
        with open(file_path, 'w') as f:
            for item in mod_pred_class.tolist():
                f.write(f"{int(item)}\n")



def produce_explanation(x, edge_index, node_idx, explainer):
  if not isinstance(node_idx, int):
      node_idx = node_idx.item()

   #subset, edge_index, inv, edge_mask



  #novel_x = data.x[node_subset]
  explanation = explainer(x, edge_index, index = node_idx)#novel_x, edge_subset, index=node_idx_mapping)

  del explainer

  ##explanation = explainer.get_explanation_node(node_idx = node_idx, x = data.x.to(device), edge_index = data.edge_index.to(device), dataset = dataset)
  return explanation

def kl_divergence(p: torch.Tensor, q: torch.Tensor, reduction: str = 'batchmean') -> torch.Tensor:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Args:
        p (torch.Tensor): First probability distribution (P).
        q (torch.Tensor): Second probability distribution (Q).
        reduction (str): Specifies the method to reduce the loss.
                         Can be 'batchmean', 'mean', or 'sum'.

    Returns:
        torch.Tensor: The KL divergence D_{KL}(P || Q).
    """
    # Ensure the distributions are tensors and convert to float for log operations
    p = p.float()
    q = q.float()

    # Add a small constant to avoid log(0) which is undefined
    eps = 1e-10
    p = p + eps
    q = q + eps

    # Compute KL divergence with specified reduction
    kl_div = F.kl_div(p.log(), q, reduction=reduction)
    return kl_div

def node_ref_mapping(original_x, index_mapping):
    """
    mapped_x = torch.zeros((len(index_mapping.keys()), original_x.shape[1])).to('cuda:0')
    # Perform the mapping
    for old_index, new_index in index_mapping.items():
        mapped_x[new_index] = original_x[old_index]
    """
    mapped_x = torch.zeros(len(index_mapping.keys())).long().to('cuda:0')
    for old_index, new_index in index_mapping.items():
        mapped_x[new_index] = old_index

    return  mapped_x

def remove_output_edges(list_useless_nodes, edge_idx):
    """
    mask = torch.ones(edge_idx.shape[1], dtype = torch.bool).to('cuda:0')
    for idx, edge in enumerate(zip(edge_idx[0,:], edge_idx[1,:])):
        if edge[0] in list_useless_nodes:
            mask[idx] = False
    return mask
    """
    useless_mask = torch.isin(edge_idx[0, :], list_useless_nodes)
    # Set the corresponding entries in the mask to False
    useless_mask = ~useless_mask
    return useless_mask

def remove_bidirectional_edges(list_relevant_nodes, edge_idx):
    #mask = torch.zeros(edge_idx.shape[1], dtype = torch.bool).to('cuda:0')
    #for idx, edge in enumerate(zip(edge_idx[0,:], edge_idx[1,:])):
    #    if edge[0] in list_relevant_nodes or edge[1] in list_relevant_nodes:
    #        mask[idx] = True
    #return mask
    source_mask = torch.isin(edge_idx[0, :], list_relevant_nodes)
    target_mask = torch.isin(edge_idx[1, :], list_relevant_nodes)

    # Combine the masks to get the final mask
    mask = source_mask | target_mask
    return mask

def measure_kl(tensor1, tensor2):
    kl_div_value = kl_divergence(tensor1, tensor2, reduction='none')  # batchmean')
    kl_div_value_mean = torch.mean(kl_div_value, dim=-1)
    suff_faithfulness = torch.exp(-(kl_div_value_mean))
    return suff_faithfulness

def sufficiency_edges(model, expl_edge_idx, x, edge_index, edge_imp, explanation, threshold = 0.5, num_hops = 2, sparsity_value = 0.1, sparsity_strategy = 'percentage', sparsity_threshold =0.5):
    with torch.no_grad():
        model.eval()
        ###Importanza degli edge del sottografo
        edge_imp = explanation.edge_imp

        ###Otteniamo k-hop subgraph
        khop_edge = explanation.enc_subgraph.edge_index
        khop_x = x[explanation.enc_subgraph.nodes]

        out1 = model.encode(khop_x, khop_edge)
        out2 = model.decode(out1, khop_edge)
        original_graph_pred = out2.sigmoid()

        ###Mapping è l'edge_index nel train_data.edge_index originale
        edge_mapping = explanation.enc_subgraph.edge_mapping
        ###Qual è l'indice dell'edge_index nel train_data.edge_index originale
        ###nel enc_subgraph



        if sparsity_strategy == 'percentage':
            _, mask = torch.topk(edge_imp, int(ceil(sparsity_value*len(edge_imp))))
        else:
            print("The sparsity strategy: ", sparsity_strategy, " is not available")
            exit(1)

        try:
            condition_mask = ~torch.isin(edge_mapping, mask)
        except:
            import pdb; pdb.set_trace()

        mask = torch.unique(torch.cat((mask, edge_mapping[condition_mask])))
        subgraph_edges = explanation.enc_subgraph.edge_index[:, mask]

        out1 = model.encode(khop_x, subgraph_edges)
        out2 = model.decode(out1, subgraph_edges)
        subgraph_graph_pred = out2.sigmoid()


        edge_mapping = edge_mapping.reshape(-1)
        confident_edge_original_graph_pred = original_graph_pred[edge_mapping]
        matches = edge_mapping.unsqueeze(1) == mask.unsqueeze(0)

        # Use torch.where to find indices where the match occurs
        # We only need the indices where there is a match
        subgraph_edge_mapping = torch.where(matches)[1]

        # Convert to the desired shape and move it to the GPU if needed
        subgraph_edge_mapping = subgraph_edge_mapping.reshape(-1).to('cuda:0')

        confident_edge_subgraph_graph_pred =subgraph_graph_pred[subgraph_edge_mapping]

        suff_faithfulness = 1-torch.abs(confident_edge_original_graph_pred-confident_edge_subgraph_graph_pred)

        return suff_faithfulness


##N.B. Dovrebbe esistere qualche funzione per estrarre num_hops in automatico
def sufficiency(model, node_idx, x, edge_index, node_imp, explanation, threshold = 0.5, num_hops = 2, sparsity_value = 0.1, sparsity_strategy = 'percentage', sparsity_threshold = 0.5):
    #kl_loss = KLDivLoss()
    with torch.no_grad():
        model.eval()
        softmax_layer = Softmax(dim= -1)

        """
        khop_info = subset, sub_edge_index, mapping, _ = \
          k_hop_subgraph([node_idx], num_hops, edge_index,
                       relabel_nodes=True, num_nodes=x.shape[0])
        """
        khop_info = explanation.enc_subgraph
        ###Questo è il mapping fra nodi (PIù CONFIDENTI) nel grafo originale(node_idx) e nodi più CONFIDENTI nel k-hop
        ###In poche parole, prendere i più confidenti nel grafo originale equivale a x[node_ix]
        ###nel k-hop invece equivale a node_ref_mapping(x, exp.node_reference)[khop_info.mapping]

        ###Indice dei nodi (confidenti) nel khop
        mapping = explanation.node_idx

        if sparsity_strategy == 'percentage':
            _, mask = torch.topk(node_imp, int(ceil(sparsity_value*len(node_imp))))
        elif sparsity_strategy == 'threshold':
            mask = torch.where(node_imp >= sparsity_threshold)[0]
            #print("Percentage retained node xai: ", mask.shape[0], " ", node_imp.shape[0], " ", mask.shape[0]/node_imp.shape[0])
        else:
            print("Strategy for picking explanation: ", sparsity_strategy, " is not available")
            exit(1)
        explanation_sparsity = mask.shape[0]/node_imp.shape[0]
        ###In mask c'è l'indice dei nodi ordinati, quindi node_imp[mask] è una lista ordinata
        ###Per mappare mask agli indici originali nel grafo
        ###su
        """
        list_useless_nodes = []
        for mapping_node in mapping:
            if mapping_node not in mask:
               #print("This node is useless for the predictions! ", mapping_node)
               mask = torch.cat((mask, mapping_node.reshape(-1)))
               list_useless_nodes.append(mapping_node)
        list_useless_nodes = torch.tensor(list_useless_nodes).to('cuda:0')
        """
        try:
            condition_mask = ~torch.isin(mapping, mask)
        except:
            import pdb; pdb.set_trace()
        list_useless_nodes = mapping[condition_mask]#torch.tensor([], device='cuda:0', dtype=mapping.dtype)#.reshape(-1)

        """
        for mapping_node in mapping:
            if not torch.any(mapping_node == mask):
                mapping_node = mapping_node.reshape(-1)
                list_useless_nodes = torch.cat((list_useless_nodes, mapping_node))
        """

        mask = torch.cat((mask, list_useless_nodes))
        #import pdb; pdb.set_trace()

        ###Nodi non importanti non devono influenzare gli altri
        mask_filtered_edges = remove_output_edges(list_useless_nodes, khop_info.edge_index)
        filtered_edges = khop_info.edge_index[:,mask_filtered_edges]
        mask_filtered_edges = remove_bidirectional_edges(mask, filtered_edges)
        filtered_edges = filtered_edges[:,mask_filtered_edges]
        filtered_nodes = x[khop_info.nodes]

        expl_prediction = model(filtered_nodes, filtered_edges)
        mapping_expl_prediction = expl_prediction
        mapping_expl_prediction_all = softmax_layer(mapping_expl_prediction)
        mapping_expl_prediction = mapping_expl_prediction_all[mapping]

        real_prediction = model(filtered_nodes, khop_info.edge_index)
        mapping_real_prediction = real_prediction
        mapping_real_prediction_all = softmax_layer(mapping_real_prediction)
        mapping_real_prediction = mapping_real_prediction_all[mapping]
        """
        ###In node classification, non possiamo non avere il nodo nella spiegazione
        mapping_filtered = torch.tensor([torch.where(mask == mapping[i].item()) for i in range(len(mapping))])
        mapping_filtered = mapping_filtered.reshape(-1)
        whole_subgraph = x[subset]

        # Filtering node features, questi sono i nodi d'interesse per il k_hop e che hanno una certa importanza
        x_filtered = whole_subgraph[mask]

        sub_edge_index = khop_info.edge_index

        edge_index_filtered, _ = subgraph(mask, sub_edge_index, relabel_nodes=True)

        pred_prob = model(x_filtered, edge_index_filtered)
        """

        #pred_original = model(whole_subgraph, sub_edge_index)

        #pred_prob_original = softmax_layer(pred_original)
        #actual_prediction = pred_prob_original[mapping]
        #print("expl_prediction: ", expl_prediction, " actual_prediction", actual_prediction, " ", mapping_filtered, " ", mapping, " ", mask)

        #kl_div_value = kl_divergence(mapping_real_prediction, mapping_expl_prediction, reduction='none')#batchmean')
        #kl_div_value_mean = torch.mean(kl_div_value, dim=-1)
        #suff_faithfulness = torch.exp(-(kl_div_value_mean))

        suff_faithfulness = measure_kl(mapping_real_prediction, mapping_expl_prediction)

        return suff_faithfulness, khop_info.nodes[mapping], explanation_sparsity#, all_suff_faithfulness#_filtered

def normalize(node_imp):
    node_imp = (node_imp - torch.min(node_imp))/(torch.max(node_imp)-torch.min(node_imp))
    return node_imp


def get_explanation(explainer, node_idx, x, edge_index, edge_idx = None, k_hop_info = None, task = 'node_classification'):
    if task == 'node_classification':
        explanation = explainer.get_explanation_node(node_idx, x, edge_index, k_hop_info = k_hop_info)
    elif task == 'link_prediction':
        explanation = explainer.get_explanation_link(edge_idx, x, edge_index, k_hop_info = k_hop_info, return_type='normalized')
    return explanation.node_imp, explanation


def process_node(explainer, node_idx, data, model, k_hop_info = None):
    node_imp, explanation = get_explanation(explainer, node_idx, data.x, data.edge_index)#, k_hop_info[node_idx])
    return sufficiency(model, node_idx, data.x, data.edge_index, node_imp, explanation=explanation)

