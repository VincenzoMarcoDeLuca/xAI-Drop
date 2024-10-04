import numpy as np
from sklearn.preprocessing import PowerTransformer
import torch
from torch_geometric.utils import subgraph

def gaussian_mapping(x):
    x = x.cpu().detach().numpy()
    x = x.reshape(-1, 1)
    pt = PowerTransformer()
    x = np.longdouble(x)
    pt.fit(x)
    gaussian_x = pt.transform(x)
    gaussian_x = ((gaussian_x - np.min(gaussian_x))/(np.max(gaussian_x)-np.min(gaussian_x)))
    gaussian_x = gaussian_x.reshape(-1)
    return torch.tensor(gaussian_x)

def edge_dropping(edge_probability):
    num_edges = edge_probability.shape[0]
    torch_prob= torch.rand(num_edges).to('cuda:0')
    edge_mask = edge_probability < torch_prob
    return edge_mask

def xai_edge_dropping(edge_probability):

    num_edges= edge_probability.shape[0]
    ###Prevent having always the same ordering
    noise = torch.rand(num_edges) * 1e-4
    noise = noise.to('cuda:0')
    edge_probability = edge_probability - noise
    edge_probability = torch.clip(edge_probability, 0.0, 1.0)

    edge_mask = torch.bernoulli(1.- edge_probability).bool().to('cuda:0')
    return edge_mask

def node_dropping(edge_index, node_probability, mask, default_drop_probability = 0.5, device = 'cuda:0'):
    num_nodes = node_probability.shape[0]
    trn_node_probability = node_probability[mask]
    ###Prevent having always the same ordering
    noise = torch.rand(trn_node_probability.shape) * 1e-4
    noise = noise.to(device)
    trn_node_probability = trn_node_probability + noise

    randomicity = torch.rand(trn_node_probability.shape)
    randomicity = randomicity.to(device)
    remove_percentage_trn = default_drop_probability

    trn_node_probability = trn_node_probability-randomicity

    less_xai_trn_node = torch.argsort(trn_node_probability)[:int(trn_node_probability.shape[0]*remove_percentage_trn)]
    most_xai_trn_node = torch.argsort(trn_node_probability)[int(trn_node_probability.shape[0]*remove_percentage_trn):]

    node_mask = torch.bernoulli(1.- node_probability).bool().to('cuda:0')

    node_mask[less_xai_trn_node] = False
    node_mask[most_xai_trn_node] = True

    edge_index, _, edge_mask = subgraph(
        node_mask,
        edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        return_edge_mask=True,
       )
    return edge_index, edge_mask, node_mask


def confidence_dropping(mapping_filtered, confidence_values, num_samples, default_probability=0.3, device = 'cuda'):
    if torch.max(confidence_values) != torch.min(confidence_values):
        mapping_filtered = mapping_filtered.to(device)
        confidence_values = (confidence_values-torch.min(confidence_values))/(torch.max(confidence_values)-torch.min(confidence_values))
        dropping_probability = gaussian_mapping(confidence_values).to(device)
        dropping_probability = (1-dropping_probability).float()
        dropping_probability = torch.clip(dropping_probability, min=0.0, max = 1.0)
        all_probabilities = torch.ones(num_samples) * default_probability
        all_probabilities = all_probabilities.to(device)
        all_probabilities[mapping_filtered] = dropping_probability
    else:
        all_probabilities = torch.ones(num_samples)*default_probability
        all_probabilities = all_probabilities.to(device)

    return all_probabilities
