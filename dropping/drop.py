from scipy import stats
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import torch
from torch_geometric.utils import subgraph
from metrics.uncertainty import most_confident


def ecdf(x):
    xs, idx = torch.sort(x)
    ys = (torch.arange(1, xs.shape[0]+1)/float(xs.shape[0])).to('cuda:0')
    y = torch.empty_like(ys).to('cuda:0')
    indices = torch.arange(ys.shape[0])
    y[idx] = ys[indices]
    return ys

def gaussian_mapping(x, normalize):
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()
        x = x.reshape(-1, 1)
    else:
        x = x.reshape(-1,1)
    pt = PowerTransformer()
    x = np.longdouble(x)
    pt.fit(x)
    gaussian_x = pt.transform(x)
    if normalize:
        gaussian_x = ((gaussian_x - np.min(gaussian_x))/(np.max(gaussian_x)-np.min(gaussian_x)))
        gaussian_x = gaussian_x.reshape(-1)
        return torch.tensor(gaussian_x)
    else:
        gaussian_x = ((gaussian_x - np.min(gaussian_x))/(np.max(gaussian_x)-np.min(gaussian_x)))
        #print("Mean gaussian BEFORE: ", np.mean(gaussian_x))
        gaussian_mean = np.mean(gaussian_x)
        gaussian_mean_shift = gaussian_mean-0.5
        if gaussian_mean_shift!=0:
            gaussian_x = gaussian_x-gaussian_mean_shift
            ###ORA HO LA MEDIA GIUSTA MA IL RANGE SBAGLIATO

            #EXPERIMENT PUT THEIR AVERAGE TO 0
            gaussian_x = gaussian_x - 0.5
            range_reduction = 1-(gaussian_mean_shift)
            gaussian_x = gaussian_x*range_reduction
            gaussian_x = gaussian_x+0.5
            final_gaussian_x = gaussian_x
        else:
            final_gaussian_x = gaussian_x
        final_gaussian_x = torch.tensor(final_gaussian_x).float().to('cuda:0')
        final_gaussian_x = final_gaussian_x.reshape(-1)
        final_gaussian_x = torch.clip(final_gaussian_x, min = 0.0, max = 1.0)
        shift = torch.mean(final_gaussian_x)
        final_gaussian_x = final_gaussian_x + (0.5-shift)
        final_gaussian_x = torch.clip(final_gaussian_x, min = 0.0, max = 1.0)

        #print("Mean gaussian UPDATE: ", torch.mean(final_gaussian_x).item(), " MIN: ", torch.min(final_gaussian_x).item(), " MAX: ", torch.max(final_gaussian_x).item())
        #if np.mean(final_gaussian_x) > 0.51 or np.mean(final_gaussian_x) < 0.49 or np.min(final_gaussian_x)<-0.1 or np.max(final_gaussian_x)>1.1:
        #    import pdb; pdb.set_trace()
        gaussian_x = final_gaussian_x
        """
        gaussian_x = gaussian_x.astype(np.float32)
        min_gaussian = np.min(gaussian_x)
        max_gaussian = np.max(gaussian_x)
        new_scale = 1/(max_gaussian-min_gaussian)
        gaussian_x = (gaussian_x*new_scale)+0.5
        """
        gaussian_x = gaussian_x.reshape(-1)
        return gaussian_x#, np.min(gaussian_x),  np.max(gaussian_x)


def node_explainability():
    explainer = Explainer(model=model,
                          # algorithm=GNNExplainer(epochs=50),
                          algorithm=CaptumExplainer('Saliency'),  # 'IntegratedGradients'),
                          explanation_type='model',
                          model_config=dict(
                              mode='multiclass_classification',
                              # mode='binary_classification',
                              task_level='node',
                              return_type='raw',
                          ),
                          node_mask_type="attributes",
                          edge_mask_type="object")

    # novel_x = data.x[node_subset]
    explanation = explainer(data.x, data.edge_index, index=node_idx)  # novel_x, edge_subset, index=node_idx_mapping)


def edge_dropping(edge_probability):
    num_edges = edge_probability.shape[0]
    torch_prob= torch.rand(num_edges).to('cuda:0')
    edge_mask = edge_probability < torch_prob
    return edge_mask

def xai_edge_dropping(edge_probability, default_drop_probability = 0.5, most_confident_edges = None):
    #if most_confident_edges.shape[0] > 1:
    #    import pdb; pdb.set_trace()
    num_edges= edge_probability.shape[0]
    ###Prevent having always the same ordering
    noise = torch.rand(num_edges) * 1e-4
    noise = noise.to('cuda:0')
    edge_probability = edge_probability - noise
    edge_probability = torch.clip(edge_probability, 0.0, 1.0)


    ##Qui dico quali nodi vanno tolti (n.b. quando fatto su training diventa irrilevante)
    edge_mask = torch.bernoulli(1.- edge_probability).bool().to('cuda:0')
    return edge_mask

def node_dropping(edge_index, node_probability, mask, default_drop_probability = 0.5):
    num_nodes = node_probability.shape[0]
    #torch_prob = torch.rand(num_nodes).to('cuda')
    #node_mask = node_probability < torch_prob
    try:
        trn_node_probability = node_probability[mask]
    except:
        import pdb; pdb.set_trace()
    ###Prevent having always the same ordering
    noise = torch.rand(trn_node_probability.shape) * 1e-4
    noise = noise.to('cuda:0')
    trn_node_probability = trn_node_probability + noise

    randomicity = torch.rand(trn_node_probability.shape)#*0.5
    randomicity = randomicity.to('cuda:0')
    remove_percentage_trn = default_drop_probability#retain_percentage_trn#retain_percentage_trn#1-torch.mean(trn_node_probability).item()

    trn_node_probability = trn_node_probability-randomicity

    #remove_percentage_trn = retain_percentage_trn

    less_xai_trn_node = torch.argsort(trn_node_probability)[:int(trn_node_probability.shape[0]*remove_percentage_trn)]
    most_xai_trn_node = torch.argsort(trn_node_probability)[int(trn_node_probability.shape[0]*remove_percentage_trn):]
    print(" Less: ", less_xai_trn_node.shape, " Most: ", most_xai_trn_node.shape, " Mean ", torch.mean(trn_node_probability+randomicity).item())
    ###Qui produco un grafo che contiene nodi okay di training e tutti quelli di val e tst

    ##Qui dico quali nodi vanno tolti (n.b. quando fatto su training diventa irrilevante)
    node_mask = torch.bernoulli(1.- node_probability).bool().to('cuda:0')
    ###I most xai trn nodes vanno mantenuti!
    try:
        node_mask[less_xai_trn_node] = False
        node_mask[most_xai_trn_node] = True
    except:
        import pdb; pdb.set_trace()


    try:
        edge_index, _, edge_mask = subgraph(
        node_mask,
        edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        return_edge_mask=True,
       )
    except:
        import pdb; pdb.set_trace()
    print("This are most_xai: ", torch.sort(most_xai_trn_node)[0])
    return edge_index, edge_mask, node_mask


def confidence_dropping(most_confident_nodes, mapping_filtered, confidence_values, num_samples, default_probability=0.3, mapping='gaussian', normalize_score=True, mean_confidence_dropping = 0.5, vary_probability = True, distribution = 'cumulative'):
    if torch.max(confidence_values) != torch.min(confidence_values):

        if mapping_filtered is not None:
            mapping_filtered = mapping_filtered.to('cuda')
        if mapping == 'gaussian':
            #if normalize:
            confidence_values = (confidence_values-torch.min(confidence_values))/(torch.max(confidence_values)-torch.min(confidence_values))
            if torch.is_tensor(confidence_values) == False:
               confidence_values = torch.tensor(confidence_values)
            if distribution == 'cumulative':
               dropping_probability = ecdf(confidence_values)
               dropping_probability = (1-dropping_probability).float()
            elif distribution == 'gaussian':
               dropping_probability = gaussian_mapping(confidence_values, normalize_score)
               dropping_probability = (1-dropping_probability).float()
            else:
               print("Distribution ", distribution, " not known")
               exit(1)
            if vary_probability:
              dropping_probability = dropping_probability -0.5 + (mean_confidence_dropping)

            dropping_probability = torch.clip(dropping_probability, min=0.0, max = 1.0)
        #print("DP: ", dropping_probability, " DP: ", dropping_probability.shape)
        elif mapping == 'cumulative':
            xs, ys, idx = ecdf(confidence_values)
            print("Mapping to be adjusted")
            exit(1)
        else:
            print("Not known mapping")
            exit(1)
        if vary_probability:
            all_probabilities = torch.ones(num_samples)*mean_confidence_dropping
        else:
            all_probabilities = torch.ones(num_samples) * default_probability
        all_probabilities = all_probabilities.to('cuda')
        if mapping_filtered is not None:
           try:
               all_probabilities[mapping_filtered] = dropping_probability
           except:
               import pdb;pdb.set_trace()
        else:
           all_probabilities[most_confident_nodes] = dropping_probability
    else:
        print("AAA: MIN XAI =", torch.max(confidence_values).item(), " == MAX XAI = ", torch.min(confidence_values).item())
        all_probabilities = torch.ones(num_samples)*default_probability
        all_probabilities = all_probabilities.to('cuda:0')
    mean_probabilities = torch.mean(all_probabilities).item()
    std_probabilities = torch.std(all_probabilities).item()
    """
    for node_expl_idx, node_idx in enumerate(mapping_filtered):
                dropping_value = dropping_probability[node_expl_idx]
                all_probabilities[node_idx] = dropping_value
            #except:
            #    import pdb; pdb.set_trace()
    #all_probabilities = torch.tensor(all_probabilities)
    """
    return all_probabilities, mean_probabilities, std_probabilities
    #all_probabilities[most_confident_nodes]

    # Fit a normal distribution to the data:

    """
    mu, std = norm.fit(ys)

    # Plot the histogram.
    plt.hist(ys, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()
    """

