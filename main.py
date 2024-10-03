import torch

from dataset.utils import Dataset

from model.node_classification.gcn_node_classification import GCN_node_classification

from model.node_classification.utils import train, test

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

import torch.nn as nn

from graphxai.explainers.grad import GradExplainer


drop_strategy = 'random'


task = 'node_cl'
dataset_name = 'Cora'
model_name = 'gcn'
dataset = Dataset(dataset_name = dataset_name, device = 'cuda:0').dataset
num_features = dataset[0].x.shape[1]
num_classes = torch.unique(dataset[0].y).shape[0]

if task == 'node_cl':
  if dataset_name == 'CiteSeer':
    num_epochs = 500
    optimizer_name = 'adam'
    lr = 0.01
    weight_decay = 0.0125
    use_bn = True #False
    hidden_channels = [32]

    scheduler_name = 'ReduceLROnPlateau'
    scheduler_patience = 1000
    scheduler_factor = 0.99
    scheduler_threshold = 0.0001
    dropout = [0.8, 0.9]  # [0.8,0.9]
    dropping_node_probability = 0.0  # 0, 0.25, 0.75]

    confidence_strategy = 'threshold'
    threshold_confidence_value = 0.5
    confidence_criterion = 'Most'
    confidence_sampling = 0.66  # 0.4

    threshold_sparsity_value = 0.01
    sparsity_value = 0.33
    sparsity_strategy = 'percentage'  # otherwise percentage
    seed = 0

    dropping_distribution = 'gaussian'



  elif dataset_name == 'Cora':
    num_epochs = 10#1500
    lr = 0.0175
    weight_decay = 0.00033 #0.0
    dropout = [0.9, 0.8] #ALL OTHER [0.8,0.825] #AUMENTARE DROPOUT SUL PRIMO, PORTA A PEGGIORARE SE WEIGHT_DECAY A 0
    use_bn = True
    scheduler_name = 'ReduceLROnPlateau'
    scheduler_patience = 1000
    scheduler_factor = 0.9
    scheduler_threshold = 0.0001
    confidence_strategy = 'threshold'
    threshold_confidence_value = 0.85
    confidence_criterion = 'Most'
    confidence_sampling = 0.66  # 0.4
    threshold_sparsity_value = 0.01
    sparsity_value = 0.5
    sparsity_strategy = 'percentage'  # otherwise percentage
    hidden_channels = [128]
    seed = 100
    dropping_node_probability = 0.5
    dropping_distribution = 'gaussian'
    optimizer_name = 'adam' #'adamw'


  elif dataset_name == 'PubMed':
    #threshold_sparsity_value = 0.01
    dropping_node_probability = 0.0
    dropping_distribution='gaussian'
    hidden_channels = [32]
    num_epochs = 1000
    lr = 0.04
    weight_decay = 0.00033
    dropout = [0.75,0.8]
    use_bn = True
    scheduler_name = 'ReduceLROnPlateau'
    scheduler_patience = 80
    scheduler_factor = 0.9
    scheduler_threshold = 0.0001
    confidence_strategy='threshold'
    threshold_confidence_value = 0.95
    confidence_criterion='Most'
    confidence_sampling = 0.66#0.4
    threshold_sparsity_value = 0.01
    sparsity_value = 0.5
    sparsity_strategy = 'percentage' #otherwise percentage
    seed = 100
    optimizer_name = 'adam'
elif task == 'link_pr':
    if dataset_name == 'Cora':
        num_epochs = 1000
        lr = 0.01  # 0.005
        weight_decay = 0.0125
        # weight_decay = 0.0075 #0.0005
        use_bn = True  # False
        hidden_channels = [32]  # [32]

        scheduler_name = 'ReduceLROnPlateau'
        scheduler_patience = 1000
        scheduler_factor = 0.99
        scheduler_threshold = 0.0001
        dropout = [0.8, 0.9]  # [0.8,0.9]
        dropping_node_probability = 0.5  # 0, 0.25, 0.75]

        confidence_strategy = 'threshold'
        threshold_confidence_value = 0.5
        confidence_criterion = 'Most'
        confidence_sampling = 0.66  # 0.4

        threshold_sparsity_value = 0.01
        sparsity_value = 0.33
        sparsity_strategy = 'percentage'  # otherwise percentage
        seed = 0

        dropping_distribution = 'gaussian'

architecture_name = 'gcn'#gcn'
dropout_edge_p = 0.5
starting_drop_epoch = 0


later_log_data = {'trn_loss':[],'val_loss':[], 'tst_loss':[], 'trn_accu':[], 'val_accu':[], 'tst_accu':[],
                  'no_edges':[], 'no_nodes':[], 'mean_probability':[], 'std_probability':[],
                  'num_confident_nodes':[], 'mean_dropping':[], 'learning_rate':[],
                  'mean_confidence_nodes':[], 'std_confidence_nodes':[],
                  'all_sufficiency_mean':[], 'all_sufficiency_std':[], 'explanation_sparsity':[], 'node_backprop':[],
                  'mean_confident_nodes_faithfulness_sufficiency':[],'std_confident_nodes_faithfulness_sufficiency':[],
                  'mean_trn_confident_nodes_faithfulness_sufficiency':[], 'std_trn_confident_nodes_faithfulness_sufficiency':[],
                  'patience_val_acc':[]}

def main_function(num_epochs, lr, weight_decay, dropout, use_bn, hidden_channels, scheduler_name, scheduler_patience, scheduler_factor, scheduler_threshold,
                  confidence_strategy, threshold_confidence_value, confidence_criterion, confidence_sampling,
                  threshold_sparsity_value, sparsity_value, sparsity_strategy, seed, optimizer_name, task = 'node_cl', id = 0):
    vary_dropping_probability = False
    torch.manual_seed(seed)

    criterion_name = 'ce'
    if criterion_name == 'ce':
        criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name

    data = dataset[0]
    best_val_acc = 0.0

    data.edge_index = data.edge_index.to('cuda')
    data.y = data.y.to('cuda')
    data.x = data.x.to('cuda')
    data.train_mask = data.train_mask.to('cuda')
    data.val_mask = data.val_mask.to('cuda')
    data.test_mask = data.test_mask.to('cuda')

    if model_name == 'gcn':
        model = GCN_node_classification(input_features = num_features,
                        hidden_channels=hidden_channels,
                        dropout=dropout,
                        num_classes=num_classes,
                        dropout_edge_p=dropout_edge_p,
                        use_bn = use_bn).to('cuda')#dropout_edge_p)
    else:
        exit(1)
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay=weight_decay)
    else:
        print("Optimizer: ", optimizer_name, " not available")
        exit(1)
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=scheduler_factor,
                                      patience=scheduler_patience,
                                      threshold=scheduler_threshold,
                                  threshold_mode='abs')
    else:
        print("Scheduler: ", scheduler_name, " is not known")
        exit(1)

    k_hop_info_list = []

    no_improvement_val_acc = 0

    explainer = GradExplainer(model, criterion)
    for epoch in range(num_epochs):

      trn_res, no_nodes, no_edges, mean_probability, std_probability, num_confident_nodes, mean_dropping, mean_confidence_nodes, std_confidence_nodes, all_sufficiency_mean, all_sufficiency_std, explanation_sparsity, node_backprop, mean_confident_faithfulness_suff, std_confident_faithfulness_suff, mean_trn_confident_faithfulness_suff, std_trn_confident_faithfulness_suff = train(model, data, data.train_mask, criterion, optimizer, num_classes,
                      confidence_sampling = confidence_sampling,
                      dropping_probability=dropping_node_probability,
                      explainer = explainer, k_hop_info_list = k_hop_info_list, epoch = epoch,
                      drop_strategy=drop_strategy, starting_drop_epoch= starting_drop_epoch,
                      threshold_confidence_value=threshold_confidence_value,
                      confidence_strategy=confidence_strategy,
                      sparsity_value = sparsity_value,
                      confidence_criterion=confidence_criterion,
                      sparsity_strategy=sparsity_strategy,
                      sparsity_threshold=threshold_sparsity_value,
                      vary_probability=vary_dropping_probability,
                      dropping_distribution=dropping_distribution)
      val_res = test(model, data, data.val_mask, criterion, num_classes)
      scheduler.step(val_res[0])


      if val_res[1] > best_val_acc:
        best_val_acc = val_res[1]
        torch.save(model.state_dict(),  dataset_name + "_" + architecture_name + "_" + task + "_" + drop_strategy + "_model_"+str(id)+".pt")
        no_improvement_val_acc=0
      else:
          no_improvement_val_acc = no_improvement_val_acc+1

      print(f"Epoch {epoch}  Trn Accuracy: {trn_res[1]:.3f}, Val Accuracy: {val_res[1]:.3f}  "
                f" LOS Trn Accuracy: {trn_res[0]:.3f}, Val Accuracy: {val_res[0]:.3f} ")
    model.load_state_dict(torch.load(dataset_name + "_" + architecture_name + "_" + task + "_" + drop_strategy + "_model_"+str(id)+".pt"))
    best_tst_res = test(model, data, data.test_mask, criterion, num_classes)
    best_tst_acc = best_tst_res[1]

    return best_tst_acc

best_tst_acc = []
for id in range(5):
    seed = id
    current_tst_acc = main_function(num_epochs = num_epochs,
              lr = lr,
              weight_decay = weight_decay,
              dropout = dropout,
              use_bn = use_bn,
              hidden_channels = hidden_channels,
              scheduler_name = scheduler_name,
              scheduler_patience = scheduler_patience,
              scheduler_factor = scheduler_factor,
              scheduler_threshold = scheduler_threshold,
              confidence_strategy = confidence_strategy,
              threshold_confidence_value = threshold_confidence_value,
              confidence_criterion = confidence_criterion,
              confidence_sampling = confidence_sampling,
              threshold_sparsity_value = threshold_sparsity_value,
              sparsity_value = sparsity_value,
              sparsity_strategy = sparsity_strategy,
              seed = seed,
              optimizer_name = optimizer_name,
              task = task,
              id = id)
    print("Experiment ID: ",id, " Test accuracy: ", current_tst_acc)
    best_tst_acc.append(current_tst_acc)

best_tst_acc = np.array(best_tst_acc)
print("Mean accuracy: ", np.mean(best_tst_acc), " ", np.std(best_tst_acc))
