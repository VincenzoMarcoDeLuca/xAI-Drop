import torch
from dataset.utils import Dataset
from model.node_classification.gcn_node_classification import GCN_node_classification
from model.node_classification.gat_node_classification import GAT_node_classification
from model.node_classification.utils import train, test
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn as nn
from graphxai.explainers.grad import GradExplainer
from config import Selector


config_path = r"config_cora_gat_node_cl.json"
config = Selector(config_path).args

def main_function(num_epochs, lr, weight_decay, dropout, use_bn, hidden_channels, scheduler_name, scheduler_patience, scheduler_factor, scheduler_threshold,
                  confidence_strategy, threshold_confidence_value, confidence_sampling,
                  threshold_sparsity_value, sparsity_value, sparsity_strategy, seed, data, dropping_distribution = 'gaussian', optimizer_name='adam', criterion_name='ce', task = 'node_cl',
                  save_model_path = "", dropping_node_probability = 0.5, model_name='gcn', dropout_edge_probability=0.0, num_heads = 1,device='cuda'):
    vary_dropping_probability = False
    torch.manual_seed(seed)
    best_val_acc = 0.0

    if model_name == 'gcn':
        model = GCN_node_classification(input_features = num_features,
                        hidden_channels=hidden_channels,
                        dropout=dropout,
                        num_classes=num_classes,
                        dropout_edge_p=dropout_edge_probability,
                        use_bn = use_bn).to(device)
    elif model_name == 'gat':
        model = GAT_node_classification(input_features = num_features,
                        hidden_channels=hidden_channels,
                        num_heads = num_heads,
                        dropout=dropout,
                        num_classes=num_classes,
                        dropout_edge_p=dropout_edge_probability,
                        use_bn = use_bn).to(device)
    else:
        exit(1)
    if criterion_name == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Loss criterion: ", criterion_name, " not known")
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


    no_improvement_val_acc = 0

    explainer = GradExplainer(model, criterion)
    for epoch in range(num_epochs):

      trn_res = train(model, data, data.train_mask, criterion, optimizer, num_classes,
                      confidence_sampling = confidence_sampling,
                      dropping_probability=dropping_node_probability,
                      explainer = explainer,
                      threshold_confidence_value=threshold_confidence_value,
                      confidence_strategy=confidence_strategy,
                      sparsity_value = sparsity_value,
                      sparsity_strategy=sparsity_strategy,
                      sparsity_threshold=threshold_sparsity_value,
                      vary_probability=vary_dropping_probability,
                      dropping_distribution=dropping_distribution)
      val_res = test(model, data, data.val_mask, criterion, num_classes)
      scheduler.step(val_res[0])


      if val_res[1] > best_val_acc:
        best_val_acc = val_res[1]
        torch.save(model.state_dict(),  save_model_path)
        no_improvement_val_acc=0
      else:
          no_improvement_val_acc = no_improvement_val_acc+1

      print(f"Epoch {epoch}  Trn Accuracy: {trn_res[1]:.3f}, Val Accuracy: {val_res[1]:.3f} "
                f" Trn Loss: {trn_res[0]:.3f}, Val Loss: {val_res[0]:.3f}")
    model.load_state_dict(torch.load(save_model_path))
    best_tst_res = test(model, data, data.test_mask, criterion, num_classes)
    best_tst_acc = best_tst_res[1]

    return best_tst_acc
dataset = Dataset(dataset_name = config.model.dataset_name, device = config.model.device).dataset
num_features = dataset[0].x.shape[1]
num_classes = torch.unique(dataset[0].y).shape[0]
data = dataset[0]
data.edge_index = data.edge_index.to(config.model.device)
data.y = data.y.to(config.model.device)
data.x = data.x.to(config.model.device)
data.train_mask = data.train_mask.to(config.model.device)
data.val_mask = data.val_mask.to(config.model.device)
data.test_mask = data.test_mask.to(config.model.device)

best_tst_acc = []
for id in range(5):
    seed = id
    current_tst_acc = main_function(num_epochs = config.model.num_epochs,
              lr = config.model.lr,
              weight_decay = config.model.weight_decay,
              dropout = config.model.dropout,
              data = data,
              use_bn = config.model.use_bn,
              hidden_channels = config.model.hidden_channels,
              scheduler_name = config.model.scheduler_name,
              scheduler_patience = config.model.scheduler_patience,
              scheduler_factor = config.model.scheduler_factor,
              scheduler_threshold = config.model.scheduler_threshold,
              confidence_strategy = config.model.confidence_strategy,
              threshold_confidence_value = config.model.threshold_confidence_value,
              confidence_sampling = config.model.confidence_sampling,
              threshold_sparsity_value = config.model.threshold_sparsity_value,
              sparsity_value = config.model.sparsity_value,
              sparsity_strategy = config.model.sparsity_strategy,
              seed = config.model.seed+id,
              optimizer_name = config.model.optimizer_name,
              criterion_name=config.model.criterion_name,
              task = config.model.task,
              device = config.model.device,
              dropout_edge_probability = config.model.dropout_edge_probability,
              dropping_node_probability = config.model.dropping_node_probability,
              model_name=config.model.model_name,
              dropping_distribution = config.model.dropping_distribution,
              num_heads = config.model.num_heads,
              save_model_path=config.model.dataset_name + "_" + config.model.model_name + "_" + config.model.task + "_model_"+str(id)+".pt",
              )
    print("Experiment ID: ",id, " Test accuracy: ", current_tst_acc)
    best_tst_acc.append(current_tst_acc)

best_tst_acc = np.array(best_tst_acc)
print("Mean accuracy: ", np.mean(best_tst_acc), " ", np.std(best_tst_acc))