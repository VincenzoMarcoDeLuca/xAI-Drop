import torch
from dataset.utils_node import Dataset
from model.node_classification import GCN_node_classification, GAT_node_classification, GIN_node_classification, train, test
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn as nn
from graphxai.explainers.grad import GradExplainer
from utils.config import Selector

config_path = r"config_cora_gin_node_cl.json"
config = Selector(config_path).args

def main_function(config, save_model_path = ""):
    torch.manual_seed(config.model.seed)
    best_val_acc = 0.0

    if config.model.model_name == 'gcn':
        model = GCN_node_classification(input_features = num_features,
                        hidden_channels=config.model.hidden_channels,
                        dropout=config.model.dropout,
                        num_classes=num_classes,
                        dropout_edge_p=config.model.dropout_edge_probability,
                        use_bn = config.model.use_bn).to(config.model.device)
    elif config.model.model_name == 'gat':
        model = GAT_node_classification(input_features = num_features,
                        hidden_channels=config.model.hidden_channels,
                        num_heads = config.model.num_heads,
                        dropout=config.model.dropout,
                        num_classes=num_classes,
                        dropout_edge_p=config.model.dropout_edge_probability,
                        use_bn = config.model.use_bn).to(config.model.device)
    elif config.model.model_name == 'gin':
        model = GIN_node_classification(input_features = num_features,
                        hidden_channels=config.model.hidden_channels,
                        dropout=config.model.dropout,
                        num_classes=num_classes,
                        dropout_edge_p=config.model.dropout_edge_probability,
                        use_bn = config.model.use_bn).to(config.model.device)
    else:
        print("Model ", config.model.model_name, " is not available")
        exit(1)
    if config.model.criterion_name == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Loss criterion: ", config.model.criterion_name, " not known")
        exit(1)
    if config.model.optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.model.lr, weight_decay=config.model.weight_decay)
    elif config.model.optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr = config.model.lr, weight_decay=config.model.weight_decay)
    else:
        print("Optimizer: ", config.model.optimizer_name, " not available")
        exit(1)
    if config.model.scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=config.model.scheduler_factor,
                                      patience=config.model.scheduler_patience,
                                      threshold=config.model.scheduler_threshold,
                                      threshold_mode='abs')
    else:
        print("Scheduler: ", config.model.scheduler_name, " is not known")
        exit(1)


    no_improvement_val_acc = 0

    explainer = GradExplainer(model, criterion)
    for epoch in range(config.model.num_epochs):

      trn_res = train(model, data, data.train_mask, criterion, optimizer, num_classes,
                      dropping_probability=config.model.dropping_node_probability,
                      explainer = explainer,
                      threshold_confidence_value=config.model.threshold_confidence_value,
                      sparsity_value = config.model.sparsity_value,
                      device = config.model.device)
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
    save_model_path = config.model.dataset_name + "_" + config.model.model_name + "_" + config.model.task + "_model_"+str(id)+".pt"
    current_tst_acc = main_function(config,save_model_path=save_model_path)
    print("Experiment ID: ",id, " Test accuracy: ", current_tst_acc)
    best_tst_acc.append(current_tst_acc)

best_tst_acc = np.array(best_tst_acc)
print("Mean accuracy: ", np.mean(best_tst_acc), " +- ", np.std(best_tst_acc))
