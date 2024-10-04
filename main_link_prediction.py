from dataset.utils_edge import GraphLoader
import torch.nn as nn
import torch.optim as optim
from model.link_prediction.gcn_link_prediction import GCN_link_prediction
from model.link_prediction.utils import train_link_predictor, eval_link_predictor
from graphxai.explainers.grad import GradExplainer
import torch
import numpy as np
from config import Selector


#FROM NESS: Node Embeddings from Static SubGraphs
#,wesplitthegraphintothreeparts:10%testing,5%validation,
# 85%training,usingrandomedge-levelsplits(RES).WethenperformREStogenerateksubgraphs
# fromtrainingset,suchthatsubgraphsdonotshareanyedges.Weusek∈[2,4,8]inourexperiments.

config_path = r"config_cora_gcn_link_pr.json"
config = Selector(config_path).args


paths= {"data": "./data/",
       "results": "./results/"}


def main_function(hidden_channels, dropout, optimizer_name, loss_name, drop_edge_probability, lr, weight_decay, num_epochs,
              dataset_name,
              n_subgraph = 4,
              model_path = "",
              threshold_confidence_value = 0.95,
              confidence_strategy = 'threshold',
              sparsity_value = 0.1,
              default_drop_probability = 0.5,
              device = 'cuda',
              id = 0):
    torch.manual_seed(id)
    ds_loader = GraphLoader(seed=id, n_subgraphs=n_subgraph, paths=paths, device=device, dataset_name=dataset_name)
    train_data = ds_loader.train_data
    val_data = ds_loader.validation_data
    tst_data = ds_loader.test_data
    model = GCN_link_prediction(train_data.x.shape[1], hidden_channels=hidden_channels, dropout_probability = dropout).to(device)
    if optimizer_name == 'adam':
       optimizer = optim.Adam(params=model.parameters(), lr=lr,weight_decay=weight_decay)
    else:
       print(optimizer_name, " is not available as optimizer")
    if loss_name == 'BCEWithLogitsLoss':
       criterion = nn.BCEWithLogitsLoss()
    else:
      print(loss_name, " is not available as loss")


    best_val_auc = 0.0
    explainer = GradExplainer(model=model, decoder=model.decode, criterion=criterion)
    for epoch in range(num_epochs):

       trn_auc, trn_ap, trn_acc, trn_loss = train_link_predictor(model = model,
                                                         train_data = train_data,
                                                         optimizer = optimizer,
                                                         criterion = criterion,
                                                         explainer=explainer,
                                                         drop_edge_probability=drop_edge_probability,
                                                         threshold_confidence_value=threshold_confidence_value,
                                                         confidence_strategy=confidence_strategy,
                                                         default_drop_probability = default_drop_probability,
                                                         sparsity_value = sparsity_value,
                                                         device = device)
       val_auc, val_ap, val_loss, val_acc = eval_link_predictor(model, val_data, criterion)
       if val_auc > best_val_auc:
              best_val_auc = val_auc
              torch.save(model.state_dict(),model_path)

       print(f"{epoch:3d} TRN AUC: {trn_auc:.3f} AP: {trn_ap:.3f} ACC: {trn_acc:.3f} Loss: {trn_loss:.3f}", end=" ")
       print(f" VAL AUC: {val_auc:.3f} AP {val_ap:.3f}  ACC: {val_acc:.3f} Loss: {val_loss:.3f}")

    model.load_state_dict(torch.load(model_path))
    tst_auc, tst_ap, tst_loss, tst_acc = eval_link_predictor(model, tst_data, criterion)

    return tst_auc, tst_ap, tst_loss, tst_acc


history_tst_auc = []
history_tst_ap = []
history_tst_loss = []
history_tst_acc = []
for id in range(5):
    model_path = config.model.dataset_name + "_" + config.model.model_name + "_" + config.model.task +str(id)+".pt"
    tst_auc, tst_ap, tst_loss, tst_acc = main_function(hidden_channels = config.model.hidden_channels,
                                                           dropout = config.model.dropout,
                                                           optimizer_name = config.model.optimizer_name,
                                                           loss_name = config.model.criterion_name,
                                                           drop_edge_probability = config.model.drop_edge_probability,
                                                           lr = config.model.lr,
                                                           weight_decay = config.model.weight_decay,
                                                           num_epochs = config.model.num_epochs,
                                                           sparsity_value = config.model.sparsity_value,
                                                           id = id,
                                                           dataset_name=config.model.dataset_name,
                                                           threshold_confidence_value=config.model.threshold_confidence_value,
                                                           confidence_strategy=config.model.confidence_strategy,
                                                           default_drop_probability = config.model.drop_edge_probability,
                                                           device = config.model.device,
                                                           n_subgraph=config.model.n_subgraphs,
                                                           model_path = model_path)
    history_tst_auc.append(tst_auc)
    history_tst_ap.append(tst_ap)
    history_tst_loss.append(tst_loss)
    history_tst_acc.append(tst_acc)
history_tst_auc = np.array(history_tst_auc)
history_tst_ap = np.array(history_tst_ap)
history_tst_loss = np.array(history_tst_loss)
history_tst_acc = np.array(history_tst_acc)
print(f"Mean AUC: {np.mean(history_tst_auc):.3f} +- {np.std(history_tst_auc):.3f}")
print(f"Mean AP: {np.mean(history_tst_ap):.3f} +- {np.std(history_tst_ap):.3f}")
print(f"Mean LOSS: {np.mean(history_tst_loss):.3f} +- {np.std(history_tst_loss):.3f}")
print(f"Mean ACC: {np.mean(history_tst_acc):.3f} +- {np.std(history_tst_acc):.3f}")

