from dataset.utils_edge import GraphLoader
import torch.nn as nn
import torch.optim as optim
from model.link_prediction import GCN_link_prediction, GAT_link_prediction,GIN_link_prediction, train_link_predictor, eval_link_predictor
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

def main_function(config, model_path,seed = 0):
    paths = {"data": "./data/",
             "results": "./results/"}
    torch.manual_seed(seed)
    ds_loader = GraphLoader(seed=seed, paths=paths, device=config.model.device, dataset_name=config.model.dataset_name)
    train_data = ds_loader.train_data
    val_data = ds_loader.validation_data
    tst_data = ds_loader.test_data
    if config.model.model_name == 'gcn':
        model = GCN_link_prediction(train_data.x.shape[1],
                                    hidden_channels=config.model.hidden_channels,
                                    dropout_probability = config.model.dropout).to(config.model.device)

    elif config.model.model_name == 'gat':
        model = GAT_link_prediction(train_data.x.shape[1],
                                    hidden_channels=config.model.hidden_channels,
                                    dropout_probability = config.model.dropout).to(config.model.device)
    elif config.model.model_name == 'gin':
        model = GIN_link_prediction(train_data.x.shape[1],
                                    hidden_channels=config.model.hidden_channels,
                                    dropout_probability = config.model.dropout).to(config.model.device)

    if config.model.optimizer_name == 'adam':
       optimizer = optim.Adam(params=model.parameters(), lr=config.model.lr,weight_decay=config.model.weight_decay)
    else:
       print(config.model.optimizer_name, " is not available as optimizer")
    if config.model.criterion_name == 'BCEWithLogitsLoss':
       criterion = nn.BCEWithLogitsLoss()
    else:
      print(config.model.criterion_name, " is not available as loss")


    best_val_auc = 0.0
    explainer = GradExplainer(model=model, decoder=model.decode, criterion=criterion)
    for epoch in range(config.model.num_epochs):

       trn_auc, trn_ap, trn_acc, trn_loss = train_link_predictor(model = model,
                                                         train_data = train_data,
                                                         optimizer = optimizer,
                                                         criterion = criterion,
                                                         explainer=explainer,
                                                         drop_edge_probability=config.model.drop_edge_probability,
                                                         threshold_confidence_value=config.model.threshold_confidence_value,
                                                         sparsity_value = config.model.sparsity_value,
                                                         device = config.model.device)
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
    seed = config.model.seed + id
    model_path = config.model.dataset_name + "_" + config.model.model_name + "_" + config.model.task +str(id)+".pt"
    tst_auc, tst_ap, tst_loss, tst_acc = main_function(config = config, model_path = model_path, seed = seed)


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

