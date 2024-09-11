from dataset.utils import Dataset
from dataset.utils_edge import GraphLoader
import torch.nn as nn
import torch.optim as optim
from model.link_prediction.gcn_link_prediction import GCN_link_prediction
from model.link_prediction.utils import train_link_predictor, eval_link_predictor
from graphxai.explainers.grad import GradExplainer
import wandb
from datetime import datetime, date
import torch

import time



wandb_log = True
task = 'link_pr'
dataset_name = 'Citeseer'
architecture_name = 'gcn'
drop_strategy = 'fair_drop'

#FROM NESS: Node Embeddings from Static SubGraphs
#,wesplitthegraphintothreeparts:10%testing,5%validation,
# 85%training,usingrandomedge-levelsplits(RES).WethenperformREStogenerateksubgraphs
# fromtrainingset,suchthatsubgraphsdonotshareanyedges.Weusek∈[2,4,8]inourexperiments.



if dataset_name == 'Cora':
       if architecture_name == 'gcn' and drop_strategy  == 'baseline':
           threshold_confidence_value = 0.95
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.0, 0.0]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.000
           num_epochs = 1000
           sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_node':
              threshold_confidence_value = 0.95
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.0, 0.0] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_edge':
              threshold_confidence_value = 0.8
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.0, 0.0] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and  drop_strategy == 'xai':
           threshold_confidence_value = 0.7
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.5, 0.5]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.000
           num_epochs = 1000
           sparsity_value = 0.1
       else:
           exit(1)
elif dataset_name == 'Citeseer':
       if architecture_name == 'gcn' and drop_strategy  == 'baseline':
           threshold_confidence_value = 0.95
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.6, 0.6]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.000
           num_epochs = 1000
           sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_node':
              threshold_confidence_value = 0.95
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.6, 0.6] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_edge':
              threshold_confidence_value = 0.8
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.9, 0.25] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and  drop_strategy == 'xai':
           threshold_confidence_value = 0.9
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.9, 0.25]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.0
           num_epochs = 1000
           sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'fair_drop':
           threshold_confidence_value = 0.9
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.0, 0.0]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.0
           num_epochs = 1000
           sparsity_value = 0.1
       else:
           exit(1)

elif dataset_name == 'PubMed':
       if architecture_name == 'gcn' and drop_strategy  == 'baseline':
           threshold_confidence_value = 0.95
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.6, 0.6]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.000
           num_epochs = 1000
           sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_node':
              threshold_confidence_value = 0.95
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.6, 0.6] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and drop_strategy == 'random_edge':
              threshold_confidence_value = 0.8
              confidence_strategy = 'threshold'
              #seed = 100
              #torch.manual_seed(seed)
              hidden_channels = [128, 128]
              dropout = [0.9, 0.25] #0.7,0.7
              #drop_strategy = 'random_node'
              optimizer_name = 'adam'
              loss_name = 'BCEWithLogitsLoss'
              drop_edge_probability = 0.5
              lr = 0.005
              weight_decay=0.000
              num_epochs = 1000
              sparsity_value = 0.1
       elif architecture_name == 'gcn' and  drop_strategy == 'xai':
           threshold_confidence_value = 0.9
           confidence_strategy = 'threshold'
           # seed = 100
           # torch.manual_seed(seed)
           hidden_channels = [128, 128]
           dropout = [0.9, 0.25]
           # drop_strategy = 'random_node'
           optimizer_name = 'adam'
           loss_name = 'BCEWithLogitsLoss'
           drop_edge_probability = 0.5
           lr = 0.005
           weight_decay = 0.0
           num_epochs = 1000
           sparsity_value = 0.1
       else:
           exit(1)

n_subgraphs = 4
paths= {"data": "./data/",
       "results": "./results/"}
device = "cuda:0"


def main_function(hidden_channels, dropout, drop_strategy, optimizer_name, loss_name, drop_edge_probability, lr, weight_decay, num_epochs,
              threshold_confidence_value = 0.95,
              confidence_strategy = 'threshold',
              sparsity_value = sparsity_value,
              id = 0):
    torch.manual_seed(id)
    ds_loader = GraphLoader(seed=id, n_subgraphs=n_subgraphs, paths=paths, device=device, dataset_name=dataset_name)
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
    best_tst_auc = 0.0
    best_val_ap = 0.0
    best_tst_ap = 0.0
    best_epoch = -1

    later_log_data = {'trn_ap':[],'trn_auc':[],'trn_loss':[],'trn_acc':[],
                      'trn_eval_ap':[],'trn_eval_auc':[], 'trn_eval_loss':[],'trn_eval_acc':[],
                      'val_ap':[],'val_auc':[],'val_loss':[],'val_acc':[],
                      'tst_ap':[],'tst_auc':[],'tst_loss':[],'tst_acc':[],
                      'num_most_confident':[], 'mean_drop_prob':[], 'std_drop_prob':[],
                      'mean_confidence':[], 'std_confidence':[]}

    explainer = GradExplainer(model=model, decoder=model.decode, criterion=criterion)
    start = time.time()
    for epoch in range(num_epochs):

       print("TRAIN. ", threshold_confidence_value)
       trn_auc, trn_ap, trn_acc, trn_loss,  num_most_confident, mean_drop_prob, std_drop_prob, mean_confidence, std_confidence = train_link_predictor(model,
                                                         train_data,
                                                         optimizer,
                                                         criterion,
                                                         explainer=explainer,
                                                         drop_edge_strategy = drop_strategy,
                                                         drop_edge_probability=drop_edge_probability,
                                                         threshold_confidence_value=threshold_confidence_value,
                                                         confidence_strategy=confidence_strategy,
                                                         sparsity_value = sparsity_value)
       #print("EVAL TRN")
       trn_eval_auc,trn_eval_ap, trn_eval_loss, trn_eval_acc = eval_link_predictor(model,train_data, criterion)
       val_auc, val_ap, val_loss, val_acc = eval_link_predictor(model, val_data, criterion)
       tst_auc, tst_ap, tst_loss, tst_acc = eval_link_predictor(model, tst_data, criterion)
       if val_auc > best_val_auc:
           if drop_strategy!='fair_drop' or epoch > 10:
              print(f"BEST RESULTS AT {epoch}, VAL auc: {val_auc:.3f} acc: {val_ap:.3f} TST auc: {tst_auc} acc: {tst_ap}")
              best_val_ap = val_ap
              best_tst_ap = tst_ap
              best_val_acc = val_acc
              best_tst_acc = tst_acc
              best_val_auc = val_auc
              best_tst_auc = tst_auc
              best_epoch = epoch
              if drop_strategy is None:
                  drop_name = 'baseline'
              else:
                  drop_name = drop_strategy
              torch.save(model.state_dict(),dataset_name + "_" + architecture_name + "_" + task + "_" + drop_name + "_model"+str(id)+".pt")

       print(f"{epoch:3d} TRN AUC: {trn_auc:.3f} AP: {trn_ap:.3f} ACC: {trn_acc:.3f} Loss: {trn_loss:.3f}", end=" ")
       print(f"TRN EVAL AUC: {trn_eval_auc:.3f} AP: {trn_eval_ap:.3f} ACC: {trn_eval_acc:.3f} Loss: {trn_eval_loss:.3f}", end=" ")
       print(f" VAL AUC: {val_auc:.3f} AP {val_ap:.3f}  ACC: {val_acc:.3f} Loss: {val_loss:.3f}", end= " ")
       print(f" TST AUC: {tst_auc:.3f} AP: {tst_ap:.3f} ACC: {tst_acc:.3f} Loss: {tst_loss:.3f} NUM_CONF: {num_most_confident}",end=" ")
       print(f" CONF Mean: {mean_confidence:.3f} Std: {std_confidence:.4f}")
       later_log_data['trn_ap'].append(trn_ap)
       later_log_data['trn_auc'].append(trn_auc)
       later_log_data['trn_loss'].append(trn_loss)
       later_log_data['trn_acc'].append(trn_acc)
       later_log_data['trn_eval_ap'].append(trn_eval_ap)
       later_log_data['trn_eval_auc'].append(trn_eval_auc)
       later_log_data['trn_eval_loss'].append(trn_eval_loss)
       later_log_data['trn_eval_acc'].append(trn_eval_acc)
       later_log_data['val_ap'].append(val_ap)
       later_log_data['val_auc'].append(val_auc)
       later_log_data['val_loss'].append(val_loss)
       later_log_data['val_acc'].append(val_acc)
       later_log_data['tst_ap'].append(tst_ap)
       later_log_data['tst_auc'].append(tst_auc)
       later_log_data['tst_loss'].append(tst_loss)
       later_log_data['tst_acc'].append(tst_acc)
       later_log_data['num_most_confident'].append(num_most_confident)
       later_log_data['mean_drop_prob'].append(mean_drop_prob)
       later_log_data['std_drop_prob'].append(std_drop_prob)
       later_log_data['mean_confidence'].append(mean_confidence)
       later_log_data['std_confidence'].append(std_confidence)

    print(f"BEST RESULTS AT {best_epoch}, VAL auc: {best_val_auc:.3f} ap: {best_val_ap:.3f} TST auc: {best_tst_auc:.3f} ap: {best_tst_ap:.3f} acc: {best_tst_acc:.3f}")
    end = time.time()
    execution_time = end-start

    return later_log_data, best_epoch, best_tst_ap, best_tst_auc,best_tst_acc, execution_time

def wandb_log_visualization(later_log_data, best_epoch, best_tst_ap, best_tst_auc, best_tst_acc, time):
              today = date.today()
              d1 = today.strftime("%d/%m/%Y")
              now = datetime.now()
              current_time = now.strftime("%H:%M:%S")
              wandb.login(key="1d3f6720d42e8e526ab2b4a6ee29929f34fd2870")
              run = wandb.init(project="XAI-DROP" + str(dataset_name) + str(architecture_name)+str(task),
                               name="PROVA_XAI-DROP" + str(dataset_name) + "_" + str(architecture_name) + " " + str(
                                      id) + str(drop_strategy) + d1 + current_time,
                               config={
                                      'drop_edge_pr': drop_edge_probability,
                                      'drop_strategy': drop_strategy,
                                      #'drop_node_pr': dropping_node_probability,
                                      'hidden_channels': hidden_channels,
                                      'dropout': dropout,
                                      'architecture_name': architecture_name,
                                      'lr': lr,
                                      'weight_decay': weight_decay,
                                      'loss_name': loss_name,
                                      'optimizer_name': optimizer_name,
                                      'num_epochs': num_epochs,
                                      'early_stopping_metric': 'val_accuracy',
                                      'threshold_confidence_value': threshold_confidence_value,
                                       'confidence_strategy': confidence_strategy
              })
              wandb.log({'BEST_TST_AP-VAL_AUC-BASED': best_tst_ap}, step=0)
              wandb.log({'BEST_TST_ACC-VAL_AUC-BASED': best_tst_acc}, step=0)

              wandb.log({'BEST_TST_AUC-VAL_AUC-BASED': best_tst_auc}, step = 0)
              wandb.log({'BEST_VAL_AUC-EPOCH': best_epoch}, step=0)
              wandb.log({'TIME_IN_SECONDS': time}, step =0)
              for i in range(num_epochs):
                     for k in later_log_data.keys():
                            wandb.log({k: later_log_data[k][i]}, step=i)
              wandb.finish()

for id in range(1,5):
    later_log_data, best_epoch, best_tst_ap, best_tst_auc, best_tst_acc, execution_time = main_function(hidden_channels = hidden_channels,
                                                           dropout = dropout,
                                                           drop_strategy = drop_strategy,
                                                           optimizer_name = optimizer_name,
                                                           loss_name = loss_name,
                                                           drop_edge_probability = drop_edge_probability,
                                                           lr = lr,
                                                           weight_decay = weight_decay,
                                                           num_epochs = num_epochs,
                                                           sparsity_value = sparsity_value,
                                                           id = id,
                                                           threshold_confidence_value=threshold_confidence_value,
                                                           confidence_strategy=confidence_strategy)

    if wandb_log:
       wandb_log_visualization(later_log_data, best_epoch, best_tst_ap, best_tst_auc, best_tst_acc, execution_time)