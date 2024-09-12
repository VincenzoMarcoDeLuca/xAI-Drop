import torch

from dataset.utils import Dataset

from model.node_classification.gcn_node_classification import GCN_node_classification

from model.node_classification.utils import train, test

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics.explainability import  measure_kl, whole_explanation, acc_sufficiency

import torch.nn as nn

from graphxai.explainers.grad import GradExplainer

import json

from torch_geometric.explain import CaptumExplainer, Explainer

from torch_geometric.utils import k_hop_subgraph

import time



from datetime import datetime, date



later_log = True



drop_strategy = 'random_edge'


task = 'node_cl'
dataset_name = 'PubMed'
model_name = 'gcn'
dataset = Dataset(dataset_name = dataset_name, device = 'cuda:0').dataset
num_features = dataset[0].x.shape[1]
num_classes = torch.unique(dataset[0].y).shape[0]

print("Num features: ", num_features, " num_classes ", num_classes)

if task == 'node_cl':
  if dataset_name == 'CiteSeer':
    num_epochs = 500
    optimizer_name = 'adam'
    lr = 0.01 # 0.005
    weight_decay = 0.0125
    #weight_decay = 0.0075 #0.0005
    use_bn = True #False
    hidden_channels = [32] #[32]

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
    num_epochs = 1500
    lr = 0.0175#0.01
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
    hidden_channels = [32]
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


later_log_data = {'trn_loss':[],'val_loss':[], 'trn_accu':[], 'val_accu':[],
                  'no_edges':[], 'no_nodes':[], 'mean_probability':[], 'std_probability':[],
                  'num_confident_nodes':[], 'mean_dropping':[], 'learning_rate':[],
                  'mean_confidence_nodes':[], 'std_confidence_nodes':[],
                  'all_sufficiency_mean':[], 'all_sufficiency_std':[], 'explanation_sparsity':[],
                  'trn_trn_loss':[], 'trn_trn_acc':[], 'node_backprop':[],
                  'mean_confident_nodes_faithfulness_sufficiency':[],'std_confident_nodes_faithfulness_sufficiency':[],
                  'mean_trn_confident_nodes_faithfulness_sufficiency':[], 'std_trn_confident_nodes_faithfulness_sufficiency':[],
                  'patience_val_acc':[]}

"""
assess_xai = False
if assess_xai:

  data_list_xai = []
  data_list_ori = []
  data_list_ori_all = []
  data_list_expl_all = []
  for index in range(4):

    file_path = f'file_expl_pred{index}.txt'
    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and add the line to the list
            data_list_xai.append(int(line))

    file_path = f'file_orig_pred_{index}.txt'
    with open(file_path, 'r') as f:
        for line in f:
            # Strip whitespace and add the line to the list
            data_list_ori.append(int(line))

    file_path = f'file_expl_all{index}.txt'
    with open(file_path, 'r') as f:
        for line in f:
            line = line[1:-2]
            f_list = [float(i) for i in line.split(",") if i.strip()]
            data_list_expl_all.append(f_list)

    file_path = f'file_orig_all_{index}.txt'
    with open(file_path, 'r') as f:
        for line in f:
            line = line[1:-2]

            f_list = [float(i) for i in line.split(",") if i.strip()]
            data_list_ori_all.append(f_list)

  data_list_xai = torch.tensor(data_list_xai)
  data_list_ori = torch.tensor(data_list_ori)
  data_list_expl_all = torch.tensor(data_list_expl_all)
  data_list_ori_all = torch.tensor(data_list_ori_all)
  softmax_layer = torch.nn.Softmax(dim=-1)
  data_list_expl_all = softmax_layer(data_list_expl_all)
  data_list_ori_all = softmax_layer(data_list_ori_all)
  correct = (data_list_xai==data_list_ori).int().sum()
  tot = data_list_xai.shape[0]
  accuracy = correct/tot
  data_list_ori_one_hot = torch.nn.functional.one_hot(data_list_ori, num_classes = num_classes)
  kl_distr = measure_kl(data_list_ori_all, data_list_expl_all)
  kl_gt = measure_kl(data_list_expl_all, data_list_ori_one_hot)
  print(f"Accuracy: {accuracy:.3f} KL D: {torch.mean(kl_distr).item():.3f} KL GT: {torch.mean(kl_gt).item():.3f}")
  exit(1)
"""
just_load = True



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

    another_best_val_acc = 0.0
    another_best_val_loss = float('inf')
    another_best_epoch = -1

    best_val_acc = 0.0
    best_val_loss = 0.0
    best_tst_loss = 0.0
    best_epoch = -1

    #print("Questo e' id")
    #torch.cuda.manual_seed(id)
    data.edge_index = data.edge_index.to('cuda')
    data.y = data.y.to('cuda')
    data.x = data.x.to('cuda')
    data.train_mask = data.train_mask.to('cuda')
    data.val_mask = data.val_mask.to('cuda')
    data.test_mask = data.test_mask.to('cuda')
    #all_node_idx = torch.tensor([i for i in range(data.x.shape[0])]).int().to('cuda:0')

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


    if just_load:
      for id in range(0,5):
        model.load_state_dict(torch.load(dataset_name + "_" + architecture_name + "_" + task + "_" + drop_strategy + "_model_"+str(id)+".pt"))
        model = model.to('cpu')
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
        data.train_mask = data.train_mask.to('cpu')
        data.val_mask = data.val_mask.to('cpu')
        data.x = data.x.to('cpu')
        data.edge_index = data.edge_index.to('cpu')
        data.y = data.y.to('cpu')
        print("Tst_res: ", tst_res)
        sparsity_expl = 0.25
        xai_file_name = "XAI_"+dataset_name + "_" + architecture_name + "_" + task + "_" + drop_strategy + "_model_"+str(id)
        only_assess=False
        if only_assess == False:
            whole_explanation(data.x, data.edge_index, model, explainer,num_classes,
                          num_hops=2, sparsity_expl=sparsity_expl, batch_size=250, file_name=xai_file_name, all=500)
        acc_sufficiency(xai_file_name)
      exit(1)
    k_hop_info_list = []

    no_improvement_val_acc = 0

    start = time.time()

    explainer = GradExplainer(model, criterion)
    for epoch in range(num_epochs):
      print(f"Current bad epoch: {scheduler.num_bad_epochs} Current lr: {optimizer.param_groups[0]['lr']} patience:{scheduler.patience}")

      trn_res, no_nodes, no_edges, mean_probability, std_probability, num_confident_nodes, mean_dropping, mean_confidence_nodes, std_confidence_nodes, all_sufficiency_mean, all_sufficiency_std, explanation_sparsity, trn_trn_res, node_backprop, mean_confident_faithfulness_suff, std_confident_faithfulness_suff, mean_trn_confident_faithfulness_suff, std_trn_confident_faithfulness_suff = train(model, data, data.train_mask, criterion, optimizer, num_classes,
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


      """
      node_imp, explanation = get_explanation(explainer, all_node_idx, data.x, data.edge_index)
      node_imp = 1 - normalize(node_imp)
      sufficiency(model, all_node_idx, data.x, data.edge_index, node_imp, 
                       explanation, threshold=threshold, num_hops=2,sparsity_value=sparsity_value)
      """

      if val_res[1] > best_val_acc:
        best_epoch = epoch
        best_val_loss = val_res[0]
        best_val_acc = val_res[1]
        best_tst_loss = tst_res[0]
        best_tst_acc = tst_res[1]
        torch.save(model.state_dict(),
        dataset_name + "_" + architecture_name + "_" + task + "_" + drop_strategy + "_model_"+str(id)+".pt")

        print("BEST VAL ACC: ", tst_res[1])

        no_improvement_val_acc=0
      else:
          no_improvement_val_acc = no_improvement_val_acc+1


      if val_res[0] < another_best_val_loss:
        another_best_epoch = epoch
        another_best_val_loss = val_res[0]
        another_best_val_acc = val_res[1]
        another_best_tst_loss = tst_res[0]
        another_best_tst_acc = tst_res[1]
        print("BEST ANOTHER VAL: ", tst_res[1])


      if later_log:
          later_log_data['trn_trn_loss'].append(trn_trn_res[0])
          later_log_data['trn_trn_acc'].append(trn_trn_res[1])
          later_log_data['trn_loss'].append(trn_res[0])
          later_log_data['val_loss'].append(val_res[0])
          later_log_data['trn_accu'].append(trn_res[1])
          later_log_data['val_accu'].append(val_res[1])
          later_log_data['no_nodes'].append(no_nodes)
          later_log_data['no_edges'].append(no_edges)
          later_log_data['mean_probability'].append(mean_probability)
          later_log_data['std_probability'].append(std_probability)
          later_log_data['num_confident_nodes'].append(num_confident_nodes)
          later_log_data['mean_dropping'].append(mean_dropping)
          later_log_data['learning_rate'].append(optimizer.param_groups[0]['lr'])
          later_log_data['mean_confidence_nodes'].append(mean_confidence_nodes)
          later_log_data['std_confidence_nodes'].append(std_confidence_nodes)
          later_log_data['all_sufficiency_mean'].append(all_sufficiency_mean)
          later_log_data['all_sufficiency_std'].append(all_sufficiency_std)
          later_log_data['explanation_sparsity'].append(explanation_sparsity)
          later_log_data['node_backprop'].append(node_backprop)
          later_log_data['mean_confident_nodes_faithfulness_sufficiency'].append(mean_confident_faithfulness_suff)
          later_log_data['std_confident_nodes_faithfulness_sufficiency'].append(std_confident_faithfulness_suff)

          later_log_data['mean_confident_nodes_faithfulness_sufficiency'].append(mean_confident_faithfulness_suff)
          later_log_data['std_confident_nodes_faithfulness_sufficiency'].append(std_confident_faithfulness_suff)

          later_log_data['mean_trn_confident_nodes_faithfulness_sufficiency'].append(mean_trn_confident_faithfulness_suff)
          later_log_data['std_trn_confident_nodes_faithfulness_sufficiency'].append(std_trn_confident_faithfulness_suff)

          later_log_data['patience_val_acc'].append(no_improvement_val_acc)


          try:
              with open('file_new.json', 'w') as file_dict:

                 file_dict.write(json.dumps(later_log_data))  # use `json.loads` to do the reverse
          except:
              print("\n\n\n\n\n\n\n\nERRORE INSPIEGABILE\n\n\n\n\n")

      if epoch%1==0:
          print(f"{epoch} ACC Trn: {trn_res[1]:.3f}, Val: {val_res[1]:.3f} Trn_trn:{trn_trn_res[1]:.3f} "
                f" LOS Trn: {trn_res[0]:.3f}, Val: {val_res[0]:.3f} Trn_trn:{trn_trn_res[0]:.3f}"
                f" DR PR mean {mean_probability:.3f} std {std_probability:.3f}"
                f" CONF mean {mean_confidence_nodes:.3f} std {std_confidence_nodes:.3f}"
                f" Cnodes {num_confident_nodes} lr: {optimizer.param_groups[0]['lr']}"
                f" nod {no_nodes} edg {no_edges} "
                f"XAI m:{all_sufficiency_mean:.5f} std:{all_sufficiency_std:.5f} sparsity:{explanation_sparsity:.3f}"
                f"node_backprop:{node_backprop} "
                f"Conf_suf:{mean_confident_faithfulness_suff:.5f} std:{std_confident_faithfulness_suff:.5f}"
                f"TRN Conf_suf:{mean_trn_confident_faithfulness_suff:.5f} std:{std_trn_confident_faithfulness_suff:.5f}"
                f"Patience:{no_improvement_val_acc}")
    end = time.time()
    training_time = end-start
    print("\n\n\n",id, " dropout: ", dropout_edge_p, " \nVAL ACC EARLY EPOCH: " , best_epoch,
          " TST: ", best_tst_loss, " ", best_tst_acc, " VAL: ", best_val_loss, " ",best_val_acc)
    print("\nVAL LOS EARLY EPOCH: ", another_best_epoch,
          " TST: ", another_best_tst_loss, " ", another_best_tst_acc, " VAL: ", another_best_val_loss, " ",another_best_val_acc)
    #print("\nTST ACC EARLY EPOCH: ", best_tst_epoch,
    #      " TST: ", best_tst_tst_loss, " ", best_tst_tst_acc, " VAL: ", best_tst_val_loss, " ", best_tst_val_acc)



print("CIAO")
for id in range(5):
    seed = id+100
    print("Sono in main_function")
    main_function(num_epochs = num_epochs,
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
