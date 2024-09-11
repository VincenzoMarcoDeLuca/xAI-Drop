from model.node_classification.engage import Encoder, projection_MLP,prediction_MLP, ModelSIM
from model.node_classification.utils import train, test
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import torch
import faiss


dataset_name = 'Cora'
from torch_geometric.datasets import Planetoid
data_dir = "./data"
import os
import numpy as np
from sklearn.preprocessing import normalize

def finetune(model, x, edge_index, y, mask, criterion, optimizer):
    model.linear.train()
    out = model.predict_class(x, edge_index)
    loss = criterion(out[mask], y[mask])
    loss.backward()
    optimizer.step()

    cl = torch.argmax(out[mask], dim=-1)
    correct = torch.sum((cl==y[mask]).int())
    acc = correct/y[mask].shape[0]
    return loss.item(), acc.item()


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def filter_adj(row, col, mask):
    return row[mask], col[mask]


def dropout_edg(edge_index,  p=0.5):


    row, col, vote = edge_index


    mask = torch.gt(vote,p).to(torch.bool)

    row, col = filter_adj(row, col, mask)


    edge_index = torch.stack([row, col], dim=0)

    return edge_index.long()


def dropout_edge_guided(edge_index, vote = None, edge_p=0.2, lambda_edge= -1):# not working
    """
    generate two positive pairs
    """
    row, col = edge_index
    if vote is None:
        vote = torch.zeros(edge_index.shape[1]).to(edge_index.device)
        new_vote = vote + edge_p
        mask1 = torch.bernoulli(new_vote).to(torch.bool)
        mask2 = torch.bernoulli(new_vote).to(torch.bool)
        row1, col1 = filter_adj(row, col, mask1)
        edge_index1 = torch.stack([row1, col1], dim=0)
        row2, col2 = filter_adj(row, col, mask2)
        edge_index2 = torch.stack([row2, col2], dim=0)
    else:
        vote = vote
        vote = torch.clamp(vote, min = 0, max = 1)
        vote_threshold = vote.mean()+ lambda_edge*vote.std()
        vote[vote>vote_threshold]= 1
        mask1 = torch.bernoulli(vote).to(torch.bool)
        mask2 = ~mask1
        mask2[vote>vote_threshold] = True
        row1, col1 = filter_adj(row, col, mask1)
        edge_index1 = torch.stack([row1, col1], dim=0)
        row2, col2 = filter_adj(row, col, mask2)
        edge_index2 = torch.stack([row2, col2], dim=0)

    return edge_index1.long(), edge_index2.long()

def drop_feature_guided(x, nodevote, node_p=0.2, lambda_node=-2):


    if  nodevote is None:
        drop_mask1 = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
        drop_mask2 = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
        x1 = x.clone()
        x2 = x.clone()
        x1[:, drop_mask1] = 0
        x2[:, drop_mask2] = 0
    else:
        new_vote = nodevote
        vote_threshold =new_vote.mean()+ lambda_node*new_vote.std()
        clamp_mask = torch.clamp(new_vote, min = 0, max = 1)
        clamp_mask[new_vote> vote_threshold] = 1

        vote_matr= clamp_mask.repeat(x.shape[1], 1)
        mask1 = torch.bernoulli(vote_matr).to(torch.bool)
        mask2 = ~mask1
        mask2[:,new_vote > vote_threshold] = True


        x1 = x*mask1.T
        x2 = x*mask2.T

    return x1, x2

def train(model, optimizer, x, edge_index,  vote, nodevote):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_index_2 = dropout_edge_guided(edge_index, vote, edge_p = edge_p, lambda_edge= lambda_edge)
    x_1, x_2 = drop_feature_guided(x, nodevote, node_p=node_p, lambda_node= lambda_node)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, x, edge_index, y, mask, criterion):
  with torch.no_grad():
    model.eval()
    #z = model(x, edge_index)
    #accuracy = label_classification_torch(z, y, num_classes)
    #accuracy = label_classification(z, y)
    out = model.predict_class(x, edge_index)
    loss = criterion(out[mask], y[mask])
    cl = torch.argmax(out[mask], dim=-1)
    correct = torch.sum((cl==y[mask]).int())
    acc = correct/y[mask].shape[0]
    return acc.item(), loss.item()



def k_near_select(emb):

    emb = emb.detach().cpu().numpy()
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    _, I = index.search(emb, k_near)
    node_avg = 0
    for i in range(1, k_near):
        node_avg = node_avg + emb[I[:,i],:]
    node_avg = normalize(node_avg, axis =1)
    emb = emb * node_avg
    emb = np.maximum(0, emb)

    return emb

def get_expl(model):

    emb = model(data.x, data.edge_index).to(device)
    with torch.no_grad():
        emb = k_near_select(emb)
        nodevote = emb.sum(1)
        nodevote = nodevote - nodevote.min()
        nodevote = nodevote / nodevote.max()
        nodevote_list = nodevote.tolist()

        edge = data.edge_index.cpu().tolist()
        vote = [nodevote_list[x]+nodevote_list[y] for x, y  in zip(*edge)]
        vote = torch.tensor(vote).to(device)
        vote = vote - vote.min()
        vote = vote / vote.max()

    return vote, torch.tensor(nodevote).to(device)

device = 'cuda:0'

os.makedirs(data_dir, exist_ok=True)
dataset = Planetoid(root=data_dir, name=dataset_name, split='public', transform=T.NormalizeFeatures())#'full' #'random'
data = dataset[0]
data.edge_index = data.edge_index.to('cuda')
data.y = data.y.to('cuda')
data.x = data.x.to('cuda')
data.train_mask = data.train_mask.to('cuda')
data.val_mask = data.val_mask.to('cuda')
data.test_mask = data.test_mask.to('cuda')

data = dataset[0]
data = data.to(device)
num_classes = torch.unique(data.y).shape[0]
print(num_classes)

device = 'cuda:0'

random.seed(1000)
learning_rate = 0.0001
num_hidden = 1024
coder_hid_num = 20
num_mid_hidden = 1024
activation = F.elu
base_model = GCNConv
model_encoder =  Encoder #EncoderGAT
num_layers = 2
num_epochs = 100
weight_decay = 0.00001
lambda_edge = -2
lambda_node = -1
edge_p = .4
node_p = .4
k_near = 4

encoder = model_encoder(dataset.num_features, coder_hid_num, num_hidden, activation,
                        base_model=base_model, k=num_layers).to(device)
prejector = projection_MLP(num_hidden, num_hidden, num_hidden)
predictor = prediction_MLP(num_hidden, num_mid_hidden, num_hidden)

model = ModelSIM(encoder, prejector, predictor, num_classes=torch.unique(data.y).shape[0]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

learning_rate2 = 0.01#0.0025 #prima 0.001
weight_decay2= 0.001

optimizer2 = torch.optim.Adam(model.linear.parameters(), lr = learning_rate2, weight_decay=weight_decay2)



criterion = torch.nn.CrossEntropyLoss()

best_micro_val = 0.0
best_micro_val_epoch = 0

inner_epoch = 500

just_load = True
if just_load == False:
  for epoch in (range(num_epochs)):
    # adjust_learning_rate(optimizer,epoch)
    acc = {}
    vote, nodevote = get_expl(model)
    trn_loss = train(model, optimizer, data.x, data.edge_index, vote, nodevote)
    print(f"{epoch:3d} Trn loss:  {trn_loss:.3f}")
  torch.save(model.state_dict(), dataset_name + "_"+"GCN"+"_node_cl_engage_1.pt")
else:
  model.load_state_dict(torch.load(dataset_name + "_"+"GCN"+"_node_cl_engage_1.pt"))

later_log_data = {'trn_f_loss':[], 'trn_f_acc':[],
                  'val_f_loss':[],'val_f_acc':[],
                  'tst_f_loss':[], 'tst_f_acc':[]}
best_val_f_acc = 0.0
best_val_f_loss = 0.0
best_tst_f_acc = 0.0
best_tst_f_loss = 0.0
best_epoch = -1
for j in range(inner_epoch):
           epoch = num_epochs

           trn_f_loss, trn_f_acc = finetune(model, data.x, data.edge_index, data.y, data.train_mask, criterion, optimizer2)

           val_f_acc, val_f_loss = test(model, data.x, data.edge_index, data.y, data.val_mask, criterion)

           tst_f_acc, tst_f_loss = test(model, data.x, data.edge_index, data.y, data.test_mask, criterion)
           if val_f_acc > best_val_f_acc:
               best_val_f_acc = val_f_acc
               best_val_f_loss = val_f_loss
               best_tst_f_acc = tst_f_acc
               best_tst_f_loss = tst_f_loss
               best_epoch = j
               print("UPDATE BEST EPOCH")
           later_log_data['trn_f_acc'].append(trn_f_acc)
           later_log_data['val_f_acc'].append(val_f_acc)
           later_log_data['tst_f_acc'].append(tst_f_acc)


           print(f'Epoch{epoch} InnerEpoch:{j} F Loss:{trn_f_loss:.3f} Acc:{trn_f_acc:.3f} '
          f'Val Loss:{val_f_loss:.3f} Val Acc:{val_f_acc:.3f} '
          f'Tst Loss:{tst_f_loss:.3f} Tst Acc:{tst_f_acc:.3f}')

print("=== Final ===")