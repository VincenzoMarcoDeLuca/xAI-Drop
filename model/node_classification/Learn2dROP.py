import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from scipy.sparse.linalg import svds
import time

dataset_name = 'Cora'

def compute_accuracy(pr, gt):
    pr_cl = torch.argmax(pr, dim=-1)
    correct = torch.sum((pr_cl == gt).int())
    total = gt.shape[0]
    return correct/total

class PTDNetGCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hiddens, denoise_hidden_1, denoise_hidden_2, gamma, zeta, dropout, L, lambda_3, SVD_PI, k_svd, input_dr_p):
        super(PTDNetGCN, self).__init__()
        self.k_svd = k_svd
        self.SVD_PI = SVD_PI
        self.lambda_3 = lambda_3
        self.hiddens = hiddens
        self.gamma = torch.tensor(gamma).to('cuda')
        self.zeta = torch.tensor(zeta).to('cuda')
        self.dropout = dropout
        self.L = L
        self.dropout_in = torch.nn.Dropout(input_dr_p)
        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        self.outer_conv = GCNConv(input_dim, hiddens[0])
        for i in range(1, len(hiddens)):
            self.convs.append(GCNConv(hiddens[i - 1], hiddens[i]))
        #self.convs.append(GCNConv(hiddens[-1], output_dim))

        # Attention mechanism layers
        self.attention_layers = torch.nn.ModuleList()
        self.nblayers = []
        self.selflayers = []
        self.attentions = []
        for _ in range(len(self.hiddens) + 1):
            self.nblayers.append(Linear(hiddens[0], denoise_hidden_1))
            self.selflayers.append(Linear(hiddens[0], denoise_hidden_1))
            attention = [Linear(denoise_hidden_1 * 2, denoise_hidden_2),
                         Linear(denoise_hidden_2, 1)] if denoise_hidden_2 > 0 else [Linear(denoise_hidden_1 * 2, 1)]
            self.attentions.append(torch.nn.ModuleList(attention))
            self.attention_layers.extend([self.nblayers[-1], self.selflayers[-1]] + attention)

    def set_fea_adj(self, num_features, edge_index, edge_weight=None):
        self.features = torch.tensor([i for i in range(num_features)])
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.adj_matrix = torch.zeros((num_features, num_features), dtype=torch.float32)
        self.adj_matrix[edge_index[0], edge_index[1]] = 1
        self.rows = edge_index[0]
        self.col = edge_index[1]

    def get_attention(self, f1_features, f2_features, layer=0, training=False):
        nb_layer = self.nblayers[layer](f1_features)
        self_layer = self.selflayers[layer](f2_features)

        concat_features = torch.cat([nb_layer, self_layer], dim=1)

        for attention_layer in self.attentions[layer]:
            concat_features = F.relu(attention_layer(concat_features))
            if training:
                concat_features = F.dropout(concat_features, p=self.dropout, training=training)

        weight = concat_features
        return weight

    def hard_concrete_sample(self, log_alpha, training=True, beta=1.0):
        if training:
            random_noise = torch.rand_like(log_alpha)
            gate_inputs = (torch.log(random_noise) - torch.log(1.0 - random_noise) + log_alpha) / beta
        else:
            gate_inputs = log_alpha
        gate_inputs = torch.sigmoid(gate_inputs)
        stretched_values = gate_inputs * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(stretched_values, 0.0, 1.0)

    def forward(self, x, edge_idx,temperature=1.0, training=True):
        #edge_index, edge_weight = add_self_loops(self.edge_index, self.edge_weight, fill_value=1.0)
        self.edge_maskes = []
        self.maskes = []
        x = self.dropout_in(x)
        x = self.outer_conv(x, edge_idx)

        for layer_index, conv in enumerate(self.convs):
            xs = []
            for _ in range(self.L):  # Using only one run (L=1 as in the original code)
                row, col = self.edge_index

                f1_features = x[row]
                f2_features = x[col]

                weight = self.get_attention(f1_features, f2_features, layer=layer_index, training=training)
                mask = self.hard_concrete_sample(weight, training=training)
                self.edge_maskes.append(weight)
                self.maskes.append(mask)
                mask = mask.squeeze()
                weight = weight.squeeze()
                masked_weight = mask.mul(weight)
                x = conv(x, self.edge_index, edge_weight=masked_weight)
                xs.append(x)

            x = torch.mean(torch.stack(xs), dim=0)

        return x

    def lossl0(self, temperature):
        l0_loss = torch.zeros(1).to('cuda')
        for i, weight in enumerate(self.edge_maskes):
            #print("Questo e' gamma: ", self.gamma, " questo e' zeta: ", self.zeta, " ratio: ", -self.gamma/self.zeta)
            log_c = torch.log(-self.gamma / self.zeta)
            out_t = temperature * log_c
            in_sig = weight - out_t
            sig_out = torch.sigmoid(in_sig)
            l0_loss += torch.mean(sig_out)
        return l0_loss

    def nuclear(self, maskes):
        nuclear_loss = torch.zeros(1)
        adj_mat = self.adj_matrix.clone().to('cuda')
        values = []
        if self.lambda_3 == 0:
            return nuclear_loss
        row, col = self.edge_index
        #adj_mat[row, col] = values


        # Loop over each mask
        for mask in self.maskes:
            mask = mask.squeeze()

            # Apply the mask to the adjacency matrix
            #support_dense = self.adj_mat * mask
            prova = torch.zeros_like(adj_mat)
            row, col = self.edge_index
            mask = mask.reshape(-1).to('cuda')
            prova[row, col] = adj_mat[row,col]*mask
            support_dense = prova


            # Compute AA = support_dense.t() * support_dense
            support_trans = support_dense.t()
            AA = torch.matmul(support_trans, support_dense)

            # If using SVD_PI
            if self.SVD_PI:
                # Convert PyTorch tensor to numpy for scipy
                support_np = support_dense.detach().cpu().numpy()

                # Perform truncated SVD using scipy svds
                k = self.k_svd
                u, s, vh = svds(support_np, k=k)

                # Convert scipy outputs to PyTorch tensors and stop gradients
                u = torch.tensor(u.copy(), requires_grad=False).to('cuda')
                s = torch.tensor(s.copy(), requires_grad=False).to('cuda')
                vh = torch.tensor(vh.copy(),requires_grad=False).to('cuda')

                for i in range(k):

                    vi = vh[i].reshape(-1, 1)  # Get the i-th singular vector and reshape
                    for _ in range(1):  # Perform iterative power method step (1 iteration)
                        vi = torch.matmul(AA, vi)
                        vi_norm = torch.norm(vi)
                        vi = vi / vi_norm

                    # Compute vmv and vv for nuclear norm
                    vmv = torch.matmul(vi.t(), torch.matmul(AA, vi))
                    vv = torch.matmul(vi.t(), vi)

                    t_vi = torch.sqrt(torch.abs(vmv / vv))
                    values.append(t_vi)

                    if k > 1:
                        # Update AA to remove the rank-one approximation contribution
                        AA_minus = torch.matmul(AA, torch.matmul(vi, vi.t()))
                        AA = AA - AA_minus

            # If SVD_PI is not enabled, compute the trace of the AA matrix
            else:
                trace = torch.trace(AA)
                values.append(trace)

        # Sum all values for the final nuclear loss
        nuclear_loss = torch.stack(values).sum()

        """
        for mask in maskes:
            mask = mask.squeeze().detach().cpu().numpy()
            row, col = self.edge_index.cpu().numpy()
            support_csc = csc_matrix((mask, (row, col)), shape=adj_matrix.shape)
            import pdb; pdb.set_trace()
            u, s, vh = svds(support_csc, k=adj_matrix.shape[0])
            nuclear_loss += torch.tensor(s).sum()
        """
        return nuclear_loss


import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os


# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset_name = 'Cora'
os.makedirs("./data", exist_ok=True)
if dataset_name == 'Cora' or dataset_name == 'CiteSeer' or dataset_name == 'PubMed':
        dataset = Planetoid(root="./data", name=dataset_name, split='public')#,transform=T.NormalizeFeatures())


# Load data
dataset = dataset[0]
features = dataset.x.to(device)
train_mask = dataset.train_mask.to(device)
val_mask = dataset.val_mask.to(device)
tst_mask = dataset.test_mask.to(device)
y_train = dataset.y[dataset.train_mask].to(device)
y_val = dataset.y[dataset.val_mask].to(device)
y_test = dataset.y[dataset.test_mask].to(device)
edge_idx = dataset.edge_index.to(device)

#adj, features, y_train, y_val, y_test, train_mask, val_mask, tst_mask = load_data(dataset_name)

# Some preprocessing
nodesize = features.shape[0]

# Convert data to PyTorch tensors
y_train_tensor = y_train
y_val_tensor = y_val
y_test_tensor = y_test
train_mask_tensor = dataset.train_mask
val_mask_tensor = dataset.val_mask
test_mask_tensor = dataset.test_mask
SVD_PI = True


best_test_acc = 0
best_val_acc_trail = 0
best_val_loss = float('inf')
best_epoch = 0
curr_step = 0
best_val_acc = 0
num_epochs = 500
init_temperature = 2.0
temperature_decay = 0.99

input_dr_p = 0.85
dropout = 0.85


denoise_hidden_1 = 8 #BEST SO FAR 8
denoise_hidden_2 = 0
gamma = 0.0
zeta = 1.01

lambda1 = 0.1
lambda_3 = 0.01  ###BEST SO FAR: 0.01
coff_consis = 0.05 #Useless
k_svd = 1
lr = 0.001 #TRY TO FURTHER REDUCE LR - BEST SO FAR: 0.001
weight_decay = 0.01 #BEST SO FAR: 0.01
hiddens = [32,8]
outL = 1
L = 1
initializer = 'he'
act = 'leaky_relu'


# Define model

model = PTDNetGCN(input_dim=features.shape[1], output_dim=torch.unique(dataset.y).shape[0],
                  hiddens=hiddens, denoise_hidden_1=denoise_hidden_1, denoise_hidden_2=denoise_hidden_2,
                  gamma=gamma, zeta=zeta, dropout=dropout, L = L, lambda_3 = lambda_3, SVD_PI = SVD_PI, k_svd = k_svd, input_dr_p=input_dr_p).to(device)

model.set_fea_adj(features.shape[0], edge_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0.0
best_tst_acc = 0.0
best_epoch = 0
best_tst_loss = 0
best_val_loss = 0
# Training loop
pytorch_total_params = sum(p.numel() for p in model.parameters())


start = time.time()
for epoch in range(num_epochs):
    temperature = max(0.05, init_temperature * (temperature_decay ** epoch))
    model.train()
    optimizer.zero_grad()

    # Forward pass with L different runs
    preds = []
    for _ in range(outL):
        output = model(features, edge_idx, temperature=temperature, training=True)
        preds.append(output.unsqueeze(0))

    # Consistency loss across different predictions
    preds_stack = torch.cat(preds, dim=0)
    mean_preds = torch.mean(preds_stack, dim=0)
    #import pdb; pdb.set_trace()
    consistency_loss = torch.norm(mean_preds - preds_stack, p=2)

    # Cross-entropy loss with training mask
    #cross_loss = masked_softmax_cross_entropy(mean_preds, y_train_tensor, train_mask_tensor)
    criterion = torch.nn.CrossEntropyLoss()

    cross_loss = criterion(mean_preds[train_mask], y_train)

    # Regularization terms
    lossL2 = torch.sum(torch.stack([torch.norm(param) for param in model.parameters()]))
    lossl0 = model.lossl0(temperature)
    nuclear = model.nuclear(model.maskes)
    nuclear = nuclear.to('cuda:0')

    # Total loss
    print(f"CROSS {cross_loss.item():.3f} "
          f" + weight_decay: {weight_decay:.3f} * l2 {lossL2.item():.3f}"
          f" + la1 {lambda1:.3f} * LOSS {lossl0.item():.3f} "
          f" + la3 {lambda_3:.3f} + NUCLEAR {nuclear.item():.3f}"
          f" + cc {coff_consis:.3f} CONS {consistency_loss.item():.3f}")
    loss = cross_loss + weight_decay * lossL2 + lambda1 * lossl0 + lambda_3 * nuclear + coff_consis * consistency_loss

    #print("Questa e' la loss: ", loss)
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Validation and test accuracy
    model.eval()
    with torch.no_grad():
        output = model(features, edge_idx, temperature=None, training=False)
        with torch.no_grad():
            model.eval()
            trn_acc = compute_accuracy(output[train_mask], y_train)
            trn_loss = criterion(output[train_mask], y_train)
            val_loss = criterion(output[val_mask], y_val)
            val_acc = compute_accuracy(output[val_mask], y_val)
            tst_loss = criterion(output[tst_mask], y_test)
            tst_acc = compute_accuracy(output[tst_mask], y_test)
            print(f"{epoch:3d} TRN - L: {loss.item():.3f} CE:{trn_loss.item():.3f} Acc: {trn_acc:.3f} VAL Acc:{val_acc:.3f} L:{val_loss} TST Acc:{tst_acc:.3f} L:{tst_loss:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_tst_acc = tst_acc
                best_epoch = epoch
                best_val_loss = val_loss
                best_tst_loss = tst_loss

print(
    f"BEST {best_epoch:3d} VAL Acc:{best_val_acc:.3f} L:{best_val_loss} TST Acc:{best_tst_acc:.3f} L:{best_tst_loss:.3f}")

end = time.time()
print("SECONDI IMPIEGATI: ", end-start, " NUM EPOCHS: ", num_epochs, " numero parametri ", pytorch_total_params)
exit(0)

# Track best validation accuracy
