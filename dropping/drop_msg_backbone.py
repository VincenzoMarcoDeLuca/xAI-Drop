import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
import torch_geometric
from torch_sparse import SparseTensor, set_diag



class ModelPretreatment:
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(ModelPretreatment, self).__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, x: Tensor, edge_index: Adj):
        # add self loop
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # normalize
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight


class BbGCN(MessagePassing):
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(BbGCN, self).__init__()
        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate)
        return y

    def message(self, x_j: Tensor, drop_rate: float):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # drop messages
        x_j = F.dropout(x_j, drop_rate)

        return x_j