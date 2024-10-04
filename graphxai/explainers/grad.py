import torch

from typing import Optional, Callable
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation


class GradExplainer(_BaseExplainer):
    """
    Vanilla Gradient Explanation for GNNs

    Args:
        model (torch.nn.Module): model on which to make predictions
            The output of the model should be unnormalized class score.
            For example, last layer = CNConv or Linear.
        criterion (torch.nn.Module): loss function
    """
    def __init__(self, model: torch.nn.Module, 
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], decoder=None):
        super().__init__(model = model, decoder = decoder)
        self.criterion = criterion

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None,
                             aggregate_node_imp = torch.sum,
                             y = None,
                             forward_kwargs: dict = {},

                             unlock_feature_imp = False,
                             k_hop_info= None,
                             **_) -> Explanation:
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
                (:default: :obj:`None`)
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
                (:default: :obj:`None`)
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the enclosing 
                subgraph. Must support `dim` argument. 
                (:default: :obj:`torch.sum`)
                (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): Additional arguments to model.forward
                beyond x and edge_index. Must be keyed on argument name. 
                (default: :obj:`{}`)

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [features,]`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop, features]`
                `edge_imp`: :obj:`None`
                `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
        """
        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        if k_hop_info is None:
            k_hop_info = subset, sub_edge_index, mapping, _ = \
               k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        else:
            subset, sub_edge_index, mapping, _ = k_hop_info
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True

        output = self.model(sub_x, sub_edge_index)
        loss = self.criterion(output[mapping], label[mapping])
        loss.backward()
        if unlock_feature_imp:
            feature_imp = sub_x.grad[torch.where(subset == node_idx[0])].squeeze(0)


        node_imp = aggregate_node_imp(sub_x.grad, dim = 1)

        ###Explanation has two different attributes:
        ###exp.node_imp -> node_importance in prediction
        ###exp.node_ref -> node_mapping
        exp = Explanation(
            #feature_imp = feature_imp, #[score_1, ]
            node_imp = node_imp, #[score_1, score_2, ...] [[], []] NxF
            node_idx = mapping   #node_idx -> node_idx is the original part (node_idx)
        )
        #We set the attribute exp.enc_subgraph
        #We find the node references as exp.mapping
        exp.set_enclosing_subgraph(k_hop_info, mapping)
        return exp

    def get_explanation_graph(self, 
                                x: torch.Tensor, 
                                edge_index: torch.Tensor,
                                label: torch.Tensor, 
                                aggregate_node_imp = torch.sum,
                                forward_kwargs: dict = {}) -> Explanation:
        """
        Explain a whole-graph prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, [n x ...]): labels to explain
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the graph. 
                Must support `dim` argument. (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method. 
                Fields are:
                `feature_imp`: :obj:`None`
                `node_imp`: :obj:`torch.Tensor, [num_nodes, features]`
                `edge_imp`: :obj:`None`
                `graph`: :obj:`torch_geometric.data.Data`
        """

        self.model.eval()
        x.requires_grad = True
        output = self.model(x, edge_index, **forward_kwargs)
        loss = self.criterion(output, label)
        loss.backward()

        node_imp = aggregate_node_imp(x.grad, dim = 1)

        exp = Explanation(
            node_imp = node_imp
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None,
                             aggregate_node_imp = torch.sum,
                             y = None,
                             forward_kwargs: dict = {},

                             unlock_feature_imp = False,
                             k_hop_info= None,
                             **_) -> Explanation:
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
                (:default: :obj:`None`)
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
                (:default: :obj:`None`)
            aggregate_node_imp (function, optional): torch function that aggregates
                all node importance feature-wise scores across the enclosing
                subgraph. Must support `dim` argument.
                (:default: :obj:`torch.sum`)
            forward_kwargs (dict, optional): Additional arguments to model.forward
                beyond x and edge_index. Must be keyed on argument name.
                (default: :obj:`{}`)

        :rtype: :class:`graphxai.Explanation`

        Returns:
            exp (:class:`Explanation`): Explanation output from the method.
                Fields are:
                `feature_imp`: :obj:`torch.Tensor, [features,]`
                `node_imp`: :obj:`torch.Tensor, [nodes_in_khop, features]`
                `edge_imp`: :obj:`None`
                `enc_subgraph`: :obj:`graphxai.utils.EnclosingSubgraph`
        """
        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        if k_hop_info is None:
            k_hop_info = subset, sub_edge_index, mapping, _ = \
               k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        else:
            subset, sub_edge_index, mapping, _ = k_hop_info
        sub_x = x[subset]

        self.model.eval()
        sub_x.requires_grad = True

        output = self.model(sub_x, sub_edge_index)
        ###Prima invece era label[mapping] ed era sbagliato
        loss = self.criterion(output[mapping], label[subset[mapping]])
        loss.backward()
        if unlock_feature_imp:
            feature_imp = sub_x.grad[torch.where(subset == node_idx[0])].squeeze(0)


        node_imp = aggregate_node_imp(sub_x.grad, dim = 1)

        ###Explanation has two different attributes:
        ###exp.node_imp -> node_importance in prediction
        ###exp.node_ref -> node_mapping
        exp = Explanation(
            #feature_imp = feature_imp, #[score_1, ]
            node_imp = node_imp, #[score_1, score_2, ...] [[], []] NxF
            node_idx = mapping   #node_idx -> node_idx is the original part (node_idx)
        )
        #We set the attribute exp.enc_subgraph
        #We find the node references as exp.mapping
        exp.set_enclosing_subgraph(k_hop_info, mapping)
        return exp

    def get_explanation_link(self,
                             expl_edge_idx: int,
                             x: torch.Tensor,
                             edge_index: torch.Tensor,
                             label: Optional[torch.Tensor] = None,
                             num_hops: Optional[int] = None,
                             aggregate_node_imp = torch.sum,
                             y = None,
                             return_type = 'normalized',
                             forward_kwargs: dict = {},


                             unlock_feature_imp = False,
                             k_hop_info= None,
                             **_) -> Explanation:
        """
        Explain a link prediction.
        """
        label = self._predict(x, edge_index,return_type=return_type,
                              forward_kwargs=forward_kwargs) if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        if k_hop_info is None:
            edge_index_involved = edge_index[:,expl_edge_idx]
            node_idx = edge_index_involved.reshape(-1)
            k_hop_info = subset, sub_edge_index, node_mapping, edge_mapping = \
               k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True, num_nodes=x.shape[0])
        else:
            subset, sub_edge_index, node_mapping, edge_mapping = k_hop_info
        sub_x = x[subset]
        if self.backbone is not None:
            self.backbone.eval()
        else:
            self.model.eval()
        sub_x.requires_grad = True
        if self.backbone is not None:
            output = self.backbone.encode(sub_x, sub_edge_index)
            output = self.backbone.decode(output, sub_edge_index).view(-1).sigmoid()

        #output = self.model(sub_x, sub_edge_index)
        loss = self.criterion(output[node_mapping], label[subset[node_mapping]])
        loss.backward()
        node_imp = aggregate_node_imp(sub_x.grad, dim = 1)
        ###Importanza di un edge come la media delle importanze dei due edge
        sub_edge_imp = node_imp[sub_edge_index]
        edge_imp = torch.mean(sub_edge_imp, dim = 0)
        edge_reference = edge_index[:,edge_mapping]
        exp = Explanation(
            #feature_imp = feature_imp, #[score_1, ]
            node_imp = node_imp, #[score_1, score_2, ...] [[], []] NxF
            node_idx = subset,#node_idx -> node_idx is the original part (node_idx)
            edge_imp = edge_imp,
            edge_reference = edge_reference
        )
        #import pdb; pdb.set_trace()
        transposed_edges = edge_index_involved.transpose(0,1)
        transposed_ref = exp.edge_reference.transpose(0,1)

        torch_tmp= (transposed_ref.unsqueeze(0) == transposed_edges.unsqueeze(1)).all(dim=2)

        # Find the indices where the rows match
        torch_indexes= torch.where(torch_tmp)[1]
        #torch_indexes = torch.stack([torch.where((transposed_ref == transposed_edges[i]).all(dim=1))[0] for i in range(transposed_edges.shape[0])])

        """
        import pdb; pdb.set_trace()

        edge_mapping_src = torch.nonzero(edge_index_involved[0].unsqueeze(1) == exp.edge_reference[0], as_tuple=False)[:, 1]
        edge_mapping_dst = torch.nonzero(edge_index_involved[1].unsqueeze(1) == exp.edge_reference[1], as_tuple=False)[:, 1]
        # Use torch.isin to find where source matches destination
        # Combining src and dst matches
        mask = torch.isin(edge_mapping_src, edge_mapping_dst)

        # Get the final valid indices
        torch_indexes = torch.nonzero(mask).squeeze()
        """
        # Use the indices to get the correct mapping
        edge_mapping = torch.stack([exp.edge_reference[:,torch_indexes]])

        #import pdb;pdb.set_trace()
        #edge_mapping_src = [torch.where(edge_index_involved[0][i] == exp.edge_reference[0])[0] for i in range(edge_index_involved.shape[1])]
        #edge_mapping_dst = [torch.where(edge_index_involved[1][i] == exp.edge_reference[1])[0] for i in range(edge_index_involved.shape[1])]
        #torch_indexes = [torch.nonzero(torch.isin(edge_mapping_src[k], edge_mapping_dst[k]))[0] for k in range(len(edge_mapping_src))]
        #edge_mapping = torch.stack([edge_mapping_src[i][torch_indexes[i]] for i in range(len(torch_indexes))])

        edge_mapping = edge_mapping.reshape(-1)
        edge_mapping = torch.unique(edge_mapping)
        exp.set_enclosing_subgraph(k_hop_info, node_mapping, torch_indexes.reshape(-1))
        return exp
