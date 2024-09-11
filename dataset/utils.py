from torch_geometric.datasets import Planetoid
import os
import torch_geometric.transforms as T

class Dataset():
    def __init__(self, dataset_name, device, data_dir="./data"):
        super(Dataset, self).__init__()
        os.makedirs(data_dir, exist_ok=True)
        if dataset_name == 'Cora' or dataset_name == 'CiteSeer' or dataset_name == 'PubMed':
            self.dataset = Planetoid(root=data_dir, name=dataset_name, split='public', transform=T.NormalizeFeatures())#'full' #'random'

        data = self.dataset[0].to(device)

        print(data.train_mask.nonzero().shape)
        print(data.val_mask.nonzero().shape)
        print(data.test_mask.nonzero().shape)
        print(data.edge_index.shape)