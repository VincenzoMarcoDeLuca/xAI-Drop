import os
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.seed import seed_everything as th_seed
from typing import Dict, Tuple


class GraphLoader:
    """
    Data loader class for graph data.
    """

    def __init__(self, seed, paths, device, dataset_name: str) -> None:
        """
        Initializes the GraphLoader.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        dataset_name : str
            Name of the dataset to load.
        kwargs : Dict[str, Any], optional
            Dictionary for additional parameters if needed, by default {}.
        """
        # Get config
        self.seed = seed
        # Set the seed
        th_seed(seed)
        # Set the paths
        self.paths = paths
        self.device =device
        # Get the datasets
        self.train_data, self.validation_data, self.test_data = self.get_dataset(dataset_name)

    def get_dataset(self, dataset_name: str) -> Tuple[Data, Data, Data]:
        """
        Returns the training, validation, and test datasets.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load.
        file_path : str
            Path to the dataset.

        Returns
        -------
        Tuple[Data, Data, Data]
            Training, validation, and test datasets.
        """

        # Initialize Graph dataset class
        self.graph_dataset = GraphDataset(paths = self.paths, device = self.device, dataset_name=dataset_name)

        # Load Training, Validation, Test datasets
        train_data, val_data, test_data = self.graph_dataset._load_data()

        # Generate static subgraphs from training set
        # train_data = self.generate_subgraphs(train_data)

        # Return
        return train_data, val_data, test_data

def get_transform(device):
    """Splits data to train, validation and test, and moves them to the device"""
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.1, #0.25
                          num_test=0.2, #0.25
                          #split_labels=True,
                          is_undirected=True,
                          add_negative_train_samples=True,
                          split_labels=True,
                          neg_sampling_ratio=1.0,
                          ),
    ])

    return transform


class GraphDataset:
    """
    Dataset class for graph data format.
    """

    def __init__(self, paths, device, dataset_name: str) -> None:
        """
        Initializes the GraphDataset.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing options and arguments.
        datadir : str
            The path to the data directory.
        dataset_name : str
            Name of the dataset to load.
        """
        self.paths = paths
        self.dataset_name = dataset_name
        self.device = device
        self.data_path = os.path.join(self.paths["data"], 'Planetoid')
        self.transform = get_transform(self.device)

    def _load_data(self) -> Tuple[Data, Data, Data]:
        """
        Loads one of many available datasets and returns features and labels.

        Returns
        -------
        Tuple[Data, Data, Data]
            Training, validation, and test datasets.
        """
        if self.dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
            # Get the dataset
            self.dataset = Planetoid(self.data_path, self.dataset_name, split="random", transform=self.transform)
        elif self.dataset_name.lower() in ['chameleon']:
            # Get the dataset
            self.dataset = WikipediaNetwork(root=self.data_path, name=self.dataset_name, transform=self.transform)
        elif self.dataset_name.lower() in ["cornell", "texas", "wisconsin"]:
            # Get the dataset
            self.dataset = WebKB(root=self.data_path, name=self.dataset_name, transform=self.transform)
        else:
            print(f"Given dataset name is not found. Check for typos, or missing condition ")
            exit()
        # Data splits
        train_data, val_data, test_data = self.dataset[0]
        # Return
        return train_data, val_data, test_data
