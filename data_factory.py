import torch
import networkx as nx
import numpy as np
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.utils import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


def get_mask(idx, length):
    """Create mask.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask


def load_data(root: str, data_name: str, split='public', **kwargs):
    if data_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(root=root, name=data_name)
        mask = dataset.get_idx_split()
        train_mask, val_mask, test_mask = mask.values()
    elif data_name in ['actor', 'chameleon', 'squirrel']:
        if data_name == 'actor':
            path = root + f'/{data_name}'
            dataset = Actor(root=path)
        else:
            dataset = WikipediaNetwork(root=root, name=data_name)
        num_nodes = dataset.data.x.shape[0]
        idx_train = []
        for j in range(dataset.num_classes):
            idx_train.extend([i for i, x in enumerate(dataset.data.y) if x == j][:20])
        idx_val = np.arange(num_nodes - 1500, num_nodes - 1000)
        idx_test = np.arange(num_nodes - 1000, num_nodes)
        label_len = dataset.data.y.shape[0]
        train_mask, val_mask, test_mask = get_mask(idx_train, label_len), get_mask(idx_val, label_len), get_mask(idx_test, label_len)
    else:
        raise NotImplementedError

    print(dataset.data)
    G = to_networkx(dataset.data)
    features = dataset.data.x
    num_features = dataset.num_features
    labels = dataset.data.y
    edge_index = dataset.data.edge_index
    motif = get_motif(edge_index)
    num_classes = dataset.num_classes
    return features, num_features, labels, edge_index, motif, (train_mask, val_mask, test_mask), num_classes


def get_motif(edge_index: torch.Tensor):
    from collections import defaultdict
    locate = defaultdict(list)
    for edge in edge_index.t():
        u, v = edge[0].item(), edge[1].item()
        locate[u].append(v)
    index = []
    for edge in edge_index.t():
        u, v = edge[0].item(), edge[1].item()
        for w in locate[v]:
            if w != u:
                index.append([u, v, w])
    index = torch.tensor(index).t()
    return index



