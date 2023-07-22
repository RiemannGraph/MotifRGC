import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import dropout_edge


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features, drop_edge=0.5, drop_feats=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_features, hidden_features))
        self.layers.append(GCNConv(hidden_features, out_features))
        self.drop_edge = drop_edge
        self.drop = nn.Dropout(drop_feats)

    def forward(self, x, edge_index):
        edge_index = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers[:-1]:
            x = self.drop(F.relu(layer(x, edge_index)))
        x = self.layers[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features, heads, drop_edge=0.5, drop_feats=0.5):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_features, hidden_features, heads, dropout=drop_feats))
        for _ in range(n_layers - 2):
            self.layers.append(GATConv(hidden_features, hidden_features, heads, dropout=drop_feats))
        self.layers.append(GATConv(hidden_features, out_features, heads, dropout=drop_feats))
        self.drop_edge = drop_edge
        self.drop = nn.Dropout(drop_feats)

    def forward(self, x, edge_index):
        edge_index = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features, drop_edge=0.5, drop_feats=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_features, hidden_features))
        self.layers.append(SAGEConv(hidden_features, out_features))
        self.drop = nn.Dropout(drop_feats)
        self.dropout_edge = drop_edge

    def forward(self, x, edge_index):
        edge_index = dropout_edge(edge_index, self.dropout_edge, training=self.training)[0]
        for layer in self.layers[: -1]:
            x = self.drop(F.relu(layer(x, edge_index)))
        x = self.layers[-1](x, edge_index)
        return x

