import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geoopt.manifolds.stereographic.math import artan_k
from geoopt.manifolds.stereographic import StereographicExact
from geoopt.manifolds import Euclidean
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter
from backbone import GCN, GAT, GraphSAGE
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


EPS = 1e-5


class RiemannianFeatures(nn.Module):
    def __init__(self, num_nodes, dimensions, d_free, init_curvature, num_factors, learnable=True):
        super(RiemannianFeatures, self).__init__()
        self.manifolds = nn.ModuleList()
        self.features = nn.ParameterList()
        for i in range(num_factors):
            if isinstance(dimensions, list):
                d = dimensions[i]
            else:
                d = dimensions
            k = init_curvature * (torch.randn(1) + 1)
            manifold = StereographicExact(k=k, learnable=learnable)
            features = ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d), manifold=manifold))
            if k != 0:
                self.init_weights(features)
            self.manifolds.append(manifold)
            self.features.append(features)

    @staticmethod
    def init_weights(w, scale=1e-4):
        w.data.uniform_(-scale, scale)
        w_norm = w.data.norm(p=2, dim=-1, keepdim=True) + EPS
        w.data = w.data / w_norm * w.manifold.radius * 0.9 * torch.rand(1)

    @staticmethod
    def normalize(x, manifold):
        x_norm = x.norm(p=2, dim=-1, keepdim=True) + EPS
        if manifold.k != 0:
            x = x / x_norm * 0.9 * torch.rand(1).to(x.device) * manifold.radius
        else:
            x = x / x_norm
        return x

    def forward(self):
        products = []
        for manifold, features in zip(self.manifolds, self.features):
            products.append(self.normalize(features, manifold))
        return products


class Model(nn.Module):
    def __init__(self, backbone, n_layers, in_features, hidden_features, embed_features, n_heads, drop_edge, drop_node,
                 num_factors, dimensions, d_embeds, temperature, device=torch.device('cuda')):
        super(Model, self).__init__()
        if backbone == 'gcn':
            self.encoder = GCN(n_layers, in_features, hidden_features, embed_features, drop_edge, drop_node)
            self.encoder2 = GCN(n_layers, d_embeds, hidden_features, embed_features, drop_edge, drop_node)
        elif backbone == 'gat':
            self.encoder = GAT(n_layers, in_features, hidden_features, embed_features, n_heads, drop_edge, drop_node)
            self.encoder2 = GAT(n_layers, d_embeds, hidden_features, embed_features, n_heads, drop_edge, drop_node)
        elif backbone == 'sage':
            self.encoder = GraphSAGE(n_layers, in_features, hidden_features, embed_features, drop_edge, drop_node)
            self.encoder2 = GraphSAGE(n_layers, d_embeds, hidden_features, embed_features, drop_edge, drop_node)
        else:
            raise NotImplementedError

        self.temperature = temperature
        self.Ws = []
        self.bias = []
        for i in range(num_factors):
            if isinstance(dimensions, list):
                d = dimensions[i]
            else:
                d = dimensions
            pre = torch.randn(d_embeds, d).to(device)
            w = pre / (torch.norm(pre, dim=-1, keepdim=True) + EPS)
            self.Ws.append(w)
            self.bias.append(2 * torch.pi * torch.rand(d_embeds).to(device))

        self.motif_cls = nn.Sequential(
            nn.Linear(3 * d_embeds, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.norm = nn.LayerNorm((num_factors+1) * embed_features)

    def forward(self, x, edge_index, motif, neg_motif, rm_features: RiemannianFeatures):
        products = rm_features()
        x = self.encoder(x, edge_index)
        laplacian = self.random_mapping(rm_features.manifolds, products)
        cl_loss = 0
        embeds = []
        for embed in laplacian:
            embed = self.encoder2(embed, edge_index)
            cl_loss = cl_loss + self.cal_cl_loss(x, embed)
            embeds.append(embed)
        loss = cl_loss / len(laplacian) + self.cal_motif_loss(laplacian, motif, neg_motif)
        embeds = torch.concat(embeds, -1)
        return self.norm(torch.concat([x, embeds], -1)), loss

    def random_mapping(self, manifolds, products):
        out = []
        for i in range(len(manifolds)):
            x = products[i]
            w = self.Ws[i]
            b = self.bias[i]
            k = manifolds[i].k
            if k == 0:
                distance = x @ w.t()
            else:
                div = torch.sum((x[:, None] - w[None]) ** 2, dim=-1)
                distance = torch.log((1 + k * torch.sum(x * x, -1, keepdim=True)) / (div + EPS) + EPS)
            n = x.shape[-1]
            z = torch.exp((n - 1) * distance / 2) * torch.cos(distance + b)
            out.append(z)
        return out

    def cal_cl_loss(self, x1, x2):
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', norm1, norm2) + EPS)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim + EPS)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim + EPS)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss

    def cal_motif_loss(self, products, motifs, neg_motifs):
        loss = 0
        for product in products:
            pos = self.motif_cls(self.nodes_from_motif(product, motifs))
            neg = self.motif_cls(self.nodes_from_motif(product, neg_motifs))
            loss = loss + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos).to(motifs.device)) + \
                   F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg).to(motifs.device))
        return loss / len(products)

    @staticmethod
    def nodes_from_motif(features, motifs):
        u, v, w = motifs[0], motifs[1], motifs[2]
        f_u = features[u]
        f_v = features[v]
        f_w = features[w]
        return torch.concat([f_u, f_v, f_w], -1)


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.1):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, edge_index):
        return self.fc(self.dropout(x))


class GCNClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.1):
        super(GCNClassifier, self).__init__()
        self.fc = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, edge_index):
        return self.fc(self.dropout(x), edge_index)


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs