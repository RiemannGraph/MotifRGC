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
from torch_geometric.utils import negative_sampling


class RiemannianFeatures(nn.Module):
    def __init__(self, num_nodes, dimensions, d_free, init_curvature, num_factors):
        super(RiemannianFeatures, self).__init__()
        self.manifolds = nn.ModuleList()
        self.features = nn.ParameterList()
        for i in range(num_factors):
            if isinstance(dimensions, list):
                d = dimensions[i]
            else:
                d = dimensions
            k = init_curvature * np.random.randn()
            manifold = StereographicExact(k=k, learnable=True)
            features = ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d), manifold=manifold))
            if k != 0:
                self.init_weights(features)
            self.manifolds.append(manifold)
            self.features.append(features)
        self.manifolds.append(Euclidean(d_free))
        self.features.append(ManifoldParameter(ManifoldTensor(torch.empty(num_nodes, d_free), manifold=Euclidean(d_free))))

    @staticmethod
    def init_weights(w, scale=1e-4):
        w.data.uniform_(-scale, scale)
        w_norm = w.data.norm(p=2, dim=-1, keepdim=True)
        w.data = w.data / w_norm * w.manifold.radius * 0.9 * torch.rand(1)

    @staticmethod
    def normalize(x, manifold):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        x = x / x_norm * 0.9 * torch.rand(1).to(x.device) * manifold.radius
        return x

    def forward(self):
        products = []
        for manifold, features in zip(self.manifolds[:-1], self.features[:-1]):
            if manifold.k != 0:
                products.append(self.normalize(features, manifold))
            else:
                products.append(features)
        products.append(self.features[-1])
        return products


EPS = 1e-5


class Model(nn.Module):
    def __init__(self, backbone, n_layers, in_features, hidden_features, embed_features, n_heads, drop_edge, drop_node,
                 num_factors, dimensions, d_embeds, device=torch.device('cuda')):
        super(Model, self).__init__()
        assert (num_factors + 1) * d_embeds == embed_features, "Embed dimensions do not match"
        if backbone == 'gcn':
            self.encoder = GCN(n_layers, in_features, hidden_features, embed_features, drop_edge, drop_node)
        elif backbone == 'gat':
            self.encoder = GAT(n_layers, in_features, hidden_features, embed_features, n_heads, drop_edge, drop_node)
        elif backbone == 'sage':
            self.encoder = GraphSAGE(n_layers, in_features, hidden_features, embed_features, drop_edge, drop_node)
        else:
            raise NotImplementedError

        self.Ws = []
        self.bias = []
        for i in range(num_factors):
            if isinstance(dimensions, list):
                d = dimensions[i]
            else:
                d = dimensions
            pre = torch.randn(d_embeds, d).to(device)
            w = pre / torch.norm(pre, dim=-1, keepdim=True)
            self.Ws.append(w)
            self.bias.append(2 * torch.pi * torch.rand(d_embeds).to(device))

        self.motif_cls = nn.Sequential(
            nn.Linear(3 * embed_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, motif, rm_features: RiemannianFeatures):
        products = rm_features()
        x = self.encoder(x, edge_index)
        laplacian = self.random_mapping(rm_features.manifolds, products)
        loss = self.cal_cl_loss(x, torch.concat(laplacian, -1)) + self.cal_motif_loss(products, motif)
        return products, loss

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
                distance = torch.log((1 + k * x @ x) / (div + EPS))
            z = torch.exp((n - 1) * distance / 2) * torch.cos(distance + b)
            out.append(z)
        return out

    def cal_cl_loss(self, x1, x2):
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', norm1, norm2)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss

    def cal_motif_loss(self, products, motifs):
        neg_motifs = negative_sampling(motifs[:-1], num_nodes=products[0].shape[0])
        neg_motifs = torch.concat([neg_motifs, motifs[-1:]], dim=0)
        loss = 0
        for product in products:
            pos = self.motif_cls(self.nodes_from_motif(product, motifs))
            neg = self.motif_cls(self.nodes_from_motif(product, neg_motifs))
            loss = loss + F.binary_cross_entropy(pos, torch.ones(pos)) + F.binary_cross_entropy(neg, torch.zeros(neg))
        return loss

    @staticmethod
    def nodes_from_motif(features, motifs):
        u, v, w = motifs[0], motifs[1], motifs[2]
        f_u = features[u]
        f_v = features[v]
        f_w = features[w]
        return torch.concat([f_u, f_v, f_w], -1)















