import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import RiemannianFeatures, Model, FermiDiracDecoder
from backbone import LinearClassifier, GNNClassifier
from utils import cal_accuracy, cal_F1, cal_AUC_AP
from data_factory import load_data, mask_edges
from sklearn.cluster import KMeans
from logger import create_logger
from geoopt.optim import RiemannianAdam
from geoopt.manifolds.stereographic.math import dist


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def pretrain(self, model, Riemann_embeds_getter, epochs, logger, r_optim, optimizer):
        for epoch in range(1, epochs + 1):
            model.train()
            Riemann_embeds_getter.train()
            embeds, loss = model(self.features, self.edge_index, self.motif, self.neg_motif, Riemann_embeds_getter)
            r_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}")

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        features, in_features, labels, edge_index, neg_edge, motif, neg_motif, masks, n_classes = load_data(self.configs.root_path, self.configs.dataset)
        edge_index = edge_index.to(device)
        neg_edge = neg_edge.to(device)
        motif = motif.to(device)
        neg_motif = neg_motif.to(device)
        features = features.to(device)
        labels = labels.to(device)
        self.masks = masks
        self.in_features = in_features
        self.n_classes = n_classes
        self.labels = labels
        self.edge_index = edge_index
        self.neg_edge = neg_edge
        self.motif = motif
        self.neg_motif = neg_motif
        self.features = features

        if self.configs.downstream_task == "NC":
            vals = []
            accs = []
            wf1s = []
            mf1s = []
        elif self.configs.downstream_task == "LP":
            aucs = []
            aps = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            Riemann_embeds_getter = RiemannianFeatures(features.shape[0], self.configs.dimensions,
                                                       self.configs.init_curvature, self.configs.num_factors,
                                                       learnable=self.configs.learnable).to(device)
            model = Model(backbone=self.configs.backbone, n_layers=self.configs.n_layers, in_features=in_features,
                          embed_features=self.configs.embed_features, hidden_features=self.configs.hidden_features,
                          n_heads=self.configs.n_heads, drop_edge=self.configs.drop_edge, drop_node=self.configs.drop_edge,
                          num_factors=self.configs.num_factors, dimensions=self.configs.dimensions, d_embeds=self.configs.d_embeds,
                          temperature=self.configs.temperature, device=device).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)

            logger.info("--------------------------Training Start-------------------------")
            if self.configs.pre_training:
                self.pretrain(model, Riemann_embeds_getter, self.configs.epochs, logger, r_optim, optimizer)

            if self.configs.downstream_task == 'NC':
                _, _, _ = self.train_lp(model, Riemann_embeds_getter, r_optim, optimizer, logger)
                best_val, test_acc, test_weighted_f1, test_macro_f1 = self.train_cls(model, Riemann_embeds_getter, r_optim, optimizer, logger)
                logger.info(
                    f"val_accuracy={best_val.item() * 100: .2f}%, test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"\t\t weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                vals.append(best_val.item())
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)
            elif self.configs.downstream_task == 'LP':
                _, test_auc, test_ap = self.train_lp(model, Riemann_embeds_getter, r_optim, optimizer, logger)
                logger.info(
                    f"test_auc={test_auc * 100: .2f}%, test_ap={test_ap * 100: .2f}%")
                aucs.append(test_auc)
                aps.append(test_ap)
            else:
                raise NotImplementedError

        if self.configs.downstream_task == "NC":
            logger.info(f"valid results: {np.mean(vals)}~{np.std(vals)}")
            logger.info(f"test results: {np.mean(accs)}~{np.std(accs)}")
            logger.info(f"test weighted-f1: {np.mean(wf1s)}~{np.std(wf1s)}")
            logger.info(f"test macro-f1: {np.mean(mf1s)}~{np.std(mf1s)}")
        elif self.configs.downstream_task == "LP":
            logger.info(f"test AUC: {np.mean(aucs)}~{np.std(aucs)}")
            logger.info(f"test AP: {np.mean(aps)}~{np.std(aps)}")

    def cal_cls_loss(self, model, edge_index, mask, features, labels):
        out = model(features, edge_index)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, model, Riemann_embeds_getter, r_optim, optimizer, logger):
        """masks = (train, val, test)"""
        device = self.device
        model_cls = GNNClassifier((self.configs.num_factors+1)*self.configs.embed_features, self.n_classes,
                                  drop=self.configs.drop_cls, backbone=self.configs.backbone).to(device)
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)

        best_acc = 0.
        early_stop_count = 0

        for epoch in range(1, self.configs.epochs_cls + 1):
            model_cls.train()
            model.train()
            Riemann_embeds_getter.train()
            features, _ = model(self.features, self.edge_index, self.motif, self.neg_motif, Riemann_embeds_getter)
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer_cls.step()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[1], features, self.labels)
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    early_stop_count = 0
                    best_acc = acc
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_cls:
                    break
        test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[2], features, self.labels)
        return best_acc, test_acc, test_weighted_f1, test_macro_f1
    
    def cal_lp_loss(self, embeddings, decoder, pos_edges, neg_edges):
        pos_scores = decoder(torch.sum((embeddings[pos_edges[0]] - embeddings[pos_edges[1]])**2, -1))
        neg_scores = decoder(torch.sum((embeddings[neg_edges[0]] - embeddings[neg_edges[1]])**2, -1))
        loss = F.binary_cross_entropy(pos_scores.clip(0.01, 0.99), torch.ones_like(pos_scores)) + \
                F.binary_cross_entropy(neg_scores.clip(0.01, 0.99), torch.zeros_like(neg_scores))
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        auc, ap = cal_AUC_AP(preds, label)
        return loss, auc, ap
    
    def train_lp(self, model, Riemann_embeds_getter, r_optim, optimizer, logger):
        val_prop = 0.05
        test_prop = 0.1
        pos_edges, neg_edges = mask_edges(self.edge_index, self.neg_edge, val_prop, test_prop)
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        early_stop_count = 0
        for g in r_optim.param_groups:
            g['lr'] = self.configs.lr_lp
        for epoch in range(1, self.configs.epochs_lp + 1):
            model.train()
            Riemann_embeds_getter.train()
            r_optim.zero_grad()
            optimizer.zero_grad()
            embeddings, _ = model(self.features, self.edge_index, self.motif, self.neg_motif, Riemann_embeds_getter)
            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, decoder, pos_edges[0], neg_edge_train)
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}")
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                Riemann_embeds_getter.eval()
                val_loss, auc, ap = self.cal_lp_loss(embeddings, decoder, pos_edges[1], neg_edges[1])
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_AUC={auc}, val_AP={ap}")
                if ap > best_ap:
                    early_stop_count = 0
                    best_ap = ap
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_lp:
                    break
        test_loss, test_auc, test_ap = self.cal_lp_loss(embeddings, decoder, pos_edges[2], neg_edges[2])
        return test_loss, test_auc, test_ap
            
            