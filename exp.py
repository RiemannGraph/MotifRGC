import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from models import RiemannianFeatures, Model, LinearClassifier, GCNClassifier, CLLoss, MotifLoss
from utils import cal_accuracy, cal_F1
from data_factory import load_data
from sklearn.cluster import KMeans
from logger import create_logger
from geoopt.optim import RiemannianAdam


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def pretrain(self, model, Riemann_embeds_getter, criterion, epochs, logger, r_optim, optimizer):
        for epoch in range(1, epochs + 1):
            model.train()
            Riemann_embeds_getter.train()
            embeds, x = model(self.features, self.edge_index, Riemann_embeds_getter)
            loss = criterion(embeds, x, self.motif)
            r_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}")

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        features, in_features, labels, edge_index, motif, masks, n_classes = load_data(self.configs.root_path, self.configs.dataset)
        edge_index = edge_index.to(device)
        motif = motif.to(device)
        features = features.to(device)
        labels = labels.to(device)
        self.edge_index = edge_index
        self.motif = motif
        self.features = features

        if self.configs.downstream_task == "NC":
            vals = []
            accs = []
            wf1s = []
            mf1s = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            Riemann_embeds_getter = RiemannianFeatures(features.shape[0], self.configs.dimensions, self.configs.d_free,
                                                       self.configs.init_curvature, self.configs.num_factors).to(device)
            model = Model(backbone=self.configs.backbone, n_layers=self.configs.n_layers, in_features=in_features,
                          embed_features=self.configs.embed_features, hidden_features=self.configs.hidden_features,
                          n_heads=self.configs.n_heads, d_free=self.configs.d_free, drop_edge=self.configs.drop_edge, drop_node=self.configs.drop_edge,
                          num_factors=self.configs.num_factors, dimensions=self.configs.dimensions, d_embeds=self.configs.d_embeds
                          ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay,
                                     stabilize=100)

            criterion_cl = CLLoss(t=self.configs.temperature)
            criterion_motif = MotifLoss(d_embeds=self.configs.d_embeds)


            logger.info("--------------------------Training Start-------------------------")
            self.pretrain(model, Riemann_embeds_getter, criterion_cl, self.configs.epochs, logger, r_optim, optimizer)
            self.pretrain(model, Riemann_embeds_getter, criterion_motif, self.configs.epochs, logger, r_optim, optimizer)

            if self.configs.downstream_task == 'NC':
                model.eval()
                Riemann_embeds_getter.eval()
                best_val, test_acc, test_weighted_f1, test_macro_f1 = self.train_cls(products.detach(), edge_index, labels, n_classes, masks)
                logger.info(
                    f"Epoch {epoch}: val_accuracy={best_val.item() * 100: .2f}%, test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"\t\t weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                vals.append(best_val.item())
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)
            elif self.configs.downstream_task == 'LP':
                pass
            else:
                raise NotImplementedError

        if self.configs.downstream_task == "NC":
            logger.info(f"valid results: {np.mean(vals)}~{np.std(vals)}")
            logger.info(f"test results: {np.mean(accs)}~{np.std(accs)}")
            logger.info(f"test weighted-f1: {np.mean(wf1s)}~{np.std(wf1s)}")
            logger.info(f"test macro-f1: {np.mean(mf1s)}~{np.std(mf1s)}")

    def cal_cls_loss(self, model, edge_index, mask, features, labels):
        out = model(features, edge_index)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, features, edge_index, labels, n_classes, masks):
        """masks = (train, val, test)"""
        device = self.device
        # model_cls = LinearClassifier(2*self.configs.embed_features, n_classes, drop=self.configs.drop_cls).to(device)
        model_cls = GCNClassifier(2*self.configs.embed_features, n_classes, drop=self.configs.drop_cls).to(device)
        optimizer = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)

        best_acc = 0.
        early_stop_count = 0
        best_model = None

        for epoch in range(1, self.configs.epochs_cls + 1):
            model_cls.train()
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, edge_index, masks[0], features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, edge_index, masks[1], features, labels)
                print(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    early_stop_count = 0
                    best_acc = acc
                    best_model = model_cls
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_cls:
                    break
        best_model.eval()
        test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model_cls, edge_index, masks[2], features, labels)
        return best_acc, test_acc, test_weighted_f1, test_macro_f1
