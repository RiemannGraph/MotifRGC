import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from models import RiemannianFeatures, Model, LinearClassifier
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

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        features, in_features, labels, edge_index, motif, masks, n_classes = load_data(self.configs.root_path, self.configs.dataset)

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
                          num_factors=self.configs.num_factors, dimensions=self.configs.dimensions, d_embeds=self.configs.d_embeds,
                          temperature=self.configs.temperature).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay,
                                     stabilize=100)

            edge_index = edge_index.to(device)
            motif = motif.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logger.info("--------------------------Training Start-------------------------")
            for epoch in range(1, self.configs.epochs + 1):
                model.train()
                Riemann_embeds_getter.train()
                products, loss = model(features, edge_index, motif, Riemann_embeds_getter)
                r_optim.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                r_optim.step()
                optimizer.step()
                logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
            if self.configs.downstream_task == 'NC':
                model.eval()
                Riemann_embeds_getter.eval()
                best_val, test_acc, test_weighted_f1, test_macro_f1 = self.train_cls(products.detach(), labels, n_classes, masks, [r_optim, optimizer])
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

    def cal_cls_loss(self, model, mask, features, labels):
        out = model(features)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, features, labels, n_classes, masks, optims):
        """masks = (train, val, test)"""
        device = self.device
        model = LinearClassifier(self.configs.embed_features, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)

        best_acc = 0.
        early_stop_count = 0
        best_model = None

        for epoch in range(1, self.configs.epochs_cls + 1):
            model.train()
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model, masks[0], features, labels)
            optimizer.zero_grad()
            for optim in optims:
                optim.zero_grad()
            loss.backward()
            optimizer.step()
            for optim in optims:
                optim.step()
            print(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")

            if epoch % self.configs.eval_freq == 0:
                model.eval()
                val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model, masks[1], features, labels)
                print(f"Epoch {epoch}: val_loss={val_loss.item()}, val_accuracy={acc}")
                if acc > best_acc:
                    early_stop_count = 0
                    best_acc = acc
                    best_model = model
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_cls:
                    break
        best_model.eval()
        test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model, masks[2], features, labels)
        return best_acc, test_acc, test_weighted_f1, test_macro_f1
