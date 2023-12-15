import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import RiemannianFeatures, Model, FermiDiracDecoder
from backbone import GNNClassifier
from utils import cal_accuracy, cal_F1, cal_AUC_AP
from data_factory import load_data, mask_edges
from logger import create_logger
from geoopt.optim import RiemannianAdam
import time
import os


class Exp:
    def __init__(self, configs):
        self.configs = configs
        self.val_prop = 0.05
        self.test_prop = 0.1
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def pretrain(self, model, Riemann_embeds_getter, epochs, logger, r_optim, optimizer):
        pos_motifs, neg_motifs = mask_edges(self.motif, self.neg_motif, 0.05, 0.1)
        pos_edges, neg_edges = mask_edges(self.edge_index, self.neg_edge, 0.05, 0.1)
        neg_motif_train = neg_motifs[0][:, np.random.randint(0, neg_motifs[0].shape[1], pos_motifs[0].shape[1])]
        for epoch in range(1, epochs + 1):
            model.train()
            Riemann_embeds_getter.train()
            embeds, loss = model(self.features, pos_edges[0], pos_motifs[0], neg_motif_train, Riemann_embeds_getter)
            r_optim.zero_grad()
            optimizer.zero_grad()   
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
        torch.save(model.state_dict(), 'pretrain.pt')

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
            if os.path.exists('pretrain.pt') and self.configs.pre_training:
                print("Loading Pretrained model")
                model.load_state_dict(torch.load('pretrain.pt'))
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
            elif self.configs.downstream_task == 'Motif':
                _, test_auc, test_ap = self.train_motif(model, Riemann_embeds_getter, r_optim, optimizer, logger)
                logger.info(
                    f"test_auc={test_auc * 100: .2f}%, test_ap={test_ap * 100: .2f}%")
                aucs.append(test_auc)
                aps.append(test_ap)
            else:
                raise NotImplementedError

        if self.configs.downstream_task == "NC":
            logger.info(f"valid results: {np.mean(vals)}~{np.std(vals)}")
            logger.info(f"best test ACC: {np.max(accs)}")
            logger.info(f"test results: {np.mean(accs)}~{np.std(accs)}")
            logger.info(f"test weighted-f1: {np.mean(wf1s)}~{np.std(wf1s)}")
            logger.info(f"test macro-f1: {np.mean(mf1s)}~{np.std(mf1s)}")
        elif self.configs.downstream_task == "LP" or self.configs.downstream_task == 'Motif':
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
        d = (self.configs.num_factors+1)*self.configs.embed_features
        model_cls = GNNClassifier(backbone=self.configs.backbone, n_layers=2, in_features=self.in_features+d,
                                  hidden_features=self.configs.hidden_features_cls, out_features=self.n_classes,
                                  n_heads=self.configs.n_heads, drop_edge=self.configs.drop_edge_cls, 
                                  drop_node=self.configs.drop_cls).to(device)
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
        r_optim = RiemannianAdam(Riemann_embeds_getter.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        best_acc = 0.
        early_stop_count = 0
        time_before_train = time.time()
        all_times = []
        for epoch in range(1, self.configs.epochs_cls + 1):
            now_time = time.time()
            model_cls.train()
            model.train()
            Riemann_embeds_getter.train()
            features, _ = model(self.features, self.edge_index, self.motif, self.neg_motif, Riemann_embeds_getter)
            features = torch.concat([self.features, features.detach()], -1)
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
            optimizer_cls.zero_grad()
            optimizer.zero_grad()
            r_optim.zero_grad()
            loss.backward()
            optimizer_cls.step()
            optimizer.step()
            r_optim.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}, time={time.time()-now_time}")
            all_times.append(time.time() - now_time)

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
        avg_train_time = np.mean(all_times)
        time_str = f"Average Time: {avg_train_time} s/epoch"
        logger.info(time_str)
        time_str = f"{self.configs.downstream_task}_{self.configs.dataset}_{time_str}\n"
        with open('time.txt', 'a') as f:
            f.write(time_str)
        f.close()
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
        pos_motifs, neg_motifs = mask_edges(self.motif, self.neg_motif, val_prop, test_prop)
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        early_stop_count = 0
        for g in r_optim.param_groups:
            g['lr'] = self.configs.lr_lp
        time_before_train = time.time()
        for epoch in range(1, self.configs.epochs_lp + 1):
            t = time.time()
            model.train()
            Riemann_embeds_getter.train()
            r_optim.zero_grad()
            optimizer.zero_grad()
            neg_motif_train = neg_motifs[0][:, np.random.randint(0, neg_motifs[0].shape[1], pos_motifs[0].shape[1])]
            embeddings, _ = model(self.features, pos_edges[0], pos_motifs[0], neg_motif_train, Riemann_embeds_getter)
            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, decoder, pos_edges[0], neg_edge_train)
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - t}")
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                Riemann_embeds_getter.eval()
                val_loss, auc, ap = self.cal_lp_loss(embeddings, decoder, pos_edges[1], neg_edges[1])
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_AUC={auc}, val_AP={ap}")
                if ap > best_ap:
                    early_stop_count = 0
                    best_ap = ap
                    embeds = embeddings.detach().cpu().numpy()
                    np.save(self.configs.save_embeds, embeds)
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_lp:
                    break
        avg_train_time = (time.time() - time_before_train) / epoch
        time_str = f"Average Time: {avg_train_time} s/epoch"
        logger.info(time_str)
        time_str = f"{self.configs.downstream_task}_{self.configs.dataset}_{time_str}\n"
        with open('time.txt', 'a') as f:
            f.write(time_str)
        f.close()
        test_loss, test_auc, test_ap = self.cal_lp_loss(embeddings, decoder, pos_edges[2], neg_edges[2])
        return test_loss, test_auc, test_ap
            
            
    def cal_motif_loss(self, embeddings, decoder, pos_motifs, neg_motifs):
            from sklearn.metrics import roc_auc_score, average_precision_score
            pos_scores1 = decoder(torch.sum((embeddings[pos_motifs[0]] - embeddings[pos_motifs[1]])**2, -1))
            pos_scores2 = decoder(torch.sum((embeddings[pos_motifs[2]] - embeddings[pos_motifs[1]])**2, -1))
            pos_scores3 = decoder(torch.sum((embeddings[pos_motifs[2]] - embeddings[pos_motifs[0]])**2, -1))
            
            neg_scores1 = decoder(torch.sum((embeddings[neg_motifs[0]] - embeddings[neg_motifs[1]])**2, -1))
            neg_scores2 = decoder(torch.sum((embeddings[neg_motifs[2]] - embeddings[neg_motifs[1]])**2, -1))
            neg_scores3 = decoder(torch.sum((embeddings[neg_motifs[2]] - embeddings[neg_motifs[0]])**2, -1))
            
            pos1 = pos_scores1 * pos_scores2 * (1 - pos_scores3)
            pos2 = pos_scores1 * pos_scores2 * pos_scores3
            pos0 = 1 - pos1 - pos2
            pos = torch.stack([pos0, pos1, pos2], dim=1)
            p_y = pos_motifs[-1].detach() + 1
            
            neg1 = neg_scores1 * neg_scores2 * (1 - neg_scores3)
            neg2 = neg_scores1 * neg_scores2 * neg_scores3
            neg0 = 1 - neg1 - neg2
            neg = torch.stack([neg0, neg1, neg2], dim=1)
            n_y = torch.zeros_like(p_y)
            
            probs = torch.concat([pos, neg], dim=0)
            label = torch.concat([p_y, n_y])
            
            loss = F.nll_loss(torch.log(probs + 1e-5), label)

            preds = probs.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            auc = roc_auc_score(label, preds, multi_class='ovo')
            aps = []
            for i in range(3):
                y = (label == i).astype(int)
                aps.append(average_precision_score(y, preds[:, i], average='macro'))
            ap = np.mean(aps)
            return loss, auc, ap
    
    def train_motif(self, model, Riemann_embeds_getter, r_optim, optimizer, logger):
        val_prop = 0.05
        test_prop = 0.1
        pos_motifs, neg_motifs = mask_edges(self.motif, self.neg_motif, val_prop, test_prop)
        pos_edges, neg_edges = mask_edges(self.edge_index, self.neg_edge, val_prop, test_prop)
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        early_stop_count = 0
        for g in r_optim.param_groups:
            g['lr'] = self.configs.lr_lp
        time_before_train = time.time()
        for epoch in range(1, self.configs.epochs_lp + 1):
            model.train()
            Riemann_embeds_getter.train()
            r_optim.zero_grad()
            optimizer.zero_grad()
            neg_motif_train = neg_motifs[0][:, np.random.randint(0, neg_motifs[0].shape[1], pos_motifs[0].shape[1])]
            embeddings, _ = model(self.features, pos_edges[0], pos_motifs[0], neg_motif_train, Riemann_embeds_getter)
            loss, auc, ap = self.cal_motif_loss(embeddings, decoder, pos_motifs[0], neg_motif_train)
            loss.backward()
            r_optim.step()
            optimizer.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}")
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                Riemann_embeds_getter.eval()
                val_loss, auc, ap = self.cal_motif_loss(embeddings, decoder, pos_motifs[1], neg_motifs[1])
                logger.info(f"Epoch {epoch}: val_loss={val_loss.item()}, val_AUC={auc}, val_AP={ap}")
                if ap > best_ap:
                    early_stop_count = 0
                    best_ap = ap
                else:
                    early_stop_count += 1
                if early_stop_count >= self.configs.patience_lp:
                    break
        avg_train_time = (time.time() - time_before_train) / epoch
        time_str = f"Average Time: {avg_train_time} s/epoch"
        logger.info(time_str)
        time_str = f"{self.configs.downstream_task}_{self.configs.dataset}_{time_str}\n"
        with open('time.txt', 'a') as f:
            f.write(time_str)
        f.close()
        test_loss, test_auc, test_ap = self.cal_motif_loss(embeddings, decoder, pos_motifs[2], neg_motifs[2])
        return test_loss, test_auc, test_ap            
            