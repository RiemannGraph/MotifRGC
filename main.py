import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
from typing import Union


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='')

# Experiment settings
parser.add_argument('--downstream_task', type=str, default='NC',
                    choices=['NC', 'LP'])
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'airport', 'amazon'])
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--save_embeds', type=str, default="./results/embeds.npy")
parser.add_argument('--log_path', type=str, default="./results/cls_Cora.log")
parser.add_argument('--pre_training', action='store_false')

# Riemannian Embeds
parser.add_argument('--num_factors', type=int, default=1, help='number of product factors')
parser.add_argument('--dimensions', type=int, nargs='+', default=[8], help='dimension of Riemannian embedding')
parser.add_argument('--d_embeds', type=int, default=8, help='dimension of laplacian features')
# parser.add_argument('--d_free', type=int, default=2, help='dimension of rotational factor')
parser.add_argument('--init_curvature', type=float, default=-1.0, help='initial curvature')
parser.add_argument('--learnable', action='store_false')

# Contrastive Learning Module
parser.add_argument('--backbone', type=str, default='gat', choices=['gcn', 'gat', 'sage'])
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--hidden_features', type=int, default=64)
parser.add_argument('--embed_features', type=int, default=32, help='dimensions of graph embedding')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--drop_node', type=float, default=0.0)
parser.add_argument('--drop_edge', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_Riemann', type=float, default=0.001)
parser.add_argument('--w_decay', type=float, default=0.)
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of contrastive loss')

# Node Classification
parser.add_argument('--drop_cls', type=float, default=0.5)
parser.add_argument('--drop_edge_cls', type=float, default=0.0)
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--w_decay_cls', type=float, default=0.0)
parser.add_argument('--epochs_cls', type=int, default=500)
parser.add_argument('--patience_cls', type=int, default=30)
parser.add_argument('--save_path_cls', type=str, default='./checkpoints/cls.pth')

# Link Prediction
parser.add_argument('--lr_lp', type=float, default=0.001)
parser.add_argument('--w_decay_lp', type=float, default=0)
parser.add_argument('--epochs_lp', type=int, default=100)
parser.add_argument('--patience_lp', type=int, default=3)
parser.add_argument('--save_path_lp', type=str, default='./checkpoints/cls.pth')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

configs = parser.parse_args()
if len(configs.dimensions) == 1:
    configs.dimensions = configs.dimensions[0]
results_dir = f"./results/{configs.version}"
log_path = f"{results_dir}/{configs.downstream_task}_{configs.backbone}_{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
configs.save_embeds = f"{results_dir}/embeds.npy"
print(f"Log path: {configs.log_path}; embeds path: {configs.save_embeds}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()

