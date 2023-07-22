import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def cal_F1(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    weighted_f1 = f1_score(trues, preds, average='weighted')
    macro_f1 = f1_score(trues, preds, average='macro')
    return weighted_f1, macro_f1