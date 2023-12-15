#!/bin/bash

python3 main.py \
--downstream_task NC  \
--dataset Citeseer  \
--root_path ./datasets  \
--eval_freq 5  \
--exp_iters 5 \
--num_factors 2 \
--dimensions 8  \
--d_embeds  8 \
--init_curvature  -1.0  \
--backbone  sage \
--epochs  200 \
--hidden_features 64  \
--embed_features  32  \
--n_layers  2 \
--drop_node 0.6 \
--drop_edge 0.4 \
--lr  1e-2  \
--lr_Riemann  1e-3 \
--w_decay 0.0 \
--n_heads 8 \
--t 1.0 \
--r 2.0 \
--temperature 0.2 \
--drop_cls  0.8 \
--drop_edge_cls 0.8 \
--lr_cls  0.001  \
--w_decay_cls 5e-4 \
--hidden_features_cls 32  \
--epochs_cls  500 \
--patience_cls  50  \
--lr_lp  1e-4  \
--w_decay_lp 5e-4 \
--epochs_lp  200 \
--patience_lp  10
# --learnable true  \   # If curvature is not learnable
# --pre_training \  #If don't pre-training 