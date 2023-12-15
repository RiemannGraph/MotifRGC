#!/bin/bash

python3 main.py \
--downstream_task NC  \
--dataset airport  \
--root_path ./datasets  \
--eval_freq 5  \
--exp_iters 5 \
--num_factors 3 \
--dimensions 8  \
--d_embeds  8 \
--init_curvature  -3.0  \
--backbone  gcn \
--epochs  200 \
--hidden_features 64  \
--embed_features  32  \
--n_layers  2 \
--drop_node 0.0 \
--drop_edge 0.0 \
--lr  0.001  \
--lr_Riemann  0.001 \
--w_decay 0.0 \
--n_heads 8 \
--t 1.0 \
--r 2.0 \
--temperature 0.2 \
--drop_cls  0.0 \
--drop_edge_cls 0.0 \
--lr_cls  0.01  \
--w_decay_cls 5e-5 \
--hidden_features_cls 32  \
--epochs_cls  500 \
--patience_cls  10  \
--lr_lp  1e-4  \
--w_decay_lp 0.0 \
--epochs_lp  300 \
--patience_lp  10
# --learnable true  \   # If curvature is not learnable
# --pre_training \  #If don't pre-training 