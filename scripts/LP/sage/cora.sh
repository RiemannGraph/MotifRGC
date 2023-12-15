#!/bin/bash

python3 main.py \
--downstream_task LP  \
--dataset Cora  \
--root_path ./datasets  \
--eval_freq 50  \
--exp_iters 5 \
--num_factors 3 \
--dimensions 8  \
--d_embeds  8 \
--init_curvature  -1.0  \
--backbone  sage \
--epochs  200 \
--hidden_features 64  \
--embed_features  32  \
--n_layers  2 \
--drop_node 0.5 \
--drop_edge 0.0 \
--lr  0.01  \
--lr_Riemann  0.001 \
--w_decay 0.0 \
--n_heads 4 \
--t 1.0 \
--r 2.0 \
--temperature 0.2 \
--lr_lp  1e-4  \
--w_decay_lp 0.0 \
--epochs_lp  200 \
--patience_lp  10
# --learnable true  \   # If curvature is not learnable
# --pre_training \  #If don't pre-training 