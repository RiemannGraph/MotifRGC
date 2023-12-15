#!/bin/bash

python3 main.py \
--downstream_task LP  \
--dataset Pubmed  \
--root_path ./datasets  \
--eval_freq 50  \
--exp_iters 5 \
--num_factors 1 \
--dimensions 8  \
--d_embeds  8 \
--init_curvature  -3.5  \
--backbone  sage \
--epochs  200 \
--hidden_features 32  \
--embed_features  16  \
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
--lr_lp  1e-3  \
--w_decay_lp 0.0 \
--epochs_lp  300 \
--patience_lp  10   \
# --pre_training