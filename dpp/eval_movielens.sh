#!/bin/bash

python3 eval_movielens.py --dataset_path ../data/ \
        --dataset movielens1M --device cuda --cuda_core 1 \
        --model_path DPP_movielens1M.pt --batch_size 32