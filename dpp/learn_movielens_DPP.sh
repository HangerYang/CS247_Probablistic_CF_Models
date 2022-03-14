#!/bin/bash


cluster_size=1
component_num=1
dataset="movielens1M"

echo "PGC Learning on ${dataset}"
python3.9 train.py --dataset_path ../data \
    --dataset $dataset --device cuda --cuda_core 0 --model DPP --max_epoch 50 \
    --batch_size 128 --lr 0.001 --weight_decay 0.0 \
    --max_cluster_size $cluster_size --component_num $component_num \
    --log_file "logs/DPP_${cluster_size}_${component_num}_${dataset}_log.txt" \
    --output_model_file "DPP_movielens1M.pt"
