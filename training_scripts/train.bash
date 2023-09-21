#!/bin/bash
source .venv/bin/activate
python main.py \
            --train \
            --city_num 100 \
            --clusters_num 5 \
            --batch_size 1024 \
            --model_type moe_mlp \
            --hidden_dim 128 \
            --n_component 1 \
            --checkpoint_interval 10
