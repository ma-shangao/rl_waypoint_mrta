#!/bin/bash
# Copyright 2023 MA Song at UCL FRL

# nohup bash -c "source training_scripts/train.bash" &

source .venv/bin/activate
python main.py \
            --train \
            --city_num 100 \
            --clusters_num 5 \
            --batch_size 1024 \
            --model_type moe_mlp \
            --hidden_dim 128 \
            --checkpoint_interval 10
