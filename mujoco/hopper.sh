#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python train.py --env_name Hopper-v2 --seed 42 --method ppo --n_step 1000000
CUDA_VISIBLE_DEVICES=2 python train.py --env_name Hopper-v2 --seed 42 --method sac --n_step 1000000
CUDA_VISIBLE_DEVICES=2 python train.py --env_name Hopper-v2 --seed 42 --method sac --is_per True --n_step 1000000
