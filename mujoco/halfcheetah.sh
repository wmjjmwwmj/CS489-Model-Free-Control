#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --env_name HalfCheetah-v2 --seed 42 --method ppo --n_step 3000000
CUDA_VISIBLE_DEVICES=1 python train.py --env_name HalfCheetah-v2 --seed 42 --method sac --n_step 3000000
CUDA_VISIBLE_DEVICES=1 python train.py --env_name HalfCheetah-v2 --seed 42 --method sac --is_per True --n_step 3000000