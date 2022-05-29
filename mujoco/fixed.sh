#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py --env_name HalfCheetah-v2 --seed 42 --method sac --n_step 3000000
CUDA_VISIBLE_DEVICES=1 python train.py --env_name Ant-v2 --seed 42 --method sac --n_step 3000000
CUDA_VISIBLE_DEVICES=1 python train.py --env_name Hopper-v2 --seed 42 --method sac --n_step 1000000