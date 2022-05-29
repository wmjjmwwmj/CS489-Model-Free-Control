#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --env_name Humanoid-v2 --seed 42 --method sac --n_step 10000000
CUDA_VISIBLE_DEVICES=0 python train.py --env_name Humanoid-v2 --seed 42 --method sac --is_per --entropy_tuning --n_step 10000000