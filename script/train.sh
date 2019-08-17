#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --config configs/fcn8s_pascal.yml