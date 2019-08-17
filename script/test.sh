#!/usr/bin/env bash
foldNo="dilated_fcn_fold0"

CUDA_VISIBLE_DEVICES=0 python -u fewshot_imprinted_finetune.py --config /group/xiangyi/Pascal/runs/dilatedfcn8s_pascal/dilated_fcn_fold0/fcn8s_pascal.yml \
--model_path /group/xiangyi/Pascal/runs/dilatedfcn8s_pascal/$foldNo/dilated_fcn8s_pascal_best_model.pkl --binary 2 \
--out_dir /p300/AdaptiveMaskedProxies/Output/releasedModel/oneShot/$foldNo/

