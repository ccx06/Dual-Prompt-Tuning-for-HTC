#!/bin/bash
PYTHON_BIN="/root/anaconda3/envs/py37_t110/bin/python"

DATE=$(date "+%Y%m%d%H")
LAMBDA1=0.5
LAMBDA2=0.2
ALPHA=0.6
BETA=0.1
NALPHA=`awk -v a="${ALPHA}" 'BEGIN{printf "%.1f\n", 1-a}'`

CUDA_VISIBLE_DEVICES=2,3 ${PYTHON_BIN} src/train_dpt.py \
    --lr 3e-5 \
    --data_dir /mnt/home/ExploreProject/DPT/data/BGC \
    --data_name bgc \
    --batch 32 \
    --update 1 \
    --mlmloss_weight 1 \
    --arch /mnt/home/pretrained_models/bert-base-uncased-bgc \
    --cont_loss_weight ${LAMBDA1} \
    --cont_negative_sample_mode hard \
    --cont_negative_num_list 2 5 8 3 \
    --cont_neg_part_weight ${NALPHA} \
    --cont_use_rank 1 \
    --cont_rank_loss_weight ${BETA} \
    --multitask_loss_weight ${LAMBDA2} \
    --seed 42 \
    --exp_name lambda1-${LAMBDA1}_lambda2-${LAMBDA2}_alpha-${ALPHA}_beta-${BETA}_${DATE}