#!/bin/bash

#cd ../..

# custom config
DATA="/wsh/data"
TRAINER=MaPLe

DATASET=$1
SEED=$2
NIMG=$3
LENGTH=$4
DEPTH=$5
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=4


DIR=output/memory_m1/${DATASET}/shots_16/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    /home/wushanghui/anaconda3/envs/maple/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --nimg ${NIMG}\
    --memorylength ${LENGTH}\
    --memorydepth ${DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    /home/wushanghui/anaconda3/envs/maple/bin/python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --nimg ${NIMG}\
    --memorylength ${LENGTH}\
    --memorydepth ${DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
