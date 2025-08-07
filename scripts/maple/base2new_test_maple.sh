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
LOADEP=30
SUB=new


COMMON_DIR=${DATASET}/shots_16/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/memory_m1/${COMMON_DIR}
DIR=output/test/memory_m1/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --nimg ${NIMG}\
    --memorylength ${LENGTH}\
    --memorydepth ${DEPTH} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --nimg ${NIMG}\
    --memorylength ${LENGTH}\
    --memorydepth ${DEPTH} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi