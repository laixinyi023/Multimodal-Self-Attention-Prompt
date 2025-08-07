#!/bin/bash

#cd ../..

# custom config
DATA="/wsh/data"
TRAINER=MaPLe

DATASET=$1
SEED1=1
SEED2=2
SEED3=3

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
SUB=new


COMMON_DIR1=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED1}
COMMON_DIR2=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED2}
COMMON_DIR3=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED3}

MODEL_DIR1=output/memory_m1/${COMMON_DIR1}
MODEL_DIR2=output/memory_m1/${COMMON_DIR2}
MODEL_DIR3=output/memory_m1/${COMMON_DIR3}

DIR1=output/test/memory_m1/test_${SUB}/${COMMON_DIR1}
DIR2=output/test/memory_m1/test_${SUB}/${COMMON_DIR2}
DIR3=output/test/memory_m1/test_${SUB}/${COMMON_DIR3}
if [ -d "$DIR1" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR1}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED1} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR1} \
    --model-dir ${MODEL_DIR1} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR1}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED1} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR1} \
    --model-dir ${MODEL_DIR1} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi



if [ -d "$DIR2" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR2}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED2} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR2} \
    --model-dir ${MODEL_DIR2} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED2} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR2} \
    --model-dir ${MODEL_DIR2} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi



if [ -d "$DIR3" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR3}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED3} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR3} \
    --model-dir ${MODEL_DIR3} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR3}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED3} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR3} \
    --model-dir ${MODEL_DIR3} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi