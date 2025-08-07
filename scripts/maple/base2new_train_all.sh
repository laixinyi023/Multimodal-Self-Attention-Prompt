#!/bin/bash

#cd ../..

# custom config
DATASET=$1


rm -rf /wsh/MemoryUnit/output/memory_m1
rm -rf /wsh/MemoryUnit/output/test

for SEED in 1 2 3
do
    rm -rf /wsh/MemoryUnit/output/memory_m1/${DATASET}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/seed${SEED}/log.txt
    bash scripts/maple/base2new_train_maple.sh ${DATASET} ${SEED} 3 4 11
    bash scripts/maple/base2new_test_maple.sh ${DATASET} ${SEED} 3 4 11
done

bash scripts/maple/parse.sh ${DATASET}

