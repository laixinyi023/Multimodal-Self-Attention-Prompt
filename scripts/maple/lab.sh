#!/bin/bash

#cd ../..

# custom config

#
rm -rf /wsh/MemoryUnit/output/memory_m1
rm -rf /wsh/MemoryUnit/output/test
for DEPTH in 11
do
  for LENGTH in 4
  do
    for DATASET in caltech101 imagenet  oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101
    do
      echo "************************************"
      echo "DEPTH:"${DEPTH}"LENGTH:"${LENGTH}   "DATASET:"${DATASET}
      echo "************************************"
        for SEED in 1 2 3
        do
        bash scripts/maple/base2new_train_maple.sh ${DATASET} ${SEED} 3 ${LENGTH} ${DEPTH}
        bash scripts/maple/base2new_test_maple.sh ${DATASET} ${SEED} 3 ${LENGTH} ${DEPTH}
        done
        bash scripts/maple/parse.sh ${DATASET}
#        rm -rf /wsh/MemoryUnit/output/memory_m1
#        rm -rf /wsh/MemoryUnit/output/test
    done
  done
done


