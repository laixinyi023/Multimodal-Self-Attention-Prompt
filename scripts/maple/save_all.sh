#!/bin/bash

#cd ../..

# custom config
DATASET=$1



for ((i=1;i<=100;i++))
  do
      bash scripts/maple/base2new_train_maple.sh ${DATASET} ${i}
#      bash scripts/maple/base2new_test_maple.sh ${DATASET} ${i}
  done

#bash scripts/maple/parse.sh ${DATASET

#rm -rf /wsh/MemoryUnit/output/memory_m1
#rm -rf /wsh/MemoryUnit/output/test