#!/bin/bash

#cd ../..

# custom config

DataSet=$1
Device=$2

for SEED in 1 2 3
do
    bash scripts/maple/base2new_train_maple.sh ${DataSet} ${SEED} ${Device}
done
