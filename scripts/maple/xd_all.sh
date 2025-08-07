
rm -rf /wsh/MemoryUnit/output/imagenet
rm -rf /wsh/MemoryUnit/output/evaluation
#
for SEED in 1 2 3
do
    bash scripts/maple/xd_train_maple.sh imagenet ${SEED}
    bash scripts/maple/xd_test_maple.sh caltech101 ${SEED}
    bash scripts/maple/xd_test_maple.sh oxford_pets ${SEED}
    bash scripts/maple/xd_test_maple.sh stanford_cars ${SEED}
    bash scripts/maple/xd_test_maple.sh oxford_flowers ${SEED}
    bash scripts/maple/xd_test_maple.sh food101 ${SEED}
    bash scripts/maple/xd_test_maple.sh fgvc_aircraft ${SEED}
    bash scripts/maple/xd_test_maple.sh sun397 ${SEED}
    bash scripts/maple/xd_test_maple.sh dtd ${SEED}
    bash scripts/maple/xd_test_maple.sh eurosat ${SEED}
    bash scripts/maple/xd_test_maple.sh ucf101 ${SEED}
done
python parse_test_res.py /wsh/MemoryUnit/output/imagenet/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/caltech101 --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/oxford_pets --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/stanford_cars --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/oxford_flowers --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/food101 --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/fgvc_aircraft --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/sun397 --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/dtd --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/eurosat --test-log
python parse_test_res.py /wsh/MemoryUnit/output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/ucf101 --test-log
