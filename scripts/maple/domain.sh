

for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh imagenetv2 ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_sketch ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_a ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_r ${SEED}
done