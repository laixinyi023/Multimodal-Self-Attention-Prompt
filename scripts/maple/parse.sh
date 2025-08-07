
#!/bin/bash

DataSet=$1


python parse_test_res.py /wsh/MemoryUnit/output/test/memory_m1/test_new/${DataSet}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx --test-log
python parse_test_res.py /wsh/MemoryUnit/output/memory_m1/${DataSet}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx --test-log