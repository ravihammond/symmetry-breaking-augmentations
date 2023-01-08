#!/bin/bash

python tools/collect_policy_differences.py \
    --train_test_splits sad_train_test_splits.json \
    --split_indexes 0,1,2,3,4,5 \
