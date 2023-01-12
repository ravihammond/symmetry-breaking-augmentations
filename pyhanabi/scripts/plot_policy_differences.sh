#!/bin/bash

#python tools/plot_policy_differences.py \
    #--load_dir similarity_data_vs_sad \
    #--compare_models br,sba,obl,rand \
    #--train_test_splits sad_train_test_splits.json \
    #--split_indexes 0,1,2,3,4 \

#python tools/plot_policy_differences.py \
    #--load_dir similarity_data_vs_obl \
    #--compare_models br,sba \
    #--train_test_splits sad_train_test_splits.json \
    #--split_indexes 0,1,2,3,4 \
    #--title "OBL Policy Differences" \
    #--ylabel "Similarities vs OBL"

python tools/plot_policy_differences.py \
    --load_dir similarity_data_vs_op \
    --compare_models br,sba \
    --train_test_splits sad_train_test_splits.json \
    --split_indexes 0,1,2,3,4 \
    --title "OP Policy Differences" \
    --ylabel "Similarities vs OP"
