#!/bin/bash


for split_indexes in $(seq 1 4)
do
    for single_policy_index in $(seq 0 6)
    do
        python tools/collect_policy_differences.py \
            --output_dir similarity_data_vs_op \
            --rollout_policies agent_groups/all_sad.json \
            --train_test_splits sad_train_test_splits.json \
            --split_indexes $split_indexes \
            --single_policy $single_policy_index \
            --compare_models br,sba,op \
            --compare_as_base op \
            --num_game 5000 \
            --num_thread 20 \
            --seed 0 
    done
done
