#!/bin/bash

# Similarity vs SAD, 6-7 splits, Test Partners

#for split_indexes in $(seq 7 9)
#do
    #for single_policy_index in $(seq 0 6)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_test_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_six.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext six \
            #--split_type test
    #done
#done

## Similarity vs SAD, 6-7 splits, Train Partners

#for split_indexes in $(seq 7 9)
#do
    #for single_policy_index in $(seq 0 5)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_train_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_six.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext six \
            #--split_type train
    #done
#done

## Similarity vs SAD, 1-12 splits, Train Partners

#for split_indexes in $(seq 8 12)
#do
    #for single_policy_index in $(seq 0 11)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_one_test_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type test 
    #done
#done

## Similarity vs SAD, 1-12 splits, Test Partners

#for split_indexes in $(seq 8 12)
#do
    #for single_policy_index in $(seq 0 0)
    #do
        #python tools/collect_policy_differences.py \
            #--output_dir acting_comp_one_train_fixed \
            #--rollout_policies agent_groups/all_sad.json \
            #--train_test_splits train_test_splits/sad_splits_one.json \
            #--split_indexes $split_indexes \
            #--single_policy $single_policy_index \
            #--compare_models sad \
            #--base_models br,sba \
            #--num_game 5000 \
            #--num_thread 20 \
            #--seed 0 \
            #--name_ext one \
            #--split_type train
    #done
#done

