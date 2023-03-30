#!/bin/bash

#python tools/plot_policy_differences.py \
    #--load_dir acting_comp_test_fixed \
    #--sad_dir sad_similarities_fixed \
    #--compare_models br,sba,sad \
    #--train_test_splits train_test_splits/sad_splits_six.json \
    #--split_indexes 0,1,2,3,4,5,6,7,8,9 \
    #--sad_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    #--name_ext six \
    #--title "SAD Test Set Similarities, 6-7 Splits" \
    #--ylabel "Similarity vs SAD" 

#python tools/plot_policy_differences.py \
    #--load_dir acting_comp_train_fixed \
    #--data_type train \
    #--compare_models br,sba \
    #--train_test_splits train_test_splits/sad_splits_six.json \
    #--split_indexes 0,1,2,3,4,5,6,7,8,9 \
    #--name_ext six \
    #--title "SAD Train Set Similarities, 6-7 Splits" \
    #--ylabel "Similarity vs SAD" 

#python tools/plot_policy_differences.py \
    #--load_dir acting_comp_one_test_fixed \
    #--sad_dir sad_similarities \
    #--name_ext one \
    #--compare_models br,sba,sad \
    #--train_test_splits train_test_splits/sad_splits_one.json \
    #--split_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    #--sad_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    #--title "SAD Test Set Similarities, 1-12 Splits" \
    #--ylabel "Similarity vs SAD" \

#python tools/plot_policy_differences.py \
    #--load_dir acting_comp_one_train \
    #--sad_dir sad_similarities \
    #--name_ext one \
    #--compare_models br,sba \
    #--train_test_splits train_test_splits/sad_splits_one.json \
    #--split_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    #--title "SAD Train Set Similarities, 1-12 Splits" \
    #--ylabel "Similarity vs SAD" \

# =============================================

python tools/plot_policy_differences.py \
    --load_dir acting_comp_obl_test_fixed \
    --data_type test \
    --compare_models br,sba \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --split_indexes 0,1,2,3,4,5,6,7,8,9 \
    --name_ext six \
    --title "All OBL Test Set Similarities, 6-7 Splits" \
    --ylabel "Similarity vs OBL" \

python tools/plot_policy_differences.py \
    --load_dir acting_comp_obl_train_fixed \
    --data_type train \
    --compare_models br,sba \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --split_indexes 0,1,2,3,4,5,6,7,8,9 \
    --name_ext six \
    --title "All OBl Train Set Similarities, 6-7 Splits" \
    --ylabel "Similarity vs OBl" 

python tools/plot_policy_differences.py \
    --load_dir acting_comp_obl_one_test_fixed \
    --data_type test \
    --name_ext one \
    --compare_models br,sba \
    --train_test_splits train_test_splits/sad_splits_one.json \
    --split_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --title "OBL Test Set Similarities, 1-12 Splits" \
    --ylabel "Similarity vs OBL" \

python tools/plot_policy_differences.py \
    --load_dir acting_comp_obl_one_train_fixed \
    --data_type train \
    --name_ext one \
    --compare_models br,sba \
    --train_test_splits train_test_splits/sad_splits_one.json \
    --split_indexes 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --title "OBL Train Set Similarities, 1-12 Splits" \
    --ylabel "Similarity vs OBL" \

# =============================================

#python tools/plot_policy_differences.py \
    #--load_dir similarity_data_vs_sad \
    #--compare_models br,sba \
    #--train_test_splits train_test_splits/sad_splits_six.json \
    #--split_indexes 0 \
    #--title "SAD Policy Differences" \
    #--ylabel "Similarity vs SAD"

#python tools/plot_policy_differences.py \
    #--load_dir similarity_data_vs_obl \
    #--compare_models br,sba \
    #--train_test_splits sad_train_test_splits.json \
    #--split_indexes 0,1,2,3,4 \
    #--title "OBL Policy Differences" \
    #--ylabel "Similarities vs OBL"

#python tools/plot_policy_differences.py \
    #--load_dir similarity_data_vs_op \
    #--compare_models br,sba \
    #--train_test_splits sad_train_test_splits.json \
    #--split_indexes 0,1,2,3,4 \
    #--title "OP Policy Differences" \
    #--ylabel "Similarities vs OP"
