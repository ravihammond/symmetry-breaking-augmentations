#!/bin/bash

#python eval_convex_hull.py \
    #--models agent_groups/all_sad.json \
    #--splits train_test_splits/sad_splits_six.json \
    #--sad_legacy 1 \
    #--num_game 1 \
    #--num_thread 10 \
    #--num_run 1 \
    #--seed 0 \
    #--device cuda:0,cuda:1,cuda:2,cuda:3 \

#python tools/eval_model_verbose.py \
    #--weight1 exps/br_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,1 \
    #--train_test_splits train_test_splits/sad_splits_six.json \
    #--split_index 0 \
    #--partner_models agent_groups/all_sad.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/sba_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,1 \
    #--train_test_splits train_test_splits/sad_splits_six.json \
    #--split_index 0 \
    #--partner_models agent_groups/all_sad.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

python tools/eval_model_verbose.py \
    --weight1 exps/ch_sad_six_1_3_6_7_8_12/model_epoch1000.pthw \
    --sad_legacy 0,1 \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --split_index 0 \
    --partner_models agent_groups/all_sad.json \
    --num_game 5000 \
    --seed 0 \
    --convex_hull 1\

python tools/eval_model_verbose.py \
    --weight1 exps/ch_sad_six_1_2_8_9_10_13/model_epoch1000.pthw \
    --sad_legacy 0,1 \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --split_index 1 \
    --partner_models agent_groups/all_sad.json \
    --num_game 5000 \
    --seed 0 \
    --convex_hull 1\

python tools/eval_model_verbose.py \
    --weight1 exps/ch_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    --sad_legacy 0,1 \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --split_index 2 \
    --partner_models agent_groups/all_sad.json \
    --num_game 5000 \
    --seed 0 \
    --convex_hull 1\

#python tools/eval_model_verbose.py \
    #--weight1 exps/br_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,1 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_op.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/sba_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,1 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_op.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/ch_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,1 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_op.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/ch_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,0 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_obl.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/sba_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,0 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_obl.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

#python tools/eval_model_verbose.py \
    #--weight1 exps/ch_sad_six_3_4_6_7_11_13/model_epoch1000.pthw \
    #--sad_legacy 0,0 \
    #--all_splits 1 \
    #--partner_models agent_groups/all_obl.json \
    #--num_game 5000 \
    #--seed 0 \
    #--convex_hull 0\

