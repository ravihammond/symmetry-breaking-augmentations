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

python tools/eval_model_verbose.py \
    --weight1 exps/sba_sad_six_1_3_6_7_8_12/model_epoch1000.pthw \
    --sad_legacy 0,1 \
    --train_test_splits train_test_splits/sad_splits_six.json \
    --partner_models agent_groups/all_sad.json \
    --num_game 1 \
    --seed 0 \
    --convex_hull 1\

