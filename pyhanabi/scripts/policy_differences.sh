#!/bin/bash

python policy_differences.py \
    --act_policy ../training_models/sad_2p_models/sad_5.pthw \
    --act_sad_legacy 1 \
    --comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw \
    --comp_names br,sba \
    --num_game 1 \
    --num_thread 1 \
    --seed 0 \
