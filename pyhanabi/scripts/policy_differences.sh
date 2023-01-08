#!/bin/bash

#python policy_differences.py \
    #--act_policy ../training_models/sad_2p_models/sad_13.pthw \
    #--act_sad_legacy 1 \
    #--comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw \
    #--comp_names br,sba \
    #--num_game 5000 \
    #--num_thread 20 \
    #--seed 0 \

python policy_differences.py \
    --act_policy ../training_models/obl1/model0.pthw \
    --act_sad_legacy 1 \
    --comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw \
    --comp_names br,sba \
    --num_game 5000 \
    --num_thread 20 \
    --seed 0 \

