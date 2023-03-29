#!/bin/bash

python policy_differences.py \
    --act_policy1 exps/sba_sad_six_1_3_6_7_8_12/model_epoch1000.pthw \
    --act_policy2 ../training_models/sad_2p_models/sad_2.pthw \
    --act_sad_legacy 0,1 \
    --base_name sba \
    --comp_policies ../training_models/sad_2p_models/sad_2.pthw \
    --comp_sad_legacy 1 \
    --comp_names sad \
    --num_game 4 \
    --num_thread 20 \
    --seed 0 \
    --verbose 1 \
    --outdir temp \

#python policy_differences.py \
    #--act_policy ../training_models/sad_2p_models/sad_2.pthw \
    #--act_sad_legacy 1 \
    #--comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw \
    #--comp_sad_legacy 0 \
    #--compare_as_base obl \
    #--num_game 1 \
    #--num_thread 20 \
    #--seed 0 \
    #--verbose 1 \

#python policy_differences.py \
    #--act_policy ../training_models/sad_2p_models/sad_2.pthw \
    #--act_sad_legacy 1 \
    #--comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw,../training_models/obl1/model0.pthw \
    #--comp_sad_legacy 0 \
    #--comp_names br,sba,obl \
    #--compare_as_base obl \
    #--num_game 1 \
    #--num_thread 20 \
    #--seed 0 \
    #--verbose 1 \

#python policy_differences.py \
    #--act_policy ../training_models/sad_2p_models/sad_2.pthw \
    #--act_sad_legacy 1 \
    #--comp_policies ../training_models/op_models/op_1.pthw \
    #--comp_sad_legacy 1 \
    #--comp_names op \
    #--num_game 5000 \
    #--num_thread 20 \
    #--seed 0 \

#python policy_differences.py \
    #--act_policy ../training_models/sad_2p_models/sad_2.pthw \
    #--act_sad_legacy 1 \
    #--rand_policy 1 \
    #--num_game 5000 \
    #--num_thread 20 \
    #--seed 0 \

#python policy_differences.py \
    #--act_policy ../training_models/obl1/model0.pthw \
    #--act_sad_legacy 1 \
    #--comp_policies exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/sba_sad_1_3_6_7_8_12/model_epoch1000.pthw,exps/ground
    #--comp_names br,sba \
    #--num_game 5000 \
    #--num_thread 20 \
    #--seed 0 \

