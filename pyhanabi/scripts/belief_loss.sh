#!/bin/bash
#--policy ../training_models/obl1/model0.pthw \
python eval_belief.py \
       --policy exps/pobl2_all_colours/model0.pthw \
       --belief_model exps/pbelief_obl1f_all_colours/model0.pthw \
       --policy_conventions conventions/all_colours.json \
       --belief_conventions conventions/all_colours.json \
       --device cuda:1 \
       --num_game 1000 \
       --batch_size 500 \
       --seed 0 \
       --loss_type response_playable \
       --colour_max 1 \
       --colour_min 0 \
       --reverse_colour 0 \
       --override 3 \
