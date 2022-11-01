#!/bin/bash
python eval_belief.py \
       --policy exps/obl1/model0.pthw \
       --belief_model exps/pbelief_obl1f_CR-P0/model0.pthw \
       --policy_conventions conventions/all_colours.json \
       --belief_conventions conventions/CR-P0.json \
       --device cuda:1 \
       --num_game 1000 \
       --batch_size 500 \
       --seed 0 \
       --xentropy_type per_card \
