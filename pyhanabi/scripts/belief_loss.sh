#!/bin/bash
python eval_belief.py \
       --policy ../training_models/obl1/model0.pthw \
       --belief_model exps/pbelief_obl1f_CR-P0/model0.pthw \
       --policy_conventions conventions/CR-P0.json \
       --belief_conventions conventions/CR-P0.json \
       --device cuda:1 \
       --num_game 1 \
       --batch_size 2 \
       --seed 0 \
       --xentropy_type response_playable \
