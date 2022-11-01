#!/bin/bash
python eval_belief.py \
       --policy exps/obl1/model0.pthw \
       --belief_model exps/pbelief_obl1f_all_colours/model0.pthw \
       --convention conventions/all_colours.json \
       --device cuda:1 \
       --num_game 1000 \
       --batch_size 500 \
       --seed 0 \
       --xentropy_type per_card \
