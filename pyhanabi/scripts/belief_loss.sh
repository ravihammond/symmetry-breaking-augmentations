#!/bin/bash
python tools/belief_xent_cross_play.py \
       --policy exps/obl1/model0.pthw \
       --belief_model exps/pbelief_obl1f_all_colours/model0.pthw \
       --convention conventions/all_colours.json \
       --num_game 1 \
       --seed 0 \
