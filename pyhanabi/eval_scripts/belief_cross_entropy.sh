#!/bin/bash

python tools/eval_model_verbose.py \
    --weight1 ../training_models/obl1/model0.pthw \
    --convention conventions/all_colours.json \
    --convention_index 0 \
    --override0 3 \
    --override0 3 \
    --belief_stats 1 \
    --num_game 1 \
    --belief_model exps/pbelief_obl1f_all_colours/model0.pthw \
