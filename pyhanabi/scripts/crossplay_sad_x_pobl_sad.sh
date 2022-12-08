#!/bin/bash
python tools/cross_play.py \
    --root1 ../models/sad_2p_models \
    --all_models1 1 \
    --sad_legacy1 1 \
    --root2 exps/pobl1_sad_all \
    --num_parameters2 13 \

