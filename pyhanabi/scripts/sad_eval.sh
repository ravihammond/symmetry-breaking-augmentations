#!/bin/bash
python tools/sad_eval_model.py \
    --weight ../models/sad_models/sad_2p_1.pthw \
    --paper sad \
    --num_game 5000 \
    #--num_thread 1 \
