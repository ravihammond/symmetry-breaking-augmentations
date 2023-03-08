#!/bin/bash

# OBL Crossplay
for crossplay_index in 0 1 2 3 4
do
    python tools/save_games_multiple.py \
        --model1 obl \
        --model2 obl \
        --num_game 5000 \
        --save 1 \
        --verbose 1 \
        --crossplay 1 \
        --crossplay_index $crossplay_index 
done
