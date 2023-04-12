#!/bin/bash

# OBL Crossplay
#for crossplay_index in 0 1 2 3 4
#do
    #python tools/save_games_multiple.py \
        #--model1 obl \
        #--model2 obl \
        #--num_game 5000 \
        #--save 1 \
        #--verbose 1 \
        #--crossplay 1 \
        #--crossplay_index $crossplay_index 
#done

#OBL vs SAD Crossplay
#for crossplay_index in 2 3
#do
    #python tools/save_games_multiple.py \
        #--model1 obl \
        #--model2 sad \
        #--num_game 5000 \
        #--save 1 \
        #--verbose 1 \
        #--crossplay 1 \
        #--crossplay_index $crossplay_index 
#done

#OP vs SAD Crossplay
#for crossplay_index in 4 5 6 7
#do
    #python tools/save_games_multiple.py \
        #--model1 op \
        #--model2 sad \
        #--num_game 5000 \
        #--save 1 \
        #--verbose 1 \
        #--crossplay 1 \
        #--crossplay_index $crossplay_index 
#done

#OBL vs OP Crossplay
for crossplay_index in 2 3
do
    python tools/save_games_multiple.py \
        --model1 obl \
        --model2 op \
        --num_game 5000 \
        --save 1 \
        --verbose 1 \
        --crossplay 1 \
        --crossplay_index $crossplay_index 
done
