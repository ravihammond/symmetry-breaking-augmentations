#!/bin/bash

# OBL 6-7 splits
#for split_index in 6 7
#do
    #python tools/save_games_multiple.py \
        #--model1 br,sba \
        #--model2 obl \
        #--split_type six \
        #--split_index $split_index \
        #--num_game 5000 \
        #--save 1 \
        #--verbose 1
#done

# SAD Crossplay
for crossplay_index in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    python tools/save_games_multiple.py \
        --model1 sad \
        --model2 sad \
        --num_game 5000 \
        --save 1 \
        --verbose 1 \
        --crossplay 1 \
        --crossplay_index $crossplay_index 
done


