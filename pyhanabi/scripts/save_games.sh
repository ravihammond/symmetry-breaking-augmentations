#!/bin/bash

python save_games.py \
    --out game_data/br \
    --weight1 exps/br_sad_1_3_6_7_8_12/model_epoch1000.pthw \
    --weight2 ../training_models/sad_2p_models/sad_2.pthw \
    --player_name br_sad_six_1_3_6_7_8_12,sad_2 \
    --data_type test \
    --sad_legacy 0,1 \
    --num_game 5000 \
    --num_thread 20 \
    --seed 0 \
    --verbose 1 \
    --save 1

#python save_games.py \
    #--out game_data \
    #--weight1 ../training_models/sad_2p_models/sad_2.pthw \
    #--weight2 ../training_models/sad_2p_models/sad_2.pthw \
    #--player_name sad_2,sad_2 \
    #--data_type train \
    #--sad_legacy 1 \
    #--num_game 1 \
    #--num_thread 20 \
    #--seed 0 \
    #--verbose 1 \
    #--save 0 
